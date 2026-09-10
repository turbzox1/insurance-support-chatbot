"""BM25 + vector retrieval, combined using reciprocal rank fusion."""

import re

from rank_bm25 import BM25Okapi

from src.retrieval.retriever import get_all_documents, load_vectorstore


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def document_key(doc):
    return doc.metadata.get("chunk_id") or (
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.page_content,
    )


class HybridRetriever:
    def __init__(self, store=None):
        self.store = store if store is not None else load_vectorstore()
        self.documents = get_all_documents(self.store)
        self.tokenized_docs = [tokenize(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs) if any(self.tokenized_docs) else None

    def bm25_search(self, query, k=20):
        if self.bm25 is None:
            return []
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.documents, scores, self.tokenized_docs), key=lambda x: x[1], reverse=True
        )
        return [
            (doc, float(score)) for doc, score, words in ranked if set(tokens).intersection(words)
        ][:k]

    def hybrid_search(self, query, k=10):
        if not self.documents or not query.strip():
            return []
        rankings = [
            self.bm25_search(query, max(20, k)),
            self.store.similarity_search_with_score(query, k=max(20, k)),
        ]
        scores, lookup = {}, {}
        for ranking in rankings:
            seen = set()
            for rank, (doc, _) in enumerate(ranking, 1):
                key = document_key(doc)
                if key in seen:
                    continue
                seen.add(key)
                scores[key] = scores.get(key, 0) + 1 / (60 + rank)
                lookup[key] = doc
        return [lookup[key] for key in sorted(scores, key=scores.get, reverse=True)[:k]]
