from rank_bm25 import BM25Okapi

from retriever import (
    get_all_documents,
    retrieve_with_scores
)


class HybridRetriever:

    def __init__(self):

        self.documents = get_all_documents()

        self.tokenized_docs = [
            doc.page_content.lower().split()
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_docs
        )

    def bm25_search(self, query, k=5):

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_results = sorted(
            zip(self.documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked_results[:k]

    def hybrid_search(self, query, k=5):

        # BM25 results
        bm25_results = self.bm25_search(
            query,
            k=k
        )

        # Vector results
        vector_results = retrieve_with_scores(
            query
        )

        merged = []

        seen = set()

        # Add BM25 results
        for doc, score in bm25_results:

            content = doc.page_content

            if content not in seen:

                merged.append(doc)

                seen.add(content)

        # Add Vector results
        for doc, score in vector_results:

            content = doc.page_content

            if content not in seen:

                merged.append(doc)

                seen.add(content)

        return merged[:k]


if __name__ == "__main__":

    print("Testing Hybrid Retrieval...")

    hybrid = HybridRetriever()

    results = hybrid.hybrid_search(
        "Who appoints Insurance Ombudsman?"
    )

    for i, doc in enumerate(
        results,
        start=1
    ):

        print(f"\nChunk {i}")

        print("-" * 50)

        print(doc.page_content[:500])