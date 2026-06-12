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

    def bm25_search(self, query, k=20):

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

    def hybrid_search(self, query, k=10):

        # Get more candidates from both retrievers
        bm25_results = self.bm25_search(
            query,
            k=20
        )

        vector_results = retrieve_with_scores(
            query
        )

        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}

        # BM25 contribution
        for rank, (doc, score) in enumerate(
            bm25_results,
            start=1
        ):

            content = doc.page_content

            rrf_scores[content] = (
                rrf_scores.get(content, 0)
                + 1 / (60 + rank)
            )

        # Vector contribution
        for rank, (doc, score) in enumerate(
            vector_results,
            start=1
        ):

            content = doc.page_content

            rrf_scores[content] = (
                rrf_scores.get(content, 0)
                + 1 / (60 + rank)
            )

        # Build lookup table
        doc_lookup = {}

        for doc, score in bm25_results:
            doc_lookup[doc.page_content] = doc

        for doc, score in vector_results:
            doc_lookup[doc.page_content] = doc

        # Sort by RRF score
        ranked_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        final_docs = []

        for content, score in ranked_docs[:k]:

            final_docs.append(
                doc_lookup[content]
            )

        return final_docs


if __name__ == "__main__":

    print("Testing Hybrid Retrieval...")

    hybrid = HybridRetriever()

    results = hybrid.hybrid_search(
        "Who appoints Insurance Ombudsman?",
        k=10
    )

    for i, doc in enumerate(
        results,
        start=1
    ):

        print(f"\nChunk {i}")

        print("-" * 50)

        print(doc.page_content[:500])