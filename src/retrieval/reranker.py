from src.config.config import RERANKER_MODEL


class Reranker:
    def __init__(self, model=None):
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(RERANKER_MODEL)
        self.model = model

    def rerank(self, query, documents, top_k=5):
        if not documents:
            return []
        scores = self.model.predict([(query, doc.page_content) for doc in documents])
        return sorted(
            [(doc, float(score)) for doc, score in zip(documents, scores)],
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
