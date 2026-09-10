"""Semantic duplicate removal plus a context budget (no LLM call)."""

from langchain_core.documents import Document

from src.retrieval.retriever import get_embeddings


class ContextCompressor:
    def __init__(self, embeddings=None):
        self.embeddings = embeddings if embeddings is not None else get_embeddings()

    def compress(self, documents, similarity_threshold=0.85, max_chars=10000):
        if not documents:
            return []
        import numpy as np

        vectors = self.embeddings.embed_documents([doc.page_content for doc in documents])
        kept, embeddings, remaining = [], [], max_chars
        for doc, vector in zip(documents, vectors):
            vector = np.asarray(vector)
            norm = np.linalg.norm(vector)
            vector = vector / norm if norm else vector
            if any(
                float(np.dot(vector, previous)) > similarity_threshold for previous in embeddings
            ):
                continue
            if remaining <= 0:
                break
            text = doc.page_content[:remaining]
            kept.append(Document(page_content=text, metadata=dict(doc.metadata)))
            embeddings.append(vector)
            remaining -= len(text)
        return kept
