"""One cached embedding model and Chroma handle per process."""

from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from src.config.config import COLLECTION_NAME, EMBEDDING_MODEL, TOP_K, VECTORSTORE_PATH


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def load_vectorstore():
    from langchain_chroma import Chroma

    if not Path(VECTORSTORE_PATH, "chroma.sqlite3").exists():
        raise RuntimeError("Knowledge base missing. Run: python -m src.retrieval.ingest")
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_PATH,
        embedding_function=get_embeddings(),
    )


def get_all_documents(store=None):
    store = store if store is not None else load_vectorstore()
    data = store.get()
    return [
        Document(page_content=text, metadata={**(metadata or {}), "chunk_id": chunk_id})
        for chunk_id, text, metadata in zip(data["ids"], data["documents"], data["metadatas"])
    ]


def initialize_retriever(k=TOP_K):
    return load_vectorstore().as_retriever(search_kwargs={"k": k})


def retrieve_with_scores(query, k=TOP_K):
    return load_vectorstore().similarity_search_with_score(query, k=k)
