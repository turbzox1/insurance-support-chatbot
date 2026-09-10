"""Explicit, repeatable ingestion. Never deletes the vectorstore directory."""

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.config import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, VECTORSTORE_PATH
from src.retrieval.document_loader import load_documents


def ingest(data_dir=None, store=None):
    documents = load_documents(data_dir)
    if not documents:
        raise ValueError("No readable documents found; existing index was left unchanged.")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    ).split_documents(documents)
    if not chunks:
        raise ValueError("No chunks extracted; existing index was left unchanged.")
    ids = []
    for chunk in chunks:
        identity = f"{chunk.metadata}|{chunk.page_content}"
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        chunk.metadata["chunk_id"] = chunk_id
        ids.append(chunk_id)
    if store is None:
        from langchain_chroma import Chroma

        from src.retrieval.retriever import get_embeddings

        store = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=VECTORSTORE_PATH,
            embedding_function=get_embeddings(),
        )
    old_ids = set(store.get()["ids"])
    for start in range(0, len(chunks), 100):
        store.add_documents(chunks[start : start + 100], ids=ids[start : start + 100])
    stale = sorted(old_ids - set(ids))
    if stale:
        store.delete(ids=stale)
    return {"pages": len(documents), "chunks": len(chunks), "removed_stale_chunks": len(stale)}


def main():
    print(ingest())


if __name__ == "__main__":
    main()
