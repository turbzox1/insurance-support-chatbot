from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config.config import (
    EMBEDDING_MODEL,
    VECTORSTORE_PATH,
    TOP_K
)


def load_vectorstore():
    """
    Load embedding model and Chroma vector database.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embedding_model
    )

    return vectorstore

def get_all_documents():

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embedding_model
    )

    data = vectorstore.get()

    documents = []

    for text, metadata in zip(
        data["documents"],
        data["metadatas"]
    ):

        from langchain_core.documents import Document

        documents.append(
            Document(
                page_content=text,
                metadata=metadata
            )
        )

    return documents

def initialize_retriever(k=TOP_K):
    """
    Standard retriever for compatibility.
    """

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever


def retrieve_with_scores(query, k=TOP_K):
    """
    Returns:
    [
        (Document, score),
        (Document, score),
        ...
    ]
    """

    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k
    )

    return results


if __name__ == "__main__":

    query = "What is the role of the Insurance Ombudsman?"

    results = retrieve_with_scores(query, k=20)

    print(f"\nTop {len(results)} Vector Search Results\n")

    for i, (doc, score) in enumerate(results, start=1):

        print("=" * 80)
        print(f"Rank {i}")
        print(f"Distance: {score}")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Page: {doc.metadata.get('page')}")
        print("-" * 80)
        print(doc.page_content[:500])