from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
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

    print("Testing Retriever With Scores...\n")

    query = "Who appoints Insurance Ombudsmen?"

    results = retrieve_with_scores(query)

    for i, (doc, score) in enumerate(results, start=1):

        print(f"\nChunk {i}")
        print("-" * 50)
        print(f"Score: {score}")
        print(doc.page_content[:500])