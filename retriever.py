from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def initialize_retriever(k=3):
    """
    Load the embedding model and existing Chroma vector database.
    Return a retriever that fetches top-k relevant chunks.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}
    )

    vectorstore = Chroma(
        persist_directory="vectorstore",
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    return retriever


# This block runs ONLY when retriever.py is executed directly
if __name__ == "__main__":

    print("Testing Retriever...")

    retriever = initialize_retriever()

    query = "What is the ombudsman process?"

    results = retriever.invoke(query)

    print("\nRetrieved Chunks:\n")

    for i, doc in enumerate(results, start=1):
        print(f"\nChunk {i}")
        print("-" * 50)
        print(doc.page_content[:500])