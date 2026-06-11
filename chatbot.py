import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from retriever import initialize_retriever
from config import LLM_MODEL
from logger import logger


# Load API key
load_dotenv()


# Gemini model
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)


# Load retriever
retriever = initialize_retriever()


def ask_question(question):

    logger.info(f"Question: {question}")

    # Retrieve relevant chunks
    docs = retriever.invoke(question)

    # Extract source information
    sources = []

    for doc in docs:

        file_name = os.path.basename(
            doc.metadata.get("source", "Unknown Source")
        )

        page_number = doc.metadata.get(
            "page_label",
            doc.metadata.get("page", "Unknown")
        )

        sources.append(
            f"{file_name} (Page {page_number})"
        )

    # Remove duplicate sources
    sources = list(dict.fromkeys(sources))

    logger.info(f"Retrieved {len(docs)} documents")
    logger.info(f"Retrieved Sources: {sources}")

    # Create context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an Insurance Support Assistant.

Answer ONLY using the information present in the context below.

If the answer is not available in the context, reply exactly:

I could not find that information in the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""

    try:

        response = llm.invoke(prompt)

        logger.info(f"Answer: {response.content}")

        # Don't show sources if answer not found
        if "I could not find that information" in response.content:
            return response.content

        source_text = "\n".join(
            [f"• {source}" for source in sources]
        )

        final_answer = (
            response.content
            + "\n\nSources:\n"
            + source_text
        )

        return final_answer

    except Exception as e:

        logger.error(f"Error: {str(e)}")

        return "An error occurred while generating the response."


if __name__ == "__main__":

    while True:

        question = input("\nAsk a Question: ")

        if question.lower() == "exit":
            break

        answer = ask_question(question)

        print("\nAnswer:\n")
        print(answer)