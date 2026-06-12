import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from retriever import retrieve_with_scores
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


def ask_question(question):

    logger.info(f"Question: {question}")

    # Retrieve relevant chunks with scores
    results = retrieve_with_scores(question)

    docs = [doc for doc, score in results]
    scores = [score for doc, score in results]

    # Calculate confidence
    average_score = sum(scores) / len(scores)

    if average_score < 0.55:
        confidence = "High"

    elif average_score < 0.75:
        confidence = "Medium"

    else:
        confidence = "Low"

    # Hallucination guard

    if confidence == "Low":

        logger.warning(
            f"Low confidence retrieval detected for question: {question}"
        )

        return (
            "I could not find reliable information in the provided documents."
            + "\n\nRetrieval Information:"
            + f"\n• Documents Retrieved: {len(docs)}"
            + f"\n• Confidence: {confidence}"
        )

    # Logging
    logger.info(f"Retrieved {len(docs)} documents")
    logger.info(f"Average Score: {average_score:.3f}")
    logger.info(f"Confidence: {confidence}")

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

    logger.info(f"Retrieved Sources: {sources}")

    # Create context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    You are an Insurance Support Assistant.

    STRICT RULES:

    1. Use ONLY the information provided in the context.
    2. Do NOT use external knowledge.
    3. Do NOT make assumptions or guesses.
    4. If the answer is partially available, answer only the available part.
    5. If the answer is not available in the context, reply exactly:

    I could not find that information in the provided documents.

    6. Do NOT mention facts that are not present in the context.
    7. Keep answers concise and factual.
    8. Do NOT explain beyond the provided information.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    try:

        response = llm.invoke(prompt)

        logger.info(f"Answer: {response.content}")

        # If answer not found, don't show sources
        if "I could not find that information" in response.content:

            return (
                response.content
                + "\n\nRetrieval Information:"
                + f"\n• Documents Retrieved: {len(docs)}"
                + f"\n• Confidence: {confidence}"
            )

        source_text = "\n".join(
            [f"• {source}" for source in sources]
        )

        final_answer = (
            response.content
            + "\n\nSources:\n"
            + source_text
            + "\n\nRetrieval Information:"
            + f"\n• Documents Retrieved: {len(docs)}"
            + f"\n• Confidence: {confidence}"
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