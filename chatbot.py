import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from context_compressor import ContextCompressor
from hybrid_retriever import HybridRetriever
from reranker import Reranker

from config import LLM_MODEL
from logger import logger


# Load API key
load_dotenv()

RETRIEVAL_HEADER = "\n\nRetrieval Information:"

# In-memory conversation history
chat_history = []


# Gemini model
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# Retrieval components
hybrid_retriever = HybridRetriever()

reranker = Reranker()
compressor = ContextCompressor()

def ask_question(question):

    logger.info(f"Question: {question}")

    # Build conversation context from last 2 exchanges
    history_context = ""

    for q, a in chat_history[-2:]:

        history_context += (
            f"Previous Question: {q}\n"
            f"Previous Answer: {a}\n\n"
        )

    retrieval_query = (
        history_context
        + f"Current Question: {question}"
    )

    logger.info(
        "Using conversation history for retrieval"
    )

    # Hybrid Retrieval
    retrieved_docs = hybrid_retriever.hybrid_search(
        retrieval_query,
        k=10
    )

    # Re-ranking
    reranked_results = reranker.rerank(
        retrieval_query,
        retrieved_docs,
        top_k=5
    )

    # Debug output
    for i, (doc, score) in enumerate(
        reranked_results,
        start=1
    ):

        print(f"\nRank {i}")
        print(f"Score: {score:.4f}")
        print("-" * 50)
        print(doc.page_content[:300])

    docs = [
        doc
        for doc, score in reranked_results
    ]   

    scores = [
        score
        for doc, score in reranked_results
    ]

    before_compression = len(docs)

    docs = compressor.compress(docs)

    after_compression = len(docs)

    logger.info(
        f"Compression: {before_compression} -> {after_compression}"
    )

    # Confidence calculation
    average_score = sum(scores) / len(scores)

    if average_score > 3:
        confidence = "High"

    elif average_score > 1:
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
            + RETRIEVAL_HEADER
            + f"\n• Documents Retrieved: {len(docs)}"
            + f"\n• Confidence: {confidence}"
        )

    logger.info(
        f"Retrieved {len(docs)} documents"
    )

    logger.info(
        f"Average Score: {average_score:.3f}"
    )

    logger.info(
        f"Confidence: {confidence}"
    )

    # Extract sources
    sources = []

    for doc in docs:

        file_name = os.path.basename(
            doc.metadata.get(
                "source",
                "Unknown Source"
            )
        )

        page_number = doc.metadata.get(
            "page_label",
            doc.metadata.get(
                "page",
                "Unknown"
            )
        )

        sources.append(
            f"{file_name} (Page {page_number})"
        )

    # Remove duplicates
    sources = list(
        dict.fromkeys(sources)
    )

    logger.info(
        f"Retrieved Sources: {sources}"
    )

    # Build context
    context = "\n\n".join(
        [
            doc.page_content
            for doc in docs
        ]
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

Conversation History:
{history_context}

Context:
{context}

Question:
{question}

Answer:
"""

    try:

        response = llm.invoke(prompt)

        logger.info(
            f"Answer: {response.content}"
        )

        # Store memory
        chat_history.append(
            (
                question,
                response.content
            )
        )

        # Keep only latest 10 exchanges
        if len(chat_history) > 10:
            chat_history.pop(0)

        # If answer not found
        if (
            "I could not find that information"
            in response.content
        ):

            return (
                response.content
                + RETRIEVAL_HEADER
                + f"\n• Documents Retrieved: {len(docs)}"
                + f"\n• Confidence: {confidence}"
            )

        source_text = "\n".join(
            [
                f"• {source}"
                for source in sources
            ]
        )

        final_answer = (
            response.content
            + "\n\nSources:\n"
            + source_text
            + RETRIEVAL_HEADER
            + f"\n• Documents Retrieved: {before_compression}"
            + f"\n• Documents After Compression: {after_compression}"
            + f"\n• Confidence: {confidence}"
        )

        return final_answer

    except Exception as e:

        logger.error(
            f"Error: {str(e)}"
        )

        return (
            "An error occurred while generating the response."
        )


if __name__ == "__main__":

    while True:

        question = input(
            "\nAsk a Question: "
        )

        if question.lower() == "exit":
            break

        answer = ask_question(
            question
        )

        print("\nAnswer:\n")

        print(answer)