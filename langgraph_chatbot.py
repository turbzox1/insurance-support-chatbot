import os
from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI

from hybrid_retriever import HybridRetriever
from config import LLM_MODEL


# Load environment variables
load_dotenv()


class ChatState(TypedDict):
    question: str
    context: str
    answer: str


# Gemini model
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# Hybrid Retriever
hybrid_retriever = HybridRetriever()


def retrieve_node(state: ChatState):

    results = hybrid_retriever.hybrid_search(
        state["question"],
        k=5
    )

    context = "\n\n".join(
        [
            doc.page_content
            for doc in results
        ]
    )

    return {
        "context": context
    }


def generate_node(state: ChatState):

    prompt = f"""
You are an Insurance Support Assistant.

STRICT RULES:

1. Use ONLY the information provided in the context.
2. Do NOT use external knowledge.
3. Do NOT make assumptions.
4. If the answer is not available, reply exactly:

I could not find that information in the provided documents.

Context:
{state["context"]}

Question:
{state["question"]}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# Build Graph
graph = StateGraph(ChatState)

graph.add_node(
    "retrieve",
    retrieve_node
)

graph.add_node(
    "generate",
    generate_node
)

graph.set_entry_point(
    "retrieve"
)

graph.add_edge(
    "retrieve",
    "generate"
)

graph.add_edge(
    "generate",
    END
)

app = graph.compile()


if __name__ == "__main__":

    while True:

        question = input(
            "\nAsk a Question: "
        )

        if question.lower() == "exit":
            break

        result = app.invoke(
            {
                "question": question
            }
        )

        print("\nAnswer:\n")

        print(
            result["answer"]
        )