import os
from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI

from hybrid_retriever import HybridRetriever
from reranker import Reranker
from context_compressor import ContextCompressor

from config import LLM_MODEL


# Load environment variables
load_dotenv()


class ChatState(TypedDict):
    question: str
    retrieved_docs: list
    reranked_docs: list
    compressed_docs: list
    answer: str
    confidence: str


# Components
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

hybrid_retriever = HybridRetriever()

reranker = Reranker()

compressor = ContextCompressor()


# -----------------------------
# Retrieval Node
# -----------------------------
def retrieve_node(state: ChatState):

    print("\n[Retrieve Node]")

    retrieved_docs = hybrid_retriever.hybrid_search(
        state["question"],
        k=10
    )

    print(
        f"Retrieved {len(retrieved_docs)} documents"
    )

    return {
        "retrieved_docs": retrieved_docs
    }


# -----------------------------
# Reranker Node
# -----------------------------
def rerank_node(state: ChatState):

    print("\n[Reranker Node]")

    reranked_docs = reranker.rerank(
        state["question"],
        state["retrieved_docs"],
        top_k=5
    )

    print(
        f"Reranked to {len(reranked_docs)} documents"
    )

    return {
        "reranked_docs": reranked_docs
    }
def fallback_node(state: ChatState):

    return {
        "answer":
        "I could not find reliable information in the provided documents."
    }

def route_confidence(
    state: ChatState
):

    if state["confidence"] == "Low":

        return "fallback"

    return "generate"

# -----------------------------
# Compression Node
# -----------------------------
def compress_node(state: ChatState):

    print("\n[Compression Node]")

    docs = [
        doc
        for doc, score
        in state["reranked_docs"]
    ]

    before = len(docs)

    compressed_docs = compressor.compress(
        docs
    )

    after = len(compressed_docs)

    print(
        f"Compression: {before} -> {after}"
    )

    return {
        "compressed_docs": compressed_docs
    }

def confidence_node(state: ChatState):

    scores = [
        score
        for doc, score
        in state["reranked_docs"]
    ]

    average_score = sum(scores) / len(scores)

    if average_score > 3:

        confidence = "High"

    elif average_score > 1:

        confidence = "Medium"

    else:

        confidence = "Low"

    print(
        f"\n[Confidence Node] "
        f"{confidence} "
        f"(avg score={average_score:.2f})"
    )

    return {
        "confidence": confidence
    }
# -----------------------------
# Generation Node
# -----------------------------
def generate_node(state: ChatState):

    print("\n[Generation Node]")

    context = "\n\n".join(
        [
            doc.page_content
            for doc in state["compressed_docs"]
        ]
    )

    prompt = f"""
You are an Insurance Support Assistant.

STRICT RULES:

1. Use ONLY the information provided in the context.
2. Do NOT use external knowledge.
3. Do NOT make assumptions.
4. If the answer is not available, reply exactly:

I could not find that information in the provided documents.

Context:
{context}

Question:
{state["question"]}

Answer:
"""

    response = llm.invoke(
        prompt
    )

    return {
        "answer": response.content
    }


# -----------------------------
# Build Graph
# -----------------------------
graph = StateGraph(ChatState)

graph.add_node(
    "retrieve",
    retrieve_node
)

graph.add_node(
    "rerank",
    rerank_node
)

graph.add_node(
    "compress",
    compress_node
)

graph.add_node(
    "confidence",
    confidence_node
)

graph.add_node(
    "generate",
    generate_node
)

graph.add_node(
    "fallback",
    fallback_node
)

graph.set_entry_point(
    "retrieve"
)

graph.add_edge(
    "retrieve",
    "rerank"
)

graph.add_edge(
    "rerank",
    "compress"
)

graph.add_edge(
    "compress",
    "confidence"
)

graph.add_conditional_edges(
    "confidence",
    route_confidence,
    {
        "generate": "generate",
        "fallback": "fallback"
    }
)

graph.add_edge(
    "generate",
    END
)

graph.add_edge(
    "fallback",
    END
)

app = graph.compile()

# -----------------------------
# Main Loop
# -----------------------------
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