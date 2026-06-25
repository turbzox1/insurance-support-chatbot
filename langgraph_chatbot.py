import os
from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from langchain_google_genai import ChatGoogleGenerativeAI

from hybrid_retriever import HybridRetriever
from reranker import Reranker
from context_compressor import ContextCompressor

from config import LLM_MODEL
from history_manager import (
    load_history,
    add_message
)

# Load environment variables
load_dotenv()


class ChatState(TypedDict):
    question: str
    rewritten_question: str
    history_context: str
    retrieved_docs: list
    reranked_docs: list
    compressed_docs: list
    answer: str
    confidence: str
    rewritten_question: str


# Components
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

hybrid_retriever = HybridRetriever()

reranker = Reranker()

compressor = ContextCompressor()

def history_node(state: ChatState):

    print("\n[History Node]")

    history = load_history()

    history = history[-1:]

    history_context = ""

    for item in history:

        history_context += (
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}\n\n"
        )

    print(
        f"Loaded {len(history)} conversations"
    )

    return {
        "history_context": history_context
    }

def rewrite_query_node(state: ChatState):

    print("\n[Query Rewrite Node]")

    prompt = f"""
You are a query rewriting assistant.

Your task is to rewrite the user's question ONLY if it depends on previous conversation.

Conversation History:
{state["history_context"]}

Current Question:
{state["question"]}

Rules:

1. If the current question is already complete and understandable, return it exactly as it is.
2. If the question contains words like "it", "they", "them", "that", "this", etc., rewrite it into a complete standalone question using the conversation history.
3. Do NOT answer the question.
4. Return ONLY the rewritten question.

Rewritten Question:
"""

    response = llm.invoke(prompt)

    rewritten_question = response.content.strip()

    print("Original Question:")
    print(state["question"])

    print("\nRewritten Question:")
    print(rewritten_question)

    return {
        "rewritten_question": rewritten_question
    }

# -----------------------------
# Retrieval Node
# -----------------------------
def retrieve_node(state: ChatState):

    print("\n[Retrieve Node]")

    retrieval_query = state["rewritten_question"]
    print("\nRetrieval Query:")
    print(retrieval_query)
    retrieved_docs = hybrid_retriever.hybrid_search(
        retrieval_query,
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

    print(
        "\n[Confidence Node] Forced High"
    )

    return {
        "confidence": "High"
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

Conversation History:
{state["history_context"]}

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

def save_history_node(state: ChatState):

    print("\n[Save History Node]")

    add_message(
        state["question"],
        state["answer"]
    )

    return {}

# -----------------------------
# Build Graph
# -----------------------------
graph = StateGraph(ChatState)

graph.add_node(
    "retrieve",
    retrieve_node
)

graph.add_node(
    "history",
    history_node
)

graph.add_node(
    "rewrite_query",
    rewrite_query_node
)

graph.add_node(
    "save_history",
    save_history_node
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

graph.set_entry_point(
    "history"
)

graph.add_edge(
    "history",
    "rewrite_query"
)

graph.add_edge(
    "rewrite_query",
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

graph.add_edge(
    "confidence",
    "generate"
)

graph.add_edge(
    "generate",
    "save_history"
)

graph.add_edge(
    "save_history",
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