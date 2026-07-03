import os
from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from web_search import WebSearchAgent
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

    intent: str

    question_type: str

    history_context: str

    retrieved_docs: list

    reranked_docs: list

    compressed_docs: list

    confidence: str

    next_action: str

    web_context: str

    answer: str

    verified: bool


# Components
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

hybrid_retriever = HybridRetriever()

reranker = Reranker()

compressor = ContextCompressor()
web_search = WebSearchAgent()

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

def query_analysis_node(state: ChatState):

    print("\n[Query Analysis Node]")

    prompt = f"""
You are a query analysis agent.

Your ONLY task is to determine whether the rewritten question is complete.

Rewritten Question:
{state["rewritten_question"]}

Conversation History:
{state["history_context"]}

Rules:

1. If the rewritten question is complete and understandable on its own, return:

COMPLETE

2. If important information is still missing or the question is ambiguous, return:

AMBIGUOUS

Return ONLY one word.

Decision:
"""

    response = llm.invoke(prompt)

    decision = response.content.strip().upper()

    if decision not in [
        "COMPLETE",
        "AMBIGUOUS"
    ]:
        decision = "COMPLETE"

    print(f"Question Type: {decision}")

    return {
        "question_type": decision
    }

def intent_detection_node(state: ChatState):

    print("\n[Intent Detection Node]")

    prompt = f"""
You are an intent classification agent.

Question:
{state["rewritten_question"]}

Classify the user's intent into ONLY one of the following:

KNOWLEDGE
LIVE_INFORMATION

LIVE_INFORMATION includes questions asking for:
- latest
- recent
- current
- today
- this week
- new announcements
- live updates
- ongoing events

Everything else is KNOWLEDGE.

Return ONLY one word.

Intent:
"""

    response = llm.invoke(prompt)

    intent = response.content.strip().upper()

    if intent not in [
        "KNOWLEDGE",
        "LIVE_INFORMATION"
    ]:
        intent = "KNOWLEDGE"

    print(f"Intent: {intent}")

    return {
        "intent": intent
    }

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

    print("\n[Confidence Node]")

    scores = [
        score
        for _, score in state["reranked_docs"]
    ]

    top_score = scores[0]

    if top_score >= 2.0:
        confidence = "HIGH"

    elif top_score >= 0.5:
        confidence = "MEDIUM"

    else:
        confidence = "LOW"

    print(f"Top Score : {top_score:.2f}")
    print(f"Confidence: {confidence}")

    return {
        "confidence": confidence
    }

def routing_node(state: ChatState):

    print("\n[Routing Node]")

    question_type = state["question_type"]
    confidence = state["confidence"]

    if question_type == "AMBIGUOUS":

        decision = "CLARIFY"

    elif confidence == "LOW":

        decision = "WEB_SEARCH"

    else:

        decision = "GENERATE"

    print(f"Question Type : {question_type}")
    print(f"Confidence    : {confidence}")
    print(f"Decision      : {decision}")

    return {
        "next_action": decision
    }

def route_next_action(state: ChatState):

    return state["next_action"]

def route_verification(state: ChatState):

    if state["verified"]:

        return "save"

    return "fallback"

def clarification_node(state: ChatState):
    print("\n[Clarification Node]")

    prompt = f"""
You are an Insurance Support Assistant.

The user's question is ambiguous or missing important information.

Conversation History:
{state["history_context"]}

Current Question:
{state["question"]}

Ask ONE short follow-up question that will help answer the user's query.

Rules:
1. Do NOT answer the question.
2. Do NOT make assumptions.
3. Ask only one clarification question.
4. Keep it polite and concise.

Clarification Question:
"""

    response = llm.invoke(prompt)

    clarification = response.content.strip()

    print(clarification)

    return {
        "answer": clarification
    }

def web_search_node(state: ChatState):

    print("\n[Web Search Node]")

    results = web_search.search(
        state["rewritten_question"],
        max_results=3
    )

    web_context = ""

    for result in results:

        web_context += (
            f"Title: {result['title']}\n"
            f"Content: {result['content']}\n"
            f"Source: {result['url']}\n\n"
        )

    print(
        f"Retrieved {len(results)} web results"
    )

    return {
        "web_context": web_context
    }

# -----------------------------
# Generation Node
# -----------------------------
def generate_node(state: ChatState):

    print("\n[Generation Node]")

    rag_context = "\n\n".join(
        [
            doc.page_content
            for doc in state["compressed_docs"]
        ]
    )

    context = rag_context

    if state.get("web_context"):

        context += (
            "\n\nExternal Web Information:\n\n"
            + state["web_context"]
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

def verification_node(state: ChatState):

    print("\n[Verification Node]")

    rag_context = "\n\n".join(
        [
            doc.page_content
            for doc in state["compressed_docs"]
        ]
    )

    context = rag_context

    if state.get("web_context"):

        context += (
            "\n\nExternal Web Information:\n\n"
            + state["web_context"]
        )

    prompt = f"""
You are a response verification agent.

Question:
{state["question"]}

Available Context:
{context}

Generated Answer:
{state["answer"]}

Your task is to verify whether the generated answer is fully supported by the provided context.

Rules:

1. Return VERIFIED if the answer is supported.

2. Return NOT_VERIFIED if the answer contains unsupported claims, hallucinations, or information missing from the context.

Return ONLY one word.

Decision:
"""

    response = llm.invoke(prompt)

    decision = response.content.strip().upper()

    verified = decision == "VERIFIED"

    print(f"Verification: {decision}")

    return {
        "verified": verified
    }

def verification_failed_node(state: ChatState):

    print("\n[Verification Failed]")

    return {

        "answer":
        (
            "I could not confidently verify the generated answer "
            "using the available information."
        )

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
    "query_analysis",
    query_analysis_node
)

graph.add_node(
    "intent_detection",
    intent_detection_node
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
    "router",
    routing_node
)

graph.add_node(
    "clarify",
    clarification_node
)

graph.add_node(
    "verification_failed",
    verification_failed_node
)

graph.add_node(
    "web_search",
    web_search_node
)

graph.add_node(
    "generate",
    generate_node
)

graph.add_node(
    "verify",
    verification_node
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
    "query_analysis"
)

graph.add_edge(
    "query_analysis",
    "intent_detection"
)

graph.add_edge(
    "intent_detection",
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
    "router"
)

graph.add_conditional_edges(
    "verify",
    route_verification,
    {
        "save": "save_history",
        "fallback": "verification_failed"
    }
)

graph.add_conditional_edges(
    "router",
    route_next_action,
    {
        "GENERATE": "generate",
        "CLARIFY": "clarify",
        "WEB_SEARCH": "web_search"
    }
)

graph.add_edge(
    "generate",
    "verify"
)

graph.add_edge(
    "save_history",
    END
)

graph.add_edge(
    "clarify",
    END
)

graph.add_edge(
    "web_search",
    "generate"
)

graph.add_edge(
    "verification_failed",
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