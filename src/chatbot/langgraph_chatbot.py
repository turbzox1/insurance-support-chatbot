import os
from typing import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from src.services.web_search import WebSearchAgent
from langchain_google_genai import ChatGoogleGenerativeAI

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.context_compressor import ContextCompressor

from src.config.config import LLM_MODEL
from src.services.history_manager import (
    load_history,
    add_message
)

# Load environment variables
load_dotenv()


class ChatState(TypedDict):

    question: str

    rewritten_question: str

    intent: str

    domain: str

    question_type: str

    history_context: str

    retrieved_docs: list

    reranked_docs: list

    compressed_docs: list

    confidence: str

    next_action: str

    web_context: str

    top_score: float

    average_score: float

    sources: list

    workflow: list

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

    # Keep only recent conversations
    history = history[-5:]

    history_context = ""

    for item in history:

        answer = item["answer"]

        # Skip fallback answers
        if (
            "I could not find" in answer
            or
            "I could not confidently verify" in answer
        ):
            continue

        history_context += (
            f"Question: {item['question']}\n"
            f"Answer: {answer}\n\n"
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

    question = state["rewritten_question"].lower()

    ambiguous_patterns = [
        "it",
        "them",
        "they",
        "this",
        "that",
        "these",
        "those"
    ]

    words = question.replace("?", "").split()

    if any(word in ambiguous_patterns for word in words):

        question_type = "AMBIGUOUS"

    else:

        question_type = "COMPLETE"

    print(f"Question Type: {question_type}")

    return {
        "question_type": question_type
    }

def intent_detection_node(state: ChatState):

    print("\n[Intent Detection Node]")

    question = state["rewritten_question"].lower()

    live_keywords = [

        "latest",

        "today",

        "current",

        "recent",

        "new",

        "this week",

        "this month",

        "news",

        "update",

        "updates",

        "announcement",

        "notifications"

    ]

    if any(keyword in question for keyword in live_keywords):

        intent = "LIVE_INFORMATION"

    else:

        intent = "KNOWLEDGE"

    print(f"Intent: {intent}")

    return {
        "intent": intent
    }

def domain_detection_node(state: ChatState):

    print("\n[Domain Detection Node]")

    prompt = f"""
You are a classifier.

Determine whether the following question is related to insurance.

Question:
{state["rewritten_question"]}

Return ONLY one word:

INSURANCE
or
NON_INSURANCE
"""

    response = llm.invoke(prompt)

    domain = response.content.strip().upper()

    print(f"Domain: {domain}")

    return {
        "domain": domain
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

    print("\nRetrieved metadata:")

    for i, doc in enumerate(retrieved_docs):
        print(f"\nDocument {i+1}")
        print(doc.metadata)

    print(f"Retrieved {len(retrieved_docs)} documents")

    # ===== DEBUG START =====
    print("\nTop retrieved documents:\n")

    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"\n----- Document {i} -----")
        print(doc.page_content[:300])   # Print first 300 characters
    # ===== DEBUG END =====

    return {
        "retrieved_docs": retrieved_docs
    }

# -----------------------------
# Reranker Node
# -----------------------------
def rerank_node(state: ChatState):

    print("\n[Reranker Node]")

    reranked_docs = reranker.rerank(
        state["rewritten_question"],
        state["retrieved_docs"],
        top_k=5
    )

    print("\nTop reranked documents:\n")

    for i, item in enumerate(reranked_docs, start=1):
        doc, score = item
        print(f"\nRank {i} | Score: {score:.4f}")
        print(doc.page_content[:300])

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

    if not scores:
        confidence = "LOW"
        print("No reranked documents found.")
        return {"confidence": confidence}

    top_score = scores[0]

    top_k = scores[:3]
    avg_score = sum(top_k) / len(top_k)

    if avg_score >= 5.0:
        confidence = "HIGH"

    elif avg_score >= 3.0:
        confidence = "MEDIUM"

    else:
        confidence = "LOW"

    print(f"Top Score      : {top_score:.2f}")
    print(f"Average Top 3  : {avg_score:.2f}")
    print(f"Confidence     : {confidence}")

    return {
    "confidence": confidence,
    "top_score": top_score,
    "average_score": avg_score
}

def routing_node(state: ChatState):

    print("\n[Routing Node]")

    domain = state["domain"]
    question_type = state["question_type"]
    intent = state["intent"]
    confidence = state["confidence"]

    if domain == "NON_INSURANCE":

        decision = "NON_INSURANCE"

    elif question_type == "AMBIGUOUS":

        decision = "CLARIFY"

    elif intent == "LIVE_INFORMATION":

        decision = "WEB_SEARCH"

    elif confidence == "LOW":

        decision = "WEB_SEARCH"

    else:

        decision = "GENERATE"

    print(f"Domain         : {domain}")
    print(f"Question Type  : {question_type}")
    print(f"Intent         : {intent}")
    print(f"Confidence     : {confidence}")
    print(f"Decision       : {decision}")

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

def non_insurance_node(state: ChatState):

    return {
        "answer":
        (
            "This chatbot is designed to answer insurance-related "
            "questions only. Please ask a question related to "
            "insurance policies, claims, grievances, IRDAI, or "
            "insurance regulations."
        )
    }

# -----------------------------
# Generation Node
# -----------------------------
def generate_node(state: ChatState):

    print("\n[Generation Node]")

    print("=" * 80)
    print("DOCUMENTS SENT TO LLM")
    print("=" * 80)

    for i, doc in enumerate(state["compressed_docs"], start=1):
        print(f"\nDocument {i}")
        print("-" * 60)
        print(doc.page_content[:1000])

    print("=" * 80)

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

Your job is to answer insurance-related questions using ONLY the provided context.

========================
RULES
========================

1. Use ONLY the provided context.
2. Never use outside knowledge.
3. Never guess or infer missing information.
4. If the answer is not present in the context, reply exactly:

I could not find that information in the provided documents.

5. Answer in clear, professional language.
6. Use bullet points whenever they improve readability.
7. Preserve important names, numbers, limits, dates and rules exactly as written.
8. Do not mention "according to the context" or "based on the document".
9. Do not mention that you are an AI assistant.
10. If multiple pieces of information are available, combine them into one complete answer.

========================
CONVERSATION HISTORY
========================

{state["history_context"]}

========================
CONTEXT
========================

{context}

========================
QUESTION
========================

{state["question"]}

========================
ANSWER
========================
"""

    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(prompt)
    print("=" * 80)

    response = llm.invoke(prompt)

    # -----------------------------------
    # Build source list
    # -----------------------------------

    sources = []
    seen = set()

    for doc in state["compressed_docs"]:

        file_name = os.path.basename(
            doc.metadata.get("source", "Unknown")
        )

        page = doc.metadata.get("page")

        if page is not None:
            page += 1  # Convert to human-readable page number

        source = (file_name, page)

        if source not in seen:
            seen.add(source)
            sources.append(source)

    # -----------------------------------
    # Append sources to answer
    # -----------------------------------

    answer = response.content.strip()

    workflow = [

        "History Loaded",

        "Query Rewritten",

        f"Intent: {state['intent']}",

        "Hybrid Retrieval",

        "Cross Encoder Reranking",

        "Context Compression",

        f"Confidence: {state['confidence']}",

        "Response Generated"

    ]

    return {
        "answer": answer,
        "sources": sources,
        "workflow": workflow
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

    question = state["question"].strip()
    answer = state["answer"].strip()

    # Don't save terminal commands
    if question.startswith("python "):
        print("Skipped: terminal command")
        return {}

    # Don't save fallback answers
    if (
        "I could not find" in answer
        or
        "I could not confidently verify" in answer
    ):
        print("Skipped: fallback answer")
        return {}

    add_message(
        question,
        answer
    )

    print("Conversation saved.")

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
    "domain_detection",
    domain_detection_node
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
    "non_insurance",
    non_insurance_node
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
    "domain_detection"
)

graph.add_edge(
    "domain_detection",
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
        "WEB_SEARCH": "web_search",
        "NON_INSURANCE": "non_insurance"
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
    "non_insurance",
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

        question = input("\nAsk a Question: ")

        if question.lower() == "exit":
            break

        result = app.invoke(
            {
                "question": question
            }
        )

        print("\nAnswer:\n")
        print(result["answer"])

        print("\nConfidence:")
        print(result.get("confidence", "N/A"))

        print("\nIntent:")
        print(result.get("intent", "N/A"))

        print("\nQuestion Type:")
        print(result.get("question_type", "N/A"))

        print("\nVerified:")
        print(result.get("verified", "N/A"))

        print("\nSources:")
        print(result.get("sources", []))

        print("\nWorkflow:")
        print(result.get("workflow", []))