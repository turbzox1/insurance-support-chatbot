import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src.chatbot.langgraph_chatbot import NO_EVIDENCE, UNVERIFIED, Chatbot, Components

DOC = Document(
    page_content="Zero Co-pay is an add-on. It cannot be bought separately.",
    metadata={"source": "pdfs/Zero Co-Pay.pdf", "page": 0},
)


class Model:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            content=reply, usage_metadata={"input_tokens": 10, "output_tokens": 5}
        )


class Retriever:
    def __init__(self, docs=None):
        self.docs = [DOC] if docs is None else docs
        self.queries = []

    def hybrid_search(self, query, k):
        self.queries.append(query)
        return self.docs


class Ranker:
    def rerank(self, query, docs, top_k):
        return [(doc, 6.0) for doc in docs]


class Compressor:
    def compress(self, docs):
        return docs


class Web:
    def search(self, query, max_results):
        return [
            {
                "content": "An official update about insurance.",
                "url": "https://irdai.gov.in/",
                "title": "IRDAI",
            }
        ]


def make_bot(replies=(), docs=None, web=None):
    model, retriever = Model(replies), Retriever(docs)
    bot = Chatbot(
        Components(
            llm=model, retriever=retriever, reranker=Ranker(), compressor=Compressor(), web=web
        )
    )
    return bot, model, retriever


def generated(answer="Zero Co-pay cannot be bought separately.", ids=None):
    return json.dumps({"answer": answer, "source_ids": ["S1"] if ids is None else ids})


@pytest.mark.parametrize(
    "question,route",
    [
        ("Write a Python program.", "NON_INSURANCE"),
        ("Give me a cake recipe.", "NON_INSURANCE"),
        ("Can I claim this?", "CLARIFY"),
        ("Is it covered?", "CLARIFY"),
        ("What about that policy?", "CLARIFY"),
    ],
)
def test_cheap_routes(question, route):
    bot, model, retriever = make_bot()
    result = bot.invoke({"question": question})
    assert result["route"] == route
    assert result["llm_calls"] == 0 and not model.prompts and not retriever.queries
    assert result["sources"] == [] and result["verified"] is None


def test_rag_generation_verification_and_metadata():
    bot, model, _ = make_bot([generated(), "VERIFIED"])
    result = bot.invoke({"question": "What is Zero Co-pay?"})
    assert result["route"] == "RAG" and result["verified"] is True
    assert result["llm_calls"] == 2 and result["token_usage"]["input_tokens"] == 20
    assert result["sources"][0]["page"] == 1
    assert result["sources"][0]["source"] == "pdfs/Zero Co-Pay.pdf"
    assert len(result["history"]) == 1
    assert set(result["workflow"]) == set(result["node_latency"])
    assert result["latency_seconds"] >= sum(result["node_latency"].values())
    assert "sources are rendered separately" in model.prompts[0]


def test_followup_and_session_isolation():
    analysis = json.dumps(
        {
            "rewritten_question": "Can I buy Zero Co-pay separately?",
            "domain": "INSURANCE",
            "intent": "KNOWLEDGE",
            "question_type": "COMPLETE",
        }
    )
    bot, _, retriever = make_bot([generated(), "VERIFIED", analysis, generated(), "VERIFIED"])
    first = bot.invoke({"question": "What is Zero Co-pay?"})
    followup = bot.invoke({"question": "Can I buy it separately?", "history": first["history"]})
    assert followup["llm_calls"] == 3
    assert retriever.queries[-1] == "Can I buy Zero Co-pay separately?"
    other = bot.invoke({"question": "Can I buy it separately?"})
    assert other["route"] == "CLARIFY" and other["history"] == []
    assert len(first["history"]) == 1


def test_web_bypasses_retrieval_and_keeps_real_url():
    bot, _, retriever = make_bot(
        [generated("An official insurance update."), "VERIFIED"], web=Web()
    )
    result = bot.invoke({"question": "What are the latest IRDAI updates?"})
    assert result["route"] == "WEB_SEARCH"
    assert not retriever.queries and result["verified"]
    assert result["sources"][0]["source"] == "https://irdai.gov.in/"
    assert result["llm_calls"] == 2


def test_web_failure_is_visible_and_no_generation():
    class BrokenWeb:
        def search(self, *args, **kwargs):
            raise ConnectionError("secret must not leak")

    bot, model, _ = make_bot(web=BrokenWeb())
    result = bot.invoke({"question": "Who is the current IRDAI Chairperson?"})
    assert result["errors"] == ["web_search: ConnectionError"]
    assert "secret" not in result["answer"]
    assert not model.prompts and not result["sources"]


def test_empty_retrieval_is_not_a_model_call():
    bot, model, _ = make_bot(docs=[])
    result = bot.invoke({"question": "Explain thispolicy insurance"})
    assert result["answer"] == NO_EVIDENCE
    assert result["confidence"] == "LOW" and not model.prompts


@pytest.mark.parametrize(
    "reply",
    [
        generated(ids=["invented"]),
        generated("Claim is guaranteed [99]."),
        generated("See https://invented.example"),
    ],
)
def test_invalid_citations_are_rejected(reply):
    bot, _, _ = make_bot([reply])
    result = bot.invoke({"question": "What is Zero Co-pay?"})
    assert result["answer"] == UNVERIFIED and result["verified"] is False
    assert result["sources"] == [] and result["history"] == []


@pytest.mark.parametrize("decision", ["NOT_VERIFIED", "VERIFIED with caveats", ""])
def test_verification_fails_closed(decision):
    bot, _, _ = make_bot([generated(), decision])
    result = bot.invoke({"question": "What is Zero Co-pay?"})
    assert result["verified"] is False and result["sources"] == []
    assert result["answer"] == UNVERIFIED


@pytest.mark.parametrize("reply", ["not json", RuntimeError("quota exhausted private detail")])
def test_generation_failure(reply):
    bot, _, _ = make_bot([reply])
    result = bot.invoke({"question": "What is Zero Co-pay?"})
    assert result["errors"] and result["llm_calls"] == 1
    assert not result["sources"] and "private detail" not in result["answer"]


def test_unfamiliar_domain_uses_one_combined_analysis():
    analysis = json.dumps(
        {
            "rewritten_question": "Tell me about gardening",
            "domain": "NON_INSURANCE",
            "intent": "KNOWLEDGE",
            "question_type": "COMPLETE",
        }
    )
    bot, _, retriever = make_bot([analysis])
    result = bot.invoke({"question": "Tell me about gardening"})
    assert result["route"] == "NON_INSURANCE" and result["llm_calls"] == 1
    assert not retriever.queries


def test_no_keys_needed_to_import_or_clarify():
    result = Chatbot().invoke({"question": "Can I claim this?"})
    assert result["route"] == "CLARIFY"


@pytest.mark.parametrize("question", ["", "   ", None, "x" * 4001])
def test_input_validation(question):
    with pytest.raises(ValueError):
        Chatbot().invoke({"question": question})


def test_stale_fields_are_not_reused():
    bot, _, _ = make_bot()
    result = bot.invoke(
        {
            "question": "Write a Python program.",
            "sources": [DOC],
            "evidence": [DOC],
            "errors": ["old error"],
            "route": "RAG",
        }
    )
    assert result["sources"] == [] and result["errors"] == []
