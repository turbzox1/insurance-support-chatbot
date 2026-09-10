from types import SimpleNamespace

import pytest

from src.evaluation.evaluation import load_dataset, run_langsmith, score_output, summarize
from src.evaluation.judge import LLMJudge

CASE = {
    "question": "What is Zero Co-pay?",
    "ground_truth": "An add-on.",
    "expected_route": "RAG",
    "expected_source": "Zero Co-Pay.pdf",
}
OUTPUT = {
    "answer": "An add-on.",
    "route": "RAG",
    "sources": [{"content": "An add-on."}],
    "retrieved": [{"source": "pdfs/Zero Co-Pay.pdf", "content": "An add-on."}],
    "latency_seconds": 0.2,
    "errors": [],
    "llm_calls": 2,
}


def test_benchmark_has_categories_and_grounded_rag_cases():
    rows = load_dataset()
    assert len(rows) >= 40
    assert {row["expected_route"] for row in rows} == {
        "RAG",
        "WEB_SEARCH",
        "CLARIFY",
        "NON_INSURANCE",
    }
    assert all(
        row["ground_truth"] and row["expected_source"]
        for row in rows
        if row["expected_route"] == "RAG"
    )


def test_deterministic_scores_do_not_invent_semantic_metrics():
    metrics = score_output(CASE, OUTPUT)
    assert metrics["routing_accuracy"] == 1 and metrics["source_hit"] == 1
    assert metrics["correctness"] is None and metrics["groundedness"] is None
    assert metrics["error"] == 0


def test_no_retrieval_is_a_miss_not_success():
    metrics = score_output(CASE, {**OUTPUT, "retrieved": [], "route": "WEB_SEARCH"})
    assert metrics["source_hit"] == 0 and metrics["routing_accuracy"] == 0


def test_judge_uses_one_call_and_nulls_unavailable_metrics():
    class Model:
        calls = 0

        def invoke(self, prompt, config):
            self.calls += 1
            return SimpleNamespace(
                content='{"correctness": 0.8, "groundedness": 0.5, "retrieval_relevance": 1, "rationale": "Some claims unsupported."}'
            )

    model = Model()
    scores = LLMJudge(model).evaluate(CASE, OUTPUT)
    assert scores["hallucination"] == 0.5 and model.calls == 1
    scores = LLMJudge(model).evaluate(
        {**CASE, "ground_truth": ""}, {**OUTPUT, "route": "WEB_SEARCH", "sources": []}
    )
    assert scores["correctness"] is None and scores["groundedness"] is None
    assert scores["retrieval_relevance"] is None


def test_judge_errors_are_reported_without_scores():
    class BrokenJudge:
        def evaluate(self, case, output):
            raise ValueError("bad output")

    scores = score_output(CASE, OUTPUT, BrokenJudge())
    assert scores["judge_error"] == "ValueError" and scores["correctness"] is None


def test_summary_reports_denominators():
    row = {"metrics": score_output(CASE, OUTPUT), "output": OUTPUT}
    result = summarize([row])
    assert result["metrics"]["correctness"] == {"mean": None, "evaluated": 0}
    assert result["metrics"]["source_hit"]["evaluated"] == 1


def test_langsmith_experiment_contract(monkeypatch):
    import langsmith

    from src.chatbot.langgraph_chatbot import Chatbot

    captured = {}

    class Client:
        def create_dataset(self, **kwargs):
            return SimpleNamespace(id="dataset")

        def create_examples(self, **kwargs):
            captured.update(kwargs)

        def evaluate(self, target, **kwargs):
            output = target({"question": "Write a Python program."})
            reference = {"expected_route": "NON_INSURANCE"}
            scores = kwargs["evaluators"][0](output, reference)
            assert scores["results"][0]["score"] == 1
            assert kwargs["max_concurrency"] == 1
            return "experiment"

    monkeypatch.setenv("LANGSMITH_API_KEY", "test-placeholder")
    monkeypatch.setattr(langsmith, "Client", Client)
    assert run_langsmith([{"question": "Write a Python program."}], Chatbot()) == "experiment"
    assert captured["inputs"] == [{"question": "Write a Python program."}]


def test_invalid_judge_score():
    class Model:
        def invoke(self, *args, **kwargs):
            return SimpleNamespace(content='{"correctness": 9}')

    with pytest.raises(ValueError):
        LLMJudge(Model()).evaluate(CASE, OUTPUT)
