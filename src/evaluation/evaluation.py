"""Benchmark runner: deterministic checks, optional semantic judge and LangSmith experiment."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.config.config import BASE_DIR
from src.services.observability import flush, log_scores, tracing

DATASET = BASE_DIR / "evaluation" / "test_dataset.csv"
ROUTES = {"RAG", "WEB_SEARCH", "CLARIFY", "NON_INSURANCE"}


def load_dataset(path=DATASET):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Benchmark is empty.")
    ids = set()
    for row in rows:
        if not row.get("id") or row["id"] in ids:
            raise ValueError("Benchmark IDs must be present and unique.")
        ids.add(row["id"])
        if not row.get("question") or row.get("expected_route") not in ROUTES:
            raise ValueError(f"Invalid benchmark case: {row['id']}")
        if row["expected_route"] == "RAG" and not all(
            row.get(k) for k in ("ground_truth", "expected_source")
        ):
            raise ValueError(f"RAG case missing reference answer/source: {row['id']}")
    return rows


def serialize_result(result):
    fields = (
        "answer",
        "route",
        "sources",
        "evidence",
        "verified",
        "confidence",
        "errors",
        "latency_seconds",
        "node_latency",
        "llm_calls",
        "token_usage",
        "run_id",
        "langfuse_trace_id",
        "workflow",
        "rewritten_question",
    )
    output = {key: result.get(key) for key in fields}
    output["retrieved"] = [
        {"content": doc.page_content, **doc.metadata} for doc in result.get("retrieved_docs", [])
    ]
    return output


def score_output(case, output, judge=None):
    expected = case.get("expected_source", "")
    retrieved_sources = [Path(item.get("source", "")).name for item in output.get("retrieved", [])]
    metrics = {
        "routing_accuracy": float(output["route"] == case["expected_route"]),
        "source_hit": float(expected in retrieved_sources)
        if expected and case["expected_route"] == "RAG"
        else None,
        "latency_seconds": output["latency_seconds"],
        "error": float(bool(output.get("errors"))),
        "correctness": None,
        "groundedness": None,
        "hallucination": None,
        "retrieval_relevance": None,
    }
    if judge:
        try:
            metrics.update(judge.evaluate(case, output))
        except Exception as exc:
            metrics["judge_error"] = type(exc).__name__
    log_scores(output.get("langfuse_trace_id"), metrics)
    return metrics


def evaluate_case(case, bot, judge=None):
    try:
        output = serialize_result(bot.invoke({"question": case["question"]}))
    except Exception as exc:
        output = {
            "answer": "",
            "route": "ERROR",
            "sources": [],
            "retrieved": [],
            "errors": [type(exc).__name__],
            "latency_seconds": None,
            "llm_calls": 0,
        }
    return {
        "id": case["id"],
        "question": case["question"],
        "category": case["category"],
        "output": output,
        "metrics": score_output(case, output, judge),
    }


def summarize(rows):
    metrics = {}
    keys = (
        "routing_accuracy",
        "source_hit",
        "correctness",
        "groundedness",
        "hallucination",
        "retrieval_relevance",
        "latency_seconds",
        "error",
    )
    for key in keys:
        values = [row["metrics"].get(key) for row in rows]
        values = [value for value in values if isinstance(value, (int, float))]
        metrics[key] = {
            "mean": sum(values) / len(values) if values else None,
            "evaluated": len(values),
        }
    return {
        "examples": len(rows),
        "metrics": metrics,
        "judge_errors": sum("judge_error" in row["metrics"] for row in rows),
        "agent_llm_calls": sum(row["output"].get("llm_calls", 0) for row in rows),
    }


def run_langsmith(cases, bot, judge=None):
    import os

    from langsmith import Client

    if not os.getenv("LANGSMITH_API_KEY"):
        raise ValueError("LANGSMITH_API_KEY is required for --langsmith.")
    client = Client()
    dataset = client.create_dataset(dataset_name=f"insurance-benchmark-{uuid4().hex[:8]}")
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"question": row["question"]} for row in cases],
        outputs=cases,
    )

    def target(inputs):
        return serialize_result(bot.invoke(inputs))

    def evaluator(outputs, reference_outputs):
        metrics = score_output(reference_outputs, outputs, judge)
        return {
            "results": [
                {"key": key, "score": value}
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            ]
            + (
                [{"key": "judge_error", "score": 1, "comment": metrics["judge_error"]}]
                if "judge_error" in metrics
                else []
            )
        }

    # SDK manages experiment upload; each example executes exactly once, sequentially.
    return client.evaluate(
        target,
        data=dataset.id,
        evaluators=[evaluator],
        experiment_prefix="insurance-chatbot",
        max_concurrency=1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--validate", action="store_true", help="Validate benchmark; no external calls"
    )
    parser.add_argument(
        "--run", action="store_true", help="Run real chatbot calls (uses API quota)"
    )
    parser.add_argument("--judge", action="store_true", help="One additional model call per case")
    parser.add_argument(
        "--langsmith", action="store_true", help="Upload a new dataset and experiment"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cases = load_dataset(args.dataset)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        cases = cases[: args.limit]
    if args.validate or not args.run:
        print(f"Validated {len(cases)} cases. No API calls made. Use --run to execute.")
        return
    from src.chatbot.langgraph_chatbot import app
    from src.evaluation.judge import LLMJudge

    judge = LLMJudge() if args.judge else None
    try:
        if args.langsmith:
            run_langsmith(cases, app, judge)
        else:
            with tracing():
                results = [evaluate_case(case, app, judge) for case in cases]
            report = {"summary": summarize(results), "results": results}
            folder = BASE_DIR / "evaluation" / "results"
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            path = folder / f"{stamp}.json"
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report["summary"], indent=2))
            print(f"Full results: {path}")
    finally:
        flush()


if __name__ == "__main__":
    main()
