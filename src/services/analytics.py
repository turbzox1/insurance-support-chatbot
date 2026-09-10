"""Local operational metrics, without questions, answers or document content."""

import json

from src.config.config import BASE_DIR
from src.services.logger import logger

ANALYTICS_FILE = BASE_DIR / "logs" / "requests.jsonl"


def record_request(result):
    record = {
        key: result.get(key)
        for key in (
            "run_id",
            "route",
            "confidence",
            "verified",
            "latency_seconds",
            "node_latency",
            "errors",
            "llm_calls",
            "token_usage",
        )
    }
    try:
        ANALYTICS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with ANALYTICS_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        logger.warning("Local analytics could not be written.")


def load_analytics(path=ANALYTICS_FILE):
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records
