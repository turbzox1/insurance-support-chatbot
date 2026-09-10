import json
from collections import Counter

from src.services.analytics import load_analytics


def main():
    rows = load_analytics()
    print(
        json.dumps(
            {
                "requests": len(rows),
                "routes": dict(Counter(row["route"] for row in rows)),
                "error_rate": sum(bool(row["errors"]) for row in rows) / len(rows)
                if rows
                else None,
                "mean_latency_seconds": sum(row["latency_seconds"] for row in rows) / len(rows)
                if rows
                else None,
                "llm_calls": sum(row["llm_calls"] for row in rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
