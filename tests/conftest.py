import socket

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    for name in (
        "GOOGLE_API_KEY",
        "TAVILY_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "false")
    from src.services import analytics

    monkeypatch.setattr(analytics, "ANALYTICS_FILE", tmp_path / "requests.jsonl")

    def blocked(*args, **kwargs):
        raise AssertionError("Network calls are forbidden in automated tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
