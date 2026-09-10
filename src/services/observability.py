"""Opt-in hosted tracing; callbacks never required for answering."""

import os
from contextlib import contextmanager
from uuid import uuid4

from src.services.logger import logger


def enabled(name):
    return os.getenv(name, "false").lower() in {"true", "1", "yes"}


@contextmanager
def tracing():
    from langsmith import tracing_context

    active = enabled("LANGSMITH_TRACING") and bool(os.getenv("LANGSMITH_API_KEY"))
    with tracing_context(enabled=active):
        yield


def langfuse_callback():
    if not enabled("LANGFUSE_ENABLED"):
        return None
    if not all(os.getenv(key) for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")):
        logger.warning("Langfuse disabled: credentials incomplete.")
        return None
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler

        if os.getenv("LANGFUSE_HOST"):
            os.environ.setdefault("LANGFUSE_BASE_URL", os.environ["LANGFUSE_HOST"])
        get_client()
        trace_id = uuid4().hex
        callback = CallbackHandler(trace_context={"trace_id": trace_id})
        callback.insurance_trace_id = trace_id
        return callback
    except Exception:
        logger.warning("Langfuse initialization failed; continuing without it.")
        return None


def log_scores(trace_id, metrics):
    if not trace_id or not enabled("LANGFUSE_ENABLED"):
        return
    try:
        from langfuse import get_client

        client = get_client()
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                client.create_score(
                    trace_id=trace_id, name=name, value=float(value), data_type="NUMERIC"
                )
    except Exception:
        logger.warning("Could not publish Langfuse evaluation scores.")


def flush():
    if enabled("LANGFUSE_ENABLED"):
        try:
            from langfuse import get_client

            get_client().flush()
        except Exception:
            logger.warning("Could not flush Langfuse traces.")
