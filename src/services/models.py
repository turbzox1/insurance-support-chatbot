"""Lazy Gemini client and strict JSON parsing shared by agent and evaluators."""

import json
import os
from functools import lru_cache

from src.config.config import LLM_MODEL


@lru_cache(maxsize=1)
def get_llm():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Set GOOGLE_API_KEY in the repository .env file.")
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0, max_retries=0, timeout=45)


def response_text(response):
    content = response.content
    if isinstance(content, str):
        return content.strip()
    return "\n".join(block.get("text", "") for block in content if isinstance(block, dict)).strip()


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value
