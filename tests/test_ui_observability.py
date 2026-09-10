from pathlib import Path

from src.config.config import BASE_DIR
from src.services.observability import langfuse_callback
from src.services.web_search import WebSearchAgent


def test_langfuse_optional(monkeypatch):
    assert langfuse_callback() is None
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    assert langfuse_callback() is None


def test_langfuse_sdk_import_contract():
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    assert callable(CallbackHandler) and callable(get_client)


def test_web_filters_unusable_results():
    class Client:
        def search(self, **kwargs):
            return {
                "results": [
                    {"url": "javascript:bad", "content": "x"},
                    {"url": "https://example.com", "content": ""},
                    {"url": "https://irdai.gov.in", "content": "IRDAI"},
                ]
            }

    assert len(WebSearchAgent(Client()).search("IRDAI")) == 1


def test_streamlit_start_and_message_and_clear():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(BASE_DIR / "src" / "app.py"), default_timeout=20).run()
    assert not app.exception
    app.chat_input[0].set_value("Write a Python program.").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert "insurance" in app.chat_message[1].markdown[0].value.lower()
    app.button[0].click().run()
    assert not app.exception and len(app.chat_message) == 0
    assert app.session_state["history"] == []


def test_no_path_mutation_in_sources():
    for path in (BASE_DIR / "src").rglob("*.py"):
        text = Path(path).read_text(encoding="utf-8")
        assert "sys.path." not in text
