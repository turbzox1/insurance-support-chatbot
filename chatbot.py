"""Compatibility API; explicitly pass history for conversational calls."""

from src.chatbot.langgraph_chatbot import app, main


def ask_question(question, history=None):
    return app.invoke({"question": question, "history": history or []})["answer"]


if __name__ == "__main__":
    main()
