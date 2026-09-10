"""Caller-owned history: no global or shared disk conversation store."""


def load_history(history=None):
    return [
        {"question": str(item["question"]), "answer": str(item["answer"])}
        for item in (history or [])[-5:]
        if isinstance(item, dict) and "question" in item and "answer" in item
    ]


def add_message(history, question, answer):
    return (load_history(history) + [{"question": question, "answer": answer}])[-5:]
