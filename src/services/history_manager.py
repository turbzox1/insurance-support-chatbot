import json
import os


HISTORY_FILE = "history/chat_history.json"


def load_history():

    if not os.path.exists(HISTORY_FILE):
        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:
        return []


def save_history(history):

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )


def add_message(question, answer):

    history = load_history()

    history.append(
        {
            "question": question,
            "answer": answer
        }
    )

    history = history[-20:]

    save_history(history)