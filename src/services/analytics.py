import json
import os


ANALYTICS_FILE = "analytics.json"


def load_analytics():

    if not os.path.exists(
        ANALYTICS_FILE
    ):

        return {
            "total_questions": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
            "total_score": 0,
            "responses_generated": 0
        }

    with open(
        ANALYTICS_FILE,
        "r"
    ) as file:

        return json.load(file)


def save_analytics(data):

    with open(
        ANALYTICS_FILE,
        "w"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


def update_analytics(
    confidence,
    score
):
    score = float(score)
    data = load_analytics()

    data["total_questions"] += 1

    data["responses_generated"] += 1

    data["total_score"] += score

    if confidence == "High":

        data["high_confidence"] += 1

    elif confidence == "Medium":

        data["medium_confidence"] += 1

    else:

        data["low_confidence"] += 1

    save_analytics(data)