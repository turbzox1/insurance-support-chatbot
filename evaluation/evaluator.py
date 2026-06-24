import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)
import pandas as pd

from chatbot import ask_question


def evaluate():

    import os

    csv_path = os.path.join(
        os.path.dirname(__file__),
        "test_dataset.csv"
    )

    dataset = pd.read_csv(
        csv_path
    )
    total = len(dataset)

    correct = 0

    for _, row in dataset.iterrows():

        question = row["question"]

        expected = str(
            row["expected_answer"]
        ).lower()

        response = ask_question(
            question
        ).lower()

        if expected in response:

            correct += 1

        print("\n------------------")
        print(
            f"Question: {question}"
        )
        print(
            f"Expected: {expected}"
        )
        print(
            f"Correct: {expected in response}"
        )

    accuracy = (
        correct / total
    ) * 100

    print("\n==================")
    print(
        f"Accuracy: {accuracy:.2f}%"
    )


if __name__ == "__main__":

    evaluate()