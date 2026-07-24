import os
import sys
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from chatbot import ask_question


def evaluate_rag():

    dataset = pd.read_csv(
        "evaluation/test_dataset.csv"
    )

    total = len(dataset)

    correct = 0

    retrieval_success = 0

    for _, row in dataset.iterrows():

        question = row["question"]

        expected = str(
            row["expected_answer"]
        ).lower()

        response = ask_question(
            question
        ).lower()

        answer_correct = (
            expected in response
        )

        if answer_correct:

            correct += 1

        if (
            "Documents Retrieved: 0"
            not in response
        ):
            retrieval_success += 1

        print("\n----------------")

        print(
            f"Question: {question}"
        )

        print(
            f"Answer Correct: {answer_correct}"
        )

    answer_accuracy = (
        correct / total
    ) * 100

    retrieval_accuracy = (
        retrieval_success / total
    ) * 100

    print("\n================")

    print(
        f"Answer Accuracy: "
        f"{answer_accuracy:.2f}%"
    )

    print(
        f"Retrieval Success: "
        f"{retrieval_accuracy:.2f}%"
    )


if __name__ == "__main__":

    evaluate_rag()