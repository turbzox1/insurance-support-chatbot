from services.analytics import load_analytics


data = load_analytics()

print("\n=== Analytics Report ===")

print(
    f"Total Questions: "
    f"{data['total_questions']}"
)

print(
    f"Responses Generated: "
    f"{data['responses_generated']}"
)

print(
    f"High Confidence: "
    f"{data['high_confidence']}"
)

print(
    f"Medium Confidence: "
    f"{data['medium_confidence']}"
)

print(
    f"Low Confidence: "
    f"{data['low_confidence']}"
)

if data["total_questions"] > 0:

    avg_score = (
        data["total_score"]
        / data["total_questions"]
    )

    print(
        f"Average Retrieval Score: "
        f"{avg_score:.2f}"
    )