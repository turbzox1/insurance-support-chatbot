from web_search import WebSearchAgent

agent = WebSearchAgent()

results = agent.search(
    "Insurance Ombudsman appointment"
)

for result in results:

    print("\nTitle:")
    print(result["title"])

    print("\nURL:")
    print(result["url"])

    print("\nContent:")
    print(result["content"][:300])

    print("-" * 50)