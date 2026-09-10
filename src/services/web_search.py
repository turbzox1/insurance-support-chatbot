import os


class WebSearchAgent:
    def __init__(self, client=None):
        self.client = client

    def search(self, query, max_results=3):
        if self.client is None:
            key = os.getenv("TAVILY_API_KEY")
            if not key:
                raise RuntimeError("Web search unavailable: set TAVILY_API_KEY in .env.")
            from tavily import TavilyClient

            self.client = TavilyClient(api_key=key)
        response = self.client.search(query=query, max_results=max_results, search_depth="basic")
        return [
            item
            for item in response.get("results", [])
            if item.get("content") and str(item.get("url", "")).startswith(("https://", "http://"))
        ]
