import os

from tavily import TavilyClient

from dotenv import load_dotenv

load_dotenv()


class WebSearchAgent:

    def __init__(self):

        self.client = TavilyClient(
            api_key=os.getenv("TAVILY_API_KEY")
        )

    def search(self, query, max_results=3):

        response = self.client.search(
            query=query,
            max_results=max_results
        )

        return response["results"]