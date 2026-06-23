import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from config import LLM_MODEL


load_dotenv()


class QueryRewriter:

    def __init__(self):

        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=os.getenv(
                "GOOGLE_API_KEY"
            ),
            temperature=0
        )

    def rewrite(
        self,
        question,
        history=""
    ):

        prompt = f"""
You are a query rewriting assistant.

Your task is to rewrite the user's question
into a standalone search query.

Rules:
1. Preserve original meaning.
2. Resolve pronouns using conversation history.
3. Keep the query concise.
4. Do not answer the question.
5. Output ONLY the rewritten query.

Conversation History:
{history}

Question:
{question}

Rewritten Query:
"""

        response = self.llm.invoke(
            prompt
        )

        return response.content.strip()


if __name__ == "__main__":

    rewriter = QueryRewriter()

    history = """
Question: What is Insurance Ombudsman?
Answer: Insurance Ombudsman is a grievance redressal mechanism.
"""

    query = "Who appoints them?"

    rewritten = rewriter.rewrite(
        query,
        history
    )

    print("\nOriginal:")
    print(query)

    print("\nRewritten:")
    print(rewritten)