import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from retriever import initialize_retriever

# Load API key
load_dotenv()
print("Step 1")

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

print("Step 2")

# Load retriever
retriever = initialize_retriever()

print("Step 3")


def ask_question(question):

    # Retrieve top 3 chunks
    docs = retriever.invoke(question)

    print("Step 4")

    # Combine retrieved chunks into context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an Insurance Support Assistant.

Answer ONLY using the information present in the context below.

If the answer is not available in the context, reply exactly:

I could not find that information in the provided documents.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    print("Step 5")

    return response.content


if __name__ == "__main__":

    while True:

        question = input("\nAsk a Question: ")

        if question.lower() == "exit":
            break

        answer = ask_question(question)

        print("\nAnswer:\n")
        print(answer)