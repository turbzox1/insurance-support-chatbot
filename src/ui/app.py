import os
import sys
import time
import streamlit as st

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.chatbot.langgraph_chatbot import app


# ----------------------------------------
# Page Configuration
# ----------------------------------------

st.set_page_config(
    page_title="Insurance Support Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ----------------------------------------
# Sidebar
# ----------------------------------------

with st.sidebar:

    st.title("🤖 Insurance Chatbot")

    st.markdown("---")

    st.subheader("System Information")

    st.write("**Framework:** LangGraph")
    st.write("**LLM:** Gemini 2.5 Flash")
    st.write("**Embedding:** BAAI/bge-small-en-v1.5")
    st.write("**Retrieval:** Hybrid (BM25 + Vector)")
    st.write("**Fusion:** Reciprocal Rank Fusion (RRF)")
    st.write("**Reranker:** BAAI/bge-reranker-base")
    st.write("**Compression:** Context Compression")
    st.write("**Query Rewrite:** Enabled")
    st.write("**Conversation Memory:** Enabled")
    st.write("**Answer Verification:** Enabled")
    st.write("**Web Search:** Enabled")
    st.write("**Domain Detection:** Enabled")

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------
# Title
# ----------------------------------------

st.title("🤖 Insurance Support Chatbot")

st.caption(
    "Agentic RAG chatbot powered by LangGraph with Hybrid Retrieval, Reranking, Verification and Web Search."
)

# ----------------------------------------
# Session State
# ----------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------
# Welcome Screen
# ----------------------------------------

if len(st.session_state.messages) == 0:

    st.info(
        """
### Welcome!

You can ask questions such as:

- Who appoints the Insurance Ombudsman?
- What is the role of the Insurance Ombudsman?
- How can I file an insurance complaint?
- What is covered under health insurance?
- What are the latest IRDAI updates?
- How are insurance claims settled?

This chatbot uses:

- LangGraph Agent Workflow
- Hybrid Retrieval (BM25 + Vector)
- Cross-Encoder Reranking
- Context Compression
- Answer Verification
- Web Search for recent insurance information
"""
    )

# ----------------------------------------
# Display Previous Messages
# ----------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------------------
# User Input
# ----------------------------------------

user_question = st.chat_input(
    "Ask an insurance-related question..."
)

if user_question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):

        start_time = time.time()

        try:

            with st.spinner("Running LangGraph workflow..."):

                result = app.invoke(
                    {
                        "question": user_question
                    }
                )

            answer = result["answer"]

            end_time = time.time()

            response_time = end_time - start_time

            st.markdown(answer)

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Confidence",
                    result.get("confidence", "N/A")
                )

                st.metric(
                    "Intent",
                    result.get("intent", "N/A")
                )

            with col2:

                st.metric(
                    "Question Type",
                    result.get("question_type", "N/A")
                )

                verified = result.get("verified")

                if verified == "N/A":
                    verified_text = "N/A"
                else:
                    verified_text = "Yes" if verified else "No"

                st.metric(
                    "Verified",
                    verified_text
                )

            st.divider()

            st.subheader("📄 Sources")

            sources = result.get("sources", [])

            if sources:

                for file_name, page in sources:

                    if page is None:
                        st.write(f"• **{file_name}**")
                    else:
                        st.write(
                            f"• **{file_name}** (Page {page})"
                        )

            else:

                st.info("No document sources available.")

            st.divider()

            st.subheader("🔄 Agent Workflow")

            workflow = result.get("workflow", [])

            if workflow:

                for step in workflow:
                    st.write(f"✅ {step}")

            else:

                st.info("Workflow information not available.")

            st.caption(
                f"⏱ Response generated in {response_time:.2f} seconds"
            )

        except Exception as e:

            answer = (
                "An unexpected error occurred while generating the response."
            )

            st.error(answer)

            st.exception(e)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )