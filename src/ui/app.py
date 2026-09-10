"""Streamlit frontend. Session state owns all conversation history."""

from uuid import uuid4

import streamlit as st

from src.chatbot.langgraph_chatbot import app


def show_sources(sources):
    if sources:
        with st.expander("Sources"):
            for source in sources:
                label = source["source"]
                if source.get("page") is not None:
                    label += f" (page {source['page']})"
                st.write(label)


def main():
    st.set_page_config(page_title="Insurance Support Chatbot", page_icon="💬")
    st.title("Insurance Support Chatbot")
    st.caption(
        "Answers grounded in supplied insurance documents, with web search for current information."
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.history = []
        st.session_state.session_id = str(uuid4())
    with st.sidebar:
        st.write("Ask about policies, claims, Zero Co-pay or insurance grievances.")
        st.caption(
            "Documents are historical references. Check current policy wording for your own coverage."
        )
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.history = []
            st.session_state.session_id = str(uuid4())
            st.rerun()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            show_sources(message.get("sources", []))
    question = st.chat_input("Ask an insurance question")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Checking available information..."):
                    result = app.invoke(
                        {
                            "question": question,
                            "history": st.session_state.history,
                            "session_id": st.session_state.session_id,
                        }
                    )
                st.session_state.history = result["history"]
                answer, sources = result["answer"], result["sources"]
                st.markdown(answer)
                show_sources(sources)
                with st.expander("Request details"):
                    st.write(
                        {
                            "route": result["route"],
                            "verified": result["verified"],
                            "retrieval_confidence": result["confidence"],
                            "seconds": round(result["latency_seconds"], 2),
                            "LLM_calls": result["llm_calls"],
                            "workflow": result["workflow"],
                        }
                    )
                    if result["errors"]:
                        st.warning("; ".join(result["errors"]))
            except Exception:
                answer, sources = "The request failed. Check your setup and try again.", []
                st.error(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


if __name__ == "__main__":
    main()
