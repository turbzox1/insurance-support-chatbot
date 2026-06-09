import streamlit as st
from chatbot import ask_question

st.set_page_config(
    page_title="Insurance Support Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Insurance Support Chatbot")
st.caption("Ask questions about insurance policies, claims, grievances, and ombudsman services.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_question = st.chat_input("Ask an insurance-related question...")

if user_question:

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer = ask_question(user_question)

        st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )