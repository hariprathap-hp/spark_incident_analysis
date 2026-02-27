import streamlit as st
from backend.core_qdrant import run_llm

st.header("🚀 Spark Incident Analysis Assistant (OpenAI + Qdrant)")

prompt = st.text_input("Prompt", placeholder="Enter your message here...")

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []

if prompt:
    with st.spinner("Generating Response..."):
        generated_response = run_llm(query=prompt)
        
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(
            generated_response["answer"]
        )

if st.session_state["chat_answers_history"]:
    for response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(response)