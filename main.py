import streamlit as st
from backend.optimized_core import run_llm

st.header("LangChain🦜🔗 SIP Protocol - Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your message here...")

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

if prompt:
    with st.spinner("Generating Response"):
        generated_response = run_llm(query=prompt, confirm_cost=True)
        formatted_response = (
            f"{generated_response['answer']}"
        )

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        #st.session_state["chat_history"].append(("human", prompt))
        #st.session_state["chat_history"].append(("ai", generated_response["answer"]))

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)