import streamlit as st
import requests
import os

MEDIQA_API_URL=os.environ.get("MEDIQA_API_URL", "http://localhost:8080")

st.title = "MediQA"
st.header = "MediQA"
st.subheader = "An answer for all your medical questions"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What do you want to know?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Get a response
    res = requests.get(MEDIQA_API_URL + "/generate", params={"question": prompt})
    if res.status_code == 200:
        answer = res.json()[0][0]["generated_text"]
    else:
        answer = "There was an issue. Please try again later"

    with st.chat_message("assistant"):
        st.markdown(answer)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})