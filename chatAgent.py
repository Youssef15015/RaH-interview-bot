import streamlit as st
import pandas as pd
import os
import pickle
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA


if not os.getenv("NIVIDA_API_KEY"):
    os.environ["NVIDIA_API_KEY"] = "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"


st.set_page_config(layout="wide")
llm = ChatNVIDIA(mode = "mixtral_8x7b")

st.subheader("Chat with your AI Assistant, Interview Bot!")

vector_store_path = "vectorstore.pkl"
with open("vectorstore.pkl", "rb") as f:
    try:
        {
            vectorstore = pickle.load(f)
        }
    except (e):
        {
            print(e)
        }



if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful AI assistant named Interview Bot..."), ("user", "{input}")]
)
chain = prompt_template | llm | StrOutputParser()


user_input = st.chat_input("Ask your question:")
if user_input and vectorstore is not None:
    st.session_state.messages.append({"role": "user", "content": user_input})

    retriever = vectorstore.as_retriever()
    context = "\n\n".join(doc.page_content for doc in retriever.get_relevant_documents(user_input))
    augmented_user_input = f"Context: {context}\n\nQuestion: {user_input}"


    response = chain.invoke({"input": augmented_user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
