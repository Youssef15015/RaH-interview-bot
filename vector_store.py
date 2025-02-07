from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
import pandas as pd
import streamlit as st
import pickle
import os
import getpass


st.set_page_config(layout="wide")
llm = ChatNVIDIA(mode = "mixtral_8x7b")
document_embedder = NVIDIAEmbeddings(model="NV-Embed-QA", mode_type="passage")
# query_embedder = NVIDIAEmbeddings(mode="embed-qa-4", model_type="query")


with st.sidebar:
    DOCS_DIR = os.path.abspath("./uploaded_docs")
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    st.subheader("Add to the Knowledge Base")
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:",
accept_multiple_files=True)
        submitted = st.form_submit_button("Upload!")


    if uploaded_files and submitted:
        for uploaded_file in uploaded_files:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            with open (os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.read())


use_existing_vector_store = st.radio("Use existing vector store",["Yes", "No"])

vector_store_path = "vectorstore.pkl"

raw_documents = DirectoryLoader(DOCS_DIR).load()

vector_store_exists = os.path.exists(vector_store_path)
vectorstore = None

if use_existing_vector_store == "Yes" and vector_store_exists:
    with open(vector_store_path, "rb") as f:
        vectorstore = pd.read_pickle(f)

else:
    if raw_documents:
        text_splitter = CharacterTextSplitter(chunk_size = 2000) 
        documents = text_splitter.split_documents(raw_documents)

        vectorstore = FAISS.from_documents(documents, document_embedder)

        with open(vector_store_path, "wb") as f:
            pickle.dump(vectorstore, f)
