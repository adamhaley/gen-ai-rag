import os
import openai
import sys

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv))

openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import PyPDFLoader



from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import streamlit as st
import pypdf
import mistralai
from uuid import uuid4

from mistralai import Mistral


CHROMA_PATH = "chroma"
DATA_PATH = "./data/e-mu_eos_4.0_manual.pdf"


loader = PyPDFLoader(DATA_PATH)
pages = loader.load()

#db = Chroma.from_documents(texts, embeddings)

# Question and Answer Area
st.header("🗣️ RAG Question Answer")
prompt = st.text_area("**Ask a question related to" + DATA_PATH + ":**")
ask = st.button(
    "🔥 Ask",
)


#text split the doc, chunk and embed
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n","\n",".","?","!"," ",""],
    )

split_doc = text_splitter.split_documents(docs)

st.write(split_doc)

embeddings = OllamaEmbeddings(
        model="mistral:latest"
        )

vectorstore = Chroma(
    collection_name="rag-chroma",
    embedding_function=embeddings,
    persist_directory="./my_chroma_data"
)

retriever = vectorstore.as_retriever()

uuids = [str(uuid4()) for _ in range(len(split_doc))]
vectorstore.add_documents(documents=split_doc, ids=uuids)
print(f"Finished ingesting file: {DATA_PATH}")

