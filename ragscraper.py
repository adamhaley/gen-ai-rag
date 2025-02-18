import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "./data/e-mu_eos_4.0_manual.pdf"


loader = PyPDFLoader(
    file_path = DATA_PATH,
    extract_images = False,
    # headers = None
    # extraction_mode = "plain",
    # extraction_kwargs = None,
)

pages = loader.load_and_split()

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
texts = text_splitter.split_documents(pages)


#db = Chroma.from_documents(texts, embeddings)

# Question and Answer Area
st.header("🗣️ RAG Question Answer")
prompt = st.text_area("**Ask a question related to" + DATA_PATH + ":**")
ask = st.button(
    "🔥 Ask",
)

print(len(texts))

for item in texts:
  print(item)

