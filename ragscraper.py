import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import streamlit as st
import pymupdf 

load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "./data/e-mu_eos_4.0_manual.pdf"


pages = pymupdf.open(DATA_PATH)


#db = Chroma.from_documents(texts, embeddings)

# Question and Answer Area
st.header("🗣️ RAG Question Answer")
prompt = st.text_area("**Ask a question related to" + DATA_PATH + ":**")
ask = st.button(
    "🔥 Ask",
)

for page in pages: # iterate the document pages
    print(page)
    text = page.get_text().encode("utf8") # get plain text (is in UTF-8)
    print(text) # write text of page
    print(bytes((12,))) # write page delimiter (form feed 0x0C)

