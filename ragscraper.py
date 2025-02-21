import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "./data/e-mu_eos_4.0_manual.pdf"


loader = PyMuPDFLoader(DATA_PATH)
docs = loader.load()

#db = Chroma.from_documents(texts, embeddings)

# Question and Answer Area
st.header("🗣️ RAG Question Answer")
prompt = st.text_area("**Ask a question related to" + DATA_PATH + ":**")
ask = st.button(
    "🔥 Ask",
)


#for page in pages: # iterate the document pages
#    print(page)
#    text = page.get_text().encode("utf8") # get plain text (is in UTF-8)
#    print(text) # write text of page
#    print(bytes((12,))) # write page delimiter (form feed 0x0C)


#text split the doc, chunk and embed
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n","\n",".","?","!"," ",""],
    )

split_doc = text_splitter.split_documents(docs)

st.write(split_doc)


