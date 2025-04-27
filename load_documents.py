import os
#import openai
import sys
import shutil

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

#openai.api_key = os.environ['OPENAI_API_KEY']

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader

#LOAD DOCUMENT
DATA_PATH = "./data/e-mu_eos_4.0_manual.pdf"
#DATA_PATH = "./data/Adam-Haley-Resume-2025.pdf"

try: 
    loader = PyPDFLoader(DATA_PATH)
    pages = loader.load()
except Exception as e:
    print(e.message, e.args)

print(len(pages))

page = pages[0]

#PRINT OUT SOME DATA PROVING LOADING WORKED
print(page.page_content[:1500])
print(page.metadata)

#SPLIT THE TEXT
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

chunk_size = 1000
chunk_overlap = 200 

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

pages = r_splitter.split_documents(pages)

print("Num pages:\n")
print(len(pages))

for page in pages:
    print(page)


#CHROMA VECTOR STORE
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model="mistral")

persist_directory = 'my_chroma_data'

#shutil.rmtree(persist_directory)

vectordb = Chroma.from_documents(
    documents=pages,
    embedding=embedding,
    persist_directory=persist_directory
)

print('number of records in chroma collection\n')
print(vectordb._collection.count())

