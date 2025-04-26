import os
import openai
import sys

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']

from langchain_community.document_loaders import PyPDFLoader

#LOAD DOCUMENT
DATA_PATH = "./data/e-mu_eos_4.0_manual.pdf"

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

chunk_size =26
chunk_overlap = 4

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

pages = r_splitter.split_documents(pages)

print(len(pages))



