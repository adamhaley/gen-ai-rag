import os
import openai
import sys

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']

from langchain_community.document_loaders import PyPDFLoader

DATA_PATH = "./data/e-mu_eos_4.0_manual.pdf"

try: 
    loader = PyPDFLoader(DATA_PATH)
    pages = loader.load()
except Exception as e:
    print(e.message, e.args)

print(len(pages))

page = pages[0]

print(page.page_content[:1500])

