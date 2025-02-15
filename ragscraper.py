import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

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

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(texts, embeddings)


print(len(texts))

for item in texts:
  print(item)

