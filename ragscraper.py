import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = "./data/e-mu_eos_4.0_manual.pdf"


loader = PyPDFLoader(
    file_path = DATA_PATH,
    extract_images = False,
    # headers = None
    # extraction_mode = "plain",
    # extraction_kwargs = None,
)

documents = loader.load()

print(documents[0])
