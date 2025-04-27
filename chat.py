import os
import openai
import sys

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model="mistral")
persist_directory = 'my_chroma_data'



'''
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
'''

print(vectordb._collection.count())

question = "How do I save a sample to disc on the emu e5000 ultra?"
docs = vectordb.max_marginal_relevance_search(question,k=5)

print(len(docs))

for res in docs:
    print(res)
