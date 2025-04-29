import os
import openai
import sys
import time

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
 

start_time = time.perf_counter()

embedding = OllamaEmbeddings(model="mistral")

'''
embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)
'''

#Using numpy dot product to compare vector similarity
sentence1 = "Friday is my favorite day of the week."
sentence2 = "The sun sets at 6:00pm"
sentence3 = "Monday is my least favorite day of the week."

embedding1 = embedding.embed_query(sentence1)
embedding2 = embedding.embed_query(sentence2)
embedding3 = embedding.embed_query(sentence3)

import numpy as np

print(np.dot(embedding1, embedding2))
print(np.dot(embedding1, embedding3))
print(np.dot(embedding2, embedding3))

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")



