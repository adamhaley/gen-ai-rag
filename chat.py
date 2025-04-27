import os
import openai
import sys

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model="mistral")
#persist_directory = 'my_chroma_data'


'''
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
'''

#micro-rag tests
texts = [
    """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
    """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
    """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
]


smalldb = Chroma.from_texts(texts, embedding=embedding)

print(smalldb._collection.count())
question = "Tell me about all-white mushrooms with large fruiting bodies"

print("Q:" + question + "\n")


ss_res = smalldb.similarity_search(question, k=1)
mmr_res = smalldb.max_marginal_relevance_search(question, k=1, fetch_k=3) 

docs = ss_res
#docs = mmr_res

#docs = vectordb.max_marginal_relevance_search(question,k=5)

print(len(docs))

for res in docs:
    print(res)
