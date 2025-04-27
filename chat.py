import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model="mistral")
persist_directory = 'my_chroma_data'


vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

#micro-rag tests
'''
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
'''

#docs = mmr_res

from langchain.chat_models import ChatOllama
llm = ChatOllama(
    model = "mistral",
    temperature = 0.8,
    num_predict = 256,
    )

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

from langchain.prompts import PromptTemplate

question = "Does Adam have management or mentoring experience?"
# Build prompt
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""

print("Q:" + question + "\n")
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Run chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

result = qa_chain({"query": question})

print(result["result"])
print(result["source_documents"][0])





#docs = vectordb.similarity_search(question,k=5)
#print(len(docs))
#print(docs[0])
#for res in docs:
#    print(res)
