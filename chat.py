import os
import sys
import time
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)



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

def rag_process_chat(input):
    #start timer
    start_time = time.perf_counter()


    question = input
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
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

def main():
    print("Chat with your documents here. Type 'exit' to quit.\n")

    user_input = input(">>> ")

    # Answer with RAG
    print(rag_process_chat(user_input))

if __name__ == "__main__":
    main()

