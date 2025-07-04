from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_openai import AzureOpenAIEmbeddings

import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_store"

# Use your working approach: full endpoint URL in azure_endpoint, model as string
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.create_documents([text])

def load_static_document():
    filepath = os.path.join("..", "doc", "the_art_of_war.txt")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    print("✅ Chunking document...")
    return chunk_text(text)

def store_chunks(chunks):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    print("📦 Stored chunks in Chroma.")
    return vectorstore

def load_vectorstore():
    return Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

def answer_question(query: str) -> str:
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=ChatGroq(api_key=os.getenv("GROQ_API_KEY"), temperature=0, model_name="llama3-8b-8192"),
        retriever=retriever,
        return_source_documents=False,
    )
    return qa.run(query)
