
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_openai import AzureOpenAIEmbeddings
from chromadb.config import Settings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_store"


# Settings to connect to Chroma server running in Docker
CHROMA_SETTINGS = Settings(
    persist_directory="chroma_store"
)


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
    filepath = os.path.join("doc", "the_art_of_war.txt")
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    print("Chunking document...")
    return chunk_text(text)

def store_chunks(chunks):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client_settings=CHROMA_SETTINGS,
        collection_name="rag_collection"
    )
    print("Stored chunks in Chroma.")
    return vectorstore

def load_vectorstore():
    return Chroma(
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
        collection_name="rag_collection"
    )

# Define your system prompt
SYSTEM_PROMPT = """You are a helpful and knowledgeable AI assistant for farmers.
You are part of a Retrieval-Augmented Generation (RAG) system and must answer questions strictly based on the content of the uploaded document provided by the user.
Do not make up answers or provide information that is not supported by the document. If the document does not contain the answer, respond with "I'm not sure based on the uploaded document."
Always be clear, concise, and use friendly language in your responses."""


# Set up memory outside the function to persist across turns
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def answer_question(query: str) -> str:
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever()

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        model_name="llama3-8b-8192"
    )
    
        # Custom prompt template with system message
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False  # Set to True if you want document context returned
    )

    # Run chain with current user query
    result = qa_chain.invoke(query)
    return result["answer"]