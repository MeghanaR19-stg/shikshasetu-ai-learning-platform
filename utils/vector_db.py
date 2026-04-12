import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

def get_vector_store(api_key, persist_directory="data/chroma_db"):
    """
    Initializes or loads a ChromaDB vector store using Gemini Embeddings.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=api_key
    )
    
    # Check if DB directory exists and has files (to avoid re-init error in some environments)
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vector_store

def add_documents_to_store(chunks, api_key, persist_directory="data/chroma_db"):
    """
    Adds document chunks to the ChromaDB store. Handles both initialization and appending.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=api_key
    )
    
    # If the directory doesn't exist, Chroma.from_documents will create it.
    # If it exists, we load it and add documents.
    if os.path.exists(persist_directory):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    
    return vector_store
