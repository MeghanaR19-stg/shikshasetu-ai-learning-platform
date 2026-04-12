import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(file_path):
    """
    Loads a PDF and splits it into chunks.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Using RecursiveCharacterTextSplitter for intelligent chunking
    # handles multiple languages well by splitting on common separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks
