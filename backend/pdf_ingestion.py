import os
import re
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.vector_db import add_documents_to_store
from langchain_core.documents import Document

RAW_BOOKS_DIR = "data/raw_books"
PROCESSED_TEXT_DIR = "data/processed_text"

def clean_text(text):
    """
    Cleans extracted text to remove headers, footers, and page numbers.
    """
    text = re.sub(r'Reprint \d{4}-\d{2}', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def process_and_index_file(file_path, api_key, batch_size=20, delay=2):
    """
    Processes a single PDF with BATCHING and DELAYS to avoid Free Tier Rate Limits.
    """
    file_name = os.path.basename(file_path)
    if os.path.getsize(file_path) == 0:
        return 0
        
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 1. Clean and Combine text
    cleaned_content = []
    for doc in documents:
        cleaned_page = clean_text(doc.page_content)
        if cleaned_page:
            cleaned_content.append(cleaned_page)
    
    full_text = "\n\n".join(cleaned_content)
    
    # 2. Save to processed_text folder
    txt_file_name = file_name.replace(".pdf", ".txt")
    processed_path = os.path.join(PROCESSED_TEXT_DIR, txt_file_name)
    with open(processed_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    # 3. Chunk for RAG
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    # Extract Chapter from filename or default to "General"
    chapter_match = re.search(r'chap[_\s]?(\d+)', file_name, re.IGNORECASE)
    chapter_val = f"Chapter {chapter_match.group(1)}" if chapter_match else "General"

    # Create documents with enriched metadata
    chunks = text_splitter.split_text(full_text)
    enriched_docs = [
        Document(
            page_content=chunk, 
            metadata={
                "source_file": file_name,
                "chapter": chapter_val,
                "text_chunk": f"Chunk {i+1}"
            }
        ) for i, chunk in enumerate(chunks)
    ]
    
    # 4. Batched Indexing into ChromaDB
    total_indexed = 0
    for i in range(0, len(enriched_docs), batch_size):
        batch = enriched_docs[i:i + batch_size]
        add_documents_to_store(batch, api_key)
        total_indexed += len(batch)
        
        # Log progress (visible in terminal or captured by st.write if called properly)
        print(f"   Indexed {total_indexed}/{len(chunks)} chunks for {file_name}...")
        
        # Delay to avoid 429 Resource Exhausted on Free Tier
        if total_indexed < len(chunks):
            time.sleep(delay)
    
    return total_indexed

def reprocess_all_existing(api_key):
    """
    Scans raw_books and re-processes everything with rate-limiting.
    """
    files = [f for f in os.listdir(RAW_BOOKS_DIR) if f.endswith('.pdf')]
    results = []
    for f in files:
        path = os.path.join(RAW_BOOKS_DIR, f)
        num_chunks = process_and_index_file(path, api_key)
        results.append((f, num_chunks))
    return results
