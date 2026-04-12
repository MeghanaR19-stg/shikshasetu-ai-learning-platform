from utils.vector_db import get_vector_store
from utils.rag_logic import get_rag_chain

def build_vector_database(api_key):
    """
    Initializes the vector database. In our setup, this is handled by get_vector_store
    which loads the existing ChromaDB or prepares it for ingestion.
    """
    return get_vector_store(api_key)

def retrieve_relevant_context(query, api_key):
    """
    Retrieves the top 3 most relevant NCERT content chunks.
    """
    vector_store = get_vector_store(api_key)
    # Using k=3 as specifically requested
    docs = vector_store.similarity_search(query, k=3)
    
    # Returning the chunks as a list of strings or the documents themselves?
    # Usually context for LLM is best as strings.
    return [doc.page_content for doc in docs]
