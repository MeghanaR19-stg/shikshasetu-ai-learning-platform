from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from utils.vector_db import get_vector_store

def get_rag_chain(api_key):
    """
    Creates a RAG chain for multilingual Q&A using the Chroma vector store.
    """
    # 1. Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        google_api_key=api_key,
        temperature=0.2
    )
    
    # 2. Setup Vector Store Retriever
    vector_store = get_vector_store(api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # 3. Define Prompt Template (Multilingual & Strict)
    template = """
    You are an expert academic assistant for NCERT textbooks.
    Answer the question based ONLY on the following context.
    If the context does not contain the answer, say "I'm sorry, I couldn't find information about this in the uploaded NCERT books."
    
    Respond in the SAME LANGUAGE as the user's question (e.g., if asked in Kannada, answer in Kannada).
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 4. Create the Chain
    def format_docs(docs):
        # Including source metadata
        return "\n\n".join([f"Source ({doc.metadata.get('source', 'Unknown')}): {doc.page_content}" for doc in docs])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def stream_response(api_key, question):
    """
    Streams the response for a typewriter effect in Streamlit.
    """
    chain = get_rag_chain(api_key)
    for chunk in chain.stream(question):
        yield chunk

def get_sources_for_query(api_key, question):
    """
    Retrieves the raw documents that were used for context.
    """
    vector_store = get_vector_store(api_key)
    docs = vector_store.similarity_search(question, k=3)
    return docs
