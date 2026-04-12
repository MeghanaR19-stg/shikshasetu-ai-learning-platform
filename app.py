import streamlit as st
import os
from backend.pdf_ingestion import process_and_index_file, reprocess_all_existing
from utils.rag_logic import stream_response, get_sources_for_query
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="ShikshaSetu: Multilingual NCERT Agent", page_icon="🎓", layout="wide")

# Sidebar for configuration
with st.sidebar:
    st.markdown("## 🎓 ShikshaSetu Settings")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.divider()
    if st.button("🗑️ Clear Knowledge Base"):
        if os.path.exists("data/chroma_db"):
            import shutil
            shutil.rmtree("data/chroma_db")
            st.success("Knowledge base cleared!")
        else:
            st.warning("Knowledge base is already empty.")
    
    if api_key and st.button("🔄 Reprocess Existing Data"):
        with st.spinner("Refining all stored books..."):
            results = reprocess_all_existing(api_key)
            st.success(f"Reprocessed {len(results)} books.")

st.title("🎓 ShikshaSetu: Multilingual NCERT Assistant")
st.markdown("""
Welcome to **ShikshaSetu**, your AI-powered companion for NCERT textbooks. 
Upload books in **English, Hindi, or Kannada** to generate structured knowledge and start learning!
""")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

tab1, tab2, tab3 = st.tabs(["💬 Chat with Agent", "📤 Manage Knowledge Base", "📂 Processed Data"])

with tab2:
    RAW_BOOKS_DIR = "data/raw_books"
    if not os.path.exists(RAW_BOOKS_DIR):
        os.makedirs(RAW_BOOKS_DIR)

    uploaded_files = st.file_uploader(
        "Upload NCERT PDFs", 
        type="pdf", 
        accept_multiple_files=True
    )

    if uploaded_files:
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar.")
        else:
            if st.button("🚀 Process and Index"):
                with st.spinner("Ingesting textbooks..."):
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(RAW_BOOKS_DIR, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        st.info(f"Processing: {uploaded_file.name}")
                        num_chunks = process_and_index_file(file_path, api_key)
                        st.write(f"✅ Indexed {num_chunks} chunks.")
                st.success("Indexing complete!")
                st.balloons()

    st.divider()
    st.subheader("📖 Locally Stored Raw Books")
    existing_books = [f for f in os.listdir(RAW_BOOKS_DIR) if f.endswith('.pdf')]
    if existing_books:
        for book in existing_books:
            st.text(f"📄 {book}")
    else:
        st.info("No raw books stored.")

with tab3:
    st.subheader("📝 Cleaned Text Files")
    st.markdown("These are the cleaned versions of your textbooks, ready for inspection.")
    PROCESSED_DIR = "data/processed_text"
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    processed_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('.txt')]
    if processed_files:
        for txt_file in processed_files:
            with st.expander(f"👁️ {txt_file}"):
                with open(os.path.join(PROCESSED_DIR, txt_file), "r", encoding="utf-8") as f:
                    st.text_area("Content", f.read(), height=300)
    else:
        st.info("No processed text available yet. Please index some books first.")

with tab1:
    st.subheader("🤖 ShikshaSetu Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your books..."):
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                try:
                    for chunk in stream_response(api_key, prompt):
                        full_response += (chunk or "")
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                    
                    with st.expander("📚 Sources"):
                        sources = get_sources_for_query(api_key, prompt)
                        for i, doc in enumerate(sources):
                            file_name = doc.metadata.get('source_file', 'Unknown')
                            chapter = doc.metadata.get('chapter', 'Unknown')
                            st.write(f"**{i+1}. {file_name} ({chapter})**")
                            st.caption(doc.page_content[:200] + "...")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
