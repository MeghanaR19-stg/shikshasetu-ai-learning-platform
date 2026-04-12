# ShikshaSetu

AI-powered multilingual learning platform using NCERT curriculum knowledge.

## Features

• Curriculum-aware RAG pipeline
• Multilingual lesson generation
• AI mentor assistant
• Quiz generation
• Student performance analytics

## Architecture

PDF Ingestion → RAG → LLM → Translation → Validation → Learning Interface

## Tech Stack

Python
Streamlit
FastAPI
ChromaDB
LangChain
Sentence Transformers

## Run Locally

pip install -r requirements.txt

Run backend:
uvicorn backend.main:app --reload

Run frontend:
streamlit run frontend/app.py
