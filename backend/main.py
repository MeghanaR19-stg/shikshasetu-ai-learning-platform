from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from backend.rag_pipeline import retrieve_relevant_context
from backend.lesson_generator import generate_multilingual_lesson
from backend.interaction_handler import process_followup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="ShikshaSetu API")

class LessonRequest(BaseModel):
    topic: str
    subject: str
    grade: str
    language: str
    api_key: str

class InteractionRequest(BaseModel):
    lesson_data: dict
    query: str
    language: str
    api_key: str

@app.post("/generate_lesson")
async def generate_lesson(request: LessonRequest):
    try:
        # 1. Retrieve Context from English sources
        context_chunks = retrieve_relevant_context(request.topic, request.api_key)
        context_text = "\n\n".join(context_chunks)
        
        # 2. "One-Shot" Lesson Generation (Consolidated Pipeline)
        # This handles Generation, Validation, and Translation in a single request.
        # This is the most stable and efficient approach for the Gemini Free Tier.
        final_lesson = generate_multilingual_lesson(
            context=context_text,
            topic=request.topic,
            grade=request.grade,
            language=request.language,
            api_key=request.api_key
        )
        
        return final_lesson
        
    except Exception as e:
        error_msg = str(e)
        if "RetryError" in error_msg:
            error_msg = "The API is currently very busy. Please wait a moment and try again."
        elif "RESOURCE_EXHAUSTED" in error_msg:
            error_msg = "API Speed Limit reached. Please try again in 1 minute."
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/ask_question")
async def ask_question(request: InteractionRequest):
    try:
        response = process_followup(
            current_lesson=request.lesson_data,
            user_query=request.query,
            language=request.language,
            api_key=request.api_key
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
