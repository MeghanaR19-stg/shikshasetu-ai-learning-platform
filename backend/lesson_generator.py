from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from tenacity import retry, wait_exponential, stop_after_attempt

class QuizQuestion(BaseModel):
    question: str = Field(description="The question text in the target language")
    options: List[str] = Field(description="4 multiple choice options in the target language")
    answer: str = Field(description="The correct option in the target language")

class LessonSchema(BaseModel):
    lesson_explanation: str = Field(description="Detailed educational content in the target language")
    key_points: List[str] = Field(description="5 detailed bullet points in the target language")
    indian_examples: List[str] = Field(description="Real-world examples from India in the target language")
    short_summary: str = Field(description="A 2-sentence wrap up in the target language")
    quiz: List[QuizQuestion] = Field(description="5 multiple choice questions in the target language")

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def generate_multilingual_lesson(context, topic, grade, language, api_key):
    """
    Consolidates Generation, Validation, and Translation into ONE single AI request.
    This is the most efficient way to maintain quality while respecting Free Tier quotas.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        google_api_key=api_key,
        temperature=0.3
    )
    
    parser = JsonOutputParser(pydantic_object=LessonSchema)
    
    template = """
    You are an expert Educational AI Tutor for ShikshaSetu, specializing in the Indian NCERT curriculum.
    
    TASK: Generate a high-quality, validated lesson in {target_language}.
    
    TOPIC: {topic}
    GRADE: {grade}
    
    NCERT SOURCE CONTEXT (FOR FACT-CHECKING):
    {context}
    
    EXECUTION STEPS (INTERNAL):
    1. ANALYZE the topic based on NCERT guidelines and the provided context.
    2. COMPOSE a high-quality lesson in English first to ensure academic precision.
    3. VALIDATE the English content for grade-appropriateness (Grade {grade}) and factual accuracy.
    4. TRANSFORM the entire lesson into {target_language}.
    5. FORMAL TONE: Use a formal academic tone (textbook style) in {target_language}.
    6. SCRIPT AUDIT: Ensure the native script for {target_language} is perfect and no English words are leaked.
    
    REQUIREMENTS:
    - CLARITY: Use 3-4 distinct paragraphs for the explanation with clear white space between them. Avoid long, dense blocks of text (no "packs of sentences").
    - Include 5 detailed key points.
    - Include 3-4 real-world examples from India (Geography, Scientists, Culture).
    - Include a 5-question Multiple Choice Quiz.
    - Ensure the JSON is strictly valid with NO trailing commas.
    
    FINAL OUTPUT: Return ONLY the structured JSON in {target_language}.
    
    {format_instructions}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = prompt | llm | parser
    
    response = chain.invoke({
        "context": context,
        "topic": topic,
        "grade": grade,
        "target_language": language,
        "format_instructions": parser.get_format_instructions()
    })
    
    return response
