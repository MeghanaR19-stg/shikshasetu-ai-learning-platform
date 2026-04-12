from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def process_followup(current_lesson, user_query, language, api_key):
    """
    Handles follow-up questions about a specific lesson.
    Uses the lesson content as context to provide accurate answers.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        google_api_key=api_key,
        temperature=0.4
    )
    
    template = """
    You are an expert educational AI tutor for Indian students on the ShikshaSetu platform.
    
    CURRENT LESSON CONTEXT:
    {lesson_text}
    
    The student is asking a follow-up question in {language}:
    "{query}"
    
    REQUIREMENTS:
    1. Respond ONLY in {language}.
    2. Maintain an encouraging and academic tone.
    3. If the question is "Simplify," explain the lesson's main concepts in much simpler terms with relatable analogies.
    4. If it's a new question, answer it based on the lesson context and NCERT curriculum standards.
    5. Be concise but helpful.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # We pass the lesson as a string block
    lesson_str = f"EXPLANATION: {current_lesson['lesson_explanation']}\nKEY POINTS: {', '.join(current_lesson['key_points'])}"
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "lesson_text": lesson_str,
        "query": user_query,
        "language": language
    })
    
    return response
