from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from backend.lesson_generator import LessonSchema

from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def validate_and_improve_lesson(lesson_data, context, grade, api_key):
    """
    Validates the generated lesson for factual accuracy, curriculum alignment,
    readability, and cultural relevance. Patches errors if found.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        google_api_key=api_key,
        temperature=0.2
    )
    
    parser = JsonOutputParser(pydantic_object=LessonSchema)
    
    template = """
    You are a Senior Curriculum Validator for ShikshaSetu.
    
    TASK: Review the following lesson for Grade {grade} students.
    
    NCERT SOURCE CONTEXT:
    {context}
    
    LESSON TO REVIEW (JSON):
    {lesson_data}
    
    VALIDATION CHECKLIST:
    1. Factual Accuracy: Are there any errors? Cross-reference with the context.
    2. Curriculum Alignment: Does it use the correct terminology from the NCERT context?
    3. Readability: Is the language appropriate for a Grade {grade} student? (Neither too hard nor too simple).
    4. Cultural Relevance: Are the examples relevant to Indian students?
    
    INSTRUCTIONS:
    - If you find errors or areas for improvement, patch them in the JSON below.
    - If the lesson is already perfect, return it as is.
    - Maintain the exact same JSON format.
    - CRITICAL: Ensure the JSON is strictly valid. DO NOT include trailing commas in lists or objects.
    
    {format_instructions}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = prompt | llm | parser
    
    response = chain.invoke({
        "grade": grade,
        "context": context,
        "lesson_data": lesson_data,
        "format_instructions": parser.get_format_instructions()
    })
    
    return response

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def validate_translation_quality(translated_data, original_english_data, language, api_key):
    """
    Verifies the quality of the translated lesson against the original English source.
    Ensures correct script, grammar, and academic professional tone.
    """
    if language.lower() == "english":
        return translated_data

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        google_api_key=api_key,
        temperature=0.1
    )
    
    parser = JsonOutputParser(pydantic_object=LessonSchema)
    
    template = """
    You are a Senior Multilingual Academic Editor for ShikshaSetu.
    
    TASK: Verify the translation quality for a {language} lesson.
    
    ORIGINAL ENGLISH LESSON (Reference):
    {original_english}
    
    TRANSLATED LESSON ({language}):
    {translated_data}
    
    QUALITY CHECKLIST:
    1. Script Accuracy: Is the {language} script 100% correct? Ensure NO English characters are leaked.
    2. Academic Tone: Does it sound like a formal NCERT textbook in {language}?
    3. Meaning Mapping: Does the translation preserve 100% of the educational value from the English version?
    4. Structure: Is the JSON structure exactly preserved?
    
    INSTRUCTIONS:
    - If you find any typos, grammatical errors, or non-academic phrasing, PATCH the {language} JSON.
    - If the translation is perfect, return it as is.
    - CRITICAL: Ensure the JSON is strictly valid. DO NOT include trailing commas.
    
    {format_instructions}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = prompt | llm | parser
    
    response = chain.invoke({
        "language": language,
        "original_english": original_english_data,
        "translated_data": translated_data,
        "format_instructions": parser.get_format_instructions()
    })
    
    return response
