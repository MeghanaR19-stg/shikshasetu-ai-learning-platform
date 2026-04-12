from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from backend.lesson_generator import LessonSchema
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def translate_lesson(lesson_data, target_language, api_key):
    """
    Translates the lesson into the target language AND performs 
    an internal quality audit to ensure script and academic tone accuracy.
    Consolidated into one step to prevent API rate limiting.
    """
    if target_language.lower() == "english":
        return lesson_data

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", 
        google_api_key=api_key,
        temperature=0.1
    )
    
    parser = JsonOutputParser(pydantic_object=LessonSchema)
    
    template = """
    You are an expert Multilingual Academic Editor specializing in Indian NCERT curriculum.
    
    TASK: Translate the following structured lesson from English into {language}.
    
    DIRECTIONS:
    1. Translate every value (Explanation, Key Points, Examples, Quiz) into {language}.
    2. Maintain a FORMAL ACADEMIC TONE (NCERT Textbook style).
    3. Use the correct NATIVE SCRIPT for {language}.
    4. QUALITY AUDIT: After translating, review your own work. Ensure NO English words "leaked" into the native text and that grammar is perfect.
    
    CRITICAL: 
    - Maintain the exact same JSON keys.
    - DO NOT include trailing commas.
    - Output ONLY the final validated JSON.
    
    LESSON DATA (JSON):
    {lesson_data}
    
    {format_instructions}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = prompt | llm | parser
    
    response = chain.invoke({
        "language": target_language,
        "lesson_data": lesson_data,
        "format_instructions": parser.get_format_instructions()
    })
    
    return response
