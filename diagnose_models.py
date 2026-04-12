import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

def list_gemini_models(api_key):
    genai.configure(api_key=api_key)
    print("Listing available models for your API key:")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # If API key is in environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Please enter your Gemini API Key: ")
    list_gemini_models(api_key)
