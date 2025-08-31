import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.base import BaseLanguageModel

try:
    from dotenv import load_dotenv
    load_dotenv()  # .env dosyasını yükle
except ImportError:
    # python-dotenv kurulmamışsa manuel olarak yükle
    def load_env_file(env_file: str = ".env"):
        """Environment değişkenlerini .env dosyasından yükler"""
        if os.path.exists(env_file):
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    load_env_file()

def load_llm_config() -> BaseLanguageModel:
    """Environment değişkenlerinden Google Gemini LLM'i yükler"""
    
    # Google Gemini için - API key'i environment'dan al
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API Key bulunamadı. Lütfen .env dosyasında GOOGLE_API_KEY değişkenini ayarlayın.")
    
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        max_tokens=1024,
        timeout=600,
    )

# Global LLM instance
llm = load_llm_config()
