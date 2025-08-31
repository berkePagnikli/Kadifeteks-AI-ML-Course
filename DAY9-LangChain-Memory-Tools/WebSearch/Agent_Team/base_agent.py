import os
from abc import ABC, abstractmethod
from typing import Any
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Agent(ABC):
    """Base class for all agents with common Google Gemini configuration and system prompt setup."""
    
    def __init__(self):
        """Initialize the agent with Google Gemini configuration."""
        
        self.llm = self._create_llm()
        self.system_prompt = self._get_system_prompt()
        self.prompt_template = self._create_prompt_template()
        self.output_parser = self._create_output_parser()
        self.chain = self._create_chain()
    
    def _create_llm(self) -> ChatGoogleGenerativeAI:
        """Create and configure the Google Gemini LLM instance."""
        
        gemini_key =  os.getenv("GOOGLE_GEMINI_KEY")

        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=gemini_key
        )
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for this agent. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _create_output_parser(self) -> Any:
        """Create the output parser for this agent. Must be implemented by subclasses."""
        pass
    
    def _create_chain(self) -> Any:
        """Create the LLM chain. Can be overridden by subclasses if needed."""
        return self.prompt_template | self.llm | self.output_parser
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute the agent's main functionality. Must be implemented by subclasses."""
        pass
