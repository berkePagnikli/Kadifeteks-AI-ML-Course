import os
from typing import Dict, List
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchRun
from base_agent import Agent
import json

class ResearcherAgent(Agent):
  def __init__(self):
    super().__init__()
    search_wrapper = GoogleSearchAPIWrapper()
    self.search_tool = GoogleSearchRun(api_wrapper=search_wrapper)

  def _get_system_prompt(self) -> str:
    """Get the system prompt for the researcher agent."""
    return """You are a Researcher Agent that finds relevant information to answer specific questions.

    You'll receive a list of research questions along with context about a problem. Your job is to:
    1. Search for relevant information for each question
    2. Summarize the findings in a clear and concise way
    3. Cite sources when possible
    4. Organize the information to be useful for solving the original problem

    Your output should follow this format:
    ```
    {
    "research_findings": [
        {
        "question": "The original research question",
        "answer": "Your detailed answer",
        },
        ...
    ],
    "additional_insights": "Any important information you discovered that wasn't explicitly asked for"
    }
    ```"""

  def _create_prompt_template(self) -> ChatPromptTemplate:
    """Create the prompt template for the researcher agent."""
    return ChatPromptTemplate.from_messages([
        SystemMessage(content=self.system_prompt),
        ("human", "Context: {context}\nResearch Questions: {questions}"),
    ])

  def _create_output_parser(self) -> StructuredOutputParser:
    """Create the output parser for the researcher agent."""
    researcher_output_schemas = [
        ResponseSchema(
            name="research_findings",
            description="List of answers to each research question with sources",
        ),
        ResponseSchema(
            name="additional_insights",
            description="Any additional important information discovered",
        ),
    ]

    return StructuredOutputParser.from_response_schemas(researcher_output_schemas)

  def run(self, context: str, questions: List[str]) -> Dict:
    """Run the researcher agent on a set of questions"""
    # First search for each question
    search_results = []
    for question in questions:
        search_result = self.search_tool.run(question)
        search_results.append(
          {"question": question, "search_result": search_result}
        )

    # Then analyze all search results together
    research_context = f"Problem context: {context}\n\nSearch results: {json.dumps(search_results, indent=2)}"
    result = self.chain.invoke(
      {"context": research_context, "questions": questions}
    )
    return result
