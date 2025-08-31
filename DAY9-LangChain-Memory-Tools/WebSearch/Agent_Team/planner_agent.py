import os
from typing import Dict, List
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from base_agent import Agent

class PlannerAgent(Agent):
    def __init__(self):
        super().__init__()
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the planner agent."""
        return """You are a Planning Agent that creates detailed solution schemas.
        
        You analyze user requests and break them down into actionable steps. Your job is to:
        1. Understand the user's problem
        2. Create a step-by-step plan to solve it
        3. Design a schema that represents the solution approach
        4. Specify what information needs to be researched
        
        Your output should follow this format:
        ```
        {
          "problem_understanding": "Brief analysis of the user's problem",
          "solution_schema": {
            "steps": ["Step 1", "Step 2", "..."],
            "components": ["Component 1", "Component 2", "..."]
          },
          "research_questions": ["Question 1", "Question 2", "..."]
        }
        ```"""
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for the planner agent."""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
    
    def _create_output_parser(self) -> StructuredOutputParser:
        """Create the output parser for the planner agent."""
        planner_output_schemas = [
            ResponseSchema(name="problem_understanding", description="Brief analysis of the user's problem"),
            ResponseSchema(name="solution_schema", description="Object containing steps and components for the solution"),
            ResponseSchema(name="research_questions", description="List of questions that need to be researched")
        ]
        
        return StructuredOutputParser.from_response_schemas(planner_output_schemas)
    
    def run(self, query: str, history: List = []) -> Dict:
        """Run the planner agent on a query"""
        result = self.chain.invoke({"input": query, "history": history})
        return result