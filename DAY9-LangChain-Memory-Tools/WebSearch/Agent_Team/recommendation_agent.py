import os
from typing import Dict
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from base_agent import Agent
import json

class RecommendationAgent(Agent):
    def __init__(self):
        super().__init__()
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the recommendation agent."""
        return """You are a Recommendation Agent that creates comprehensive, detailed implementation plans in beautifully formatted markdown.
            You'll receive a problem description, a solution schema, and research findings. Your job is to:
            Create a detailed, structured implementation report with clear sections, examples, and actionable steps in markdown format.

            Format your response EXACTLY like this:
            
				### Executive Summary
				Brief overview...
				
				### Detailed Implementation Plan
				
				#### Section 1
				**Objective:**...
				**Key Components:**...
				**Implementation Details:**...
				**Challenges & Solutions:**...
				---
				[More sections...]
				
				### Tools & Resources
				- **Tool**: Description...
				
				### Key Risks & Mitigations
				- **Risk**: Mitigation...
				
				### Expected Outcomes
				- Outcome 1...
				
				### Next Steps
				1. Step 1...
            
            Important Formatting Notes: 
            - Use ONLY MARKDOWN FORMATTING, no JSON or code blocks
            - Never wrap the response in ```markdown tags
            - Maintain the exact header structure"""
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for the recommendation agent."""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            ("human", "Problem: {problem}\nSolution Schema: {schema}\nResearch Findings: {findings}"),
        ])
    
    def _create_output_parser(self) -> StrOutputParser:
        """Create the output parser for the recommendation agent."""
        return StrOutputParser()
    
    def run(self, problem: str, schema: Dict, findings: Dict) -> str:
        """Run the recommendation agent and return markdown text"""
        return self.chain.invoke({
            "problem": problem,
            "schema": json.dumps(schema, indent=2),
            "findings": json.dumps(findings, indent=2)
        })