"""
Agent Team Package

This package contains AI agents for planning, research, and recommendations.
All agents inherit from the base Agent class for consistent configuration.
"""

from .base_agent import Agent
from .planner_agent import PlannerAgent
from .research_agent import ResearcherAgent
from .recommendation_agent import RecommendationAgent
from .team_compiler import AgentTeam

__all__ = [
    'Agent',
    'PlannerAgent', 
    'ResearcherAgent',
    'RecommendationAgent',
    'AgentTeam'
]
