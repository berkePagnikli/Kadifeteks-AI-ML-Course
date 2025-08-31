from typing import Dict
from langchain_core.messages import HumanMessage, AIMessage
import json
from planner_agent import PlannerAgent
from research_agent import ResearcherAgent
from recommendation_agent import RecommendationAgent

class TeamCompiler:
    def __init__(self):
        self.planner = PlannerAgent()
        self.researcher = ResearcherAgent()
        self.recommender = RecommendationAgent()
        self.conversation_history = []
    
    def add_to_history(self, role, content):
        """Add a message to the conversation history"""
        if role == "human":
            self.conversation_history.append(HumanMessage(content=content))
        elif role == "ai":
            self.conversation_history.append(AIMessage(content=content))
    
    def solve(self, user_query: str) -> Dict:
        """Run the full agent team workflow to solve a user query"""
        # Step 1: Plan the solution
        print("🧠 Planner Agent: Analyzing problem and creating solution schema...")
        plan_result = self.planner.run(user_query, self.conversation_history)
        print(f"\nProblem Understanding: {plan_result['problem_understanding']}")
        print(f"Solution Schema: {json.dumps(plan_result['solution_schema'], indent=2)}")
        print(f"Research Questions: {json.dumps(plan_result['research_questions'], indent=2)}\n")
        
        # Step 2: Conduct research based on the plan
        print("🔍 Researcher Agent: Searching for information...")
        research_result = self.researcher.run(
            plan_result['problem_understanding'],
            plan_result['research_questions']
        )
        print(f"\nResearch Findings: {json.dumps(research_result['research_findings'], indent=2)}")
        print(f"Additional Insights: {research_result['additional_insights']}\n")
        
        # Step 3: Generate comprehensive recommendation
        print("📝 Recommendation Agent: Creating detailed implementation plan...")
        return self.recommender.run(
            user_query,
            plan_result['solution_schema'],
            research_result
        )

if __name__ == "__main__":
    agent_team = TeamCompiler()
    user_query = "how do I increase efficiency in my work from home environment?"
    solution = agent_team.solve(user_query)

    print("\n🛠️ Solution Generated Successfully!")
    print(solution)