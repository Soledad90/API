"""
Entry point for the LLM-driven trading agent.

Usage
-----
    python main.py

Environment
-----------
Set OPENROUTER_API_KEY in a .env file (copy .env.example as a template).
"""

from src.agent import run_agent_llm

if __name__ == "__main__":
    run_agent_llm()
