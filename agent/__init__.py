"""Agent module - Core agent implementation.

Public API for creating and running ReAct-style agents.

Usage:
    from nkit.agent import Agent, Step
    
    agent = Agent(llm=my_llm)
    result = agent.run("Task description")
"""

from .core import Agent, Step

__all__ = ["Agent", "Step"]
