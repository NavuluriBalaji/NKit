"""Top-level package for nkit - Modular AI Agent Framework.

This module provides a unified API for the nkit framework, exporting components
from all submodules for convenient access.

Usage:
    from nkit import Agent, Tool  # Public API
    from nkit.agent import Agent, Step  # Direct from agent module
    from nkit.crews import Crew  # New modular API
    from nkit.tasks import Task, TaskManager
    from nkit.llms import OpenAILLM
"""

__version__ = "0.2.0"

# Import from new modular structure
from .agent import Agent, Step
from .tools import Tool, ToolRegistry
from .memory import Memory
from .utils import setup_logger
from .chain import Chain, LLMChain
from .legacy.llm_adapter import LLMAdapter, CallableLLMAdapter
from .legacy.prompt import PromptTemplate

# Expose modular components in public API
__all__ = [
    # Core agent components
    "Agent",
    "Step",
    "Tool",
    "ToolRegistry",
    "setup_logger",
    "Memory",
    "Chain",
    "LLMChain",
    "LLMAdapter",
    "CallableLLMAdapter",
    "PromptTemplate",
    # New modular modules
    "agent",
    "tasks",
    "crews", 
    "llms",
    "knowledge",
    "events",
    "hooks",
    "telemetry",
    "cli",
]
