"""Top-level package for nkit - Modular AI Agent Framework.

This module provides a unified API for the nkit framework, exporting components
from all submodules for convenient access.

Supports two reasoning modes:
- ReAct (default): Iterative reasoning loop (LLM called multiple times)
- PoT (Program of Thought): Plan once, execute deterministically (LLM called once)

Usage:
    from nkit import Agent, Tool  # Public API
    from nkit import ThoughtPlanner, ThoughtExecutor  # PoT components
    from nkit.agent import Agent, Step  # Direct from agent module
"""

__version__ = "0.3.0"

# Import from core modules
from .agent import Agent, Step
from .tools import Tool, ToolRegistry
from .memory import Memory
from .observer import LiveObserver
from .utils import setup_logger
from .chain import Chain, LLMChain
from .legacy.llm_adapter import LLMAdapter, CallableLLMAdapter
from .legacy.prompt import PromptTemplate

# Import PoT components (optional, graceful fallback if not available)
try:
    from .program import ThoughtStep, ThoughtProgram, StepStatus
    from .planner import ThoughtPlanner, PlanningError
    from .executor import ThoughtExecutor, ExecutionError, ToolTimeoutError
    HAS_POT = True
except ImportError:
    HAS_POT = False

# Import safety and audit components
try:
    from .safety import SafetyGate, SafetyViolation
    from .audit import WhyLog
except ImportError:
    pass

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
    "LiveObserver",
    # PoT components (if available)
    "ThoughtStep",
    "ThoughtProgram",
    "StepStatus",
    "ThoughtPlanner",
    "ThoughtExecutor",
    "PlanningError",
    "ExecutionError",
    "ToolTimeoutError",
    # Safety and audit
    "SafetyGate",
    "SafetyViolation",
    "WhyLog",
    # Module names for direct import
    "agent",
    "tasks",
    "crews", 
    "llms",
    "knowledge",
    "events",
    "hooks",
    "telemetry",
    "cli",
    "observer",
    "program",
    "planner",
    "executor",
    "safety",
    "audit",
]
