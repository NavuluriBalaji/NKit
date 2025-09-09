"""Top-level compatibility package for NKit.

This module re-exports the public API from the project's core implementation
so users can import using `import nkit` or `from nkit import Agent` without
needing to know the internal package layout.
"""
from importlib import import_module

_core = import_module('nkit.nbagents')

_exports = [
    "Agent", "Tool", "ToolRegistry", "BuiltinTools", "Step", "setup_logger",
    "Memory", "Chain", "LLMAdapter", "CallableLLMAdapter", "PromptTemplate",
    "LLMChain",
]

for _name in _exports:
    try:
        globals()[_name] = getattr(_core, _name)
    except AttributeError:
        # Some helpers live in subpackages, import as needed
        if _name == "Memory":
            from .memory import Memory as MemoryImpl
            globals()[_name] = MemoryImpl
        elif _name in ("Chain", "LLMChain"):
            from .chain import Chain as ChainImpl, LLMChain as LLMChainImpl
            globals()["Chain"] = ChainImpl
            globals()["LLMChain"] = LLMChainImpl
        elif _name in ("LLMAdapter", "CallableLLMAdapter", "PromptTemplate"):
            from .llm_adapter import LLMAdapter, CallableLLMAdapter
            from .prompt import PromptTemplate
            globals()["LLMAdapter"] = LLMAdapter
            globals()["CallableLLMAdapter"] = CallableLLMAdapter
            globals()["PromptTemplate"] = PromptTemplate
        elif _name in ("Tool", "ToolRegistry", "BuiltinTools"):
            from .tools import Tool, ToolRegistry
            from .tools.builtin_tools import BuiltinTools
            globals()["Tool"] = Tool
            globals()["ToolRegistry"] = ToolRegistry
            globals()["BuiltinTools"] = BuiltinTools

__all__ = _exports
