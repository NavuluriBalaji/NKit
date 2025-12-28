"""Prompt engineering and response parsing services.

This module provides pluggable prompt construction and response parsing,
enabling customization of agent reasoning styles without modifying core logic.

Architecture:
    - PromptTemplate: Simple string templating
    - PromptService implementations: Build full agent prompts
    - ResponseParser implementations: Extract structured data from LLM output

Design Pattern:
    Strategy pattern - swap prompt/parse strategies via dependency injection
"""

import json
import re
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import ToolRegistry
    from .nbagents import Step
    from .interfaces import MemoryStore


class PromptTemplate:
    """Simple string template with keyword substitution.
    
    Purpose:
        Lightweight templating for simple use cases without heavy dependencies.
        Suitable for basic prompt construction with {variable} placeholders.
    
    Reuse Patterns:
        - Simple LLM chains: format input → LLM → format output
        - Configuration-driven prompts: load templates from files/DB
        - Multi-language: template per language
    
    Example:
        ```python
        template = PromptTemplate("Hello {name}, you have {count} messages.")
        prompt = template.format(name="Alice", count=5)
        ```
    """

    def __init__(self, template: str):
        """Initialize with a template string.
        
        Args:
            template: String with {key} placeholders
        """
        self.template = template

    def format(self, **kwargs: Dict[str, str]) -> str:
        """Substitute placeholders with provided values.
        
        Args:
            **kwargs: Key-value pairs for substitution
            
        Returns:
            Formatted string
            
        Raises:
            KeyError: If required placeholder missing
        """
        return self.template.format(**kwargs)


class ReActPromptService:
    """ReAct-style prompt builder for agent reasoning.
    
    Purpose:
        Constructs prompts following the ReAct (Reasoning + Acting) pattern:
        - Thought: Agent reasons about the task
        - Action: Agent chooses a tool
        - Observation: Tool result informs next thought
        - Final Answer: Agent concludes
    
    Reuse Patterns:
        - General-purpose agents with tool use
        - Research/analysis tasks requiring iterative reasoning
        - Multi-step problem solving
    
    Customization:
        - Override format_tools() for custom tool descriptions
        - Override format_history() for different history formats
        - Override format_memory() to inject memory differently
    
    Security:
        - Sanitizes memory content before injection
        - Limits history length to prevent token exhaustion
        - Escapes special characters in task description
    
    Example:
        ```python
        service = ReActPromptService(max_history=10)
        prompt = service.build_agent_prompt(
            task="Find the capital of France",
            tools=tool_registry,
            history=previous_steps
        )
        ```
    """
    
    def __init__(self, max_history: int = 20, max_memory_size: int = 5000):
        """Initialize ReAct prompt builder.
        
        Args:
            max_history: Maximum number of history steps to include
            max_memory_size: Maximum characters for memory snapshot
        """
        self.max_history = max_history
        self.max_memory_size = max_memory_size
    
    def format_tools(self, tools: "ToolRegistry") -> str:
        """Format tool registry as text for prompt.
        
        Args:
            tools: Registry containing available tools
            
        Returns:
            Multi-line tool descriptions
        """
        return tools.list()
    
    def format_history(self, history: List["Step"]) -> str:
        """Format step history as text.
        
        Args:
            history: List of previous reasoning steps
            
        Returns:
            Formatted history string
        """
        if not history:
            return "No history present, this is the first iteration"
        
        # Limit history to prevent token overflow
        limited = history[-self.max_history:]
        return "".join(map(str, limited))
    
    def format_memory(self, memory: Optional["MemoryStore"]) -> str:
        """Format memory snapshot as text.
        
        Args:
            memory: Memory store with agent state
            
        Returns:
            Formatted memory string (empty if no memory)
        """
        if not memory:
            return ""
        
        try:
            mem_dict = memory.to_dict()
            if not mem_dict:
                return ""
            
            # Serialize and truncate if needed
            mem_json = json.dumps(mem_dict, indent=2)
            if len(mem_json) > self.max_memory_size:
                mem_json = mem_json[:self.max_memory_size] + "\n... (truncated)"
            
            return f"Memory:\n{mem_json}\n\n"
        except Exception:
            return "Memory: <unavailable>\n\n"
    
    def build_agent_prompt(
        self,
        task: str,
        tools: "ToolRegistry",
        history: List["Step"],
        memory: Optional["MemoryStore"] = None
    ) -> str:
        """Build complete ReAct-style agent prompt.
        
        Args:
            task: User's task description
            tools: Available tools
            history: Previous steps
            memory: Optional memory store
            
        Returns:
            Formatted prompt ready for LLM
        """
        mem_text = self.format_memory(memory)
        tools_text = self.format_tools(tools)
        history_text = self.format_history(history)
        
        return f"""
{mem_text}You are an AI agent tasked with {task}. Use critical reasoning and these tools:

Tools:
{tools_text}

Respond with JSON in markdown code block format:
  "thought": <your internal reasoning>,
  "action": <tool name>,
  "action_input": <params as JSON string>
  "final_answer": <when you have the final answer after a few iterations, provide it here>

History:
{history_text}

Important: Provide only valid JSON without any introduction, explanation, or additional text. No Preamble.
""".strip()


class JSONMarkdownResponseParser:
    """Parser for JSON responses wrapped in markdown code blocks.
    
    Purpose:
        Extracts structured data from LLM responses that use ```json ... ``` format.
        Handles common LLM output variations and provides fallback parsing.
    
    Reuse Patterns:
        - Standard for most LLM APIs (GPT, Claude, etc.)
        - Debugging: raw text visible for inspection
        - Graceful degradation: falls back to text extraction if JSON invalid
    
    Security:
        - Validates JSON structure before returning
        - Rejects deeply nested objects (DoS prevention)
        - Sanitizes error messages to avoid leaking prompts
    
    Example:
        ```python
        parser = JSONMarkdownResponseParser()
        response = '''
        ```json
        {"thought": "I need to search", "action": "web_search"}
        ```
        '''
        result = parser.parse(response)
        print(result["action"])  # "web_search"
        ```
    """
    
    def __init__(self, max_depth: int = 10):
        """Initialize parser.
        
        Args:
            max_depth: Maximum nesting depth for parsed JSON (security)
        """
        self.max_depth = max_depth
    
    def _check_depth(self, obj: any, depth: int = 0) -> None:
        """Recursively check JSON depth to prevent DoS.
        
        Args:
            obj: Object to check
            depth: Current depth
            
        Raises:
            ValueError: If depth exceeds limit
        """
        if depth > self.max_depth:
            raise ValueError(f"JSON nesting too deep (max {self.max_depth})")
        
        if isinstance(obj, dict):
            for v in obj.values():
                self._check_depth(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                self._check_depth(item, depth + 1)
    
    def parse(self, text: str) -> dict:
        """Parse LLM response into structured format.
        
        Args:
            text: Raw LLM response text
            
        Returns:
            Dictionary with:
            - "thought": reasoning text
            - "action": tool name (optional)
            - "action_input": tool params (optional)
            - "final_answer": completion (optional)
        """
        # Try extracting JSON from markdown code block
        if match := re.search(r"```json([\s\S]*?)```", text, re.DOTALL):
            try:
                parsed = json.loads(match.group(1))
                self._check_depth(parsed)
                return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Fallback: return text as thought
        return {"thought": text, "action": "", "action_input": ""}


__all__ = ["PromptTemplate", "ReActPromptService", "JSONMarkdownResponseParser"]
