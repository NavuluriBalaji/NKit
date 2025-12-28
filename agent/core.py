"""Core agent implementation with ReAct-style reasoning.

This module provides the main Agent class that:
- Iteratively reasons about tasks using LLMs
- Executes tools to gather information
- Maintains memory across iterations
- Returns final answers

Architecture:
    Agent uses dependency injection for all components:
    - LLM: callable or LLMAdapter
    - ToolRegistry: manages available tools
    - PromptService: builds prompts
    - ResponseParser: parses LLM outputs
    - Memory: stores state

Design Principles (SOLID):
    - Single Responsibility: Agent orchestrates; delegates formatting, parsing, execution
    - Open/Closed: Extend via plugins (tools, memory, prompts) without modifying Agent
    - Liskov Substitution: Swap any component implementing the interface
    - Interface Segregation: Small, focused interfaces (see interfaces.py)
    - Dependency Inversion: Agent depends on abstractions, not concrete implementations
"""

import asyncio
import json
import logging
import re
from typing import Callable, List, Optional, Any, TYPE_CHECKING

from ..utils import is_async_function, run_sync_or_async, setup_logger
from ..tools import Tool, ToolRegistry
from ..memory import Memory
from ..legacy.prompt import ReActPromptService, JSONMarkdownResponseParser

if TYPE_CHECKING:
    from .interfaces import MemoryStore, PromptService, ResponseParser


logger = setup_logger("nkit.nbagents")


class Step:
    """Represents one reasoning iteration in the agent's execution.
    
    Purpose:
        Captures the agent's thought process, chosen action, and observation
        for a single step. Used to build execution history for subsequent iterations.
    
    Reuse Patterns:
        - Debugging: trace agent's reasoning
        - Auditing: log decision-making process
        - Training: collect (thought, action, result) tuples
        - Evaluation: compare agent strategies
    
    Attributes:
        index: Step number in sequence (1-based)
        thought: Agent's reasoning text
        action: Tool name chosen (None if final answer)
        input: Tool parameters dict (None if final answer)
        obs: Tool execution result (None if not executed yet)
    
    Example:
        ```python
        step = Step("I need current time", index=1)
        step.set_action("get_time", {})
        step.set_obs("2025-12-28 10:30:00")
        print(step)  # Formatted output
        ```
    """
    
    def __init__(self, thought: str, index: int = 1):
        """Initialize a reasoning step.
        
        Args:
            thought: Agent's reasoning text
            index: Step number (default 1)
        """
        self.index = index
        self.thought = thought
        self.action: Optional[str] = None
        self.input: Optional[dict] = None
        self.obs: Optional[str] = None

    def set_action(self, action: str, input: dict) -> None:
        """Record the action taken in this step.
        
        Args:
            action: Tool name
            input: Tool parameters
        """
        self.action = action
        self.input = input

    def set_obs(self, obs: str) -> None:
        """Record the observation (tool result).
        
        Args:
            obs: Tool execution result
        """
        self.obs = obs

    def __str__(self) -> str:
        """Format step as human-readable string."""
        parts = [f"\n--- Iteration:{self.index} ---", f"thought: {self.thought}"]
        if self.action:
            parts.append(f"action: {self.action}")
        if self.input:
            parts.append(f"action_input: {self.input}")
        if self.obs:
            parts.append(f"observation: {self.obs}")
        return "\n".join(parts) + "\n"


class Agent:
    """ReAct-style agent with iterative reasoning and tool execution.
    
    Purpose:
        Orchestrates LLM-driven task completion by:
        1. Building prompts with task, tools, history, memory
        2. Calling LLM to get reasoning + action
        3. Executing chosen tool
        4. Repeating until final answer or max steps
    
    Architecture:
        - **Dependency Injection**: All components injected via constructor
        - **Strategy Pattern**: Swap prompt/parser strategies
        - **Extensibility**: Add tools via registry or decorator
    
    Reuse Patterns:
        - Research: multi-step information gathering
        - Analysis: iterative data exploration
        - Automation: tool orchestration workflows
        - QA: answer complex questions with tool support
    
    Security:
        - Tool input validation (via ToolRegistry)
        - Memory key sanitization (via Memory implementations)
        - LLM response validation (via ResponseParser)
        - Max steps/retries to prevent infinite loops
    
    Example (basic):
        ```python
        from nkit import Agent
        
        def my_llm(prompt: str) -> str:
            return llm_api_call(prompt)
        
        agent = Agent(llm=my_llm)
        result = agent.run("What is 2+2?")
        print(result)
        ```
    
    Example (advanced with DI):
        ```python
        from nkit import Agent
        from nkit.memory import JSONFileMemory
        from nkit.prompt import ReActPromptService
        from nkit.tools import ToolRegistry, Tool
        
        # Custom components
        memory = JSONFileMemory("./session.json")
        registry = ToolRegistry(include_builtin=False)
        registry.register(Tool("calculator", lambda x, y: x + y))
        prompt_service = ReActPromptService(max_history=5)
        
        agent = Agent(
            llm=my_llm,
            registry=registry,
            memory=memory,
            prompt_service=prompt_service,
            max_steps=15
        )
        
        result = agent.run("Calculate 42 + 58")
        ```
    
    Example (with decorator):
        ```python
        agent = Agent(llm=my_llm)
        
        @agent.tool("greet", "Greet a user")
        def greet(name: str) -> str:
            return f"Hello, {name}!"
        
        agent.run("Greet Alice")
        ```
    """
    
    def __init__(
        self,
        llm: Callable[[str], str],
        max_steps: int = 10,
        max_retries: int = 3,
        include_builtin_tools: bool = True,
        log_level: str = "INFO",
        memory: Optional["MemoryStore"] = None,
        registry: Optional[ToolRegistry] = None,
        prompt_service: Optional["PromptService"] = None,
        response_parser: Optional["ResponseParser"] = None,
    ):
        """Initialize agent with dependencies.
        
        Args:
            llm: LLM callable accepting prompt (str) and returning response (str).
                 Can be sync or async.
            max_steps: Maximum reasoning iterations before failing (default 10).
            max_retries: Tool execution retry attempts (default 3).
            include_builtin_tools: Auto-register built-in tools if True (default True).
                                   Ignored if custom registry provided.
            log_level: Logging level: DEBUG, INFO, WARNING, ERROR (default INFO).
            memory: Memory store instance. Defaults to in-memory Memory().
            registry: Tool registry. Defaults to new ToolRegistry(include_builtin_tools).
            prompt_service: Prompt builder. Defaults to ReActPromptService().
            response_parser: LLM response parser. Defaults to JSONMarkdownResponseParser().
        
        Design Note:
            All dependencies have sensible defaults, preserving backward compatibility.
            Override any component to customize behavior without modifying Agent code.
        
        Security Note:
            - Validate llm is callable
            - max_steps and max_retries prevent runaway execution
            - Tool execution is isolated (no shell=True by default in built-ins)
        """
        if not callable(llm):
            raise TypeError("llm must be callable")
        if max_steps < 1 or max_retries < 1:
            raise ValueError("max_steps and max_retries must be positive")
        
        self.llm = llm
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.is_llm_async = is_async_function(llm)
        
        # Dependency injection with defaults
        self.registry = registry or ToolRegistry(include_builtin=include_builtin_tools)
        self.prompt_service = prompt_service or ReActPromptService()
        self.parser = response_parser or JSONMarkdownResponseParser()
        self.memory = memory or Memory()
        
        self.logger = setup_logger(f"nkit.agent", log_level)
        self.logger.info(f"Agent initialized with {len(self.registry.tools)} tools")

    def tool(self, name: str, desc: str = None):
        """Decorator to register a function as a tool.
        
        Purpose:
            Convenient syntax for adding custom tools inline.
        
        Args:
            name: Tool identifier (used by LLM in actions)
            desc: Human-readable description for prompt
        
        Returns:
            Decorator function
        
        Example:
            ```python
            @agent.tool("sum", "Add two numbers")
            def add(a: int, b: int) -> int:
                return a + b
            ```
        """
        return self.registry.decorator(name, desc)

    def add_tool(self, name: str, func: Callable, desc: str = None) -> None:
        """Programmatically register a tool.
        
        Args:
            name: Tool identifier
            func: Callable (sync or async) to execute
            desc: Description for prompt
        
        Example:
            ```python
            def multiply(x: int, y: int) -> int:
                return x * y
            
            agent.add_tool("multiply", multiply, "Multiply two numbers")
            ```
        """
        self.registry.register(Tool(name, func, desc))

    async def _execute_with_retry(self, tool: Tool, inputs: dict) -> str:
        """Execute a tool with automatic retries on failure.
        
        Purpose:
            Handles transient errors (network, rate limits, etc.) gracefully.
        
        Args:
            tool: Tool instance to execute
            inputs: Parameters for tool
        
        Returns:
            Tool result as string
        
        Retries:
            - Attempts up to self.max_retries times
            - Returns error message if all attempts fail
            - Logs each attempt for debugging
        
        Security:
            - Tool.execute() is responsible for input validation
            - Exceptions are caught and logged (no uncaught crashes)
        """
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Tool '{tool.name}' attempt {attempt + 1}/{self.max_retries}")
                result = await tool.execute(**inputs)
                return str(result)
            except Exception as e:
                self.logger.warning(f"Tool '{tool.name}' attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    error_msg = f"Error after {self.max_retries} retries: {e}"
                    self.logger.error(error_msg)
                    return error_msg
                continue
        return "Unexpected retry failure"

    async def _retry_llm(self, prompt: str, prev_response: str = None) -> dict:
        """Retry LLM call with corrective prompt if response is malformed.
        
        Purpose:
            Handles LLM output format errors by providing feedback and retrying.
        
        Args:
            prompt: Original prompt
            prev_response: Previous malformed response
        
        Returns:
            Parsed response dict
        
        Raises:
            Exception: If max retries reached without valid response
        
        Design Note:
            Uses few-shot correction: shows LLM its error and asks for fix.
        """
        for attempt in range(self.max_retries):
            if attempt > 0:
                retry_prompt = f"""
Your previous response was not in the correct JSON format:
{prev_response}

Please provide a valid JSON response as specified in the original prompt:
{prompt}
"""
                self.logger.debug(f"LLM retry attempt {attempt}")
                response = re.sub(r'<think>.*?</think>', '', 
                                await run_sync_or_async(self.llm, retry_prompt), 
                                flags=re.DOTALL)
                resp_dict = self.parser.parse(response)
                if resp_dict.get("thought") and (resp_dict.get("action") or resp_dict.get("final_answer")):
                    self.logger.info("LLM retry successful")
                    print(f"\n{'=' * 15} LLM Response After Retrying {'=' * 15}\n{response}\n\n")
                    return resp_dict
                prev_response = response
        raise Exception("Max LLM retries reached without valid response")

    async def run_async(self, task: str) -> str:
        """Execute agent task asynchronously (main logic).
        
        Purpose:
            Core reasoning loop:
            1. Build prompt with task + tools + history + memory
            2. Call LLM
            3. Parse response
            4. Execute tool if action present
            5. Record step
            6. Repeat until final_answer or max_steps
        
        Args:
            task: User's task description (natural language)
        
        Returns:
            Final answer string
        
        Raises:
            Exception: If max steps reached or unrecoverable error
        
        Design Note:
            - Async-first design for I/O efficiency
            - History limited by PromptService (token management)
            - Memory updated on completion (last_answer key)
        
        Security Note:
            - Task description should be sanitized by caller if from untrusted source
            - Max steps prevents infinite loops
            - Tool execution failures are logged but don't crash agent
        """
        self.logger.info(f"Starting agent run for task: {task[:100]}...")
        history: List[Step] = []
        
        for i in range(self.max_steps):
            # Build prompt using injected service
            prompt = self.prompt_service.build_agent_prompt(task, self.registry, history, self.memory)
            self.logger.debug(f"Iteration {i + 1}/{self.max_steps}")
            print(f'\nIteration: {i + 1}\n{"=" * 15} PROMPT {"=" * 15}\n{prompt}\n')

            # Call LLM (handles sync/async)
            response = re.sub(r'<think>.*?</think>', '', 
                            await run_sync_or_async(self.llm, prompt), 
                            flags=re.DOTALL)
            print(f"\n{'=' * 15} LLM Response {'=' * 15}\n{response}\n\n")
            
            # Parse response using injected parser
            resp_dict = self.parser.parse(response)

            # Validate and retry if malformed
            if not (thought := resp_dict.get("thought")) or not (resp_dict.get("action") or resp_dict.get("final_answer")):
                self.logger.warning("Invalid LLM response format, retrying...")
                resp_dict = await self._retry_llm(prompt, response)
                thought = resp_dict.get("thought")

            step = Step(thought, i + 1)
            history.append(step)

            # Check for completion
            if final := resp_dict.get("final_answer"):
                self.logger.info("Agent completed successfully with final answer")
                try:
                    self.memory.set("last_answer", final)
                except Exception as e:
                    self.logger.debug(f"Failed to write final answer to memory: {e}")
                return final

            # Execute tool if action present
            if action := resp_dict.get("action"):
                # Parse action_input (may be string or dict)
                inputs = resp_dict["action_input"] if isinstance(resp_dict["action_input"], dict) else json.loads(
                    resp_dict["action_input"])
                
                tool = self.registry.get(action)
                if tool:
                    obs = await self._execute_with_retry(tool, inputs)
                else:
                    obs = f"Tool '{action}' not found"
                    self.logger.error(f"Tool not found: {action}")
                
                step.set_action(action, inputs)
                step.set_obs(obs)
            else:
                error_msg = "No action or final answer provided"
                self.logger.error(error_msg)
                raise Exception(error_msg)

        error_msg = f"Max steps ({self.max_steps}) reached without completion"
        self.logger.error(error_msg)
        raise Exception(error_msg)

    def run(self, task: str) -> str:
        """Execute agent task (sync wrapper for run_async).
        
        Purpose:
            Provides synchronous interface for convenience.
            Handles event loop management automatically.
        
        Args:
            task: User's task description
        
        Returns:
            Final answer string
        
        Design Note:
            - Detects existing event loop to avoid nest-asyncio issues
            - Uses ThreadPoolExecutor if loop already running
            - Otherwise runs asyncio.run() directly
        
        Example:
            ```python
            agent = Agent(llm=my_llm)
            answer = agent.run("Find capital of France")
            print(answer)
            ```
        """
        try:
            # Check if event loop is running
            loop = asyncio.get_running_loop()
            # If we get here, we're in an async context
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.run_async(task))
                return future.result()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            return asyncio.run(self.run_async(task))


__all__ = ["Agent", "Step"]
