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
import uuid
from typing import Callable, List, Optional, Any, TYPE_CHECKING

from ..utils import is_async_function, run_sync_or_async, setup_logger
from ..tools import Tool, ToolRegistry
from ..memory import Memory
from ..legacy.prompt import ReActPromptService, JSONMarkdownResponseParser

if TYPE_CHECKING:
    from .interfaces import MemoryStore, PromptService, ResponseParser


logger = setup_logger("nkit.nbagents")

# Try to import PoT components (optional, for PoT mode)
try:
    from ..planner import ThoughtPlanner
    from ..executor import ThoughtExecutor
    HAS_POT = True
except ImportError:
    HAS_POT = False
    ThoughtPlanner = None
    ThoughtExecutor = None


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
        max_steps: int = 20,
        max_retries: int = 3,
        include_builtin_tools: bool = True,
        log_level: str = "INFO",
        memory: Optional["MemoryStore"] = None,
        registry: Optional[ToolRegistry] = None,
        prompt_service: Optional["PromptService"] = None,
        response_parser: Optional["ResponseParser"] = None,
        observer: Optional[Any] = None,
        safety_gate: Optional[Any] = None,
        why_log: Optional[Any] = None,
        reasoning_mode: str = "react",  # "react" or "pot"
        planner: Optional[Any] = None,
        executor: Optional[Any] = None,
    ):
        """Initialize agent with dependencies.
        
        Args:
            llm: LLM callable accepting prompt (str) and returning response (str).
            max_steps: Maximum reasoning iterations (default 20).
            max_retries: Tool execution retry attempts (default 3).
            include_builtin_tools: Auto-register built-in tools (default True).
            log_level: Logging level: DEBUG, INFO, WARNING, ERROR (default INFO).
            memory: Memory store instance (default in-memory Memory()).
            registry: Tool registry (default ToolRegistry with built-ins).
            prompt_service: Prompt builder (default ReActPromptService).
            response_parser: LLM response parser (default JSONMarkdownResponseParser).
            observer: LiveObserver for event streaming (optional).
            safety_gate: SafetyGate for pre-execution verification (optional).
            why_log: WhyLog for audit trail (optional).
            reasoning_mode: "react" (iterative) or "pot" (plan once, execute) (default "react").
            planner: ThoughtPlanner for PoT mode (optional, auto-created if needed).
            executor: ThoughtExecutor for PoT mode (optional, auto-created if needed).
        
        Design Note:
            Agent supports two reasoning modes:
            - ReAct (default): LLM called multiple times iteratively
            - PoT (Program of Thought): LLM called once to generate complete plan
            
            All dependencies have sensible defaults, preserving backward compatibility.
        """
        if not callable(llm):
            raise TypeError("llm must be callable")
        if max_steps < 1 or max_retries < 1:
            raise ValueError("max_steps and max_retries must be positive")
        if reasoning_mode not in ("react", "pot"):
            raise ValueError("reasoning_mode must be 'react' or 'pot'")
        
        self.llm = llm
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.reasoning_mode = reasoning_mode
        self.is_llm_async = is_async_function(llm)
        
        # Dependency injection with defaults
        self.registry = registry or ToolRegistry(include_builtin=include_builtin_tools)
        self.prompt_service = prompt_service or ReActPromptService()
        self.parser = response_parser or JSONMarkdownResponseParser()
        self.memory = memory or Memory()
        self.observer = observer
        self.safety_gate = safety_gate
        self.why_log = why_log
        
        # Session tracking for audit trail
        self.session_id = str(uuid.uuid4())
        
        # PoT mode components (lazy initialization if needed)
        self.planner = planner
        self.executor = executor
        
        self.logger = setup_logger(f"nkit.agent", log_level)
        self.logger.info(f"Agent initialized with {len(self.registry.tools)} tools (mode: {reasoning_mode}, session: {self.session_id})")

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
                # Enforce tool execution timeout (30s max)
                result = await asyncio.wait_for(tool.execute(**inputs), timeout=30.0)
                result_str = str(result)
                # Enforce result length limit (10KB approx 10000 chars)
                if len(result_str) > 10000:
                    result_str = result_str[:9997] + "..."
                return result_str
            except asyncio.TimeoutError:
                self.logger.warning(f"Tool '{tool.name}' timed out after 30s")
                if attempt == self.max_retries - 1:
                    return f"Error: Tool '{tool.name}' exceeded 30s timeout on all retries."
                continue
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
            4. Execute tool if action present (with SafetyGate pre-check)
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
            - SafetyGate intercepts dangerous actions
            - WhyLog captures full reasoning chain
            - Observer emits real-time events
        
        Security Note:
            - Task description should be sanitized by caller if from untrusted source
            - Max steps prevents infinite loops
            - Tool execution failures are logged but don't crash agent
            - SafetyGate blocks destructive actions before execution
            - All actions logged to WhyLog for audit trail
        """
        self.logger.info(f"Starting agent run for task: {task[:100]}... (session: {self.session_id})")
        if self.observer:
            self.observer.emit("agent.start", task=task, max_steps=self.max_steps, session_id=self.session_id)
        
        history: List[Step] = []
        
        try:
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

                if self.observer:
                    self.observer.emit("agent.reasoning", thought=thought, step=i+1, goal=task, session_id=self.session_id)

                step = Step(thought, i + 1)
                history.append(step)

                # Check for completion
                if final := resp_dict.get("final_answer"):
                    self.logger.info("Agent completed successfully with final answer")
                    try:
                        self.memory.set("last_answer", final)
                    except Exception as e:
                        self.logger.debug(f"Failed to write final answer to memory: {e}")
                    
                    # Log to WhyLog
                    if self.why_log:
                        try:
                            self.why_log.log(
                                session_id=self.session_id,
                                event_type="completion",
                                goal=task,
                                thought=thought,
                                action=None,
                                result=final,
                                why="Agent reached final answer"
                            )
                        except Exception as e:
                            self.logger.debug(f"Failed to log to WhyLog: {e}")
                    
                    if self.observer:
                        self.observer.emit("agent.end", final_answer=final, total_steps=i+1, session_id=self.session_id)
                    return final

                # Execute tool if action present
                if action := resp_dict.get("action"):
                    # Parse action_input (may be string or dict)
                    inputs = resp_dict["action_input"] if isinstance(resp_dict["action_input"], dict) else json.loads(
                        resp_dict["action_input"])
                    
                    tool = self.registry.get(action)
                    
                    # PRE-EXECUTION: Safety check via SafetyGate
                    gate_blocked = False
                    gate_blocked_reason = None
                    human_approved = False
                    
                    if self.safety_gate and tool:
                        try:
                            self.logger.debug(f"Running safety gate check for tool '{action}'")
                            gate_result = self.safety_gate.evaluate(
                                tool_name=action,
                                tool_args=inputs,
                                goal=task,
                                why=thought
                            )
                            gate_blocked = not gate_result.get("allowed", True)
                            gate_blocked_reason = gate_result.get("reason", "Blocked by SafetyGate")
                            human_approved = gate_result.get("human_approved", False)
                            
                            if gate_blocked:
                                self.logger.warning(f"SafetyGate blocked tool '{action}': {gate_blocked_reason}")
                        except Exception as e:
                            self.logger.error(f"SafetyGate evaluation failed: {e}")
                    
                    if self.observer:
                        self.observer.emit(
                            "tool.before", 
                            tool_name=action, 
                            args=inputs, 
                            why=thought,
                            session_id=self.session_id,
                            blocked=gate_blocked,
                            reason=gate_blocked_reason
                        )
                    
                    success = False
                    obs = ""
                    
                    if gate_blocked:
                        # Tool was blocked by SafetyGate
                        obs = f"Tool '{action}' was blocked: {gate_blocked_reason}"
                        if human_approved:
                            obs += " (but human approved execution)"
                        self.logger.warning(obs)
                    elif tool:
                        # Tool execution with retry
                        obs = await self._execute_with_retry(tool, inputs)
                        success = "Unexpected retry failure" not in str(obs) and "Error after" not in str(obs)
                    else:
                        # Tool not found
                        obs = f"Tool '{action}' not found"
                        self.logger.error(f"Tool not found: {action}")
                    
                    if self.observer:
                        self.observer.emit(
                            "tool.after", 
                            tool_name=action, 
                            result=obs, 
                            success=success,
                            blocked=gate_blocked,
                            session_id=self.session_id
                        )
                    
                    # Log to WhyLog
                    if self.why_log:
                        try:
                            self.why_log.log(
                                session_id=self.session_id,
                                event_type="tool_execution",
                                goal=task,
                                thought=thought,
                                action=action,
                                result=obs[:500],  # Truncate for audit log
                                why=thought,
                                was_blocked=gate_blocked,
                                human_approved=human_approved
                            )
                        except Exception as e:
                            self.logger.debug(f"Failed to log to WhyLog: {e}")
                    
                    step.set_action(action, inputs)
                    step.set_obs(obs)
                else:
                    error_msg = "No action or final answer provided"
                    self.logger.error(error_msg)
                    
                    if self.observer:
                        self.observer.emit("agent.error", error=error_msg, step=i+1, session_id=self.session_id)
                    
                    raise Exception(error_msg)

            # Max steps reached
            error_msg = f"Max steps ({self.max_steps}) reached without completion"
            self.logger.error(error_msg)
            
            if self.observer:
                self.observer.emit("agent.error", error=error_msg, total_steps=self.max_steps, session_id=self.session_id)
            
            if self.why_log:
                try:
                    self.why_log.log(
                        session_id=self.session_id,
                        event_type="error",
                        goal=task,
                        thought="Max steps reached",
                        action=None,
                        result=error_msg,
                        why="Agent exceeded maximum iterations"
                    )
                except Exception as e:
                    self.logger.debug(f"Failed to log to WhyLog: {e}")
            
            raise Exception(error_msg)
            
        except Exception as e:
            # Catch all exceptions and emit error event
            error_str = str(e)
            self.logger.error(f"Agent execution failed: {error_str}")
            
            if self.observer:
                self.observer.emit("agent.error", error=error_str, session_id=self.session_id)
            
            if self.why_log:
                try:
                    self.why_log.log(
                        session_id=self.session_id,
                        event_type="error",
                        goal=task,
                        thought="Exception occurred",
                        action=None,
                        result=error_str,
                        why=f"Exception: {type(e).__name__}"
                    )
                except Exception as log_e:
                    self.logger.debug(f"Failed to log exception to WhyLog: {log_e}")
            
            raise

    def run(self, task: str) -> str:
        """Execute agent task using configured reasoning mode.
        
        Purpose:
            Dispatcher that selects reasoning mode (ReAct or PoT) and executes.
            Provides synchronous interface for convenience.
            Handles event loop management automatically.
        
        Args:
            task: User's task description
        
        Returns:
            Final answer string
        
        Design Note:
            - Detects existing event loop to avoid nest-asyncio issues
            - Dispatches to run_async (ReAct) or run_pot (PoT)
            - Uses ThreadPoolExecutor if loop already running
            - Otherwise runs asyncio.run() directly
        
        Example:
            ```python
            agent = Agent(llm=my_llm)  # default: ReAct mode
            answer = agent.run("Find capital of France")
            
            agent_pot = Agent(llm=my_llm, reasoning_mode="pot")
            answer = agent_pot.run("Find capital of France")  # PoT mode
            ```
        """
        # Dispatch to appropriate reasoning mode
        if self.reasoning_mode == "pot":
            return self._run_pot_sync(task)
        else:
            return self._run_react_sync(task)

    def _run_react_sync(self, task: str) -> str:
        """Execute ReAct mode (iterative reasoning)."""
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

    def _run_pot_sync(self, task: str) -> str:
        """Execute PoT mode (plan once, execute deterministically)."""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.run_pot_async(task))
                return future.result()
        except RuntimeError:
            return asyncio.run(self.run_pot_async(task))

    async def run_pot_async(self, task: str) -> str:
        """Execute task using Program of Thought (plan once).
        
        Purpose:
            LLM generates complete execution plan ONCE, then deterministic
            executor runs each step. LLM is NOT called again during execution.
        
        Args:
            task: User's task description
        
        Returns:
            Final answer string
        
        Raises:
            RuntimeError: If PoT components not available or planning fails
        """
        if not HAS_POT:
            raise RuntimeError("Program of Thought not available. Install planner and executor components.")
        
        # Lazy initialization of planner/executor if not provided
        if self.planner is None:
            if ThoughtPlanner is None:
                raise RuntimeError("ThoughtPlanner not available")
            self.planner = ThoughtPlanner(self.llm, self.registry)
        
        if self.executor is None:
            if ThoughtExecutor is None:
                raise RuntimeError("ThoughtExecutor not available")
            self.executor = ThoughtExecutor(
                self.registry,
                observer=self.observer,
                safety_gate=self.safety_gate,
                audit_log=self.why_log,
                max_retries=self.max_retries
            )
        
        self.logger.info(f"Starting PoT agent for: {task[:100]}... (session: {self.session_id})")
        
        try:
            # Step 1: Planning (LLM called ONCE)
            program = self.planner.plan(task, self.session_id)
            self.logger.info(f"Plan generated: {len(program.steps)} steps, confidence={program.confidence:.2f}")
            
            # Step 2: Execution (deterministic, no LLM calls)
            result = await self.executor.execute(program)
            
            self.logger.info("PoT execution completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"PoT execution failed: {e}")
            if self.observer:
                self.observer.emit("agent.error", error=str(e), session_id=self.session_id)
            raise


__all__ = ["Agent", "Step"]
