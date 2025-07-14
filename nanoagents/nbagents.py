import asyncio
import inspect
import json
import logging
import re
import requests
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, Any
from functools import wraps


def setup_logger(name: str = "nbagents", level: str = "INFO") -> logging.Logger:
    """Setup logger for nbagents."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


logger = setup_logger()

def is_async_function(func: Callable) -> bool:
    """Check if function is async."""
    return asyncio.iscoroutinefunction(func)

async def run_sync_or_async(func: Callable, *args, **kwargs) -> Any:
    """Run function whether it's sync or async."""
    if is_async_function(func):
        return await func(*args, **kwargs)
    else:
        return func(*args, **kwargs)


def _parse_schema(func):
    """Parse function signature for schema."""
    return [{'name': n, 'type': p.annotation.__name__ if p.annotation != inspect.Parameter.empty else 'str'}
            for n, p in inspect.signature(func).parameters.items() if n != 'self']


class Tool:
    """Represents an agent tool with async support."""

    def __init__(self, name: str, func: Callable, desc: str = None):
        self.name = name
        self.func = func
        self.desc = desc or func.__doc__.strip() if func.__doc__ else f"{name} tool"
        self.schema = _parse_schema(func)
        self.is_async = is_async_function(func)

    async def execute(self, **kwargs):
        """Execute tool with async support."""
        logger.debug(f"Executing tool '{self.name}' with args: {kwargs}")
        try:
            result = await run_sync_or_async(self.func, **kwargs)
            logger.debug(f"Tool '{self.name}' completed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool '{self.name}' failed: {e}")
            raise

    def __str__(self): 
        return f"Tool: {self.name}\nDesc: {self.desc}\nArgs: {self.schema}\nAsync: {self.is_async}"


class ToolRegistry:
    """Manages tool collection with built-in tools support."""

    def __init__(self, include_builtin: bool = True): 
        self.tools: Dict[str, Tool] = {}
        if include_builtin:
            self._register_builtin_tools()

    def register(self, tool: Tool): 
        logger.info(f"Registered tool: {tool.name}")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]: 
        return self.tools.get(name)

    def list(self) -> str: 
        return "\n\n".join(map(str, self.tools.values()))

    def _register_builtin_tools(self):
        """Register built-in tools."""
        builtin_tools = [
            ("web_search", BuiltinTools.web_search, "Search the web for information"),
            ("read_file", BuiltinTools.read_file, "Read content from a file"),
            ("write_file", BuiltinTools.write_file, "Write content to a file"),
            ("list_files", BuiltinTools.list_files, "List files in a directory"),
            ("get_time", BuiltinTools.get_current_time, "Get current date and time"),
            ("async_web_search", BuiltinTools.async_web_search, "Async web search"),
        ]
        
        for name, func, desc in builtin_tools:
            self.register(Tool(name, func, desc))

    def decorator(self, name: str, desc: str = None):
        def wrap(func):
            self.register(Tool(name, func, desc))
            return func
        return wrap


class Step:
    """Single reasoning step."""

    def __init__(self, thought: str, index: int = 1):
        self.index = index
        self.thought = thought
        self.action = None
        self.input = None
        self.obs = None

    def set_action(self, action: str, input: dict): self.action, self.input = action, input

    def set_obs(self, obs: str): self.obs = obs

    def __str__(self):
        return f"\n--- Iteration:{self.index} ---\nthought: {self.thought}\n" + \
            (f"action: {self.action}\n" if self.action else "") + \
            (f"action_input: {self.input}\n" if self.input else "") + \
            (f"observation: {self.obs}\n" if self.obs else "")


class ResponseParser:
    """Parses LLM responses."""

    @staticmethod
    def parse(text: str) -> dict:
        if match := re.search(r"```json([\s\S]*?)```", text, re.DOTALL):
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return {"thought": text, "action": "", "action_input": ""}


class PromptFormatter:
    """Formats LLM prompts."""

    @staticmethod
    def format(task: str, tools: ToolRegistry, history: List[Step]) -> str:
        return f"""
You are an AI agent tasked with {task}. Use critical reasoning and these tools:

Tools:
{tools.list()}

Respond with JSON in markdown code block format:
  "thought": <your internal reasoning>,
  "action": <tool name>,
  "action_input": <params as JSON string>
  "final_answer": <when you have the final answer after a few iterations, provide it here>

History:
{"".join(map(str, history)) if history else "No history present, this is the first iteration"}

Important: Provide only valid JSON without any introduction, explanation, or additional text. No Preamble.
""".strip()


class Agent:
    """ReACT-based agent with async support, retry mechanism, and built-in tools."""

    def __init__(self, llm: Callable[[str], str], max_steps: int = 10, max_retries: int = 3, 
                 include_builtin_tools: bool = True, log_level: str = "INFO"):
        self.llm, self.max_steps, self.max_retries = llm, max_steps, max_retries
        self.registry = ToolRegistry(include_builtin_tools)
        self.parser = ResponseParser()
        self.is_llm_async = is_async_function(llm)
        
        # Setup agent-specific logger
        self.logger = setup_logger(f"nbagents.agent", log_level)
        self.logger.info(f"Agent initialized with {len(self.registry.tools)} tools")

    def tool(self, name: str, desc: str = None):
        """Decorator to register a tool."""
        return self.registry.decorator(name, desc)

    def add_tool(self, name: str, func: Callable, desc: str = None):
        """Add a tool to the registry."""
        self.registry.register(Tool(name, func, desc))

    async def _execute_with_retry(self, tool: Tool, inputs: dict) -> str:
        """Execute tool with retry mechanism and async support."""
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Tool execution attempt {attempt + 1}/{self.max_retries}")
                result = await tool.execute(**inputs)
                return str(result)
            except Exception as e:
                self.logger.warning(f"Tool execution attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    error_msg = f"Error after {self.max_retries} retries: {e}"
                    self.logger.error(error_msg)
                    return error_msg
                continue
        return "Unexpected retry failure"

    async def _retry_llm(self, prompt: str, prev_response: str = None) -> dict:
        """Retry LLM call with better prompt."""
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
                    print(f"\n{15 * '='} LLM Response After Retrying {15 * '='}\n{response}\n\n")
                    return resp_dict
                prev_response = response
        raise Exception("Max LLM retries reached")

    async def run_async(self, task: str) -> str:
        """Run the agent asynchronously."""
        self.logger.info(f"Starting async agent run for task: {task[:100]}...")
        history: List[Step] = []
        
        for i in range(self.max_steps):
            prompt = PromptFormatter.format(task, self.registry, history)
            self.logger.debug(f"Iteration {i + 1}/{self.max_steps}")
            print(f'\nIteration: {i + 1}\n{15 * "="} PROMPT {15 * "="}\n{prompt}\n')

            response = re.sub(r'<think>.*?</think>', '', 
                            await run_sync_or_async(self.llm, prompt), 
                            flags=re.DOTALL)
            print(f"\n{15 * '='} LLM Response {15 * '='}\n{response}\n\n")
            resp_dict = self.parser.parse(response)

            if not (thought := resp_dict.get("thought")) or not (resp_dict.get("action") or resp_dict.get("final_answer")):
                self.logger.warning("Invalid LLM response format, retrying...")
                resp_dict = await self._retry_llm(prompt, response)
                thought = resp_dict.get("thought")

            step = Step(thought, i + 1)
            history.append(step)

            if final := resp_dict.get("final_answer"):
                self.logger.info("Agent completed successfully with final answer")
                return final

            if action := resp_dict.get("action"):
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

        error_msg = "Max steps reached without completion"
        self.logger.error(error_msg)
        raise Exception(error_msg)

    def run(self, task: str) -> str:
        """Run the agent (sync wrapper for async)."""
        if asyncio.get_event_loop().is_running():
            # If already in an event loop, create a new one
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.run_async(task))
                return future.result()
        else:
            return asyncio.run(self.run_async(task))


class BuiltinTools:
    """Collection of built-in tools."""
    
    @staticmethod
    def web_search(query: str, num_results: int = 3) -> str:
        """Search the web for information."""
        try:
            # Simple web search using DuckDuckGo Instant Answer API
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data.get('AbstractText'):
                return f"Search result: {data['AbstractText']}"
            elif data.get('Answer'):
                return f"Answer: {data['Answer']}"
            else:
                return f"No direct answer found for query: {query}"
        except Exception as e:
            return f"Web search failed: {e}"
    
    @staticmethod
    def read_file(file_path: str) -> str:
        """Read content from a file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return f"File not found: {file_path}"
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return f"File content ({len(content)} characters):\n{content}"
        except Exception as e:
            return f"Error reading file: {e}"
    
    @staticmethod
    def write_file(file_path: str, content: str) -> str:
        """Write content to a file."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            return f"Error writing file: {e}"
    
    @staticmethod
    def list_files(directory: str = ".") -> str:
        """List files in a directory."""
        try:
            path = Path(directory)
            if not path.exists():
                return f"Directory not found: {directory}"
            files = [f.name for f in path.iterdir() if f.is_file()]
            dirs = [f.name + "/" for f in path.iterdir() if f.is_dir()]
            all_items = sorted(dirs + files)
            return f"Contents of {directory}:\n" + "\n".join(all_items)
        except Exception as e:
            return f"Error listing directory: {e}"
    
    @staticmethod
    def get_current_time() -> str:
        """Get the current date and time."""
        return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    @staticmethod
    async def async_web_search(query: str, num_results: int = 3) -> str:
        """Async version of web search."""
        import aiohttp
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    data = await response.json()
            
            if data.get('AbstractText'):
                return f"Search result: {data['AbstractText']}"
            elif data.get('Answer'):
                return f"Answer: {data['Answer']}"
            else:
                return f"No direct answer found for query: {query}"
        except Exception as e:
            return f"Async web search failed: {e}"
