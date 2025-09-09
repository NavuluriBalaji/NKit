from typing import Callable, Dict, Optional
from .utils import parse_schema, is_async_function, run_sync_or_async
import logging

logger = logging.getLogger("nbagents.tools")


class Tool:
    """Represents an agent tool with async support."""

    def __init__(self, name: str, func: Callable, desc: str = None):
        self.name = name
        self.func = func
        self.desc = desc or (func.__doc__.strip() if func.__doc__ else f"{name} tool")
        self.schema = parse_schema(func)
        self.is_async = is_async_function(func)

    async def execute(self, **kwargs):
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
    """Manages tool collection with optional built-in registration."""

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
        # import here to avoid circular imports
        from .builtin_tools import BuiltinTools
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
