from typing import Callable, Dict, Optional
from ..utils import parse_schema, is_async_function, run_sync_or_async
import logging

logger = logging.getLogger("nkit.tools")


class Tool:
    def __init__(self, name: str, func: Callable, desc: str = None):
        self.name = name
        self.func = func
        self.desc = desc or (func.__doc__.strip() if func.__doc__ else f"{name} tool")
        self.schema = parse_schema(func)
        self.is_async = is_async_function(func)

    async def execute(self, **kwargs):
        """Execute tool with proper error handling.
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
            
        Raises:
            ValidationError: If inputs are invalid
            ToolError: If tool execution fails
            Exception: For unexpected errors
        """
        logger.debug(f"Executing tool '{self.name}' with args: {kwargs}")
        try:
            result = await run_sync_or_async(self.func, **kwargs)
            logger.debug(f"Tool '{self.name}' completed successfully")
            return result
        except Exception as e:
            # Import here to avoid circular dependency
            from .builtin_tools import ValidationError, ToolError
            
            if isinstance(e, (ValidationError, ToolError)):
                logger.warning(f"Tool '{self.name}' input/execution error: {e}")
                raise
            else:
                logger.error(f"Tool '{self.name}' failed unexpectedly: {e}")
                raise

    def __str__(self): 
        return f"Tool: {self.name}\nDesc: {self.desc}\nArgs: {self.schema}\nAsync: {self.is_async}"


class ToolRegistry:
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
        from .builtin_tools import BuiltinTools
        from .dynamic import get_codeact_tools
        
        # Create a default instance of BuiltinTools
        tools_instance = BuiltinTools()
        
        # Register bound methods as tools
        builtin_tools = [
            ("web_search", tools_instance.web_search, "Search the web for information"),
            ("read_file", tools_instance.read_file, "Read content from a file"),
            ("write_file", tools_instance.write_file, "Write content to a file"),
            ("list_files", tools_instance.list_files, "List files in a directory"),
            ("get_time", tools_instance.get_current_time, "Get current date and time"),
            ("async_web_search", tools_instance.async_web_search, "Async web search"),
        ]
        for name, func, desc in builtin_tools:
            self.register(Tool(name, func, desc))

        # Dynamically inject CodeACT capabilities
        try:
            for codeact_tool in get_codeact_tools(self):
                self.register(codeact_tool)
        except Exception as e:
            logger.warning(f"Failed to bind CodeACT tools: {e}")

    def decorator(self, name: str, desc: str = None):
        def wrap(func):
            self.register(Tool(name, func, desc))
            return func
        return wrap


__all__ = ["Tool", "ToolRegistry"]
