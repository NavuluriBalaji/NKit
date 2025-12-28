"""Utility functions for async handling, schema parsing, and logging.

This module provides shared utilities used throughout the framework.

Purpose:
    - Unified async/sync function detection and execution
    - Function schema introspection for tool registration
    - Centralized logging configuration

Reuse:
    Import and use these utilities in any custom component to maintain
    consistency with the framework's patterns.
"""

import asyncio
import inspect
import logging
from typing import Any, Callable


def setup_logger(name: str = "nkit", level: str = "INFO") -> logging.Logger:
    """Setup and configure a logger instance.
    
    Purpose:
        Provides consistent logging format across all components.
        Creates logger with stream handler if not already configured.
    
    Args:
        name: Logger name (hierarchical, e.g., "nkit.agent")
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    
    Reuse Patterns:
        - Component logging: `logger = setup_logger("nkit.my_component")`
        - Debug mode: `setup_logger(level="DEBUG")`
        - Production: `setup_logger(level="WARNING")`
    
    Design Note:
        Checks for existing handlers to avoid duplicate logging.
    
    Example:
        ```python
        from nkit.utils import setup_logger
        
        logger = setup_logger("my_app", "DEBUG")
        logger.debug("Detailed info")
        logger.info("Normal operation")
        ```
    """
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


def is_async_function(func: Callable) -> bool:
    """Check if a function is async (coroutine).
    
    Purpose:
        Detect async functions to handle them appropriately (await vs direct call).
    
    Args:
        func: Function or callable to check
    
    Returns:
        True if func is async, False otherwise
    
    Reuse Patterns:
        - Tool registration: detect sync/async tools
        - LLM adapters: handle sync/async LLM clients
        - Node handlers: detect async graph nodes
    
    Example:
        ```python
        async def fetch(): ...
        def process(): ...
        
        is_async_function(fetch)    # True
        is_async_function(process)  # False
        ```
    """
    return asyncio.iscoroutinefunction(func)


async def run_sync_or_async(func: Callable, *args, **kwargs) -> Any:
    """Execute a function, handling both sync and async seamlessly.
    
    Purpose:
        Unified execution interface that works with any callable.
        Automatically awaits async functions.
    
    Args:
        func: Function to execute (sync or async)
        *args: Positional arguments
        **kwargs: Keyword arguments
    
    Returns:
        Function result
    
    Reuse Patterns:
        - Tool execution: support both sync and async tools
        - LLM calls: support any LLM client
        - Callbacks: user-provided hooks
    
    Security Note:
        Caller is responsible for validating func and arguments.
    
    Example:
        ```python
        def sync_fn(x): return x * 2
        async def async_fn(x): return x * 2
        
        result1 = await run_sync_or_async(sync_fn, 5)   # 10
        result2 = await run_sync_or_async(async_fn, 5)  # 10
        ```
    """
    if is_async_function(func):
        return await func(*args, **kwargs)
    else:
        return func(*args, **kwargs)


def parse_schema(func: Callable) -> list:
    """Extract parameter schema from function signature.
    
    Purpose:
        Introspect function parameters to generate tool schemas.
        Used by Tool class to document expected arguments.
    
    Args:
        func: Function to introspect
    
    Returns:
        List of dicts with 'name' and 'type' keys for each parameter
    
    Design Note:
        - Skips 'self' parameter (for methods)
        - Defaults to 'str' type if annotation missing
        - Only captures parameter name and type (not defaults or validation)
    
    Reuse Patterns:
        - Tool registration: auto-generate schemas
        - Documentation: list tool parameters
        - Validation: check required params (future enhancement)
    
    Example:
        ```python
        def add(x: int, y: int) -> int:
            return x + y
        
        schema = parse_schema(add)
        # [{'name': 'x', 'type': 'int'}, {'name': 'y', 'type': 'int'}]
        ```
    """
    return [
        {
            'name': n,
            'type': p.annotation.__name__ if p.annotation != inspect.Parameter.empty else 'str'
        }
        for n, p in inspect.signature(func).parameters.items()
        if n != 'self'
    ]


__all__ = [
    "setup_logger",
    "is_async_function", 
    "run_sync_or_async",
    "parse_schema",
]
