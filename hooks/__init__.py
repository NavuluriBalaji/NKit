"""Lifecycle hooks for agent execution control.

This module provides hooks for intercepting and modifying agent behavior:
- Pre/post execution hooks
- Error handling hooks
- Tool execution hooks
- Custom validation/transformation

Architecture:
    - HookManager: Coordinates hook registration and execution
    - Hook types: Predefined lifecycle points
    - Chain pattern: Multiple hooks execute in sequence
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union


class HookType(Enum):
    """Predefined hook execution points."""
    # Agent lifecycle
    BEFORE_AGENT_RUN = "before_agent_run"
    AFTER_AGENT_RUN = "after_agent_run"
    ON_AGENT_ERROR = "on_agent_error"
    
    # Task lifecycle
    BEFORE_TASK = "before_task"
    AFTER_TASK = "after_task"
    ON_TASK_ERROR = "on_task_error"
    
    # Tool execution
    BEFORE_TOOL_CALL = "before_tool_call"
    AFTER_TOOL_CALL = "after_tool_call"
    ON_TOOL_ERROR = "on_tool_error"
    
    # LLM calls
    BEFORE_LLM_CALL = "before_llm_call"
    AFTER_LLM_CALL = "after_llm_call"
    ON_LLM_ERROR = "on_llm_error"
    
    # Memory operations
    BEFORE_MEMORY_READ = "before_memory_read"
    AFTER_MEMORY_READ = "after_memory_read"
    BEFORE_MEMORY_WRITE = "before_memory_write"
    AFTER_MEMORY_WRITE = "after_memory_write"


@dataclass
class HookContext:
    """Context passed to hooks.
    
    Attributes:
        hook_type: Type of hook being executed
        data: Hook-specific data (input/output/error)
        metadata: Additional context (agent_id, task_id, etc.)
        can_modify: Whether hook can modify data
    """
    hook_type: HookType
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    can_modify: bool = True


class HookResult:
    """Result of hook execution.
    
    Attributes:
        should_continue: Whether to continue execution
        modified_data: Modified data to use (if can_modify=True)
        error: Error to raise (if should_continue=False)
    """
    
    def __init__(
        self,
        should_continue: bool = True,
        modified_data: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ):
        self.should_continue = should_continue
        self.modified_data = modified_data
        self.error = error


class HookManager:
    """Manages lifecycle hooks for agent components.
    
    Purpose:
        Enables aspect-oriented programming for agents:
        - Inject custom logic at key points
        - Validate/transform inputs/outputs
        - Implement cross-cutting concerns (logging, metrics, auth)
    
    Features:
        - Multiple hooks per lifecycle point
        - Sequential execution with early termination
        - Data modification support
        - Sync/async hooks
    
    Reuse Patterns:
        - Validation: Reject invalid inputs in before_tool_call
        - Logging: Record all tool calls in after_tool_call
        - Retry: Implement custom retry logic in on_error hooks
        - Caching: Cache LLM responses in after_llm_call
    
    Example:
        ```python
        manager = HookManager()
        
        # Validate tool inputs
        @manager.register(HookType.BEFORE_TOOL_CALL)
        def validate_tool_input(ctx: HookContext) -> HookResult:
            if not ctx.data.get("arguments"):
                return HookResult(
                    should_continue=False,
                    error=ValueError("Missing arguments")
                )
            return HookResult(should_continue=True)
        
        # Log all tool calls
        @manager.register(HookType.AFTER_TOOL_CALL)
        async def log_tool_call(ctx: HookContext):
            print(f"Tool {ctx.data['tool_name']} returned: {ctx.data['result']}")
        ```
    """
    
    def __init__(self):
        """Initialize hook manager."""
        self.hooks: Dict[HookType, List[Callable]] = {}
    
    def register(self, hook_type: HookType, priority: int = 0):
        """Decorator to register a hook.
        
        Args:
            hook_type: Type of lifecycle hook
            priority: Execution priority (lower = earlier, default 0)
        
        Returns:
            Decorator function
        
        Example:
            ```python
            @manager.register(HookType.BEFORE_AGENT_RUN, priority=-10)
            def setup_logging(ctx):
                # Runs before other hooks
                pass
            ```
        """
        def decorator(func: Callable[[HookContext], Union[HookResult, None]]):
            if hook_type not in self.hooks:
                self.hooks[hook_type] = []
            
            # Store with priority for sorting
            self.hooks[hook_type].append((priority, func))
            # Sort by priority (ascending)
            self.hooks[hook_type].sort(key=lambda x: x[0])
            
            return func
        return decorator
    
    async def execute_hooks(
        self,
        hook_type: HookType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        can_modify: bool = True
    ) -> HookResult:
        """Execute all hooks for a given type.
        
        Args:
            hook_type: Type of hook to execute
            data: Data to pass to hooks (may be modified)
            metadata: Additional context
            can_modify: Whether hooks can modify data
        
        Returns:
            HookResult with aggregated modifications
        
        Raises:
            Exception: If any hook returns should_continue=False with an error
        
        Note:
            Hooks execute sequentially. If any hook returns should_continue=False,
            execution stops and the error is raised.
        """
        if hook_type not in self.hooks:
            return HookResult(should_continue=True)
        
        ctx = HookContext(
            hook_type=hook_type,
            data=data.copy(),
            metadata=metadata or {},
            can_modify=can_modify
        )
        
        # Execute hooks in priority order
        for priority, hook_func in self.hooks[hook_type]:
            try:
                # Call hook (sync or async)
                if asyncio.iscoroutinefunction(hook_func):
                    result = await hook_func(ctx)
                else:
                    result = hook_func(ctx)
                
                # If hook returns None, assume success
                if result is None:
                    result = HookResult(should_continue=True)
                
                # Check if we should stop
                if not result.should_continue:
                    if result.error:
                        raise result.error
                    return result
                
                # Apply modifications if allowed
                if can_modify and result.modified_data:
                    ctx.data.update(result.modified_data)
            
            except Exception as e:
                # If an exception occurs in a hook, stop execution
                return HookResult(should_continue=False, error=e)
        
        # All hooks succeeded, return final data
        return HookResult(should_continue=True, modified_data=ctx.data)
    
    def execute_hooks_sync(
        self,
        hook_type: HookType,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        can_modify: bool = True
    ) -> HookResult:
        """Synchronous hook execution.
        
        Args:
            hook_type: Type of hook to execute
            data: Data to pass to hooks
            metadata: Additional context
            can_modify: Whether hooks can modify data
        
        Returns:
            HookResult with aggregated modifications
        """
        try:
            loop = asyncio.get_running_loop()
            # Already in async context, create task
            future = asyncio.ensure_future(
                self.execute_hooks(hook_type, data, metadata, can_modify)
            )
            return asyncio.run_coroutine_threadsafe(future, loop).result()
        except RuntimeError:
            # No event loop, run in new loop
            return asyncio.run(
                self.execute_hooks(hook_type, data, metadata, can_modify)
            )
    
    def remove_hook(self, hook_type: HookType, func: Callable) -> None:
        """Remove a registered hook.
        
        Args:
            hook_type: Hook type to remove from
            func: Hook function to remove
        """
        if hook_type in self.hooks:
            self.hooks[hook_type] = [
                (p, f) for p, f in self.hooks[hook_type] if f != func
            ]
    
    def clear_hooks(self, hook_type: Optional[HookType] = None) -> None:
        """Clear hooks.
        
        Args:
            hook_type: Specific hook type to clear (None = clear all)
        """
        if hook_type:
            self.hooks[hook_type] = []
        else:
            self.hooks.clear()


# Global hook manager instance
_global_manager: Optional[HookManager] = None


def get_global_hook_manager() -> HookManager:
    """Get or create global hook manager singleton.
    
    Returns:
        Global HookManager instance
    
    Usage:
        ```python
        from nkit.hooks import get_global_hook_manager, HookType
        
        manager = get_global_hook_manager()
        
        @manager.register(HookType.BEFORE_TOOL_CALL)
        def validate_input(ctx):
            if "dangerous" in str(ctx.data):
                return HookResult(
                    should_continue=False,
                    error=ValueError("Dangerous input detected")
                )
        ```
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = HookManager()
    return _global_manager


__all__ = [
    "HookType",
    "HookContext",
    "HookResult",
    "HookManager",
    "get_global_hook_manager"
]
