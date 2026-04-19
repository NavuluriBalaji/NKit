"""Observer module for live streaming agent decisions.

This module provides the LiveObserver class, which acts as the core event bus
for the NKit real-time execution stream. Developers can subscribe to these
events to monitor, log, or intercept agent behaviors as they happen.

Supported Events:
- agent.start: When the agent begins a task
- agent.reasoning: The agent's thought process before acting
- tool.before: Before a tool is executed (interceptable)
- tool.after: After a tool returns its result
- agent.end: When the agent completes the task
"""

import time
import inspect
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

@dataclass
class Event:
    """Represents a structured live event in the NKit execution stream."""
    name: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


class LiveObserver:
    """Live streaming event observer for NKit agents.
    
    Allows external code to intercept and monitor the internal execution
    loop of ReAct agents in real time. Supports both synchronous and asynchronous
    handlers on a per-event basis.
    
    Example:
        ```python
        observer = LiveObserver()
        
        @observer.on("tool.before")
        async def watch(event):
            print(f"[ABOUT TO] {event['tool_name']} because: {event['why']}")
            
        agent = Agent(llm=my_llm, observer=observer)
        ```
    """
    
    def __init__(self):
        """Initialize empty observer registries."""
        self._handlers: Dict[str, List[Callable]] = {}

    def on(self, event_name: str) -> Callable:
        """Decorator to register a handler for a specific event.
        
        Args:
            event_name: e.g., 'agent.start', 'tool.before', etc.
            
        Returns:
            The decorator function.
        """
        def decorator(func: Callable):
            if event_name not in self._handlers:
                self._handlers[event_name] = []
            self._handlers[event_name].append(func)
            return func
        return decorator

    def emit(self, event_name: str, **data) -> None:
        """Emit an event synchronously, running any handlers.
        
        If a handler is async, this function will try to schedule it
        in the current event loop. Use `aemit` for proper async propagation.
        
        Args:
            event_name: The event to trigger.
            **data: Key-value data attached to the event.
        """
        event = Event(name=event_name, data=data)
        handlers = self._handlers.get(event_name, [])
        
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(handler(event))
                    except RuntimeError:
                        asyncio.run(handler(event))
                else:
                    handler(event)
            except Exception as e:
                # Observers must not crash the agent loop under any circumstances
                print(f"[Observer Error] Handler for '{event_name}' failed: {e}")

    async def aemit(self, event_name: str, **data) -> None:
        """Emit an event asynchronously, properly awaiting async handlers.
        
        Args:
            event_name: The event to trigger.
            **data: Key-value data attached to the event.
        """
        event = Event(name=event_name, data=data)
        handlers = self._handlers.get(event_name, [])
        
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                print(f"[Observer Error] Async handler for '{event_name}' failed: {e}")

__all__ = ["Event", "LiveObserver"]
