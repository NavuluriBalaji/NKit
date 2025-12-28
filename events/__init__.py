"""Event system for agent coordination and observability.

This module provides pub/sub event bus for:
- Inter-agent communication
- Lifecycle hooks
- Monitoring/telemetry
- Debug logging

Architecture:
    - EventBus: Central pub/sub coordinator
    - Event types: Predefined event categories
    - Async delivery: Non-blocking event propagation
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(Enum):
    """Predefined event categories."""
    # Agent lifecycle
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    
    # Task lifecycle
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    
    # Tool execution
    TOOL_CALLED = "tool.called"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    
    # LLM calls
    LLM_CALLED = "llm.called"
    LLM_RESPONDED = "llm.responded"
    LLM_FAILED = "llm.failed"
    
    # Memory operations
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    
    # Crew coordination
    CREW_STARTED = "crew.started"
    CREW_COMPLETED = "crew.completed"
    
    # Custom events
    CUSTOM = "custom"


@dataclass
class Event:
    """Represents an event in the system.
    
    Attributes:
        type: Event category
        data: Event payload
        source: Component that emitted the event
        timestamp: Event creation time
        id: Unique event identifier
        metadata: Additional context
    """
    type: EventType
    data: Dict[str, Any]
    source: str
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Central event pub/sub coordinator.
    
    Purpose:
        Enables loose coupling between components via event-driven architecture.
        Components publish events without knowing subscribers.
    
    Features:
        - Async event delivery (non-blocking)
        - Type-based subscription filtering
        - Event history for replay/debugging
        - Wildcard subscriptions
    
    Reuse Patterns:
        - Monitoring: Subscribe to all events for logging
        - Telemetry: Count events for metrics
        - Debugging: Replay event history
        - Coordination: Agents communicate via events
    
    Example:
        ```python
        bus = EventBus()
        
        # Subscribe to agent events
        @bus.subscribe(EventType.AGENT_STARTED)
        async def on_agent_start(event):
            print(f"Agent {event.source} started")
        
        # Publish event
        await bus.publish(Event(
            type=EventType.AGENT_STARTED,
            data={"task": "analyze data"},
            source="analyst_agent"
        ))
        ```
    """
    
    def __init__(self, enable_history: bool = True, max_history: int = 1000):
        """Initialize event bus.
        
        Args:
            enable_history: Store events for replay
            max_history: Maximum events to keep in history
        """
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.wildcard_subscribers: List[Callable] = []
        self.enable_history = enable_history
        self.max_history = max_history
        self.history: List[Event] = []
    
    def subscribe(self, event_type: Optional[EventType] = None):
        """Decorator to subscribe to event type.
        
        Args:
            event_type: Event type to subscribe to (None for all events)
        
        Returns:
            Decorator function
        
        Example:
            ```python
            @bus.subscribe(EventType.TASK_COMPLETED)
            async def on_task_done(event):
                print(f"Task {event.data['task_id']} done")
            ```
        """
        def decorator(func: Callable[[Event], Any]):
            if event_type is None:
                self.wildcard_subscribers.append(func)
            else:
                if event_type not in self.subscribers:
                    self.subscribers[event_type] = []
                self.subscribers[event_type].append(func)
            return func
        return decorator
    
    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        
        Note:
            Subscribers are called asynchronously without blocking.
            Exceptions in subscribers are logged but don't affect other subscribers.
        """
        # Store in history
        if self.enable_history:
            self.history.append(event)
            if len(self.history) > self.max_history:
                self.history.pop(0)
        
        # Notify type-specific subscribers
        type_subscribers = self.subscribers.get(event.type, [])
        
        # Notify wildcard subscribers
        all_subscribers = type_subscribers + self.wildcard_subscribers
        
        # Call subscribers asynchronously
        if all_subscribers:
            tasks = []
            for subscriber in all_subscribers:
                if asyncio.iscoroutinefunction(subscriber):
                    tasks.append(subscriber(event))
                else:
                    # Wrap sync functions
                    tasks.append(asyncio.to_thread(subscriber, event))
            
            # Execute without waiting (fire and forget)
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def publish_sync(self, event: Event) -> None:
        """Synchronous publish (creates async task).
        
        Args:
            event: Event to publish
        
        Note:
            Use this in sync contexts. Creates background task for async delivery.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish(event))
        except RuntimeError:
            # No event loop, run in new loop
            asyncio.run(self.publish(event))
    
    def get_history(
        self,
        event_type: Optional[EventType] = None,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """Retrieve event history with optional filtering.
        
        Args:
            event_type: Filter by event type
            source: Filter by source component
            limit: Maximum events to return
        
        Returns:
            List of matching events (most recent first)
        """
        filtered = self.history
        
        if event_type:
            filtered = [e for e in filtered if e.type == event_type]
        
        if source:
            filtered = [e for e in filtered if e.source == source]
        
        return list(reversed(filtered[-limit:]))
    
    def clear_history(self) -> None:
        """Clear event history."""
        self.history.clear()


# Global event bus instance
_global_bus: Optional[EventBus] = None


def get_global_event_bus() -> EventBus:
    """Get or create global event bus singleton.
    
    Returns:
        Global EventBus instance
    
    Usage:
        ```python
        from nkit.events import get_global_event_bus, EventType
        
        bus = get_global_event_bus()
        
        @bus.subscribe(EventType.AGENT_STARTED)
        async def log_agent_start(event):
            print(f"Agent started: {event.source}")
        ```
    """
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


__all__ = ["Event", "EventType", "EventBus", "get_global_event_bus"]
