"""Task management system for agent workflows.

This module provides task definition, execution, dependencies, and lifecycle management
similar to CrewAI's task system.

Architecture:
    - Task: Unit of work with inputs, outputs, dependencies
    - TaskOutput: Execution result with metadata
    - TaskManager: Orchestrates task execution with dependency resolution
    - TaskCallback: Lifecycle hooks for monitoring

Design Pattern:
    - Builder pattern for task configuration
    - Observer pattern for callbacks
    - DAG execution for dependencies
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent.base import Agent


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskOutput:
    """Output from task execution.
    
    Attributes:
        task_id: Unique task identifier
        result: Task execution result
        status: Execution status
        duration: Execution time in seconds
        error: Error message if failed
        metadata: Additional context (agent used, tools called, etc.)
    """
    task_id: str
    result: Any
    status: TaskStatus
    duration: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Task:
    """Represents a unit of work to be executed by an agent.
    
    Purpose:
        Encapsulates task description, dependencies, validation, and callbacks.
        Can be executed standalone or as part of a crew workflow.
    
    Reuse Patterns:
        - Sequential workflows: Chain tasks with dependencies
        - Parallel execution: Independent tasks run concurrently
        - Conditional tasks: Skip based on previous results
        - Retry logic: Auto-retry on failure
    
    Example:
        ```python
        task = Task(
            description="Analyze Q3 sales data",
            expected_output="Summary report with insights",
            agent=analyst_agent,
            dependencies=[data_fetch_task]
        )
        
        result = task.execute(context={"year": 2024})
        print(result.result)
        ```
    """
    
    def __init__(
        self,
        description: str,
        expected_output: str,
        agent: Optional["Agent"] = None,
        dependencies: Optional[List["Task"]] = None,
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[List] = None,
        async_execution: bool = False,
        callback: Optional[Callable[[TaskOutput], None]] = None,
        max_retries: int = 0,
        retry_delay: float = 1.0,
    ):
        """Initialize a task.
        
        Args:
            description: What the agent should accomplish
            expected_output: Format/content expected in result
            agent: Agent assigned to execute (can be set later)
            dependencies: Tasks that must complete first
            context: Additional context/variables for agent
            tools: Override agent's default tools
            async_execution: Execute asynchronously if True
            callback: Function called with TaskOutput after execution
            max_retries: Number of retry attempts on failure
            retry_delay: Seconds to wait between retries
        """
        self.id = str(uuid.uuid4())
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.dependencies = dependencies or []
        self.context = context or {}
        self.tools = tools
        self.async_execution = async_execution
        self.callback = callback
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.status = TaskStatus.PENDING
        self.output: Optional[TaskOutput] = None
    
    def execute(self, context: Optional[Dict[str, Any]] = None) -> TaskOutput:
        """Execute the task synchronously.
        
        Args:
            context: Runtime context merged with task context
        
        Returns:
            TaskOutput with result and metadata
        """
        return asyncio.run(self.execute_async(context))
    
    async def execute_async(self, context: Optional[Dict[str, Any]] = None) -> TaskOutput:
        """Execute the task asynchronously.
        
        Args:
            context: Runtime context merged with task context
        
        Returns:
            TaskOutput with result and metadata
        """
        if not self.agent:
            raise ValueError(f"Task {self.id} has no agent assigned")
        
        # Merge contexts
        full_context = {**self.context, **(context or {})}
        
        # Build task prompt
        prompt = self._build_prompt(full_context)
        
        # Execute with retries
        start_time = time.time()
        self.status = TaskStatus.RUNNING
        
        for attempt in range(self.max_retries + 1):
            try:
                result = await self._execute_with_agent(prompt)
                duration = time.time() - start_time
                
                self.status = TaskStatus.COMPLETED
                self.output = TaskOutput(
                    task_id=self.id,
                    result=result,
                    status=TaskStatus.COMPLETED,
                    duration=duration,
                    metadata={
                        "agent": self.agent.__class__.__name__,
                        "attempt": attempt + 1,
                        "context": full_context,
                    }
                )
                
                if self.callback:
                    self.callback(self.output)
                
                return self.output
                
            except Exception as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    duration = time.time() - start_time
                    self.status = TaskStatus.FAILED
                    self.output = TaskOutput(
                        task_id=self.id,
                        result=None,
                        status=TaskStatus.FAILED,
                        duration=duration,
                        error=str(e),
                        metadata={"agent": self.agent.__class__.__name__, "attempts": attempt + 1}
                    )
                    
                    if self.callback:
                        self.callback(self.output)
                    
                    raise
        
        raise RuntimeError("Unreachable code")
    
    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build agent prompt from task description and context."""
        prompt_parts = [
            f"Task: {self.description}",
            f"Expected Output: {self.expected_output}",
        ]
        
        if context:
            prompt_parts.append(f"Context: {context}")
        
        # Add dependency outputs
        if self.dependencies:
            dep_results = []
            for dep in self.dependencies:
                if dep.output and dep.output.status == TaskStatus.COMPLETED:
                    dep_results.append(f"- {dep.description}: {dep.output.result}")
            if dep_results:
                prompt_parts.append("Previous Task Results:\n" + "\n".join(dep_results))
        
        return "\n\n".join(prompt_parts)
    
    async def _execute_with_agent(self, prompt: str) -> str:
        """Execute prompt with assigned agent."""
        if hasattr(self.agent, 'run_async'):
            return await self.agent.run_async(prompt)
        else:
            return self.agent.run(prompt)


class TaskManager:
    """Manages task execution with dependency resolution.
    
    Purpose:
        Orchestrates multiple tasks, resolving dependencies and executing
        in correct order (topological sort of DAG).
    
    Features:
        - Automatic dependency resolution
        - Parallel execution of independent tasks
        - Failure handling (skip, retry, stop)
        - Progress tracking
    
    Example:
        ```python
        manager = TaskManager()
        
        task1 = Task("Fetch data", "CSV file", agent=fetcher)
        task2 = Task("Analyze data", "Report", agent=analyst, dependencies=[task1])
        task3 = Task("Visualize", "Charts", agent=visualizer, dependencies=[task2])
        
        results = manager.execute_tasks([task1, task2, task3])
        ```
    """
    
    def __init__(self):
        """Initialize task manager."""
        self.tasks: List[Task] = []
    
    def add_task(self, task: Task) -> "TaskManager":
        """Add a task to be managed."""
        self.tasks.append(task)
        return self
    
    def execute_tasks(self, tasks: Optional[List[Task]] = None) -> List[TaskOutput]:
        """Execute tasks with dependency resolution (sync)."""
        return asyncio.run(self.execute_tasks_async(tasks))
    
    async def execute_tasks_async(self, tasks: Optional[List[Task]] = None) -> List[TaskOutput]:
        """Execute tasks with dependency resolution (async).
        
        Args:
            tasks: Tasks to execute (uses self.tasks if None)
        
        Returns:
            List of TaskOutputs in execution order
        """
        tasks = tasks or self.tasks
        if not tasks:
            return []
        
        # Build dependency graph
        task_map = {t.id: t for t in tasks}
        completed = set()
        outputs = []
        
        while len(completed) < len(tasks):
            # Find tasks ready to execute (all dependencies completed)
            ready = [
                t for t in tasks
                if t.id not in completed
                and all(dep.id in completed for dep in t.dependencies)
            ]
            
            if not ready:
                # Circular dependency or all remaining tasks have incomplete deps
                pending = [t for t in tasks if t.id not in completed]
                raise RuntimeError(f"Cannot resolve dependencies for tasks: {[t.id for t in pending]}")
            
            # Execute ready tasks (in parallel if async)
            if any(t.async_execution for t in ready):
                results = await asyncio.gather(*[t.execute_async() for t in ready], return_exceptions=True)
            else:
                results = [await t.execute_async() for t in ready]
            
            # Process results
            for task, result in zip(ready, results):
                if isinstance(result, Exception):
                    task.status = TaskStatus.FAILED
                    task.output = TaskOutput(
                        task_id=task.id,
                        result=None,
                        status=TaskStatus.FAILED,
                        duration=0.0,
                        error=str(result)
                    )
                outputs.append(task.output)
                completed.add(task.id)
        
        return outputs


__all__ = ["Task", "TaskOutput", "TaskStatus", "TaskManager"]
