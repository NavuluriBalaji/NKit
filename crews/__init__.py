"""Multi-agent orchestration and crew management.

This module provides CrewAI-style multi-agent coordination:
- Sequential execution
- Hierarchical delegation
- Parallel task processing
- Inter-agent communication

Architecture:
    - Crew: Container for agents and tasks
    - Process: Execution strategy (sequential, hierarchical, parallel)
    - CrewOutput: Aggregated results
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from ..events import get_global_event_bus, Event, EventType


class ProcessType(Enum):
    """Crew execution strategies."""
    SEQUENTIAL = "sequential"  # Tasks run one after another
    HIERARCHICAL = "hierarchical"  # Manager delegates to workers
    PARALLEL = "parallel"  # Tasks run concurrently


@dataclass
class Agent:
    """Represents an agent in a crew.
    
    Attributes:
        role: Agent's role/specialty
        goal: Agent's objective
        backstory: Agent's background/context
        tools: Available tools
        llm: Language model to use
        verbose: Enable debug logging
    """
    role: str
    goal: str
    backstory: str
    tools: List[Any] = field(default_factory=list)
    llm: Optional[Any] = None
    verbose: bool = False
    
    # Internal state
    _agent_impl: Optional[Any] = None
    
    def __post_init__(self):
        """Initialize agent implementation if needed."""
        # Import here to avoid circular dependency
        if self._agent_impl is None and self.llm is not None:
            from ..nbagents import Agent as NkitAgent
            self._agent_impl = NkitAgent(llm_client=self.llm, tools=self.tools)
    
    async def execute_task(self, task: Any, context: Optional[Dict] = None) -> str:
        """Execute a task.
        
        Args:
            task: Task to execute
            context: Additional context from previous tasks
        
        Returns:
            Task result
        """
        # Build prompt with role, goal, and task
        prompt = f"""Role: {self.role}
Goal: {self.goal}
Backstory: {self.backstory}

Task: {task.description}
Expected Output: {task.expected_output}
"""
        
        # Add context if available
        if context:
            prompt += f"\n\nContext from previous tasks:\n{context}"
        
        # Execute with agent implementation
        if self._agent_impl:
            if hasattr(self._agent_impl, 'arun'):
                result = await self._agent_impl.arun(prompt)
            else:
                result = self._agent_impl.run(prompt)
        else:
            # Fallback: just echo for testing
            result = f"[{self.role}] Completed task: {task.description}"
        
        if self.verbose:
            print(f"[{self.role}] Result: {result}")
        
        return result


@dataclass
class Task:
    """Represents a task in a crew workflow.
    
    Attributes:
        description: Task description
        expected_output: Expected result format
        agent: Agent assigned to task
        dependencies: Task dependencies
        context: Additional context
    """
    description: str
    expected_output: str
    agent: Optional[Agent] = None
    dependencies: List['Task'] = field(default_factory=list)
    context: Optional[Dict[str, Any]] = None


@dataclass
class CrewOutput:
    """Aggregated crew execution results.
    
    Attributes:
        tasks_output: Results from each task
        raw: Raw output string
        token_usage: Token consumption tracking
        duration: Execution time
    """
    tasks_output: List[Dict[str, Any]] = field(default_factory=list)
    raw: str = ""
    token_usage: Dict[str, int] = field(default_factory=dict)
    duration: float = 0.0


class Crew:
    """Multi-agent crew orchestration.
    
    Purpose:
        Coordinate multiple agents to complete complex workflows:
        - Divide work among specialists
        - Share context between agents
        - Handle dependencies
        - Aggregate results
    
    Features:
        - Multiple execution strategies (sequential, hierarchical, parallel)
        - Automatic context passing
        - Event emission for monitoring
        - Flexible agent assignment
    
    Reuse Patterns:
        - Sequential: Research -> Analysis -> Report
        - Hierarchical: Manager assigns tasks to specialists
        - Parallel: Multiple independent analyses
    
    Example:
        ```python
        # Define agents
        researcher = Agent(
            role="Researcher",
            goal="Find relevant information",
            backstory="Expert in web research"
        )
        
        analyst = Agent(
            role="Analyst",
            goal="Analyze data",
            backstory="Data analysis expert"
        )
        
        # Define tasks
        research_task = Task(
            description="Research latest AI trends",
            expected_output="List of 5 trends with descriptions",
            agent=researcher
        )
        
        analysis_task = Task(
            description="Analyze trends and identify patterns",
            expected_output="Pattern analysis report",
            agent=analyst,
            dependencies=[research_task]
        )
        
        # Create crew
        crew = Crew(
            agents=[researcher, analyst],
            tasks=[research_task, analysis_task],
            process=ProcessType.SEQUENTIAL
        )
        
        # Execute
        result = await crew.kickoff_async()
        print(result.raw)
        ```
    """
    
    def __init__(
        self,
        agents: List[Agent],
        tasks: List[Task],
        process: ProcessType = ProcessType.SEQUENTIAL,
        verbose: bool = False,
        manager_llm: Optional[Any] = None
    ):
        """Initialize crew.
        
        Args:
            agents: List of agents in crew
            tasks: List of tasks to execute
            process: Execution strategy
            verbose: Enable debug logging
            manager_llm: LLM for hierarchical manager (if process=HIERARCHICAL)
        """
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.verbose = verbose
        self.manager_llm = manager_llm
        self.event_bus = get_global_event_bus()
    
    async def kickoff_async(self) -> CrewOutput:
        """Execute crew asynchronously.
        
        Returns:
            CrewOutput with aggregated results
        """
        import time
        start_time = time.time()
        
        # Emit crew started event
        await self.event_bus.publish(Event(
            type=EventType.CREW_STARTED,
            data={"num_agents": len(self.agents), "num_tasks": len(self.tasks)},
            source="crew"
        ))
        
        # Execute based on process type
        if self.process == ProcessType.SEQUENTIAL:
            output = await self._execute_sequential()
        elif self.process == ProcessType.HIERARCHICAL:
            output = await self._execute_hierarchical()
        elif self.process == ProcessType.PARALLEL:
            output = await self._execute_parallel()
        else:
            raise ValueError(f"Unknown process type: {self.process}")
        
        # Calculate duration
        output.duration = time.time() - start_time
        
        # Emit crew completed event
        await self.event_bus.publish(Event(
            type=EventType.CREW_COMPLETED,
            data={"duration": output.duration, "num_tasks": len(output.tasks_output)},
            source="crew"
        ))
        
        return output
    
    def kickoff(self) -> CrewOutput:
        """Execute crew synchronously.
        
        Returns:
            CrewOutput with aggregated results
        """
        return asyncio.run(self.kickoff_async())
    
    async def _execute_sequential(self) -> CrewOutput:
        """Execute tasks sequentially.
        
        Each task receives context from previous tasks.
        
        Returns:
            CrewOutput with results
        """
        tasks_output = []
        context = {}
        
        for idx, task in enumerate(self.tasks):
            if self.verbose:
                print(f"\n=== Task {idx + 1}/{len(self.tasks)}: {task.description} ===")
            
            # Use assigned agent or default to first agent
            agent = task.agent or self.agents[0]
            
            # Execute task
            result = await agent.execute_task(task, context=context)
            
            # Store output
            task_output = {
                "task": task.description,
                "agent": agent.role,
                "result": result
            }
            tasks_output.append(task_output)
            
            # Update context for next task
            context[f"task_{idx}"] = result
        
        # Build raw output
        raw = "\n\n".join(
            f"Task: {t['task']}\nAgent: {t['agent']}\nResult: {t['result']}"
            for t in tasks_output
        )
        
        return CrewOutput(tasks_output=tasks_output, raw=raw)
    
    async def _execute_hierarchical(self) -> CrewOutput:
        """Execute tasks hierarchically.
        
        Manager agent delegates tasks to worker agents.
        
        Returns:
            CrewOutput with results
        """
        if not self.manager_llm:
            raise ValueError("manager_llm required for hierarchical process")
        
        # Create manager agent
        manager = Agent(
            role="Manager",
            goal="Delegate tasks efficiently to team",
            backstory="Experienced project manager",
            llm=self.manager_llm,
            verbose=self.verbose
        )
        
        tasks_output = []
        
        # Manager decides task assignment
        if self.verbose:
            print("\n=== Manager delegating tasks ===")
        
        # For simplicity, assign tasks round-robin
        # In production, manager could use LLM to decide
        for idx, task in enumerate(self.tasks):
            agent_idx = idx % len(self.agents)
            agent = self.agents[agent_idx]
            
            if self.verbose:
                print(f"Task {idx + 1} -> {agent.role}")
            
            # Execute task
            result = await agent.execute_task(task)
            
            tasks_output.append({
                "task": task.description,
                "agent": agent.role,
                "result": result
            })
        
        # Manager aggregates results
        aggregation_task = Task(
            description=f"Aggregate results from {len(tasks_output)} tasks",
            expected_output="Consolidated final report"
        )
        
        context = {f"task_{i}": t["result"] for i, t in enumerate(tasks_output)}
        final_result = await manager.execute_task(aggregation_task, context=context)
        
        return CrewOutput(tasks_output=tasks_output, raw=final_result)
    
    async def _execute_parallel(self) -> CrewOutput:
        """Execute tasks in parallel.
        
        Tasks without dependencies run concurrently.
        
        Returns:
            CrewOutput with results
        """
        # Build dependency graph
        task_results = {}
        
        async def execute_task_with_deps(task: Task, idx: int):
            """Execute task after dependencies complete."""
            # Wait for dependencies
            if task.dependencies:
                dep_results = {}
                for dep in task.dependencies:
                    dep_idx = self.tasks.index(dep)
                    dep_results[f"dep_{dep_idx}"] = await task_results[dep_idx]
                context = dep_results
            else:
                context = {}
            
            # Execute task
            agent = task.agent or self.agents[idx % len(self.agents)]
            result = await agent.execute_task(task, context=context)
            
            return {
                "task": task.description,
                "agent": agent.role,
                "result": result
            }
        
        # Create futures for all tasks
        task_futures = {}
        for idx, task in enumerate(self.tasks):
            future = asyncio.create_task(execute_task_with_deps(task, idx))
            task_futures[idx] = future
            task_results[idx] = future
        
        # Wait for all tasks
        tasks_output = []
        for idx in range(len(self.tasks)):
            output = await task_futures[idx]
            tasks_output.append(output)
        
        # Build raw output
        raw = "\n\n".join(
            f"Task: {t['task']}\nAgent: {t['agent']}\nResult: {t['result']}"
            for t in tasks_output
        )
        
        return CrewOutput(tasks_output=tasks_output, raw=raw)


__all__ = [
    "ProcessType",
    "Agent",
    "Task",
    "CrewOutput",
    "Crew"
]
