"""Program of Thought (PoT) execution model for NKit agents.

This module defines the core data structures for PoT-based agent execution:
- ThoughtStep: A single step in the execution plan
- ThoughtProgram: The complete execution plan
- StepStatus: Lifecycle status of a step

PoT differs from ReAct in that the LLM generates a COMPLETE execution plan
ONCE at the beginning, then a deterministic executor runs each step.
The LLM is not called again during execution.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class StepStatus(Enum):
    """Lifecycle status of a ThoughtStep during execution."""
    PENDING = "pending"          # Not yet executed
    RUNNING = "running"          # Currently executing
    COMPLETE = "complete"        # Successfully completed
    FAILED = "failed"            # Failed after retries
    BLOCKED = "blocked"          # Blocked by SafetyGate
    SKIPPED = "skipped"          # Skipped due to on_failure='skip'


@dataclass
class ThoughtStep:
    """A single step in a Program of Thought execution plan.
    
    Represents a discrete tool invocation with its reasoning, dependencies,
    failure handling, and execution state.
    
    Attributes:
        step_id: Unique identifier for this step (1-indexed position)
        tool_name: Name of the tool to execute (must match ToolRegistry)
        args: Arguments to pass to the tool (dict of key-value pairs)
        why: Reasoning explaining why this step is needed
        depends_on: List of step_ids that must complete before this step
        on_failure: How to handle failure ("abort", "skip", or "retry")
        expected_output: Natural language description of expected result
        status: Current lifecycle status (default PENDING)
        result: Result from tool execution (set when status=COMPLETE)
        error: Error message if status=FAILED or status=BLOCKED
        duration_ms: Execution time in milliseconds (set when complete)
    """
    step_id: int
    tool_name: str
    args: Dict[str, Any]
    why: str
    depends_on: List[int] = field(default_factory=list)
    on_failure: str = "abort"  # "abort" | "skip" | "retry"
    expected_output: str = ""
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None

    def __post_init__(self):
        """Validate step after initialization."""
        if self.on_failure not in ("abort", "skip", "retry"):
            raise ValueError(f"on_failure must be 'abort', 'skip', or 'retry', got '{self.on_failure}'")
        if not isinstance(self.args, dict):
            raise ValueError(f"args must be dict, got {type(self.args)}")
        if not isinstance(self.depends_on, list):
            raise ValueError(f"depends_on must be list, got {type(self.depends_on)}")

    def is_ready(self, completed_step_ids: set) -> bool:
        """Check if all dependencies are satisfied.
        
        Args:
            completed_step_ids: Set of step_ids that have completed
            
        Returns:
            True if all dependencies are complete or no dependencies exist
        """
        return all(dep_id in completed_step_ids for dep_id in self.depends_on)

    def mark_complete(self, result: Any, duration_ms: float) -> None:
        """Mark step as successfully completed.
        
        Args:
            result: The result returned by the tool
            duration_ms: Execution time in milliseconds
        """
        self.status = StepStatus.COMPLETE
        self.result = result
        self.duration_ms = duration_ms

    def mark_failed(self, error: str) -> None:
        """Mark step as failed.
        
        Args:
            error: Error message describing the failure
        """
        self.status = StepStatus.FAILED
        self.error = error

    def mark_blocked(self, reason: str) -> None:
        """Mark step as blocked by SafetyGate.
        
        Args:
            reason: Reason for blocking
        """
        self.status = StepStatus.BLOCKED
        self.error = reason

    def mark_skipped(self) -> None:
        """Mark step as skipped."""
        self.status = StepStatus.SKIPPED

    def to_dict(self) -> Dict[str, Any]:
        """Serialize step to dictionary (for audit logging).
        
        Returns:
            Dictionary representation of the step
        """
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "args": self.args,
            "why": self.why,
            "depends_on": self.depends_on,
            "on_failure": self.on_failure,
            "expected_output": self.expected_output,
            "status": self.status.value,
            "result": str(self.result)[:500] if self.result is not None else None,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ThoughtProgram:
    """Complete execution plan generated by LLM once at the beginning.
    
    The LLM analyzes the goal, tool descriptions, and constraints ONCE,
    then produces this structured execution plan. The executor deterministically
    runs each step. The LLM is not consulted again during execution.
    
    Attributes:
        goal: The original user task
        session_id: UUID tracking this agent run across all components
        reasoning: LLM's high-level explanation of the approach
        confidence: LLM's confidence score (0.0 - 1.0)
        steps: List of ThoughtStep objects in order
        created_at: ISO 8601 timestamp when plan was created
    """
    goal: str
    session_id: str
    reasoning: str
    confidence: float
    steps: List[ThoughtStep]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def __post_init__(self):
        """Validate program after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0.0-1.0, got {self.confidence}")
        if not self.steps:
            raise ValueError("Program must have at least one step")
        
        # Verify step_ids are sequential starting from 1
        step_ids = [s.step_id for s in self.steps]
        if step_ids != list(range(1, len(self.steps) + 1)):
            raise ValueError(f"Step IDs must be sequential 1..N, got {step_ids}")

    def is_complete(self) -> bool:
        """Check if all steps are finished (completed or skipped).
        
        Returns:
            True if all steps have terminal status
        """
        return all(
            s.status in (StepStatus.COMPLETE, StepStatus.SKIPPED)
            for s in self.steps
        )

    def next_ready_steps(self) -> List[ThoughtStep]:
        """Get all steps ready to execute (dependencies satisfied, not yet run).
        
        Returns:
            List of ThoughtStep objects that can execute now
        """
        completed_ids = {
            s.step_id for s in self.steps 
            if s.status in (StepStatus.COMPLETE, StepStatus.SKIPPED)
        }
        
        return [
            s for s in self.steps
            if s.status == StepStatus.PENDING and s.is_ready(completed_ids)
        ]

    def has_failures(self) -> bool:
        """Check if any step has failed.
        
        Returns:
            True if any step is in FAILED or BLOCKED status
        """
        return any(s.status in (StepStatus.FAILED, StepStatus.BLOCKED) for s in self.steps)

    def summary(self) -> Dict[str, int]:
        """Get count of steps by status.
        
        Returns:
            Dictionary with keys: pending, running, complete, failed, blocked, skipped
        """
        counts = {
            "pending": 0,
            "running": 0,
            "complete": 0,
            "failed": 0,
            "blocked": 0,
            "skipped": 0,
        }
        for step in self.steps:
            counts[step.status.value] += 1
        return counts

    def get_step(self, step_id: int) -> Optional[ThoughtStep]:
        """Retrieve a step by ID.
        
        Args:
            step_id: The step ID to look up
            
        Returns:
            ThoughtStep or None if not found
        """
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize program to dictionary (for audit logging).
        
        Returns:
            Dictionary representation of the program
        """
        return {
            "goal": self.goal,
            "session_id": self.session_id,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "steps": [s.to_dict() for s in self.steps],
            "summary": self.summary(),
        }


__all__ = ["ThoughtStep", "ThoughtProgram", "StepStatus"]
