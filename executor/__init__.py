"""Thought Executor for deterministic program execution.

ThoughtExecutor takes a ThoughtProgram and runs each step in order,
resolving dependencies, handling failures, and collecting results.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from ..program import ThoughtStep, ThoughtProgram, StepStatus
from ..tools import ToolRegistry


class ExecutionError(Exception):
    """Raised when execution fails critically."""
    pass


class ToolTimeoutError(Exception):
    """Raised when a tool exceeds its timeout."""
    pass


class ThoughtExecutor:
    """Executes a ThoughtProgram step-by-step.
    
    The executor:
    1. Validates program safety with SafetyGate
    2. Iterates until all steps complete or blocking error occurs
    3. Resolves $step_N dependencies in args
    4. Executes tools with retry logic and timeouts
    5. Emits observer events for monitoring
    6. Logs to audit trail
    7. Returns final result
    
    The LLM is NOT involved during execution - it's purely deterministic.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        observer: Optional[Any] = None,
        safety_gate: Optional[Any] = None,
        audit_log: Optional[Any] = None,
        max_retries: int = 3,
    ):
        """Initialize executor with dependencies.
        
        Args:
            tool_registry: ToolRegistry with available tools
            observer: LiveObserver for event streaming (optional)
            safety_gate: SafetyGate for pre-execution checks (optional)
            audit_log: WhyLog for audit trail (optional)
            max_retries: Max attempts per tool (default 3)
        """
        self.tool_registry = tool_registry
        self.observer = observer
        self.safety_gate = safety_gate
        self.audit_log = audit_log
        self.max_retries = max_retries
        
        # Store step results for $step_N resolution
        self._result_store: Dict[int, Any] = {}

    async def execute(self, program: ThoughtProgram) -> str:
        """Execute a ThoughtProgram.
        
        Args:
            program: The ThoughtProgram to execute
            
        Returns:
            Final result as string
            
        Raises:
            ExecutionError: If execution fails critically
        """
        # Pre-execution: Safety check all steps
        if self.safety_gate:
            try:
                self.safety_gate.inspect_program(program)
            except Exception as e:
                self._emit("agent.error", error=f"Safety check failed: {e}")
                raise ExecutionError(f"Program blocked by safety gate: {e}")
        
        # Emit start event
        self._emit("agent.start", 
                   goal=program.goal, 
                   session_id=program.session_id,
                   total_steps=len(program.steps),
                   reasoning=program.reasoning,
                   confidence=program.confidence)
        
        # Emit reasoning event
        self._emit("agent.reasoning",
                   reasoning=program.reasoning,
                   confidence=program.confidence,
                   session_id=program.session_id)
        
        # Execute steps until complete
        completed_steps = 0
        while not program.is_complete():
            ready_steps = program.next_ready_steps()
            
            if not ready_steps:
                # No ready steps but not complete = deadlock or all blocked
                if program.has_failures():
                    break  # Failures occurred, stop
                else:
                    raise ExecutionError("Deadlock: no ready steps but program incomplete")
            
            # Execute each ready step
            for step in ready_steps:
                await self._execute_step(step, program)
                
                if step.status == StepStatus.COMPLETE:
                    completed_steps += 1
                    # Store result for dependency resolution
                    self._result_store[step.step_id] = step.result
                elif step.status == StepStatus.FAILED:
                    # Check on_failure policy
                    if step.on_failure == "abort":
                        self._emit("agent.error", 
                                   error=f"Step {step.step_id} failed: {step.error}",
                                   session_id=program.session_id)
                        raise ExecutionError(f"Step {step.step_id} failed: {step.error}")
                    elif step.on_failure == "skip":
                        step.mark_skipped()
                elif step.status == StepStatus.BLOCKED:
                    # Step was blocked by safety
                    if step.on_failure == "abort":
                        self._emit("agent.error",
                                   error=f"Step {step.step_id} blocked: {step.error}",
                                   session_id=program.session_id)
                        raise ExecutionError(f"Step {step.step_id} blocked: {step.error}")
                    elif step.on_failure == "skip":
                        step.mark_skipped()
        
        # Get final result from last step
        last_step = program.steps[-1]
        final_result = str(last_step.result) if last_step.result is not None else "No result"
        
        # Emit end event
        self._emit("agent.end",
                   final_answer=final_result,
                   session_id=program.session_id,
                   total_steps=len(program.steps),
                   completed_steps=completed_steps)
        
        # Log program completion
        if self.audit_log:
            try:
                self.audit_log.log_program(program)
            except Exception as e:
                # Don't fail if audit logging fails
                pass
        
        return final_result

    async def _execute_step(self, step: ThoughtStep, program: ThoughtProgram) -> None:
        """Execute a single step with retry logic and safety checks.
        
        Args:
            step: The ThoughtStep to execute
            program: The parent ThoughtProgram (for context)
        """
        step.status = StepStatus.RUNNING
        
        # Resolve args (replace $step_N with actual results)
        try:
            resolved_args = self._resolve_args(step.args)
        except Exception as e:
            step.mark_failed(f"Failed to resolve args: {e}")
            return
        
        # Emit before event
        self._emit("tool.before",
                   tool_name=step.tool_name,
                   args=resolved_args,
                   why=step.why,
                   step_id=step.step_id,
                   session_id=program.session_id)
        
        # Secondary safety check on this step
        if self.safety_gate:
            try:
                self.safety_gate.check_step(step, program.goal)
            except Exception as e:
                step.mark_blocked(str(e))
                self._emit("tool.after",
                           tool_name=step.tool_name,
                           result=f"Blocked: {e}",
                           success=False,
                           blocked=True,
                           duration_ms=0.0,
                           step_id=step.step_id,
                           session_id=program.session_id)
                if self.audit_log:
                    try:
                        self.audit_log.log_blocked(program.session_id, step, str(e))
                    except:
                        pass
                return
        
        # Execute with retry logic
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Get tool
                tool = self.tool_registry.get(step.tool_name)
                if not tool:
                    raise ExecutionError(f"Tool '{step.tool_name}' not found")
                
                # Execute with timeout
                try:
                    result = await asyncio.wait_for(
                        tool.execute(**resolved_args),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    raise ToolTimeoutError(f"Tool exceeded 30s timeout")
                
                # Success
                duration_ms = (time.time() - start_time) * 1000
                step.mark_complete(result, duration_ms)
                
                self._emit("tool.after",
                           tool_name=step.tool_name,
                           result=str(result)[:500],
                           success=True,
                           duration_ms=duration_ms,
                           step_id=step.step_id,
                           session_id=program.session_id)
                
                if self.audit_log:
                    try:
                        self.audit_log.log_step(program.session_id, step, result)
                    except:
                        pass
                return
                
            except ToolTimeoutError as e:
                last_error = str(e)
                if attempt == self.max_retries - 1:
                    # Last attempt
                    break
                # Retry with exponential backoff
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                last_error = str(e)
                if attempt == self.max_retries - 1:
                    # Last attempt
                    break
                # Retry with exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        # All retries failed
        duration_ms = (time.time() - start_time) * 1000
        step.mark_failed(last_error or "Unknown error")
        
        self._emit("tool.after",
                   tool_name=step.tool_name,
                   result=f"Failed: {last_error}",
                   success=False,
                   duration_ms=duration_ms,
                   step_id=step.step_id,
                   session_id=program.session_id)
        
        if self.audit_log:
            try:
                self.audit_log.log_error(program.session_id, step, last_error)
            except:
                pass

    def _resolve_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve $step_N references in args.
        
        Args:
            args: Arguments possibly containing $step_N placeholders
            
        Returns:
            Resolved arguments with actual results
            
        Raises:
            ValueError: If referenced step not found
        """
        resolved = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$step_"):
                # Extract step ID
                try:
                    step_id = int(value[6:])  # Remove "$step_" prefix
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid step reference: {value}")
                
                if step_id not in self._result_store:
                    raise ValueError(f"Step {step_id} not executed yet")
                
                resolved[key] = self._result_store[step_id]
            else:
                resolved[key] = value
        
        return resolved

    def _emit(self, event_name: str, **data) -> None:
        """Emit observer event safely.
        
        Args:
            event_name: Event name (e.g., 'agent.start')
            **data: Event data
        """
        if self.observer:
            try:
                self.observer.emit(event_name, **data)
            except Exception as e:
                # Don't let observer failures crash execution
                pass


__all__ = ["ThoughtExecutor", "ExecutionError", "ToolTimeoutError"]
