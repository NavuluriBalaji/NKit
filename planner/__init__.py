"""Thought Planner for Program of Thought execution.

ThoughtPlanner takes a goal and available tools, then calls the LLM
ONCE to generate a complete structured execution plan (ThoughtProgram).
The plan is then deterministically executed by ThoughtExecutor.
"""

import json
import re
from typing import Any, Dict, List, Optional
from ..program import ThoughtStep, ThoughtProgram, StepStatus
from ..tools import ToolRegistry


class PlanningError(Exception):
    """Raised when planning fails (invalid JSON, missing tools, etc.)."""
    pass


class ThoughtPlanner:
    """Plans agent execution by calling LLM once to generate complete program.
    
    The planner:
    1. Formats available tools for the LLM
    2. Builds planning prompt with goal and tools
    3. Calls LLM.complete() ONCE with temperature=0.2 (deterministic)
    4. Parses JSON response into ThoughtProgram
    5. Validates all tools exist in registry
    6. Returns executable ThoughtProgram
    
    The LLM is NOT called during execution - only at planning time.
    """
    PLAN_PROMPT = """
You are a planning agent. Given a goal and available tools, produce
a complete Program of Thought  a structured execution plan.

You think ONCE and plan COMPLETELY. The executor will run your plan.
You will NOT be called again during execution.

Available Tools:
{tools}

Goal: {goal}

Rules:
1. tool_name must exactly match a tool from the available tools list
2. args must be concrete values, no placeholders
3. Use $step_N in args to reference output of a previous step
4. If you use $step_N in args, N must be in depends_on
5. why must explain reasoning, not restate the action
6. on_failure: "abort" if critical, "retry" if flaky, "skip" if optional
7. Minimum steps only  every step must earn its place
8. Respond with ONLY valid JSON, no markdown, no explanation outside JSON

Respond with ONLY valid JSON, no markdown wrapper:

{{
  "reasoning": "high level explanation of your approach",
  "confidence": 0.0 to 1.0,
  "steps": [
    {{
      "step_id": 1,
      "tool_name": "tool_name_here",
      "args": {{"key": "value"}},
      "why": "specific reason this step is needed",
      "depends_on": [],
      "on_failure": "retry",
      "expected_output": "what this step returns"
    }}
  ]
}}
"""

    def __init__(self, llm: Any, tool_registry: ToolRegistry):
        """Initialize planner with LLM and tools.
        
        Args:
            llm: LLM provider (must have complete() method)
            tool_registry: ToolRegistry with available tools
        """
        self.llm = llm
        self.tool_registry = tool_registry

    def plan(self, goal: str, session_id: str) -> ThoughtProgram:
        """Generate execution plan by calling LLM once.
        
        Args:
            goal: The user's task/goal
            session_id: UUID for tracking this run
            
        Returns:
            ThoughtProgram with complete execution plan
            
        Raises:
            PlanningError: If JSON parsing fails, tools are missing, or LLM fails
        """
        # Format tools for LLM
        tools_str = self._format_tools()
        
        # Build prompt
        prompt = self.PLAN_PROMPT.format(tools=tools_str, goal=goal)
        
        # Call LLM once (temperature=0.2 for determinism, if supported)
        try:
            response = self.llm.complete(prompt)
        except Exception as e:
            raise PlanningError(f"LLM call failed: {e}")
        
        # Parse response into ThoughtProgram
        try:
            program = self._parse_response(response, goal, session_id)
        except Exception as e:
            raise PlanningError(f"Failed to parse LLM response: {e}")
        
        # Validate tools exist
        for step in program.steps:
            if step.tool_name not in self.tool_registry.tools:
                raise PlanningError(
                    f"Tool '{step.tool_name}' (step {step.step_id}) not found in registry. "
                    f"Available: {', '.join(self.tool_registry.tools.keys())}"
                )
        
        return program

    def _format_tools(self) -> str:
        """Format tool descriptions for LLM prompt.
        
        Returns:
            Markdown-formatted tool list
        """
        lines = []
        for name, tool in self.tool_registry.tools.items():
            lines.append(f"- **{name}**: {tool.desc}")
            if tool.schema:
                lines.append(f"  Args: {tool.schema}")
        return "\n".join(lines)

    def _parse_response(self, response: str, goal: str, session_id: str) -> ThoughtProgram:
        """Parse LLM JSON response into ThoughtProgram.
        
        Args:
            response: Raw LLM response (may contain markdown fences)
            goal: The original goal
            session_id: Session ID for tracking
            
        Returns:
            Parsed ThoughtProgram
            
        Raises:
            ValueError: If JSON is invalid or required fields missing
        """
        # Remove markdown code fences if present
        response = response.strip()
        if response.startswith("```"):
            # Remove ```json or ``` prefix and ``` suffix
            response = re.sub(r'^```(?:json)?\s*', '', response)
            response = re.sub(r'\s*```$', '', response)
        
        # Parse JSON
        data = json.loads(response)
        
        # Extract required fields
        if "reasoning" not in data:
            raise ValueError("Response missing 'reasoning' field")
        if "confidence" not in data:
            raise ValueError("Response missing 'confidence' field")
        if "steps" not in data:
            raise ValueError("Response missing 'steps' field")
        
        # Parse steps
        steps = []
        for step_data in data["steps"]:
            step = ThoughtStep(
                step_id=step_data.get("step_id"),
                tool_name=step_data.get("tool_name"),
                args=step_data.get("args", {}),
                why=step_data.get("why", ""),
                depends_on=step_data.get("depends_on", []),
                on_failure=step_data.get("on_failure", "abort"),
                expected_output=step_data.get("expected_output", ""),
            )
            steps.append(step)
        
        # Create and return program
        program = ThoughtProgram(
            goal=goal,
            session_id=session_id,
            reasoning=data["reasoning"],
            confidence=float(data["confidence"]),
            steps=steps,
        )
        
        return program


__all__ = ["ThoughtPlanner", "PlanningError"]


