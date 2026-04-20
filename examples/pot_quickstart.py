"""Quick Start: Minimal PoT Example (Copy & Paste Ready).

The simplest possible PoT example to get started.
Just change the tasks and tools - everything else stays the same.
"""

import sys
import os
import asyncio
import json

# Add NKit to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from NKit.tools import ToolRegistry, Tool
from NKit.planner import ThoughtPlanner
from NKit.executor import ThoughtExecutor


# ============================================================================
# STEP 1: Define your tools
# ============================================================================

def get_time():
    """Get current time."""
    from datetime import datetime
    return datetime.now().isoformat()

def add_numbers(a: int, b: int):
    """Add two numbers."""
    return a + b

def multiply_numbers(a: int, b: int):
    """Multiply two numbers."""
    return a * b

# Register tools
registry = ToolRegistry(include_builtin=False)
registry.register(Tool("get_time", get_time, "Get current time"))
registry.register(Tool("add", add_numbers, "Add two numbers"))
registry.register(Tool("multiply", multiply_numbers, "Multiply two numbers"))


# ============================================================================
# STEP 2: Create an LLM that returns your plan
# ============================================================================

class SimpleLLM:
    """Mock LLM that returns a hardcoded plan."""
    
    def complete(self, prompt: str) -> str:
        # Define your workflow here
        plan = {
            "reasoning": "Calculate some numbers and show the time",
            "confidence": 0.95,
            "steps": [
                {
                    "step_id": 1,
                    "tool_name": "get_time",
                    "args": {},
                    "why": "Get current time",
                    "depends_on": [],
                    "on_failure": "abort"
                },
                {
                    "step_id": 2,
                    "tool_name": "add",
                    "args": {"a": 10, "b": 20},
                    "why": "Add 10 and 20",
                    "depends_on": [],
                    "on_failure": "abort"
                },
                {
                    "step_id": 3,
                    "tool_name": "multiply",
                    "args": {"a": "$step_2", "b": 2},
                    "why": "Multiply result by 2",
                    "depends_on": [2],
                    "on_failure": "abort"
                }
            ]
        }
        return json.dumps(plan)
    
    def __call__(self, prompt: str) -> str:
        return self.complete(prompt)


# ============================================================================
# STEP 3: Run it!
# ============================================================================

async def main():
    """Main execution."""
    print("\n" + "="*60)
    print("  PoT Quick Start Example")
    print("="*60 + "\n")
    
    # Create components
    llm = SimpleLLM()
    planner = ThoughtPlanner(llm, registry)
    executor = ThoughtExecutor(registry)
    
    # Plan (LLM called once)
    print("📋 Planning...")
    program = planner.plan("Do some work", session_id="quickstart")
    print(f"✅ Plan: {len(program.steps)} steps\n")
    
    # Execute (no more LLM calls)
    print("🔄 Executing...")
    result = await executor.execute(program)
    
    print(f"\n✅ Result:\n{result}\n")


# ============================================================================
# STEP 4: Customize this!
# ============================================================================

"""
To customize:

1. Add your own tools:
   def my_tool(param1: str, param2: int):
       '''Do something'''
       return result
   
   registry.register(Tool("my_tool", my_tool, "Description"))

2. Modify the plan in SimpleLLM.complete():
   - Add more steps
   - Change tool names to match your tools
   - Use $step_N to reference previous results
   - Set on_failure to "abort", "skip", or "retry"

3. Change the goal:
   program = planner.plan("YOUR GOAL HERE", session_id="id")

4. Add error handling:
   try:
       result = await executor.execute(program)
   except Exception as e:
       print(f"Error: {e}")

5. Add an observer for detailed output:
   from NKit.observer import LiveObserver
   observer = LiveObserver()
   executor = ThoughtExecutor(registry, observer=observer)
"""


if __name__ == "__main__":
    asyncio.run(main())
