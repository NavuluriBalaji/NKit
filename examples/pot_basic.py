"""Complete Program of Thought (PoT) example.

This demonstrates NKit agents using Program of Thought execution:
- LLM generates COMPLETE execution plan ONCE
- Deterministic executor runs each step in order
- Full observability, safety, and audit trail

Contrasts with ReAct:
- ReAct: Multiple LLM calls (adaptive but slow)
- PoT: Single LLM call (fast, deterministic)
"""

import sys
import os
import asyncio
import json

# Add NKit root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from NKit.agent import Agent
from NKit.tools import ToolRegistry, Tool
from NKit.observer import LiveObserver
from NKit.program import ThoughtStep, ThoughtProgram, StepStatus
from NKit.planner import ThoughtPlanner
from NKit.executor import ThoughtExecutor


def demo_llm_stub():
    """Mock LLM that returns a pre-defined plan."""
    class MockLLM:
        def complete(self, prompt: str) -> str:
            # Simulate an LLM response with a complete execution plan
            plan = {
                "reasoning": "User wants time and to check files. I'll get time first.",
                "confidence": 0.95,
                "steps": [
                    {
                        "step_id": 1,
                        "tool_name": "get_time",
                        "args": {},
                        "why": "Get current date and time",
                        "depends_on": [],
                        "on_failure": "abort"
                    }
                ]
            }
            return json.dumps(plan)
        
        def __call__(self, prompt: str) -> str:
            # Support both .complete() and direct call
            return self.complete(prompt)
    
    return MockLLM()



def setup_observer() -> LiveObserver:
    """Create observer with event handlers."""
    observer = LiveObserver()
    
    @observer.on("agent.start")
    def on_start(event):
        print(f"\n{'='*70}")
        print(f"🚀 AGENT START")
        print(f"  Goal: {event['goal'][:60]}...")
        print(f"  Total steps: {event['total_steps']}")
        print(f"  Reasoning: {event.get('reasoning', 'N/A')[:80]}...")
        print(f"{'='*70}\n")
    
    @observer.on("agent.reasoning")
    def on_reasoning(event):
        print(f"📋 PLAN GENERATED")
        print(f"  Reasoning: {event['reasoning'][:100]}")
        print(f"  Confidence: {event['confidence']:.1%}\n")
    
    @observer.on("tool.before")
    def on_tool_before(event):
        blocked = event.get("blocked", False)
        if blocked:
            reason = event.get("reason", "Unknown")
            print(f"  🚫 BLOCKED (Step {event['step_id']}): {reason}")
        else:
            print(f"  ⚙️  STEP {event['step_id']}: {event['tool_name']}")
            print(f"      Why: {event['why']}")
            args = event.get("args", {})
            if args:
                print(f"      Args: {args}")
    
    @observer.on("tool.after")
    def on_tool_after(event):
        success = event.get("success", False)
        duration = event.get("duration_ms", 0)
        result_preview = str(event.get("result", ""))[:60]
        
        if success:
            print(f"      ✅ SUCCESS ({duration:.0f}ms)")
            print(f"      Result: {result_preview}...")
        else:
            print(f"      ❌ FAILED ({duration:.0f}ms)")
            print(f"      Error: {result_preview}...")
        print()
    
    @observer.on("agent.end")
    def on_end(event):
        print(f"\n{'='*70}")
        print(f"🎯 AGENT COMPLETE")
        print(f"  Final answer: {event['final_answer'][:100]}...")
        print(f"  Total steps: {event['total_steps']}")
        print(f"{'='*70}\n")
    
    @observer.on("agent.error")
    def on_error(event):
        print(f"\n❌ ERROR: {event['error']}\n")
    
    return observer


async def demo_pot_execution():
    """Demonstrate PoT execution with manual components."""
    print("\n" + "="*70)
    print("DEMO: Program of Thought (PoT) - Manual Execution")
    print("="*70)
    
    try:
        # Create components
        registry = ToolRegistry(include_builtin=True)
        llm = demo_llm_stub()
        observer = setup_observer()
        
        # Create planner
        planner = ThoughtPlanner(llm, registry)
        
        # Plan the task (LLM called ONCE)
        try:
            program = planner.plan(
                goal="What is the current time? Then list files in the current directory.",
                session_id="demo-session-001"
            )
            print(f"✅ Plan generated: {len(program.steps)} steps")
            print(f"   Reasoning: {program.reasoning}")
            print(f"   Confidence: {program.confidence:.1%}\n")
        except Exception as e:
            print(f"❌ Planning failed: {e}")
            print(f"   Note: This is expected if the LLM response format doesn't match\n")
            return
        
        # Execute plan deterministically
        executor = ThoughtExecutor(
            registry,
            observer=observer,
            max_retries=1
        )
        
        try:
            result = await executor.execute(program)
            print(f"\n✅ PoT execution complete")
            print(f"   Result: {result[:100]}...\n")
        except Exception as e:
            print(f"\n❌ Execution failed: {e}\n")
    except Exception as e:
        print(f"❌ Demo setup failed: {e}\n")
        import traceback
        traceback.print_exc()


def demo_agent_pot_mode():
    """Demonstrate PoT via Agent class (high-level API)."""
    print("\n" + "="*70)
    print("DEMO: Program of Thought via Agent Class")
    print("="*70)
    
    # Create agent in PoT mode
    agent = Agent(
        llm=demo_llm_stub(),
        reasoning_mode="pot",
        observer=setup_observer()
    )
    
    # Run task
    try:
        result = agent.run("Get current time and list files")
        print(f"\n✅ Agent completed: {result[:80]}...\n")
    except Exception as e:
        print(f"\n❌ Agent failed: {e}\n")


def demo_comparison():
    """Show ReAct vs PoT comparison."""
    print("\n" + "="*70)
    print("DEMO: ReAct vs PoT Comparison")
    print("="*70)
    
    print("""
ReAct (Iterative):
  ┌─ LLM Call 1 ──────┐
  │ "I need time"     │
  └─────────────────────┘
           ↓
    Execute: get_time()
           ↓
  ┌─ LLM Call 2 ──────┐
  │ "Now I need files"│
  └─────────────────────┘
           ↓
    Execute: list_files()
           ↓
  ┌─ LLM Call 3 ──────┐
  │ "Final answer is.."│
  └─────────────────────┘

  Pros: Adaptive, can change plan
  Cons: Slower (many LLM calls), complex

Program of Thought (PoT):
  ┌─ LLM Call 1 (ONLY) ────────────────────────┐
  │ Plan: Step 1: get_time()                   │
  │       Step 2: list_files()                 │
  │ (Complete execution plan as JSON)          │
  └──────────────────────────────────────────────┘
           ↓
    Deterministic Executor
    ┌─────────────────────┐
    │ Step 1: get_time()  │
    │ Step 2: list_files()│
    └─────────────────────┘
           ↓
      Final Result

  Pros: Fast (1 LLM call), predictable, auditable
  Cons: Less adaptive, requires good planning
    """)


def main():
    """Run all demos."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  NKit: Program of Thought (PoT) Agent Framework".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    # Show comparison
    demo_comparison()
    
    # Test planner directly first
    print("\nTesting Planner Component...")
    print("-" * 70)
    try:
        registry = ToolRegistry(include_builtin=True)
        llm = demo_llm_stub()
        
        # Test if LLM returns valid JSON
        prompt = "test"
        response = llm.complete(prompt)
        print(f"LLM Response: {response[:100]}...")
        
        # Test planner
        planner = ThoughtPlanner(llm, registry)
        program = planner.plan("Test goal", session_id="test-123")
        print(f"✅ Planner works: {len(program.steps)} steps, confidence={program.confidence:.1%}\n")
    except Exception as e:
        print(f"❌ Planner test failed: {e}\n")
        import traceback
        traceback.print_exc()
    
    # Run async PoT demo (with error handling)
    print("\nRunning PoT Execution Demo...")
    print("-" * 70)
    try:
        asyncio.run(demo_pot_execution())
    except Exception as e:
        print(f"❌ PoT demo failed: {e}\n")
    
    # Run agent-based demo
    print("\nRunning Agent-Based PoT Demo...")
    print("-" * 70)
    try:
        demo_agent_pot_mode()
    except Exception as e:
        print(f"❌ Agent demo failed: {e}\n")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Program of Thought Advantages")
    print("="*70)
    print("""
✅ Single LLM Call: Plan generated once, not called during execution
✅ Deterministic: Same input → same execution path (predictable)
✅ Observable: Every step can be monitored and logged
✅ Safe: Pre-execute safety checks on entire plan
✅ Auditable: Full plan + execution trail in WhyLog
✅ Fast: No latency waiting for LLM responses during execution
✅ Cost-effective: Fewer LLM API calls = lower costs

Use PoT when:
  • Task is well-defined
  • You want predictability
  • Speed matters
  • Compliance/audit trails required
  • Limited LLM budget

Use ReAct when:
  • Task is exploratory
  • You need adaptivity
  • Clarification from user possible
  • Plan might change based on results
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
