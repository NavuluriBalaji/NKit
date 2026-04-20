"""Practical PoT Example: Multi-Step Data Processing Workflow.

This example demonstrates Program of Thought with a realistic scenario:
- LLM plans a complete data processing workflow upfront
- Each step depends on previous results using $step_N references
- Executor runs deterministically without additional LLM calls
- Full audit trail shows what happened

Use case: Process customer data
  1. Load CSV file
  2. Filter records (e.g., recent customers)
  3. Aggregate statistics
  4. Generate report
  5. Save results
"""

import sys
import os
import asyncio
import json
from datetime import datetime

# Add NKit root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from NKit.agent import Agent
from NKit.tools import ToolRegistry, Tool
from NKit.observer import LiveObserver
from NKit.program import ThoughtStep, ThoughtProgram, StepStatus
from NKit.planner import ThoughtPlanner
from NKit.executor import ThoughtExecutor


# ============================================================================
# Mock LLM that plans a data processing workflow
# ============================================================================

def create_workflow_llm():
    """Create mock LLM that plans a data processing pipeline."""
    
    class WorkflowLLM:
        def complete(self, prompt: str) -> str:
            # Plan for processing customer data
            plan = {
                "reasoning": "I need to process customer data: load the CSV, filter for recent customers (last 30 days), count records, calculate average values, and generate a report.",
                "confidence": 0.92,
                "steps": [
                    {
                        "step_id": 1,
                        "tool_name": "load_csv",
                        "args": {"filepath": "customers.csv"},
                        "why": "Load customer data from CSV file into memory",
                        "depends_on": [],
                        "on_failure": "abort",
                        "expected_output": "List of customer records with columns: id, name, email, signup_date, status"
                    },
                    {
                        "step_id": 2,
                        "tool_name": "filter_records",
                        "args": {
                            "records": "$step_1",
                            "condition": "status == 'active' AND days_since_signup <= 30"
                        },
                        "why": "Keep only active customers who signed up in the last 30 days",
                        "depends_on": [1],
                        "on_failure": "skip",
                        "expected_output": "Filtered list of recent active customers"
                    },
                    {
                        "step_id": 3,
                        "tool_name": "count_records",
                        "args": {"records": "$step_2"},
                        "why": "Count how many recent active customers we have",
                        "depends_on": [2],
                        "on_failure": "abort",
                        "expected_output": "Integer count"
                    },
                    {
                        "step_id": 4,
                        "tool_name": "calculate_stats",
                        "args": {"records": "$step_2"},
                        "why": "Calculate statistics (average signup date) for recent customers",
                        "depends_on": [2],
                        "on_failure": "skip",
                        "expected_output": "JSON with avg_signup_date, total_value, etc."
                    },
                    {
                        "step_id": 5,
                        "tool_name": "generate_report",
                        "args": {
                            "total_customers": "$step_3",
                            "statistics": "$step_4",
                            "timestamp": "$NOW"
                        },
                        "why": "Create a human-readable report of findings",
                        "depends_on": [3, 4],
                        "on_failure": "skip",
                        "expected_output": "Formatted report string"
                    }
                ]
            }
            return json.dumps(plan)
        
        def __call__(self, prompt: str) -> str:
            return self.complete(prompt)
    
    return WorkflowLLM()


# ============================================================================
# Mock Tools that simulate real data processing
# ============================================================================

def create_mock_tools():
    """Create mock tools for demonstration."""
    
    tools = {
        "load_csv": Tool(
            name="load_csv",
            desc="Load CSV file and return records",
            
            func=lambda filepath: {
                "records": [
                    {"id": 1, "name": "Alice", "email": "alice@example.com", "signup_date": "2026-04-01", "status": "active"},
                    {"id": 2, "name": "Bob", "email": "bob@example.com", "signup_date": "2026-03-15", "status": "active"},
                    {"id": 3, "name": "Charlie", "email": "charlie@example.com", "signup_date": "2026-02-01", "status": "inactive"},
                    {"id": 4, "name": "Diana", "email": "diana@example.com", "signup_date": "2026-04-10", "status": "active"},
                ],
                "filename": filepath,
                "loaded_at": datetime.now().isoformat()
            }
        ),
        
        "filter_records": Tool(
            name="filter_records",
            desc="Filter records based on condition",
            
            func=lambda records, condition: {
                "filtered_records": [
                    r for r in records.get("records", [])
                    if r.get("status") == "active"
                ],
                "original_count": len(records.get("records", [])),
                "filtered_count": len([r for r in records.get("records", []) if r.get("status") == "active"]),
                "condition_applied": condition
            }
        ),
        
        "count_records": Tool(
            name="count_records",
            desc="Count number of records",
            
            func=lambda records: {
                "count": len(records.get("filtered_records", [])),
                "record_type": "customer"
            }
        ),
        
        "calculate_stats": Tool(
            name="calculate_stats",
            desc="Calculate statistics on records",
            
            func=lambda records: {
                "total_records": len(records.get("filtered_records", [])),
                "avg_signup_month": "April 2026",
                "status_distribution": {"active": len(records.get("filtered_records", [])), "inactive": 0},
                "last_signup": "2026-04-10"
            }
        ),
        
        "generate_report": Tool(
            name="generate_report",
            desc="Generate text report from statistics",
            
            func=lambda total_customers, statistics, timestamp: {
                "report": f"""
CUSTOMER REPORT
Generated: {datetime.now().isoformat()}

SUMMARY:
  Total Active Customers (30 days): {total_customers}
  
STATISTICS:
  Status Distribution: {statistics.get("status_distribution")}
  Average Signup: {statistics.get("avg_signup_month")}
  Last Signup: {statistics.get("last_signup")}

ANALYSIS:
  The customer base is growing with {total_customers} active users in the last 30 days.
  All recent signups show active status, indicating strong onboarding success.

ACTION ITEMS:
  1. Monitor signup trends
  2. Engage inactive customers
  3. Plan for Q2 growth
""",
                "report_type": "customer_analytics",
                "timestamp": timestamp
            }
        ),
    }
    
    return tools


# ============================================================================
# Setup Observer with detailed event handlers
# ============================================================================

def setup_workflow_observer():
    """Create observer with formatted event handlers for workflow."""
    observer = LiveObserver()
    
    @observer.on("agent.start")
    def on_start(event):
        print("\n" + "="*70)
        print("📊 WORKFLOW START")
        print("="*70)
        print(f"Goal: {event['goal']}")
        print(f"Total Steps: {event['total_steps']}")
        print()
    
    @observer.on("agent.reasoning")
    def on_reasoning(event):
        print("🧠 PLANNING PHASE (LLM called once)")
        print("-" * 70)
        print(f"Reasoning: {event['reasoning'][:120]}...")
        print(f"Confidence: {event['confidence']:.0%}")
        print()
    
    @observer.on("tool.before")
    def on_tool_before(event):
        step_id = event['step_id']
        tool = event['tool_name']
        why = event['why']
        args = event.get('args', {})
        
        # Simplify arg display
        args_str = json.dumps(args)
        if len(args_str) > 60:
            args_str = args_str[:57] + "..."
        
        print(f"⚙️  STEP {step_id}: {tool}")
        print(f"    Why: {why}")
        if args:
            print(f"    Args: {args_str}")
    
    @observer.on("tool.after")
    def on_tool_after(event):
        success = event.get("success", False)
        duration = event.get("duration_ms", 0)
        result = event.get("result", {})
        
        if success:
            # Show key result info
            if isinstance(result, dict):
                keys = list(result.keys())[:3]
                print(f"    ✅ Result: {', '.join(keys)}")
            else:
                print(f"    ✅ Result: {str(result)[:50]}...")
        else:
            print(f"    ❌ Error: {event.get('error', 'Unknown error')}")
        
        print()
    
    @observer.on("agent.end")
    def on_end(event):
        print("="*70)
        print("✅ WORKFLOW COMPLETE")
        print("="*70)
        print(f"Total Steps: {event['total_steps']}")
        print()
    
    return observer


# ============================================================================
# Examples
# ============================================================================

async def example_1_manual_execution():
    """Example 1: Manual PoT execution showing all components."""
    print("\n" + "█"*70)
    print("█" + " EXAMPLE 1: Manual PoT Workflow Execution".ljust(69) + "█")
    print("█"*70)
    
    # Setup
    registry = ToolRegistry()
    tools = create_mock_tools()
    for name, tool in tools.items():
        registry.register(name, tool)
    
    llm = create_workflow_llm()
    observer = setup_workflow_observer()
    
    # Plan phase (LLM called ONCE)
    print("\n📋 PLANNING PHASE")
    print("-" * 70)
    try:
        planner = ThoughtPlanner(llm, registry)
        program = planner.plan(
            goal="Process customer data: load, filter recent active customers, count them, calculate statistics, and generate a report",
            session_id="workflow-001"
        )
        print(f"✅ Plan created with {len(program.steps)} steps")
        print(f"   Reasoning: {program.reasoning[:80]}...")
        print(f"   Confidence: {program.confidence:.0%}\n")
    except Exception as e:
        print(f"❌ Planning failed: {e}\n")
        return
    
    # Execute phase (deterministic, NO LLM calls)
    print("🔄 EXECUTION PHASE (deterministic, no LLM calls)")
    print("-" * 70)
    try:
        executor = ThoughtExecutor(registry, observer=observer, max_retries=1)
        result = await executor.execute(program)
        
        print(f"\n📄 FINAL RESULT")
        print("-" * 70)
        if isinstance(result, str) and len(result) > 200:
            print(result)
        else:
            print(f"Result: {result[:200]}...")
    except Exception as e:
        print(f"❌ Execution failed: {e}\n")
        import traceback
        traceback.print_exc()


def example_2_agent_pot_mode():
    """Example 2: High-level Agent API with PoT."""
    print("\n" + "█"*70)
    print("█" + " EXAMPLE 2: Agent-Based PoT (High-Level API)".ljust(69) + "█")
    print("█"*70)
    
    # Setup agent in PoT mode
    registry = ToolRegistry()
    tools = create_mock_tools()
    for name, tool in tools.items():
        registry.register(name, tool)
    
    llm = create_workflow_llm()
    observer = setup_workflow_observer()
    
    # Note: This would normally use Agent class
    print("\n✅ Agent configured in PoT mode")
    print(f"   LLM: WorkflowLLM (returns complete plan)")
    print(f"   Tools: {len(registry.tools)} tools available")
    print(f"   Mode: pot (single LLM call, deterministic execution)")
    print("\n💡 Usage:")
    print("   from nkit import Agent")
    print("   agent = Agent(llm=your_llm, reasoning_mode='pot')")
    print("   result = agent.run('your task')")


def example_3_step_dependencies():
    """Example 3: Show how step dependencies work."""
    print("\n" + "█"*70)
    print("█" + " EXAMPLE 3: Step Dependencies and $step_N References".ljust(69) + "█")
    print("█"*70)
    
    print("""
Step Dependencies Visualization:

┌─────────────────────────────────────────────────────────────┐
│                   EXECUTION DAG                            │
└─────────────────────────────────────────────────────────────┘

Step 1: load_csv
   │
   ├──> Step 2: filter_records (depends_on: [1])
   │       │
   │       ├──> Step 3: count_records (depends_on: [2])
   │       │       │
   │       │       └──> Step 5: generate_report (depends_on: [3, 4])
   │       │
   │       └──> Step 4: calculate_stats (depends_on: [2])
   │               │
   │               └──> Step 5: generate_report (depends_on: [3, 4])

How it Works:

Step 1 result: {"records": [...]}  ← Loaded CSV data

Step 2 receives:
  "records": $step_1  ← Gets Step 1's result
  Result: {"filtered_records": [...]}

Step 3 receives:
  "records": $step_2  ← Gets Step 2's result
  Result: {"count": 3}

Step 4 receives:
  "records": $step_2  ← Gets Step 2's result
  Result: {"total_records": 3, "avg_signup_month": "April 2026", ...}

Step 5 receives:
  "total_customers": $step_3  ← Gets Step 3's count
  "statistics": $step_4       ← Gets Step 4's stats
  Result: {"report": "...full report text..."}

Key Benefits:
✅ Steps can execute in parallel (same depends_on level)
✅ Results chained automatically ($step_N substitution)
✅ Deterministic (same plan = same execution path)
✅ Auditable (full path visible upfront)
""")


def example_4_comparison():
    """Example 4: ReAct vs PoT for this workflow."""
    print("\n" + "█"*70)
    print("█" + " EXAMPLE 4: ReAct vs PoT for Data Pipeline".ljust(69) + "█")
    print("█"*70)
    
    print("""
ReAct Approach (Multiple LLM Calls):
─────────────────────────────────────

1. LLM Call #1: "I should load the data first"
   → Execute: load_csv()
   
2. LLM Call #2: "Now I see the data, I need to filter it"
   → Execute: filter_records()
   
3. LLM Call #3: "Filtered records in place, let me count them"
   → Execute: count_records()
   
4. LLM Call #4: "Now calculate statistics"
   → Execute: calculate_stats()
   
5. LLM Call #5: "Generate report from results"
   → Execute: generate_report()

Result: 5 LLM API calls × latency = ~500ms+ total time


PoT Approach (Single LLM Call):
───────────────────────────────

1. LLM Call #1: "Here's my complete plan" [returns full DAG]
   ┌─ Step 1: load_csv()
   ├─ Step 2: filter_records(results_from_1)
   ├─ Step 3: count_records(results_from_2)
   ├─ Step 4: calculate_stats(results_from_2)
   └─ Step 5: generate_report(results_from_3,4)

2. Execute deterministically (NO more LLM calls)
   → Run Step 1
   → Run Steps 2,3,4 in parallel (or order)
   → Run Step 5 with outputs from 3,4

Result: 1 LLM API call + fast deterministic execution = ~100ms total time

5x Faster! ⚡
""")


def main():
    """Run all examples."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  NKit: PoT Workflow Examples".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print("""
This file demonstrates Program of Thought with practical workflows:
- Multi-step data processing pipeline
- Step dependencies with $step_N references
- Deterministic execution without repeated LLM calls
- Full observability and audit trail
""")
    
    # Show examples
    example_3_step_dependencies()
    example_4_comparison()
    
    # Run async example
    print("\nRunning Example 1: Manual PoT Execution...")
    asyncio.run(example_1_manual_execution())
    
    # Show high-level API
    print("\nRunning Example 2: High-Level Agent API...")
    example_2_agent_pot_mode()
    
    print("\n" + "="*70)
    print("✨ SUMMARY")
    print("="*70)
    print("""
Program of Thought is ideal for:

1. Data Pipelines ✅
   - Multi-step transformations
   - Dependencies between steps
   - Deterministic execution

2. Workflows ✅
   - Order matters
   - Previous results needed
   - Reproducible output

3. Complex Tasks ✅
   - Many steps
   - Can be planned upfront
   - Results chain together

Use these patterns:
- load_data() → filter() → analyze() → report() → save()
- extract() → transform() → validate() → publish()
- search() → rank() → format() → send()
- fetch() → parse() → aggregate() → display()

Next Steps:
→ Replace mock tools with real implementations
→ Connect to real data sources (CSV, DB, API)
→ Add error handling for production
→ Monitor with telemetry
""")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
