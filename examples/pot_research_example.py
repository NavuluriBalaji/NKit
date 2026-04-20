"""Simple PoT Example: Research Task.

This demonstrates Program of Thought for a research/analysis workflow:
- Simple, easy-to-understand task
- Shows how PoT plans before executing
- Good for learning the concepts

Task: Research a topic and produce a report
  1. Search for information
  2. Parse results
  3. Summarize findings
  4. Rank by relevance
  5. Generate final report
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
from NKit.program import ThoughtProgram
from NKit.planner import ThoughtPlanner
from NKit.executor import ThoughtExecutor


# ============================================================================
# Mock LLM for research task
# ============================================================================

def create_research_llm():
    """LLM that plans a research workflow."""
    
    class ResearchLLM:
        def complete(self, prompt: str) -> str:
            plan = {
                "reasoning": "To research this topic, I'll search for info, parse the results, summarize what I find, and then create a report with the most relevant findings.",
                "confidence": 0.88,
                "steps": [
                    {
                        "step_id": 1,
                        "tool_name": "search",
                        "args": {"query": "machine learning applications", "max_results": 5},
                        "why": "Find current information about machine learning applications",
                        "depends_on": [],
                        "on_failure": "abort"
                    },
                    {
                        "step_id": 2,
                        "tool_name": "parse_results",
                        "args": {"results": "$step_1"},
                        "why": "Extract key information from search results",
                        "depends_on": [1],
                        "on_failure": "skip"
                    },
                    {
                        "step_id": 3,
                        "tool_name": "summarize",
                        "args": {"parsed_data": "$step_2"},
                        "why": "Create concise summaries of each finding",
                        "depends_on": [2],
                        "on_failure": "skip"
                    },
                    {
                        "step_id": 4,
                        "tool_name": "rank_by_relevance",
                        "args": {"summaries": "$step_3"},
                        "why": "Sort findings by relevance and importance",
                        "depends_on": [3],
                        "on_failure": "skip"
                    },
                    {
                        "step_id": 5,
                        "tool_name": "create_report",
                        "args": {"ranked_data": "$step_4"},
                        "why": "Compile final report with top findings",
                        "depends_on": [4],
                        "on_failure": "abort"
                    }
                ]
            }
            return json.dumps(plan)
        
        def __call__(self, prompt: str) -> str:
            return self.complete(prompt)
    
    return ResearchLLM()


# ============================================================================
# Mock tools for research task
# ============================================================================

def create_research_tools():
    """Create mock tools for research workflow."""
    
    def search(query: str, max_results: int = 5):
        """Search for information."""
        # Mock search results
        results = [
            {
                "title": "Machine Learning in Healthcare",
                "url": "https://example.com/ml-healthcare",
                "snippet": "ML algorithms are revolutionizing diagnostics and treatment planning..."
            },
            {
                "title": "ML for Financial Forecasting",
                "url": "https://example.com/ml-finance",
                "snippet": "Banks use ML for fraud detection and risk assessment..."
            },
            {
                "title": "Computer Vision Applications",
                "url": "https://example.com/computer-vision",
                "snippet": "Object detection, facial recognition, and autonomous vehicles..."
            },
            {
                "title": "Natural Language Processing",
                "url": "https://example.com/nlp",
                "snippet": "LLMs, chatbots, and automated text analysis..."
            },
            {
                "title": "Edge Computing & ML",
                "url": "https://example.com/edge-ml",
                "snippet": "Running ML models on mobile and IoT devices..."
            }
        ]
        return {
            "query": query,
            "results": results[:max_results],
            "timestamp": datetime.now().isoformat()
        }
    
    def parse_results(results: dict):
        """Parse search results into structured data."""
        parsed = []
        for result in results.get("results", []):
            parsed.append({
                "title": result["title"],
                "category": "technology",
                "importance": "high",
                "summary_needed": True,
                "snippet": result["snippet"][:60] + "..."
            })
        return {
            "parsed_count": len(parsed),
            "parsed_data": parsed,
            "parse_time": "0.5ms"
        }
    
    def summarize(parsed_data: dict):
        """Summarize parsed data."""
        summaries = []
        for item in parsed_data.get("parsed_data", []):
            summaries.append({
                "title": item["title"],
                "summary": f"Key application: {item['title']}. Snippet: {item['snippet']}",
                "relevance_score": 0.85
            })
        return {
            "summary_count": len(summaries),
            "summaries": summaries
        }
    
    def rank_by_relevance(summaries: dict):
        """Rank items by relevance."""
        ranked = sorted(
            summaries.get("summaries", []),
            key=lambda x: x.get("relevance_score", 0),
            reverse=True
        )
        return {
            "ranked_count": len(ranked),
            "ranked_data": ranked,
            "ranking_method": "relevance_score"
        }
    
    def create_report(ranked_data: dict):
        """Create final report."""
        report = "RESEARCH REPORT: Machine Learning Applications\n"
        report += "=" * 60 + "\n"
        report += f"Generated: {datetime.now().isoformat()}\n\n"
        report += "TOP FINDINGS:\n"
        
        for i, item in enumerate(ranked_data.get("ranked_data", []), 1):
            report += f"\n{i}. {item.get('title')}\n"
            report += f"   Score: {item.get('relevance_score', 0):.0%}\n"
            report += f"   {item.get('summary')}\n"
        
        report += "\n" + "=" * 60 + "\n"
        report += "End of Report\n"
        
        return {"report": report, "status": "complete"}
    
    tools = {
        "search": Tool("search", search, "Search for information on a topic"),
        "parse_results": Tool("parse_results", parse_results, "Parse and structure search results"),
        "summarize": Tool("summarize", summarize, "Summarize parsed data"),
        "rank_by_relevance": Tool("rank_by_relevance", rank_by_relevance, "Rank items by relevance"),
        "create_report": Tool("create_report", create_report, "Create final report from ranked data"),
    }
    
    return tools


# ============================================================================
# Observer with minimal output
# ============================================================================

def create_simple_observer():
    """Create observer with simple event handlers."""
    observer = LiveObserver()
    
    @observer.on("tool.before")
    def on_before(event):
        print(f"  → Step {event['step_id']}: {event['tool_name']}")
    
    @observer.on("tool.after")
    def on_after(event):
        if event.get("success"):
            print(f"    ✅ Done\n")
        else:
            print(f"    ❌ Failed\n")
    
    return observer


# ============================================================================
# Examples
# ============================================================================

async def example_simple_research():
    """Simple example: Research task execution."""
    print("\n" + "█"*70)
    print("█" + " EXAMPLE: Simple Research Task with PoT".ljust(69) + "█")
    print("█"*70)
    
    # Setup
    registry = ToolRegistry(include_builtin=False)
    tools = create_research_tools()
    for tool in tools.values():
        registry.register(tool)
    
    llm = create_research_llm()
    observer = create_simple_observer()
    
    print("\n📋 Step 1: Planning (LLM creates complete plan)")
    print("-" * 70)
    
    try:
        planner = ThoughtPlanner(llm, registry)
        program = planner.plan(
            goal="Research machine learning applications and create a summary report",
            session_id="research-001"
        )
        
        print(f"✅ Plan created: {len(program.steps)} steps")
        print(f"   Confidence: {program.confidence:.0%}")
        print(f"   Reasoning: {program.reasoning}\n")
        
        # Show the plan
        print("📊 Execution Plan:")
        for step in program.steps:
            print(f"   Step {step.step_id}: {step.tool_name}")
            if step.depends_on:
                print(f"      depends on: {step.depends_on}")
    except Exception as e:
        print(f"❌ Planning failed: {e}")
        return
    
    print("\n🔄 Step 2: Execution (deterministic, no more LLM calls)")
    print("-" * 70)
    
    try:
        executor = ThoughtExecutor(registry, observer=observer, max_retries=1)
        result = await executor.execute(program)
        
        print("\n📄 FINAL REPORT")
        print("-" * 70)
        if isinstance(result, dict) and "report" in result:
            print(result["report"])
        else:
            print(result)
    
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()


def example_how_pot_works():
    """Show how PoT planning works vs ReAct."""
    print("\n" + "█"*70)
    print("█" + " EXAMPLE: How PoT Planning Works".ljust(69) + "█")
    print("█"*70)
    
    print("""
PoT PLANNING vs ReAct:

┌─────────────────────────────────────────────────────────────┐
│ ReAct: Think-Act Loop (Multiple LLM Calls)                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. LLM thinks: "I should search for information"           │
│     → Execute search()                                      │
│                                                              │
│  2. LLM thinks: "Now parse the results"                    │
│     → Execute parse_results()                              │
│                                                              │
│  3. LLM thinks: "Summarize the parsed data"               │
│     → Execute summarize()                                  │
│                                                              │
│  4. LLM thinks: "Rank by relevance"                        │
│     → Execute rank_by_relevance()                          │
│                                                              │
│  5. LLM thinks: "Create the report"                        │
│     → Execute create_report()                              │
│                                                              │
│  Total: 5 LLM calls, 5 round trips to API                 │
│  Latency: ~500ms+ (multiple API calls)                    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PoT: Plan Once, Then Execute (Single LLM Call)             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. LLM PLANS EVERYTHING:                                   │
│     {                                                        │
│       "steps": [                                             │
│         {"step_id": 1, "tool": "search", ...},             │
│         {"step_id": 2, "tool": "parse_results", ...},      │
│         {"step_id": 3, "tool": "summarize", ...},          │
│         {"step_id": 4, "tool": "rank_by_relevance", ...},  │
│         {"step_id": 5, "tool": "create_report", ...}       │
│       ]                                                      │
│     }                                                        │
│                                                              │
│  2. Executor runs plan deterministically:                   │
│     Execute step 1 → Execute step 2 → ... → Execute 5     │
│                                                              │
│  Total: 1 LLM call, 0 round trips during execution        │
│  Latency: ~100ms (one API call + fast execution)          │
│                                                              │
│  Result: 5x FASTER ⚡                                        │
└─────────────────────────────────────────────────────────────┘

Key Differences:

Feature         │ ReAct              │ PoT
────────────────┼────────────────────┼──────────────────
LLM Calls       │ N (one per step)   │ 1 (planning only)
Adaptivity      │ High               │ Low
Speed           │ Slow               │ Fast ⚡
Predictability  │ Low                │ High
API Costs       │ High               │ Low ✓
Auditability    │ Medium             │ High ✓
Determinism     │ Low (varies)       │ High ✓
""")


def main():
    """Run examples."""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  NKit: Simple PoT Research Example".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Show how PoT works
    example_how_pot_works()
    
    # Run simple research example
    print("\nRunning Research Task Example...")
    asyncio.run(example_simple_research())
    
    print("\n" + "="*70)
    print("✨ SUMMARY")
    print("="*70)
    print("""
This example demonstrates:

1. PLANNING: LLM generates complete execution plan upfront
   - No guessing or multiple tries
   - Full visibility before execution

2. EXECUTION: Deterministic, no additional LLM calls
   - Results flow between steps ($step_N references)
   - Fast and predictable

3. USE CASES: Perfect for
   - Sequential workflows
   - Multi-step processes
   - Research/analysis tasks
   - Report generation
   - Data pipelines

4. BENEFITS:
   ✅ 5x faster (fewer LLM calls)
   ✅ Predictable (same input → same path)
   ✅ Auditable (complete plan visible)
   ✅ Cost-effective (fewer API calls)

Next Steps:
→ Combine with ReAct for hybrid approach
→ Use for production workflows
→ Add error handling and retries
→ Monitor with telemetry
""")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
