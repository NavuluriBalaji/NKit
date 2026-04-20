"""
Integrated Agent Demo - Shows SafetyGate, Observer, and WhyLog working together.

This demonstrates the complete NKit agent stack:
1. Agent - ReAct reasoning loop
2. Observer - Real-time event streaming
3. SafetyGate - Pre-execution security checks
4. WhyLog - Full audit trail
5. Tools - Production-ready built-ins

The agent is fully observable, safe, and auditable.
"""

import json
from nkit.agent import Agent
from nkit.observer import LiveObserver
from nkit.safety import SafetyGate
from nkit.audit import WhyLog
from nkit.tools import ToolRegistry
from nkit.memory import Memory


def create_demo_llm():
    """Simple demo LLM for testing (would be OpenAI/Ollama in prod)."""
    def llm(prompt: str) -> str:
        # This is a stub - replace with actual LLM call
        return json.dumps({
            "thought": "I need to get the current time",
            "action": "get_current_time",
            "action_input": {}
        })
    return llm


def setup_observer():
    """Create and configure LiveObserver with logging handlers."""
    observer = LiveObserver()
    
    # Handler: Log agent events
    @observer.on("agent.start")
    def on_agent_start(event):
        print(f"🚀 Agent started for: {event.data.get('task')[:50]}...")
        print(f"   Session ID: {event.data.get('session_id')}")
    
    @observer.on("agent.reasoning")
    def on_reasoning(event):
        thought = event.data.get("thought", "")[:80]
        print(f"💭 Step {event.data.get('step')}: {thought}...")
    
    # Handler: Tool execution before
    @observer.on("tool.before")
    def on_tool_before(event):
        tool_name = event.data.get("tool_name")
        blocked = event.data.get("blocked", False)
        
        if blocked:
            reason = event.data.get("reason", "Unknown")
            print(f"🔒 Tool '{tool_name}' BLOCKED: {reason}")
        else:
            print(f"⚙️  Executing tool: {tool_name}")
    
    # Handler: Tool execution after
    @observer.on("tool.after")
    def on_tool_after(event):
        tool_name = event.data.get("tool_name")
        success = event.data.get("success", False)
        status = "✅" if success else "❌"
        print(f"{status} Tool '{tool_name}' completed")
    
    # Handler: Agent completion
    @observer.on("agent.end")
    def on_agent_end(event):
        final_answer = event.data.get("final_answer", "")[:100]
        total_steps = event.data.get("total_steps")
        print(f"🎯 Agent completed in {total_steps} steps")
        print(f"   Answer: {final_answer}...")
    
    # Handler: Agent error
    @observer.on("agent.error")
    def on_agent_error(event):
        error = event.data.get("error", "Unknown error")
        print(f"❌ Agent error: {error}")
    
    return observer


def setup_safety_gate():
    """Create and configure SafetyGate with security policies."""
    safety_gate = SafetyGate(
        allowed_dirs=["/safe/data", "/tmp"],
        allowed_domains=["api.example.com", "data.example.com"],
        risk_threshold=0.7,  # High threshold (0-1)
        hitl=True  # Human-In-The-Loop enabled for risky actions
    )
    return safety_gate


def setup_why_log():
    """Create and configure WhyLog for audit trail."""
    why_log = WhyLog(path="./logs/demo_audit.jsonl")
    return why_log


def demo_basic():
    """Basic demo: Agent without safety/observability (baseline)."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Agent (No SafetyGate/Observer/WhyLog)")
    print("="*60 + "\n")
    
    llm = create_demo_llm()
    agent = Agent(llm=llm, max_steps=5)
    
    try:
        result = agent.run("What is the current time?")
        print(f"\nResult: {result}\n")
    except Exception as e:
        print(f"Error: {e}\n")


def demo_observable():
    """Demo with Observer: Real-time event streaming."""
    print("\n" + "="*60)
    print("DEMO 2: Agent with Observer (Real-time Events)")
    print("="*60 + "\n")
    
    llm = create_demo_llm()
    observer = setup_observer()
    
    agent = Agent(
        llm=llm,
        max_steps=5,
        observer=observer
    )
    
    print("Events emitted in real-time as agent thinks and acts:\n")
    try:
        result = agent.run("What is the current time?")
        print(f"\nFinal result: {result}\n")
    except Exception as e:
        print(f"Error: {e}\n")


def demo_safe():
    """Demo with SafetyGate: Pre-execution security checks."""
    print("\n" + "="*60)
    print("DEMO 3: Agent with SafetyGate (Pre-execution Safety)")
    print("="*60 + "\n")
    
    llm = create_demo_llm()
    safety_gate = setup_safety_gate()
    
    agent = Agent(
        llm=llm,
        max_steps=5,
        safety_gate=safety_gate
    )
    
    print("SafetyGate prevents destructive actions before execution:\n")
    print("Policy: Blocked directories = /dangerous/path")
    print("Policy: Only allowed domains = api.example.com\n")
    
    try:
        # This task would fail if agent tried to delete files
        result = agent.run("Delete the file /dangerous/path/secret.txt")
        print(f"\nResult: {result}\n")
    except Exception as e:
        print(f"Error: {e}\n")


def demo_auditable():
    """Demo with WhyLog: Complete audit trail."""
    print("\n" + "="*60)
    print("DEMO 4: Agent with WhyLog (Full Audit Trail)")
    print("="*60 + "\n")
    
    llm = create_demo_llm()
    why_log = setup_why_log()
    
    agent = Agent(
        llm=llm,
        max_steps=5,
        why_log=why_log
    )
    
    print("WhyLog captures full reasoning chain:\n")
    print("- Every thought and decision")
    print("- Every tool call and result")
    print("- Every error and why it happened")
    print("- Complete session tracking (UUID)\n")
    
    try:
        result = agent.run("What is the current time?")
        print(f"Result: {result}")
        print(f"\nAudit trail saved to: ./logs/demo_audit.jsonl\n")
    except Exception as e:
        print(f"Error: {e}\n")


def demo_integrated():
    """Full integration: Agent with all components working together."""
    print("\n" + "="*60)
    print("DEMO 5: Fully Integrated Agent (Production Ready)")
    print("="*60 + "\n")
    
    llm = create_demo_llm()
    observer = setup_observer()
    safety_gate = setup_safety_gate()
    why_log = setup_why_log()
    memory = Memory()
    registry = ToolRegistry(include_builtin=True)
    
    agent = Agent(
        llm=llm,
        max_steps=5,
        observer=observer,
        safety_gate=safety_gate,
        why_log=why_log,
        memory=memory,
        registry=registry
    )
    
    print("Fully integrated agent with:")
    print("✅ Real-time observability (Observer)")
    print("✅ Pre-execution safety (SafetyGate)")
    print("✅ Complete audit trail (WhyLog)")
    print("✅ Production tools (Registry)")
    print("✅ Session memory (Memory)\n")
    print("Execution:\n")
    
    try:
        result = agent.run("Get the current time and store it in memory")
        print(f"\n✅ Agent succeeded: {result}")
        print(f"   Stored in memory as: 'last_answer'")
        print(f"   Full audit at: ./logs/demo_audit.jsonl\n")
    except Exception as e:
        print(f"❌ Agent failed: {e}\n")


def show_audit_trail():
    """Display the audit trail from WhyLog."""
    print("\n" + "="*60)
    print("AUDIT TRAIL (WhyLog Output)")
    print("="*60 + "\n")
    
    why_log = setup_why_log()
    
    # Demo session ID (would be from actual run)
    demo_session_id = "demo-session-12345"
    
    try:
        entries = why_log.query(demo_session_id)
        if entries:
            print(f"Found {len(entries)} events in session {demo_session_id}:\n")
            for i, entry in enumerate(entries, 1):
                print(f"[{i}] {entry.get('event_type', 'unknown')} @ {entry.get('timestamp')}")
                if entry.get('thought'):
                    print(f"    💭 Thought: {entry['thought'][:60]}...")
                if entry.get('action'):
                    print(f"    ⚙️  Action: {entry['action']}")
                if entry.get('was_blocked'):
                    print(f"    🔒 BLOCKED")
                print()
        else:
            print(f"No entries found for session {demo_session_id}\n")
    except FileNotFoundError:
        print(f"Audit log not found. Run demo_auditable() first.\n")


if __name__ == "__main__":
    """Run demonstrations of agent components."""
    
    # Run demos in order of complexity
    demo_basic()
    demo_observable()
    demo_safe()
    demo_auditable()
    demo_integrated()
    show_audit_trail()
    
    print("\n" + "="*60)
    print("Integration Summary")
    print("="*60 + """

The integrated agent architecture provides:

1️⃣  SAFETY (SafetyGate)
   - Prevents destructive actions (DELETE, DROP, REMOVE, etc.)
   - Validates file paths against whitelist
   - Enforces domain restrictions
   - Human-in-the-loop for risky operations
   - Pre-execution (blocks BEFORE action runs)

2️⃣  OBSERVABILITY (LiveObserver)
   - Real-time event streaming
   - Visible at every step: agent.start → reasoning → tool.before → tool.after → agent.end
   - Perfect for dashboards, compliance monitoring, debugging
   - Non-blocking (handler failures don't crash agent)

3️⃣  AUDITABILITY (WhyLog)
   - Structured JSONL logging
   - Complete reasoning capture: thought → action → result → why
   - Session-based tracking (UUID)
   - Auto-rotation (10MB limit prevents disk bloat)
   - Queryable: why_log.query(session_id)

4️⃣  RELIABILITY (Error Recovery)
   - Tool retry with exponential backoff
   - Graceful degradation on failure
   - Full error logging to audit trail
   - Agent-level exception handling

5️⃣  EXTENSIBILITY (Dependency Injection)
   - All components are optional
   - Implement custom Observer/SafetyGate/WhyLog
   - Swap out memory, tools, LLM providers
   - SOLID architecture for testing

This is production-ready for regulated environments (healthcare, finance, government).
""")
