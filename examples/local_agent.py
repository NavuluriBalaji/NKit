"""Example 1: Local Agent.

Demonstrates running an agent entirely locally using Ollama, 
while observing its decisions live using LiveObserver and auditing 
them with WhyLog.

Usage: python examples/local_agent.py
"""

import sys
import uuid
from pathlib import Path

# Fix import path for running directly from examples dir
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from NKit.agent.core import Agent
from NKit.llms import OllamaLLM
from NKit.observer import LiveObserver
from NKit.audit import WhyLog

# 1. Setup Live Observer
observer = LiveObserver()

@observer.on("agent.reasoning")
def watch_thinking(event):
    print(f"\n[🧠 THINKING] Step {event['step']}: {event['thought']}")

@observer.on("tool.before")
def watch_tool(event):
    print(f"[🛠 ABOUT TO USE: {event['tool_name']}]")
    print(f"  └─ Why: {event['why']}")
    print(f"  └─ Args: {event['args']}")

@observer.on("tool.after")
def watch_tool_result(event):
    res = str(event['result'])
    short_res = res[:80] + "..." if len(res) > 80 else res
    print(f"[✅ TOOL RESULT] {short_res}")


def main():
    print("="*60)
    print("NKit Local Agent Demo (Ollama)")
    print("="*60)

    # 2. Setup LLM and Log
    # Make sure you run `ollama serve` and `ollama pull llama3` before running this
    try:
        llm = OllamaLLM(model="llama3")
        if not llm.health_check():
            print("WARNING: Ollama isn't running or reachable at localhost:11434")
    except Exception as e:
        print(f"Ollama failed: {e}")

    session_id = str(uuid.uuid4())
    audit_log = WhyLog("./logs/local_audit.jsonl")

    # 3. Create Agent
    agent = Agent(
        llm=llm.complete, # Pass the complete function
        observer=observer
    )

    # Simple custom tool
    @agent.tool("calculator", "Evaluate simple math expressions (e.g. '5 * 2', '100 / 4')")
    def calculator(expression: str) -> str:
        try:
            # Danger: eval is used here purely as a quick dummy tool!
            # Use safe eval logic in production.
            return str(eval(expression, {"__builtins__": None}, {}))
        except Exception as e:
            return f"Error evaluating: {e}"

    # 4. Integrate WhyLog with Observer
    @observer.on("agent.start")
    def _log_start(e):
        audit_log.log(session_id, "agent.start", e["task"])
        
    @observer.on("tool.before")
    def _log_tool(e):
        # We need a fallback goal string here if intercepted globally
        audit_log.log(session_id, "tool.before", "Runtime Task", action=e["tool_name"], why=e["why"])
        
    @observer.on("agent.end")
    def _log_end(e):
        audit_log.log(session_id, "agent.end", "Runtime Task", result=e["final_answer"])

    # 5. Run it
    task = "Find the sum of 534 and 921, then multiply it by 4."
    print(f"\nTask: {task}")
    
    try:
        result = agent.run(task)
        print(f"\n🎉 FINAL ANSWER: {result}")
    except Exception as e:
        print(f"\nAgent failed: {e}")

    # 6. Replay log
    print("\n\n--- Audit Trail Replay ---")
    audit_log.replay(session_id)

if __name__ == "__main__":
    main()
