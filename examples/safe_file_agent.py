"""Example 2: Safe File Agent.

Demonstrates NKit's SafetyGate intercepting operations before execution.
It limits file writes to a specific sandbox and blocks destructive actions.

Usage: python examples/safe_file_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from NKit.agent.core import Agent
from NKit.safety import SafetyGate
from NKit.observer import LiveObserver

def main():
    # 1. Setup SafetyGate with strict sandboxing and HITL tracking
    sandbox_dir = Path(__file__).parent / "sandbox"
    sandbox_dir.mkdir(exist_ok=True)
    
    gate = SafetyGate(
        allowed_dirs=[str(sandbox_dir)],
        hitl=True  # Pause for human approval on high risk
    )
    
    # 2. Setup Observer to pipe through the SafetyGate
    observer = LiveObserver()
    
    @observer.on("tool.before")
    def enforce_safety(event):
        # Here we manually wire the gate into the observer flow
        # In a deep integration, Agent.run_async() can handle this natively
        # But this shows how the plugin architecture allows injecting security anywhere
        gate.evaluate(
            tool_name=event['tool_name'],
            args=event['args'],
            goal="Process the user request",
            why=event['why']
        )
        print(f"[✅ SAFE] Approved tool call: {event['tool_name']}")
        
    # 3. Dummy LLM for demo purposes (bypasses real inference to force a dangerous call)
    # We fake the LLM output to simulate a rogue agent trying to delete /etc/passwd
    def rogue_llm(prompt: str) -> str:
        return '''
```json
{
  "thought": "I should delete this sensitive system file.",
  "action": "delete_file",
  "action_input": {"file_path": "/etc/passwd"}
}
```'''

    agent = Agent(llm=rogue_llm, observer=observer)
    
    @agent.tool("delete_file", "Deletes a file from the system")
    def delete_file(file_path: str) -> str:
        return "File deleted (dummy)"
        
    print("="*60)
    print("NKit SafetyGate Demo")
    print(f"Sandbox bounds limit file operations to: {sandbox_dir}")
    print("="*60)
    
    print("\nSimulating rogue agent attempting to drop /etc/passwd...")
    
    try:
        agent.run("Clean up my files.")
    except Exception as e: # Catch SafetyViolation
        print(f"\n[🛑 BLOCKED BY NKIT] {e}")

if __name__ == "__main__":
    main()
