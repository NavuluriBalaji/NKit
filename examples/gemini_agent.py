"""Example: Gemini Agent.

Demonstrates running an agent using Google Gemini via the pure urllib HTTP adapter.

Usage: 
1. set GEMINI_API_KEY=your_key_here
2. python examples/gemini_agent.py
"""

import sys
import os
from pathlib import Path

# Fix import path for running directly from examples dir
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from NKit.agent.core import Agent
from NKit.llms import GeminiLLM
from NKit.observer import LiveObserver

def main():
    print("="*60)
    print("NKit Gemini Agent Demo")
    print("="*60)
    
    if not os.environ.get("GEMINI_API_KEY"):
        print("🛑 Error: Please set the GEMINI_API_KEY environment variable.")
        print("Example (Windows): $env:GEMINI_API_KEY='AizaSy...'")
        sys.exit(1)

    try:
        # 1. Setup Gemini LLM using flash model by default
        llm = GeminiLLM(model="gemini-2.5-flash")
        
        # 2. Add an observer to stream thoughts live
        observer = LiveObserver()
        
        @observer.on("agent.reasoning")
        def watch_thinking(event):
            thought = event.get('thought', '')
            # Truncate thought for cleaner terminal output
            print(f"\n[🧠 GEMINI] {thought[:100]}..." if len(thought) > 100 else f"\n[🧠 GEMINI] {thought}")

        @observer.on("tool.before")
        def watch_tool(event):
             print(f"[⚙️ EXECUTING TOOL] {event['tool_name']} => {event['args']}")

        # 3. Create and execute the Agent
        agent = Agent(llm=llm.complete, observer=observer)
        
        task = "Tell me a very brief fact about quantum entanglement, then check the current time."
        print(f"\nTask: {task}")
        
        result = agent.run(task)
        
        print("\n" + "="*60)
        print("🎉 GEMINI FINAL ANSWER:")
        print("="*60)
        print(result)
        
    except Exception as e:
        print(f"\nExecution failed: {e}")

if __name__ == "__main__":
    main()
