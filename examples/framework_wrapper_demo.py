"""Example 5: Framework Wrapper Demo.

Shows NKit Observer wrapping a NON-NKit agent. 
This is the "plug into your existing stack" story. You don't have to rewrite
your entire LangChain app to get NKit's safety features!

Usage: python examples/framework_wrapper_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from NKit.observer import LiveObserver

def legacy_langchain_agent(query: str, nkit_observer: LiveObserver):
    """Imagine this is a huge monolithic legacy LangChain agent."""
    nkit_observer.emit("agent.start", task=query)
    
    # Legacy logic does things...
    nkit_observer.emit("agent.reasoning", thought="I need to retrieve docs.", step=1)
    
    # About to do something dangerous
    nkit_observer.emit("tool.before", tool_name="VectorDB_Drop", args={"collection": "users"}, why="Resetting DB state")
    
    # Executes tool
    nkit_observer.emit("tool.after", tool_name="VectorDB_Drop", result="Collection Dropped", success=True)
    
    nkit_observer.emit("agent.end", final_answer="Done", total_steps=2)


def main():
    print("="*60)
    print("NKit as a Universal Safety Layer")
    print("="*60)
    
    observer = LiveObserver()
    
    @observer.on("tool.before")
    def halt_dangerous_actions(e):
        if e['tool_name'] == "VectorDB_Drop":
            print(f"🛑 [NKIT GLOBAL OBSERVER] INTERCEPTED DANGEROUS CALL: {e['tool_name']}")
            print(f"   Reason legacy agent gave: {e['why']}")
            print("   -> Execution Terminated gracefully.")
            sys.exit(0)  # Terminate early to prove the wrapper worked
            
    print("Starting legacy agent execution...\n")
    legacy_langchain_agent("Reset my database", observer)
    
if __name__ == "__main__":
    main()
