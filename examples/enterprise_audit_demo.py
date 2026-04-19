"""Example 4: Enterprise Audit Demo.

Demostrates how you pitch NKit to a compliance officer or enterprise client.
Shows a logistics/inventory query workflow heavily monitored by WhyLog.

Usage: python examples/enterprise_audit_demo.py
"""

import sys
import uuid
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from NKit.agent.core import Agent
from NKit.audit import WhyLog

def mock_enterprise_llm(prompt: str) -> str:
    """A mock LLM to guarantee predictable path execution for the demo."""
    if "[MOCK DB RESULT] rows: 42" in prompt or "observation: [MOCK DB RESULT]" in prompt:
        return '''
```json
{
  "thought": "I have the count, returning the final answer.",
  "final_answer": "There are 42 transmission parts in stock."
}
```'''
    elif "Count parts" in prompt:
         return '''
```json
{
  "thought": "I need to query the inventory DB for parts matching 'transmission'.",
  "action": "query_db",
  "action_input": {"sql": "SELECT count(*) FROM parts WHERE type='transmission'"}
}
```'''
    else:
        return '''
```json
{
  "thought": "I have the count, returning the final answer.",
  "final_answer": "There are 42 transmission parts in stock."
}
```'''

def main():
    print("="*80)
    print("NKit Enterprise Workflow: Automotive Inventory Audit")
    print("="*80)
    
    session_id = str(uuid.uuid4())
    log_path = "./logs/enterprise_audit.jsonl"
    audit_log = WhyLog(log_path)
    
    goal = "Count parts for 'transmission' in inventory."
    
    agent = Agent(llm=mock_enterprise_llm)
    
    @agent.tool("query_db", "Query the inventory SQL database securely.")
    def query_db(sql: str) -> str:
        # We manually push audit entries here to simulate the core integration
        audit_log.log(session_id, "tool.before", goal, action="query_db", why="To gather part stock counts.")
        return "[MOCK DB RESULT] rows: 42"

    print(f"Goal: {goal}\nProcessing...")
    
    try:
        # Pre-log start
        audit_log.log(session_id, "agent.start", goal)
        res = agent.run(goal)
        audit_log.log(session_id, "agent.end", goal, result=res)
        print(f"Workflow Complete: {res}\n")
    except Exception as e:
        print(f"Error: {e}")
        
    print("\n" + "#"*80)
    print("THIS IS WHAT YOU SHOW THE COMPLIANCE TEAM:")
    print("#"*80)
    audit_log.replay(session_id)
    
if __name__ == "__main__":
    main()
