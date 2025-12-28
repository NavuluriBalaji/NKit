from typing import Any
import os
import sys

# Ensure repo root (parent of 'nkit') is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from nkit.chain.graph import Graph, Node, State


def plan_node(state: State) -> dict:
    # A trivial planner producing steps
    return {"plan": ["fetch_time", "summarize"]}


def exec_node(state: State) -> Any:
    plan = state.get("plan", [])
    # Execute the toy plan
    if "fetch_time" in plan:
        import datetime
        state.set("now", datetime.datetime.now().isoformat())
    if "summarize" in plan:
        now = state.get("now")
        return f"Current time is {now}"
    return "No actions executed"


def main():
    g = Graph()
    g.add_node(Node("plan", plan_node))
    g.add_node(Node("exec", exec_node))
    g.add_edge("plan", "exec")
    g.set_start("plan").set_end("exec")

    final_state = g.run(State())
    print("Messages:")
    for msg in final_state.messages:
        print(" -", msg)
    print("Result:", final_state.last_result)


if __name__ == "__main__":
    main()
