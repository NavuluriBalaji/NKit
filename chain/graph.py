from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from NKit.utils import is_async_function, run_sync_or_async
from NKit.memory import Memory


@dataclass
class State:
    """Execution state passed between graph nodes.

    - `data`: arbitrary dict for intermediate values
    - `memory`: optional shared memory store
    - `messages`: textual logs/messages for debugging or agent communication
    - `last_result`: last node's output (string or structured)
    - `error`: error message if a node failed
    """
    data: Dict[str, Any] = field(default_factory=dict)
    memory: Optional[Memory] = None
    messages: List[str] = field(default_factory=list)
    last_result: Any = None
    error: Optional[str] = None

    def log(self, msg: str) -> None:
        self.messages.append(msg)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


class Node:
    """A graph node encapsulating a handler.

    The `handler` receives `State` and may return:
      - `State` (updated)
      - `dict` (merged into `state.data`)
      - any other value (stored in `state.last_result`)

    Handlers can be sync or async.
    """

    def __init__(self, name: str, handler: Callable[[State], Union[State, Dict[str, Any], Any]]):
        self.name = name
        self.handler = handler
        self.is_async = is_async_function(handler)

    async def run(self, state: State) -> State:
        try:
            result = await run_sync_or_async(self.handler, state)
            if isinstance(result, State):
                state = result
            elif isinstance(result, dict):
                state.data.update(result)
                state.last_result = result
            else:
                state.last_result = result
            return state
        except Exception as e:
            state.error = f"Node '{self.name}' failed: {e}"
            return state


class Edge:
    """Directed edge from `src` to `dst` with optional condition predicate."""

    def __init__(self, src: str, dst: str, condition: Optional[Callable[[State], bool]] = None):
        self.src = src
        self.dst = dst
        self.condition = condition

    def matches(self, state: State) -> bool:
        if self.condition is None:
            return True
        try:
            return bool(self.condition(state))
        except Exception:
            return False


class Graph:
    """A lightweight execution graph similar in spirit to LangGraph.

    Supports conditional routing and sync/async node handlers.
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.start: Optional[str] = None
        self.end: Optional[str] = None

    def add_node(self, node: Node) -> "Graph":
        self.nodes[node.name] = node
        if self.start is None:
            self.start = node.name
        return self

    def add_edge(self, src: str, dst: str, condition: Optional[Callable[[State], bool]] = None) -> "Graph":
        self.edges.append(Edge(src, dst, condition))
        return self

    def set_start(self, name: str) -> "Graph":
        self.start = name
        return self

    def set_end(self, name: str) -> "Graph":
        self.end = name
        return self

    def next_nodes(self, current: str, state: State) -> List[str]:
        """Resolve next nodes based on edges and conditions.
        Returns a list to allow for future fan-out/concurrency.
        """
        nxt: List[str] = []
        for e in self.edges:
            if e.src == current and e.matches(state):
                nxt.append(e.dst)
        return nxt

    async def arun(self, state: Optional[State] = None) -> State:
        if not self.start:
            raise ValueError("Graph has no start node")
        state = state or State()

        current = self.start
        visited = set()
        while current:
            if current in visited and current == self.end:
                # prevent infinite loops on end
                break
            visited.add(current)

            node = self.nodes.get(current)
            if not node:
                state.error = f"Missing node: {current}"
                break

            state.log(f"Running node: {current}")
            state = await node.run(state)
            if state.error:
                state.log(state.error)
                break

            if self.end and current == self.end:
                state.log("Reached end node")
                break

            nexts = self.next_nodes(current, state)
            if not nexts:
                # No outgoing edges; stop
                state.log("No next nodes; halting")
                break
            # For now, choose first. Future: parallel fan-out.
            current = nexts[0]

        return state

    def run(self, state: Optional[State] = None) -> State:
        return asyncio.run(self.arun(state))
