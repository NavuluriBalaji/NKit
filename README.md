# NKit — Production Safety Layer for Agentic AI

> **Live observability, pre-execution safety, and audit trails for any agent framework.**

NKit is a nano, minimalistic, production-ready framework for building ReAct agents, or wrapping your existing ones. It focuses natively on ensuring that agents are safe, compliant, and observable *before* they execute destructive actions, solving the biggest blockers to enterprise AI adoption.

---

## 🛑 Why NKit? (The Missing Safety Layer)
Most agent frameworks focus on chaining together LLM queries but treat production safety as an afterthought. NKit was built specifically to solve three critical problems:

1. **Pre-Execution Intent Verification:** Other frameworks let tools run immediately. NKit's `SafetyGate` pauses and evaluates an agent's intent *before* execution, blocking misaligned goals or destructive actions automatically (with an optional HITL fallback).
2. **The `Why Log`:** Classic logging tells you "Agent executed query `DROP TABLE users`". NKit's `WhyLog` creates a structured JSONL audit trail capturing the exact chain of thought that led to that hallucinated conclusion.
3. **Live Decision Streaming:** NKit ditches post-mortem log scraping in favor of a real-time event bus (`LiveObserver`), allowing developers and compliance teams to monitor decisions *live*.

---

## ⚡ Quick Start

```python
from nkit.agent import Agent
from nkit.llms import OllamaLLM
from nkit.observer import LiveObserver

# 1. Start live observer
observer = LiveObserver()

@observer.on("tool.before")
def watch_intent(e):
    print(f"Agent attempting {e['tool_name']} because: {e['why']}")

# 2. Hook up local LLM
llm = OllamaLLM(model="llama3")

# 3. Mount and run
agent = Agent(llm=llm.complete, observer=observer)
agent.run("Summarize the current market trends")
```

---

## 🏗 Core Concepts

* **Observer (`nkit.observer`)**: Intercepts the core ReAct loop allowing developers to use asynchronous `@observer.on()` listeners to monitor the start, thoughts, tool pre-execution, and results of actions dynamically.
* **SafetyGate (`nkit.safety`)**: Pre-execution middleware that heuristically and conditionally blocks path-escapes, destructive keywords, and untrained domain accesses. Supports HITL (Human-in-the-loop) approvals.
* **WhyLog (`nkit.audit`)**: Extensively formats the trace-history into a rotating 10MB `audit.jsonl` log file, embedding the thought-process behind every single external action.

---

## 🧠 Supported LLM Providers

NKit provides pure, bloat-free `urllib`-based LLM adapters ensuring you aren't dragging in multi-megabyte SDKs just to query an API:

* `OllamaLLM`: Calls localhost (llama3, phi3, mistral).
* `OpenAILLM`: Calls OpenAI (gpt-4o) with native exponential backoff.
* `AnthropicLLM`: Calls Anthropic (claude-3-opus).
* `OpenRouterLLM`: Universal model passthrough to api.openrouter.ai.

---

## 📚 Examples
Explore the `/examples` directory to see NKit in action:

1. **`local_agent.py`**: A fully local agent using Ollama and the WhyLog.
2. **`safe_file_agent.py`**: Demonstrates the SafetyGate isolating an agent within a sandbox.
3. **`multi_agent_research.py`**: Multi-agent task orchestration tracked via Observer.
4. **`enterprise_audit_demo.py`**: An enterprise use-case mimicking a logistics compliance workflow.
5. **`framework_wrapper_demo.py`**: Wraps a mock LangChain agent into the NKit LiveObserver to prove interoperability.

---

## ✅ Production Checklist
Out of the box, NKit provides:
- [x] Pre-Execution Safety Interceptors & Human-in-The-Loop.
- [x] Asynchronous real-time observation streaming.
- [x] Centralized, stateful Audit Logging (`WhyLogs`).
- [x] Hard bounds on executions: 30s Tool Timeouts, 10KB Result Sizing, Max Retries limiting.
- [x] Security-principled path sandboxing and String parsing.

---

## 🗺 Architecture

```text
 ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
 │   User Prompt   │ ────▶ │  Agent / Crew   │ ────▶ │   LLM Adapter   │
 └─────────────────┘       └─────────────────┘       └─────────────────┘
                                   │
                                   ▼
                         ┌─────────────────┐ (Triggers agent.reasoning)
                         │  Live Observer  │
                         └─────────────────┘
                                   │
                                   ▼
                         ┌─────────────────┐ (Evaluates Risk & HITL)
                         │   Safety Gate   │
                         └─────────────────┘
                                   │
                                   ▼
                         ┌─────────────────┐ (Logs the "Why")
                         │     WhyLog      │
                         └─────────────────┘
                                   │
                                   ▼
                         ┌─────────────────┐ (Timeout bounds & limits)
                         │ Tool Execution  │
                         └─────────────────┘
```

---

## 🤝 Contributing
NKit adheres strictly to SOLID design principles. No single module should become a God class. Features must be injected as plugins via interfaces. If extending NKit, remember its north star: **Security, safety, and observability above all else.** Pull requests are welcome!
