## Core Philosophy — Follow This Always
- **Nano and minimalistic** — every line earns its place, no bloat
- **Security first** — validation and limits are built in, not bolted on
- **Plugin architecture** — swap memory, LLM, tools via constructor injection
- **SOLID principles** — single responsibility, no god classes
- **Graceful degradation** — tool failures log and continue, never crash agent
- **Dependency injection** — Agent depends on abstractions, not concretions

## Code Style Rules — Always Follow
1. Every public class and function has a docstring with: purpose, args, returns, example
2. Every LLM call has timeout + retry (max 3, exponential backoff)
3. Every tool execution has timeout (default 30s) + result size limit (10KB)
4. No secrets in logs — scrub API keys, tokens, passwords before any log write
5. All file paths go through PathValidator before any file operation
6. All external string inputs go through StringValidator
7. Thread-safe where state is shared
8. Type hints on all function signatures
9. No bare except clauses — always catch specific exceptions
10. Follow existing patterns in the codebase — don't invent new patterns

## Key Abstractions — Understand These Before Writing Any Code

### Agent (agent/)
The core orchestrator. Runs the ReAct loop.
Depends on: LLM, ToolRegistry, MemoryStore, PromptService (all injected)
Does NOT directly handle: logging, safety checks, event emission
Those are handled by Observer, SafetyGate, WhyLog injected at construction.

```python
agent = Agent(
    llm=OllamaLLM(model="llama3"),
    memory=JSONFileMemory("./session.json"),
    observer=LiveObserver(),
    safety_gate=SafetyGate(allowed_dirs=["./data"]),
    audit_log=WhyLog(path="./logs/audit.jsonl")
)
```

### LiveObserver (observer/)
Real-time event stream. Emits events BEFORE execution, not after.
This is the key differentiator — pre-execution visibility.

Five events in order:
- agent.start     → {task, agent_id, timestamp}
- agent.reasoning → {thought, goal, step_number, timestamp}
- tool.before     → {tool_name, args, why, timestamp}  ← BEFORE execution
- tool.after      → {tool_name, result, success, duration_ms, timestamp}
- agent.end       → {final_answer, total_steps, duration_ms, timestamp}

```python
observer = LiveObserver()

@observer.on("tool.before")
def watch(event):
    print(f"ABOUT TO: {event['tool_name']} because: {event['why']}")
```

### SafetyGate (safety/)
Runs BEFORE every tool execution. Checks if action matches original goal.
Blocks and raises SafetyViolation if misaligned.
Supports HITL (human-in-the-loop) mode for high-risk actions.

Rules enforced:
- Destructive actions (delete, drop, remove, truncate) need explicit confirmation
- File writes outside allowed_dirs are always blocked
- Network calls to non-whitelisted domains are flagged
- Actions contradicting the original goal keywords are blocked

```python
gate = SafetyGate(
    allowed_dirs=["./data", "./output"],
    allowed_domains=["api.openai.com"],
    risk_threshold=0.7,
    hitl=True
)
```

### WhyLog (audit/)
Structured JSONL audit trail. Every entry answers: what, why, and was it safe.
Rotates at 10MB. Supports replay — prints human-readable narrative of a session.

Every entry contains:
- timestamp, session_id, event_type, goal (original task — on every entry)
- thought (agent reasoning at this moment)
- action, result, why
- was_blocked (bool), human_approved (bool or null)

```python
log = WhyLog(path="./logs/audit.jsonl")
log.replay(session_id)  # prints step-by-step narrative
```

### LLM Providers (llms/)
All implement BaseLLM with: complete(prompt) → str, stream(prompt) → Iterator,
health_check() → bool

Providers:
- OllamaLLM(model="llama3", timeout=30)           # local
- OpenAILLM(model="gpt-4o", api_key="...")         # cloud
- AnthropicLLM(model="claude-3-5-haiku-...", ...)  # cloud
- OpenRouterLLM(model="...", api_key="...")         # any model

### ToolRegistry (tools/)
Manages tool registration and lookup.
Tools are plain Python functions decorated with @agent.tool or registered via registry.

### MemoryStore (memory/)
Interface: get(key, default), set(key, value), clear()
Implementations: InMemoryStore, JSONFileMemory

### HookManager (hooks/)
Lifecycle hooks that fire around agent and tool execution.
Observer and SafetyGate plug in here — they are hooks, not core logic.

## The Five Production Examples (always keep runnable)

examples/local_agent.py
→ OllamaLLM + LiveObserver + WhyLog. Shows live decision stream in terminal.

examples/safe_file_agent.py  
→ SafetyGate blocking write outside sandbox. Shows HITL approval prompt.

examples/multi_agent_research.py
→ ResearchAgent + SummaryAgent sharing memory. Cross-agent audit trail.

examples/enterprise_audit_demo.py
→ Logistics/inventory workflow. Full WhyLog replay at end. Enterprise demo.

examples/framework_wrapper_demo.py
→ NKit Observer wrapping a non-NKit agent. Shows "plug into any stack" story.

## LLM Provider Usage Pattern
```python
# Local (Ollama must be running)
from nkit.llms import OllamaLLM
llm = OllamaLLM(model="llama3.2", timeout=30)

# Cloud
from nkit.llms import OpenAILLM, AnthropicLLM
llm = OpenAILLM(model="gpt-4o", api_key="sk-...")
llm = AnthropicLLM(model="claude-3-5-haiku-20251022", api_key="sk-ant-...")
```

## Error Handling Pattern — Always Use This
```python
try:
    result = tool.execute(args)
except ToolTimeoutError as e:
    # Log with context, continue agent loop
    self.audit_log.log_error(session_id, tool_name, str(e))
    return ToolResult(success=False, error=str(e), result=None)
except SafetyViolation as e:
    # Log, do not continue — this is intentional blocking
    self.audit_log.log_blocked(session_id, tool_name, str(e))
    raise
except Exception as e:
    # Unexpected — log full context, graceful degradation
    logger.error(f"Unexpected tool error: {tool_name}, step: {step}, goal: {goal}, error: {e}")
    return ToolResult(success=False, error=str(e), result=None)
```

## Security Rules — Non-Negotiable
- NEVER log API keys, tokens, or passwords
- NEVER execute shell commands unless explicitly sandboxed
- ALWAYS validate file paths through PathValidator
- ALWAYS validate string inputs through StringValidator  
- ALWAYS enforce max_steps (default 20) to prevent infinite loops
- ALWAYS enforce tool result size limit (default 10KB)
- Destructive filesystem operations require SafetyGate confirmation

## What Copilot Should NEVER Do in This Codebase
- Add new dependencies without strong justification
- Create god classes that handle multiple responsibilities
- Write bare try/except without specific exception types
- Skip docstrings on public interfaces
- Emit events after execution when they should be before
- Bypass PathValidator for any file operation
- Log raw LLM responses that might contain secrets
- Add complexity that breaks the "nano and minimalistic" philosophy

## Testing Patterns
```python
# Unit test pattern for NKit
def test_safety_gate_blocks_destructive_action():
    gate = SafetyGate(allowed_dirs=["./sandbox"])
    with pytest.raises(SafetyViolation):
        gate.check(
            tool_name="file_write",
            args={"path": "/etc/passwd", "content": "hack"},
            goal="summarize document"
        )

# Integration test pattern
def test_observer_receives_tool_before_event():
    events = []
    observer = LiveObserver()
    
    @observer.on("tool.before")
    def capture(event):
        events.append(event)
    
    agent = Agent(llm=MockLLM(), observer=observer)
    agent.run("test task")
    
    assert any(e["event_type"] == "tool.before" for e in events)
    assert events[0]["why"] is not None  # why is always populated
```

## CLI Commands Reference
```bash
nkit run "task description" --llm ollama:llama3 --tools web,file
nkit replay ./logs/audit.jsonl --session <session_id>
nkit health
nkit tools list
nkit chat --llm ollama:llama3
```

## Quick Start (What New Developers See First)
```python
from nkit import Agent
from nkit.llms import OllamaLLM
from nkit.observer import LiveObserver
from nkit.safety import SafetyGate
from nkit.audit import WhyLog

agent = Agent(
    llm=OllamaLLM(model="llama3.2"),
    observer=LiveObserver(),
    safety_gate=SafetyGate(allowed_dirs=["./data"]),
    audit_log=WhyLog(path="./logs/audit.jsonl")
)

result = agent.run("Analyze the sales data in ./data/q3.csv")
```

## Session and Versioning
- session_id: uuid4, generated per agent.run() call
- All audit entries carry session_id for cross-entry correlation
- Package version lives in pyproject.toml only — single source of truth
- Changelog maintained in CHANGELOG.md — update with every meaningful change

## Modules Not Yet Built (Build These Next)
- nkit/observer/__init__.py  ← highest priority
- nkit/safety/__init__.py    ← second priority  
- nkit/audit/__init__.py     ← third priority
- nkit/llms/ollama.py        ← needed for local model support
- nkit/llms/openai.py        ← needed for cloud support
- nkit/llms/anthropic.py     ← needed for cloud support
- nkit/llms/openrouter.py    ← needed for model flexibility
- tests/                     ← entire test suite needs building

## Author Context
Built by Balaji — associate software engineer and AI engineer specializing in
enterprise AI across automotive and logistics domains. Also the author of 
nbagents (published on PyPI). NKit is the production-hardened evolution of 
that work, focused on making agentic AI safe and auditable for enterprise use.