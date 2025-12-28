"""Migration Guide: Legacy to Modular Architecture

This guide helps you transition from the original nbagents.py monolithic design
to the new modular architecture inspired by CrewAI.

## What Changed?

### Before: Monolithic Design
- Single `nbagents.py` with all core logic
- Limited extensibility
- Hard-coded dependencies
- Basic multi-agent support

### After: Modular Architecture
- 12+ specialized modules
- Plugin-based extensibility
- Dependency injection
- Professional multi-agent orchestration
- Built-in observability

## Backward Compatibility

**Good news:** Your existing code continues to work! The legacy API is fully supported.

```python
# This still works
from nkit import Agent, Tool

agent = Agent(llm_client=my_llm)
result = agent.run("task")
```

## Migration Patterns

### 1. Basic Agent → Modern Agent

**Before:**
```python
from nkit import Agent, Tool

def my_tool(x: str) -> str:
    return f"Result: {x}"

agent = Agent(
    llm_client=my_llm,
    tools=[Tool(name="tool", func=my_tool)]
)

result = agent.run("Do something")
```

**After (Enhanced):**
```python
from nkit.nbagents import Agent  # or from nkit.agent when refactored
from nkit.tools import Tool
from nkit.events import get_global_event_bus, EventType
from nkit.telemetry import get_metrics_collector

# Set up observability
bus = get_global_event_bus()
metrics = get_metrics_collector()

@bus.subscribe(EventType.AGENT_STARTED)
async def log_start(event):
    print(f"Agent started: {event.source}")

# Create agent (same as before)
agent = Agent(
    llm_client=my_llm,
    tools=[Tool(name="tool", func=my_tool)]
)

# Run with automatic telemetry
result = agent.run("Do something")

# Check metrics
print(f"Agent calls: {metrics.get_counter('agent.runs')}")
```

### 2. Sequential Tasks → Task Manager

**Before:**
```python
# Manual chaining
result1 = agent1.run("First task")
result2 = agent2.run(f"Second task using: {result1}")
result3 = agent3.run(f"Third task using: {result2}")
```

**After:**
```python
from nkit.tasks import Task, TaskManager

# Define tasks with dependencies
task1 = Task(
    description="First task",
    expected_output="Initial data",
    agent=agent1
)

task2 = Task(
    description="Second task",
    expected_output="Processed data",
    agent=agent2,
    dependencies=[task1]  # Automatic context passing
)

task3 = Task(
    description="Third task",
    expected_output="Final result",
    agent=agent3,
    dependencies=[task2]
)

# Execute with dependency resolution
manager = TaskManager()
results = await manager.execute_tasks_async([task1, task2, task3])

# Or synchronously
results = manager.execute_tasks([task1, task2, task3])
```

### 3. Multi-Agent → Crew

**Before:**
```python
from nkit.agents.orchestrator import MultiAgentOrchestrator, Role

agents = [agent1, agent2, agent3]
roles = [
    Role("Researcher", "Find information"),
    Role("Analyst", "Analyze data"),
    Role("Writer", "Write report")
]

orchestrator = MultiAgentOrchestrator(agents, roles)
result = orchestrator.run("Create a research report")
```

**After:**
```python
from nkit.crews import Crew, Agent, Task, ProcessType

# Define specialized agents
researcher = Agent(
    role="Researcher",
    goal="Find comprehensive information",
    backstory="Expert researcher with 10 years experience",
    tools=[web_search_tool],
    llm=my_llm
)

analyst = Agent(
    role="Analyst",
    goal="Analyze data and identify insights",
    backstory="Data scientist specializing in trend analysis",
    tools=[data_analysis_tool],
    llm=my_llm
)

writer = Agent(
    role="Writer",
    goal="Create engaging reports",
    backstory="Technical writer with communication expertise",
    tools=[formatting_tool],
    llm=my_llm
)

# Define workflow
research_task = Task(
    description="Research AI agent frameworks",
    expected_output="List of 10 frameworks with descriptions",
    agent=researcher
)

analysis_task = Task(
    description="Analyze frameworks and identify trends",
    expected_output="Analysis report with 3 key insights",
    agent=analyst,
    dependencies=[research_task]
)

writing_task = Task(
    description="Write comprehensive report",
    expected_output="2000-word technical report",
    agent=writer,
    dependencies=[analysis_task]
)

# Execute crew
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=ProcessType.SEQUENTIAL,
    verbose=True
)

result = crew.kickoff()
print(result.raw)
```

### 4. Custom Memory → Pluggable Memory

**Before:**
```python
from nkit.memory import Memory

# Basic in-memory only
memory = Memory()
agent = Agent(llm_client=my_llm, memory=memory)
```

**After:**
```python
from nkit.memory import JSONFileMemory
from nkit.interfaces import MemoryStore

# Use persistent memory
memory = JSONFileMemory(filepath="agent_memory.json")

# Or create custom backend
class RedisMemory(MemoryStore):
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def get(self, key: str) -> Any:
        return self.redis.get(key)
    
    def set(self, key: str, value: Any) -> None:
        self.redis.set(key, value)

redis_memory = RedisMemory(my_redis_client)
agent = Agent(llm_client=my_llm, memory=redis_memory)
```

### 5. RAG Integration → Knowledge Base

**Before:**
```python
from nkit.retrieval import InMemoryRetriever

# Basic retrieval
retriever = InMemoryRetriever()
retriever.add_document("doc1", "Content about Python")
results = retriever.retrieve("Python programming")
```

**After:**
```python
from nkit.knowledge import KnowledgeBase, Document, SentenceChunker
from nkit.knowledge import SimpleEmbeddingProvider

# Advanced knowledge base
kb = KnowledgeBase(
    chunker=SentenceChunker(max_chunk_size=500),
    embedding_provider=SimpleEmbeddingProvider()  # or OpenAIEmbeddings
)

# Add documents with metadata
docs = [
    Document(
        content="Python is a programming language...",
        metadata={"source": "python_docs", "version": "3.10"}
    ),
    Document(
        content="JavaScript is used for web development...",
        metadata={"source": "js_docs", "version": "ES2021"}
    )
]

for doc in docs:
    kb.add_document(doc)

# Semantic search with scores
results = kb.search("programming languages", top_k=5)
for chunk, score in results:
    print(f"Score: {score:.2f}")
    print(f"Source: {chunk.metadata['source']}")
    print(f"Content: {chunk.content}\n")

# Save/load
kb.save("./knowledge_base")
kb.load("./knowledge_base")
```

### 6. LLM Clients → Unified LLM Adapters

**Before:**
```python
# Direct API calls, no abstraction
import openai

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

**After:**
```python
from nkit.llms import OpenAILLM, OllamaLLM, AnthropicLLM, LLMConfig

# Unified interface
config = LLMConfig(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

llm = OpenAILLM(config, api_key=my_key)

# Same interface for all providers
response = llm.call(prompt)

# Or async
response = await llm.acall(prompt)

# Or streaming
async for chunk in llm.stream(prompt):
    print(chunk, end="")

# Easy provider switching
# llm = OllamaLLM(LLMConfig(model="llama2"))
# llm = AnthropicLLM(LLMConfig(model="claude-3"))
```

### 7. Monitoring → Comprehensive Telemetry

**Before:**
```python
# Manual logging
import logging

logger = logging.getLogger(__name__)
logger.info("Agent started")

start_time = time.time()
result = agent.run(task)
duration = time.time() - start_time

logger.info(f"Agent completed in {duration:.2f}s")
```

**After:**
```python
from nkit.telemetry import get_metrics_collector, get_tracer, get_cost_tracker

metrics = get_metrics_collector()
tracer = get_tracer()
cost_tracker = get_cost_tracker()

# Automatic tracing
with tracer.span("agent_execution") as span:
    result = agent.run(task)
    span.metadata["result_length"] = len(result)

# Metrics
metrics.increment("agent.tasks_completed")
metrics.histogram("agent.latency", value=duration)

# Cost tracking
cost_tracker.record_usage(
    model="gpt-4",
    prompt_tokens=100,
    completion_tokens=50
)

# Get analytics
stats = metrics.get_stats("agent.latency")
print(f"Average: {stats['avg']:.2f}s")
print(f"P95: {stats['p95']:.2f}s")

print(f"Total cost: ${cost_tracker.get_total_cost():.4f}")
print(f"Total tokens: {cost_tracker.get_total_tokens()}")
```

### 8. Event Handling → Event Bus

**Before:**
```python
# Custom callbacks
def on_tool_call(tool_name, args):
    print(f"Calling {tool_name} with {args}")

# No standard way to hook into events
```

**After:**
```python
from nkit.events import get_global_event_bus, EventType

bus = get_global_event_bus()

# Subscribe to any event
@bus.subscribe(EventType.TOOL_CALLED)
async def log_tool_call(event):
    print(f"Tool: {event.data['tool_name']}")
    print(f"Args: {event.data['arguments']}")

@bus.subscribe(EventType.LLM_CALLED)
async def track_llm_usage(event):
    model = event.data.get("model")
    tokens = event.data.get("tokens", 0)
    # Send to monitoring system

@bus.subscribe()  # Subscribe to all events
async def debug_all(event):
    print(f"Event: {event.type} from {event.source}")
```

### 9. Validation → Lifecycle Hooks

**Before:**
```python
# Manual validation before tool calls
def safe_tool_call(tool, args):
    if not validate_args(args):
        raise ValueError("Invalid arguments")
    return tool(args)
```

**After:**
```python
from nkit.hooks import get_global_hook_manager, HookType, HookResult

manager = get_global_hook_manager()

# Validation hook
@manager.register(HookType.BEFORE_TOOL_CALL)
def validate_tool_input(ctx):
    args = ctx.data.get("arguments", {})
    
    # Check for dangerous patterns
    if "rm -rf" in str(args):
        return HookResult(
            should_continue=False,
            error=SecurityError("Dangerous command detected")
        )
    
    # Sanitize inputs
    sanitized = {k: sanitize(v) for k, v in args.items()}
    return HookResult(
        should_continue=True,
        modified_data={"arguments": sanitized}
    )

# Caching hook
cache = {}

@manager.register(HookType.AFTER_LLM_CALL)
async def cache_llm_response(ctx):
    prompt = ctx.data.get("prompt")
    response = ctx.data.get("response")
    cache[prompt] = response

@manager.register(HookType.BEFORE_LLM_CALL)
async def check_cache(ctx):
    prompt = ctx.data.get("prompt")
    if prompt in cache:
        return HookResult(
            should_continue=False,
            modified_data={"response": cache[prompt]}
        )
```

## CLI Usage

**New:** Project scaffolding

```bash
# Create project structure
python -m nkit.cli create project my_agent_system

# Navigate to project
cd my_agent_system

# Create agent template
python -m nkit.cli create agent data_analyst

# Create crew template
python -m nkit.cli create crew research_crew

# Project structure
# my_agent_system/
# ├── agents/
# │   └── data_analyst.py
# ├── crews/
# │   └── research_crew.py
# ├── tools/
# ├── data/
# ├── config.json
# └── README.md
```

## Best Practices

### 1. Use Type Hints
```python
from typing import List, Dict, Any

def my_tool(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Tool with proper type hints."""
    return [{"result": f"Item {i}"} for i in range(limit)]
```

### 2. Leverage Async
```python
# Prefer async for I/O-bound operations
async def research_task():
    results = await asyncio.gather(
        web_search("topic A"),
        web_search("topic B"),
        web_search("topic C")
    )
    return aggregate(results)
```

### 3. Monitor Everything
```python
from nkit.telemetry import get_metrics_collector, get_tracer

metrics = get_metrics_collector()
tracer = get_tracer()

@tracer.span("critical_operation")
def important_function():
    metrics.increment("function.calls")
    # ... work ...
    metrics.histogram("function.latency", value=duration)
```

### 4. Use Events for Coordination
```python
from nkit.events import get_global_event_bus, Event, EventType

bus = get_global_event_bus()

# Agent A emits event
await bus.publish(Event(
    type=EventType.CUSTOM,
    data={"message": "Analysis complete", "results": data},
    source="agent_a"
))

# Agent B receives event
@bus.subscribe(EventType.CUSTOM)
async def on_analysis_complete(event):
    if event.source == "agent_a":
        # Agent B starts next phase
        await agent_b.start_work(event.data["results"])
```

### 5. Document Everything
```python
def my_tool(query: str) -> str:
    \"\"\"Search the knowledge base.
    
    Purpose:
        Retrieves relevant documents matching the query.
    
    Args:
        query: Search query string
    
    Returns:
        Formatted search results
    
    Raises:
        ValueError: If query is empty
    
    Example:
        >>> results = my_tool("Python frameworks")
        >>> print(results)
    \"\"\"
    pass
```

## Breaking Changes

None! The new architecture is fully backward compatible. You can:

1. Keep using the old API
2. Gradually adopt new modules
3. Mix old and new code
4. Migrate at your own pace

## Next Steps

1. Read `MODULAR_ARCHITECTURE.md` for detailed design
2. Check `examples/` for working demos
3. Explore each module's docstrings
4. Start with `crews` and `tasks` for immediate value
5. Add telemetry for production monitoring
6. Implement custom plugins as needed

## Getting Help

- Check module docstrings
- See `QUICKREF.md` for quick reference
- Review examples in `examples/`
- Read inline documentation

## Summary

The modular architecture provides:
- ✅ Professional multi-agent orchestration
- ✅ Flexible task management
- ✅ Unified LLM interface
- ✅ Knowledge base with RAG
- ✅ Event-driven coordination
- ✅ Comprehensive observability
- ✅ Plugin architecture
- ✅ CLI tooling
- ✅ Full backward compatibility

Start using the new features today while keeping your existing code running!
"""

if __name__ == "__main__":
    print(__doc__)
