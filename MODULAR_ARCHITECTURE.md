"""nkit Framework - Modular AI Agent Architecture

# Overview

nkit is a production-grade Python framework for building AI agents and multi-agent systems, inspired by CrewAI and LangChain's architectural patterns.

## Architecture

The framework follows a modular design with clear separation of concerns:

```
nkit/
├── agent/          - Core agent implementation
├── tasks/          - Task management and orchestration
├── crews/          - Multi-agent coordination
├── llms/           - LLM provider adapters
├── tools/          - Tool registry and execution
├── memory/         - Memory backends
├── knowledge/      - Knowledge base with embeddings
├── events/         - Event system for observability
├── hooks/          - Lifecycle hooks
├── telemetry/      - Metrics, tracing, cost tracking
├── security/       - Input validation and safety
├── retrieval/      - RAG document retrieval
├── chain/          - Graph-based workflows
└── cli/            - Command-line interface
```

## Key Features

### 1. Task Management (tasks/)
- Dependency resolution with DAG execution
- Parallel and sequential execution
- Retry logic and error handling
- Output validation

### 2. LLM Adapters (llms/)
- Unified interface for multiple providers
- OpenAI, Ollama, Anthropic support
- Async/sync execution
- Streaming support
- Token counting and cost tracking

### 3. Multi-Agent Crews (crews/)
- Sequential execution (pipeline)
- Hierarchical delegation (manager + workers)
- Parallel execution (concurrent tasks)
- Context sharing between agents

### 4. Knowledge Base (knowledge/)
- Document chunking strategies
- Embedding generation
- Vector similarity search
- Integration with RAG

### 5. Event System (events/)
- Pub/sub event bus
- Lifecycle event tracking
- Agent coordination
- Debugging and monitoring

### 6. Lifecycle Hooks (hooks/)
- Pre/post execution hooks
- Error handling hooks
- Input/output transformation
- Custom validation

### 7. Telemetry (telemetry/)
- Metrics collection (counters, gauges, histograms)
- Distributed tracing with spans
- Token usage and cost tracking
- Performance monitoring

### 8. CLI (cli/)
- Project scaffolding
- Agent/crew templates
- Configuration management

## Quick Start

### Installation

```bash
pip install nkit  # (when published)
# or for development:
pip install -e .
```

### Create a Simple Agent

```python
from nkit.nbagents import Agent
from nkit.tools import Tool

# Define a tool
def search_web(query: str) -> str:
    return f"Search results for: {query}"

# Create agent
agent = Agent(
    llm_client=None,  # Configure your LLM
    tools=[Tool(name="search", func=search_web)]
)

# Run task
result = agent.run("Find information about AI agents")
print(result)
```

### Create a Crew

```python
from nkit.crews import Crew, Agent, Task, ProcessType

# Define agents
researcher = Agent(
    role="Researcher",
    goal="Find relevant information",
    backstory="Expert in web research and data gathering"
)

writer = Agent(
    role="Writer",
    goal="Create engaging content",
    backstory="Professional content writer"
)

# Define tasks
research_task = Task(
    description="Research latest AI trends",
    expected_output="List of 5 trends with descriptions",
    agent=researcher
)

writing_task = Task(
    description="Write a blog post about the trends",
    expected_output="800-word blog post",
    agent=writer,
    dependencies=[research_task]
)

# Create and run crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=ProcessType.SEQUENTIAL,
    verbose=True
)

result = crew.kickoff()
print(result.raw)
```

### Task Management

```python
from nkit.tasks import Task, TaskManager

# Create tasks with dependencies
task1 = Task(
    description="Load data",
    agent=data_loader,
    expected_output="DataFrame"
)

task2 = Task(
    description="Analyze data",
    agent=analyst,
    expected_output="Analysis report",
    dependencies=[task1]  # Runs after task1
)

task3 = Task(
    description="Visualize results",
    agent=visualizer,
    expected_output="Charts",
    dependencies=[task2]
)

# Execute with dependency resolution
manager = TaskManager()
results = await manager.execute_tasks_async([task1, task2, task3])
```

### Knowledge Base with RAG

```python
from nkit.knowledge import KnowledgeBase, Document, SentenceChunker
from nkit.knowledge import SimpleEmbeddingProvider

# Create knowledge base
kb = KnowledgeBase(
    chunker=SentenceChunker(max_chunk_size=500),
    embedding_provider=SimpleEmbeddingProvider()
)

# Add documents
doc = Document(
    content="Python is a high-level programming language...",
    metadata={"source": "python_docs", "category": "programming"}
)
kb.add_document(doc)

# Search
results = kb.search("What is Python?", top_k=3)
for chunk, score in results:
    print(f"Score: {score:.2f} - {chunk.content[:100]}...")
```

### Event System

```python
from nkit.events import get_global_event_bus, EventType

bus = get_global_event_bus()

# Subscribe to events
@bus.subscribe(EventType.AGENT_STARTED)
async def log_agent_start(event):
    print(f"Agent {event.source} started at {event.timestamp}")

@bus.subscribe(EventType.TOOL_CALLED)
async def track_tool_usage(event):
    tool_name = event.data.get("tool_name")
    print(f"Tool used: {tool_name}")

# Events are automatically published by framework components
```

### Lifecycle Hooks

```python
from nkit.hooks import get_global_hook_manager, HookType, HookResult

manager = get_global_hook_manager()

# Validate tool inputs
@manager.register(HookType.BEFORE_TOOL_CALL)
def validate_tool_input(ctx):
    if not ctx.data.get("arguments"):
        return HookResult(
            should_continue=False,
            error=ValueError("Missing arguments")
        )
    return HookResult(should_continue=True)

# Log all tool executions
@manager.register(HookType.AFTER_TOOL_CALL)
async def log_tool_execution(ctx):
    print(f"Tool {ctx.data['tool_name']} completed in {ctx.data['duration']:.2f}s")
```

### Telemetry and Monitoring

```python
from nkit.telemetry import get_metrics_collector, get_tracer, get_cost_tracker

# Metrics
metrics = get_metrics_collector()
metrics.increment("agent.tasks_completed")
metrics.histogram("agent.latency", value=1.25)

# Tracing
tracer = get_tracer()
with tracer.span("agent_execution") as span:
    result = agent.run(task)
    span.metadata["result_length"] = len(result)

# Cost tracking
cost_tracker = get_cost_tracker()
cost_tracker.record_usage(
    model="gpt-4",
    prompt_tokens=150,
    completion_tokens=75
)
print(f"Total cost: ${cost_tracker.get_total_cost():.4f}")
```

### CLI Usage

```bash
# Create new project
python -m nkit.cli create project my_agent_project

# Create agent template
cd my_agent_project
python -m nkit.cli create agent data_analyst

# Create crew template
python -m nkit.cli create crew research_crew
```

## Design Principles

### 1. Modularity
Each module has a single, well-defined responsibility and can be used independently.

### 2. Extensibility
All core interfaces use abstract base classes, allowing easy customization:
- Custom LLM providers
- Custom memory backends
- Custom embedding providers
- Custom chunking strategies

### 3. Type Safety
Comprehensive type hints throughout for better IDE support and error detection.

### 4. Async-First
Full async/sync support for high-performance applications.

### 5. Observability
Built-in events, hooks, and telemetry for production monitoring.

## Plugin Architecture

### Custom LLM Provider

```python
from nkit.llms import BaseLLM

class CustomLLM(BaseLLM):
    def call(self, prompt: str) -> str:
        # Your implementation
        pass
    
    async def acall(self, prompt: str) -> str:
        # Async implementation
        pass
```

### Custom Memory Backend

```python
from nkit.interfaces import MemoryStore

class RedisMemory(MemoryStore):
    def get(self, key: str) -> Any:
        # Redis implementation
        pass
    
    def set(self, key: str, value: Any) -> None:
        # Redis implementation
        pass
```

### Custom Embedding Provider

```python
from nkit.knowledge import EmbeddingProvider

class HuggingFaceEmbeddings(EmbeddingProvider):
    def embed(self, texts: List[str]) -> List[List[float]]:
        # Use sentence-transformers
        pass
```

## Migration from Legacy Code

The framework maintains backward compatibility with the original `nbagents.py` API:

```python
# Old code still works
from nkit.nbagents import Agent

agent = Agent(llm_client=my_llm)
result = agent.run("task")
```

New code can gradually adopt modular components:

```python
# Modern approach
from nkit.agent import Agent  # (when refactored)
from nkit.crews import Crew
from nkit.tasks import Task

# Use new features
crew = Crew(agents=[...], tasks=[...])
```

## Testing

```python
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# With coverage
pytest --cov=nkit tests/
```

## Documentation

- `ARCHITECTURE.md` - Detailed architecture documentation
- `QUICKREF.md` - Quick reference guide
- `SUMMARY.md` - Implementation summary
- Module docstrings - Inline documentation with examples

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

[Your License Here]

## Roadmap

- [ ] Move core agent logic to `agent/` module
- [ ] Add vector database integrations (Pinecone, Weaviate, Chroma)
- [ ] OpenAI function calling support
- [ ] Streaming responses
- [ ] Web UI for agent monitoring
- [ ] More LLM provider adapters (Cohere, AI21, etc.)
- [ ] Advanced planning strategies (MCTS, beam search)
- [ ] Multi-modal support (images, audio)

## Credits

Inspired by:
- CrewAI - Multi-agent orchestration patterns
- LangChain - Tool and chain abstractions
- OpenAI - Agent and tool calling APIs
"""

__version__ = "0.2.0"
__author__ = "nkit contributors"

__all__ = [
    # Core
    "Agent",
    "Tool",
    "ToolRegistry",
    
    # Tasks
    "Task",
    "TaskManager",
    
    # Crews
    "Crew",
    "ProcessType",
    
    # LLMs
    "BaseLLM",
    "OpenAILLM",
    "OllamaLLM",
    "AnthropicLLM",
    
    # Memory
    "Memory",
    "JSONFileMemory",
    
    # Knowledge
    "KnowledgeBase",
    "Document",
    "Chunker",
    
    # Events
    "EventBus",
    "EventType",
    "get_global_event_bus",
    
    # Hooks
    "HookManager",
    "HookType",
    "get_global_hook_manager",
    
    # Telemetry
    "MetricsCollector",
    "Tracer",
    "CostTracker",
    "get_metrics_collector",
    "get_tracer",
    "get_cost_tracker",
]
