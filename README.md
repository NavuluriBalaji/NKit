# NKit Agentic Framework

**Production-ready agent framework with plugin architecture, SOLID principles, and comprehensive security.**

## Features

- ✅ **Plugin & Play**: Swap memory, retrieval, prompts without code changes
- ✅ **RAG Support**: Built-in retrieval systems for context-augmented agents
- ✅ **Security First**: Input validation, path sanitization, resource limits
- ✅ **SOLID Design**: Dependency injection, interface-based extensibility
- ✅ **Graph Orchestration**: LangGraph-style DAG execution with conditional routing
- ✅ **Multi-Agent**: CrewAI-style orchestration for agent teams
- ✅ **Async/Sync**: Seamless handling of sync and async functions
- ✅ **Comprehensive Docs**: Every function explains WHY and HOW to reuse

## Quick Start

### Basic Agent

```python
from nkit import Agent

def my_llm(prompt: str) -> str:
    # Your LLM API call here
    return llm_response

agent = Agent(llm=my_llm)
result = agent.run("What is the capital of France?")
print(result)
```

### RAG-Enabled Agent

```python
from nkit import Agent
from nkit.retrieval import InMemoryRetriever
from nkit.memory import JSONFileMemory

# Setup knowledge base
retriever = InMemoryRetriever()
retriever.add_documents([
    {"content": "Paris is the capital of France", "metadata": {"source": "geo.txt"}},
])

# Custom prompt service with RAG
from nkit.prompt import ReActPromptService

class RAGPromptService(ReActPromptService):
    def __init__(self, retriever, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
    
    def build_agent_prompt(self, task, tools, history, memory=None):
        docs = self.retriever.retrieve(task, top_k=3)
        context = "\n".join([d["content"] for d in docs])
        base = super().build_agent_prompt(task, tools, history, memory)
        return f"Context:\n{context}\n\n{base}"

# Create agent with plugins
agent = Agent(
    llm=my_llm,
    memory=JSONFileMemory("./session.json"),
    prompt_service=RAGPromptService(retriever)
)

result = agent.run("What is the capital of France?")
```

### Graph Orchestration

```python
from nkit.chain import Graph, Node, State

def plan(state: State):
    return {"plan": ["analyze", "summarize"]}

def execute(state: State):
    plan = state.get("plan")
    return f"Executed: {plan}"

g = Graph()
g.add_node(Node("plan", plan)).add_node(Node("exec", execute))
g.add_edge("plan", "exec")

final = g.run(State())
print(final.last_result)
```

### Custom Tools with Security

```python
from nkit import Agent
from nkit.security import PathValidator, ToolInputValidator

# Path validation
path_val = PathValidator(allowed_dirs=["./data"])

@agent.tool("safe_read", "Read file safely")
def read_file(file_path: str) -> str:
    validated = path_val.validate_path(file_path)
    with open(validated, 'r') as f:
        return f.read()
```

## Architecture Overview

```
nkit/
├── nbagents.py           # Core Agent with ReAct reasoning
├── interfaces.py         # Abstract interfaces (MemoryStore, PromptService, etc.)
├── memory/               # Memory backends (Memory, JSONFileMemory)
├── prompt.py             # Prompt services (ReActPromptService, PromptTemplate)
├── tools/                # Tool system (Tool, ToolRegistry, BuiltinTools)
├── retrieval.py          # RAG retrievers (InMemoryRetriever, JSONDocumentRetriever)
├── security.py           # Input validators (PathValidator, StringValidator)
├── chain/                # Graph orchestration (Graph, Node, Edge, State)
├── agents/               # Multi-agent coordination (MultiAgentOrchestrator)
└── examples/             # Runnable demos
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for comprehensive design documentation.

## SOLID Principles

### Single Responsibility
- `Agent`: Orchestration only
- `PromptService`: Prompt construction
- `ResponseParser`: Output parsing
- `ToolRegistry`: Tool management

### Open/Closed
- Extend via plugins (tools, memory, prompts)
- No core modification needed

### Liskov Substitution
- Swap any `MemoryStore` implementation
- Swap any `PromptService` implementation

### Interface Segregation
- Small, focused interfaces
- Implement only what you need

### Dependency Inversion
- `Agent` depends on abstractions (protocols/ABCs)
- Inject dependencies via constructor

## Security Features

- **Input Validation**: Path traversal prevention, character whitelisting
- **Resource Limits**: Max steps, retries, memory size, history length
- **Injection Prevention**: Key validation, prompt sanitization, JSON depth limits
- **Least Privilege**: Directory whitelisting, no shell execution by default

Example:
```python
from nkit.security import PathValidator

validator = PathValidator(allowed_dirs=["/data"])
safe_path = validator.validate_path("/data/file.txt")  # OK
validator.validate_path("/etc/passwd")  # raises ValueError
```

## Plugin Development

### Custom Memory Backend

```python
from nkit.interfaces import MemoryStore

class RedisMemory:
    def get(self, key, default=None): ...
    def set(self, key, value): ...
    # Implement protocol methods

agent = Agent(llm=my_llm, memory=RedisMemory("redis://localhost"))
```

### Custom Retrieval System

```python
from nkit.interfaces import RetrievalSystem

class VectorDBRetriever:
    def retrieve(self, query, top_k=5, filters=None): ...
    def add_documents(self, documents): ...

retriever = VectorDBRetriever(index="my-index")
prompt_service = RAGPromptService(retriever)
agent = Agent(llm=my_llm, prompt_service=prompt_service)
```

## Examples

```bash
# Basic graph demo
python nkit/examples/demo_graph.py

# RAG agent with persistent memory
python nkit/examples/demo_rag_agent.py
```

## Requirements

- Python 3.10+
- Optional: `requests`, `aiohttp` for built-in web search tools

```bash
pip install requests aiohttp
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): Complete design documentation
- [interfaces.py](interfaces.py): Interface definitions with usage examples
- [security.py](security.py): Security validators and best practices

Every module includes comprehensive docstrings explaining:
- **Purpose**: What the component does
- **Reuse Patterns**: How to use in different scenarios
- **Security**: What to watch out for
- **Examples**: Working code snippets

## Why This Design?

**Problem:**  
"I want to plug and play a RAG system, or add a memory layer, with clean, secure, refactored code."

**Solution:**  
1. **Plugin Architecture**: Inject any component via constructor
2. **Interface-Based**: Swap implementations without code changes
3. **Security Built-In**: Validators, sanitizers, resource limits
4. **SOLID Compliance**: Each component has one job, easily extensible
5. **Comprehensive Docs**: Every function explains WHY and HOW

**Result:**  
You can build a RAG Q&A agent with persistent memory and custom tools in ~20 lines, then swap to a vector DB and Redis backend by changing 2 parameters.

## License

MIT

## Contributing

Issues and PRs welcome! See [ARCHITECTURE.md](ARCHITECTURE.md) for design guidelines.
