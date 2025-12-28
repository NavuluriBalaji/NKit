# NKit Agentic Framework - Complete Architecture Guide

## Overview

NKit is a production-ready agentic framework built on SOLID principles, providing a plugin architecture for LLM-driven agents with tool execution, memory, and RAG capabilities.

**Design Philosophy:**
- **Plugin & Play**: Swap any component (memory, retrieval, prompts) without touching core code
- **Security First**: Input validation, path sanitization, resource limits at every layer
- **SOLID Compliance**: Clean abstractions, dependency injection, extensible interfaces
- **Documentation**: Every function explains WHY and HOW to reuse it

---

## Architecture Components

### 1. Core Agent (`nkit/nbagents.py`)

**Purpose:**  
Orchestrates iterative LLM reasoning with tool execution (ReAct pattern).

**Key Classes:**
- `Agent`: Main orchestrator with dependency injection
- `Step`: Captures one reasoning iteration (thought → action → observation)

**Extensibility:**
```python
# Basic usage
agent = Agent(llm=my_llm_function)

# Advanced with DI
agent = Agent(
    llm=my_llm,
    registry=custom_tool_registry,
    memory=persistent_memory,
    prompt_service=rag_prompt_service,
    response_parser=custom_parser,
    max_steps=15
)
```

**Why Refactored This Way:**
- **SRP**: Agent delegates formatting (PromptService), parsing (ResponseParser), execution (ToolRegistry)
- **OCP**: Extend behavior via plugins, not code changes
- **DIP**: Depends on abstractions (interfaces), not concrete implementations
- Security: Validates LLM callable, enforces max steps/retries

---

### 2. Interfaces (`nkit/interfaces.py`)

**Purpose:**  
Abstract base classes and protocols defining plugin contracts.

**Core Interfaces:**
- `MemoryStore`: Storage backend protocol
- `PromptService`: Prompt construction strategy
- `ResponseParser`: LLM output parsing strategy
- `RetrievalSystem`: RAG/document retrieval interface
- `ToolValidator`: Input validation/sanitization

**Why Created:**
- **LSP**: Any implementation can substitute the interface
- **ISP**: Small, focused contracts (single-purpose interfaces)
- Enables mocking, testing, and third-party plugins

**Usage Example:**
```python
from nkit.interfaces import MemoryStore

class RedisMemory:  # Implements MemoryStore protocol
    def get(self, key, default=None): ...
    def set(self, key, value): ...
    # ... other methods
```

---

### 3. Memory Backends (`nkit/memory/__init__.py`)

**Implementations:**
- `Memory`: In-memory dict (fast, ephemeral)
- `JSONFileMemory`: File-based persistence

**Why Multiple Backends:**
- Development: Use `Memory` (no I/O)
- Production: Use `JSONFileMemory` or implement `RedisMemory`/`PostgresMemory`
- Security: Key validation prevents injection attacks

**Usage:**
```python
# Persistent session memory
memory = JSONFileMemory("./user_session.json")
agent = Agent(llm=my_llm, memory=memory)
```

---

### 4. Prompt Services (`nkit/prompt.py`)

**Implementations:**
- `PromptTemplate`: Simple string templates
- `ReActPromptService`: ReAct-style agent prompts
- `RAGPromptService` (in examples): Injects retrieved context

**Why Separated:**
- **SRP**: Prompt engineering is a distinct concern
- **OCP**: Swap prompt strategies without changing Agent
- A/B testing: Test different prompt formats easily

**Custom Prompt Service:**
```python
from nkit.prompt import ReActPromptService

class ChainOfThoughtPromptService(ReActPromptService):
    def build_agent_prompt(self, task, tools, history, memory=None):
        # Custom logic to inject step-by-step reasoning template
        ...
```

---

### 5. Tools & Registry (`nkit/tools/`)

**Components:**
- `Tool`: Wraps a function with metadata (name, desc, schema, sync/async)
- `ToolRegistry`: Manages tool collection, provides decorator API
- `BuiltinTools` (`builtin_tools.py`): Web search, file ops, time

**Why Decoupled:**
- Original `nbagents.py` had duplicate `Tool`/`ToolRegistry` definitions
- Now centralized in `tools/` package
- **OCP**: Add tools via registry, no core modifications

**Usage:**
```python
# Decorator API
@agent.tool("calculator", "Add two numbers")
def add(a: int, b: int) -> int:
    return a + b

# Programmatic API
agent.add_tool("greet", lambda name: f"Hello, {name}!")
```

---

### 6. RAG / Retrieval (`nkit/retrieval.py`)

**Implementations:**
- `InMemoryRetriever`: Keyword-based search (fast, simple)
- `JSONDocumentRetriever`: File-persisted retrieval index

**Why Built:**
- Core feature for RAG agents (inject relevant context)
- Implements `RetrievalSystem` interface
- Extensible to vector DBs (Pinecone, Chroma, FAISS)

**Usage:**
```python
retriever = InMemoryRetriever()
retriever.add_documents([
    {"content": "Paris is the capital of France", "metadata": {"source": "geo.txt"}},
])

docs = retriever.retrieve("capital of France", top_k=3)
```

**Plugin Integration:**
```python
from nkit.prompt import ReActPromptService

class RAGPromptService(ReActPromptService):
    def __init__(self, retriever, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever
    
    def build_agent_prompt(self, task, tools, history, memory=None):
        # Retrieve relevant docs
        docs = self.retriever.retrieve(task, top_k=3)
        context = "\n".join([d["content"] for d in docs])
        
        # Inject into prompt
        base = super().build_agent_prompt(task, tools, history, memory)
        return f"Context:\n{context}\n\n{base}"
```

---

### 7. Security (`nkit/security.py`)

**Validators:**
- `PathValidator`: Prevents path traversal (`../../etc/passwd`)
- `StringValidator`: Length/character whitelisting
- `ToolInputValidator`: Composite validation for tool params

**Why Critical:**
- File tools can expose entire filesystem without validation
- LLM-generated inputs are untrusted by default
- **Defense in Depth**: Validate at tool boundary

**Usage:**
```python
from nkit.security import PathValidator, ToolInputValidator

path_val = PathValidator(allowed_dirs=["./data", "/tmp"])
validator = ToolInputValidator(
    param_validators={"file_path": path_val},
    required=["file_path"]
)

# In tool execution
inputs = validator.validate("read_file", {"file_path": "/etc/passwd"})  # raises ValueError
```

---

### 8. Graph Orchestration (`nkit/chain/graph.py`)

**Components:**
- `State`: Execution context passed between nodes
- `Node`: Wraps a handler function (sync/async)
- `Edge`: Directed edge with optional condition
- `Graph`: DAG executor with conditional routing

**Why Built:**
- LangGraph-style orchestration for complex workflows
- Beyond linear chains: branching, fan-out, conditional logic
- Complements `Agent` (use Graph for multi-agent coordination)

**Usage:**
```python
from nkit.chain import Graph, Node, State

def plan(state: State):
    return {"plan": ["step1", "step2"]}

def execute(state: State):
    plan = state.get("plan")
    # Execute plan
    return "Done"

g = Graph()
g.add_node(Node("plan", plan))
g.add_node(Node("exec", execute))
g.add_edge("plan", "exec")
final = g.run(State())
```

---

### 9. Multi-Agent Orchestration (`nkit/agents/orchestrator.py`)

**Component:**
- `MultiAgentOrchestrator`: CrewAI-style sequential agent coordination

**Why Built:**
- Coordinate multiple agents with roles (planner, executor, reviewer)
- Passes context through shared memory
- Enables division of labor

**Usage:**
```python
from nkit.agents.orchestrator import MultiAgentOrchestrator, Role

planner = Agent(llm=planner_llm)
executor = Agent(llm=executor_llm)

orch = MultiAgentOrchestrator(
    [planner, executor],
    roles=[Role("planner", "Break down task"), Role("executor", "Execute steps")]
)
result = orch.run("Analyze Q3 sales data")
```

---

## SOLID Principles Applied

### Single Responsibility Principle (SRP)
**Before:**
- `Agent` handled prompt formatting, parsing, tool registration, logging
- Duplicate `Tool`/`ToolRegistry` in multiple files

**After:**
- `Agent`: Orchestration only
- `PromptService`: Prompt construction
- `ResponseParser`: Output parsing
- `ToolRegistry`: Tool management
- `Memory`: State storage
- Each class has ONE reason to change

### Open/Closed Principle (OCP)
**Implementation:**
- Add tools via `@agent.tool()` decorator or `registry.register()`
- Swap memory backend by passing different `MemoryStore` implementation
- Change prompt style by injecting custom `PromptService`
- **No core code modification needed**

### Liskov Substitution Principle (LSP)
**Guarantee:**
- Any `MemoryStore` implementation works with `Agent`
- Any `PromptService` implementation produces valid prompts
- Any `RetrievalSystem` implementation provides documents
- Swap implementations without breaking behavior

### Interface Segregation Principle (ISP)
**Design:**
- `MemoryStore`: 5 focused methods (get, set, append, clear, to_dict)
- `PromptService`: 1 method (build_agent_prompt)
- `ResponseParser`: 1 method (parse)
- **No fat interfaces**—components only implement what they need

### Dependency Inversion Principle (DIP)
**Architecture:**
- `Agent` depends on `MemoryStore` protocol, not concrete `Memory` class
- `Agent` depends on `PromptService` ABC, not `ReActPromptService`
- **High-level orchestration depends on abstractions**
- Low-level implementations depend on those same abstractions

---

## Security Features

### 1. Input Validation
- **Path Validator**: Whitelist directories, prevent traversal
- **String Validator**: Length limits, character whitelisting, pattern blacklisting
- **Tool Input Validator**: Per-parameter validation

### 2. Resource Limits
- `max_steps`: Prevent infinite loops
- `max_retries`: Limit retry attempts
- `max_memory_size`: Truncate memory snapshots in prompts
- `max_history`: Limit step history to prevent token exhaustion

### 3. Injection Prevention
- Memory key validation (alphanumeric + safe chars only)
- Prompt sanitization (escape special characters)
- JSON depth limits in parser (prevent DoS)

### 4. Least Privilege
- File tools require explicit directory whitelisting
- No shell=True in built-in tools
- Tool execution isolated (exceptions caught, logged)

---

## Plugin Development Guide

### Creating a Custom Memory Backend

```python
# Example: Redis-backed memory
import redis
from nkit.interfaces import MemoryStore

class RedisMemory:
    def __init__(self, redis_url: str):
        self.client = redis.from_url(redis_url)
    
    def get(self, key: str, default=None):
        val = self.client.get(key)
        return json.loads(val) if val else default
    
    def set(self, key: str, value):
        self.client.set(key, json.dumps(value))
    
    # ... implement append, clear, to_dict

# Usage
memory = RedisMemory("redis://localhost:6379")
agent = Agent(llm=my_llm, memory=memory)
```

### Creating a Custom Retrieval System

```python
# Example: Pinecone vector DB retriever
from nkit.interfaces import RetrievalSystem
import pinecone

class PineconeRetriever:
    def __init__(self, index_name: str, embeddings_fn):
        self.index = pinecone.Index(index_name)
        self.embeddings_fn = embeddings_fn
    
    def retrieve(self, query: str, top_k: int = 5, filters=None):
        vec = self.embeddings_fn(query)
        results = self.index.query(vec, top_k=top_k, filter=filters)
        return [{"content": r.metadata["text"], "score": r.score} for r in results.matches]
    
    def add_documents(self, documents):
        vectors = [(doc["id"], self.embeddings_fn(doc["content"]), doc["metadata"]) 
                   for doc in documents]
        self.index.upsert(vectors)

# Usage
retriever = PineconeRetriever("my-index", openai_embed)
prompt_service = RAGPromptService(retriever)
agent = Agent(llm=my_llm, prompt_service=prompt_service)
```

### Creating a Custom Prompt Service

```python
from nkit.prompt import ReActPromptService

class FewShotPromptService(ReActPromptService):
    def __init__(self, examples, **kwargs):
        super().__init__(**kwargs)
        self.examples = examples
    
    def build_agent_prompt(self, task, tools, history, memory=None):
        base = super().build_agent_prompt(task, tools, history, memory)
        
        # Inject few-shot examples
        examples_text = "Examples:\n" + "\n".join(self.examples)
        return f"{base}\n\n{examples_text}"

# Usage
examples = [
    "Q: What is 2+2? A: {\"thought\": \"Simple math\", \"final_answer\": \"4\"}",
]
service = FewShotPromptService(examples)
agent = Agent(llm=my_llm, prompt_service=service)
```

---

## Testing & Examples

### Run Demos

```bash
# Basic graph demo
python nkit/examples/demo_graph.py

# RAG-enabled agent with persistence
python nkit/examples/demo_rag_agent.py
```

### Example Output (RAG Demo)
```
[Query] What is the capital of France?
**Retrieved Context:**
1. Paris is the capital and largest city of France...

[Answer] The capital of France is Paris, located on the Seine River.

Memory persisted to: ./demo_session.json
```

---

## Future Enhancements

### Planned Features
1. **Parallel Graph Execution**: Fan-out to multiple nodes, join with reducers
2. **Checkpointing**: Save/resume Graph state for long-running workflows
3. **Event Bus**: Pub/sub for agent communication and observability
4. **Rate Limiting**: Token-bucket for tool calls and LLM requests
5. **Vector DB Integrations**: Pinecone, Chroma, Weaviate plugins
6. **LLM Adapters**: Pre-built adapters for OpenAI, Anthropic, Ollama

### Extensibility Roadmap
- **Plugin Registry**: Discover and load plugins from `nkit_plugins/` directory
- **Configuration Files**: YAML/JSON for agent setup (no code)
- **Monitoring**: Built-in telemetry (Prometheus, OpenTelemetry)

---

## Why This Architecture?

### Problem Solved
Your original request: *"Let's say we have a RAG system—if I want to build one for users, I should plug and play. Similarly, if I want to include a memo layer, I should be able to. Clean, refactored code with security."*

### Solution Delivered
1. **RAG Plugin**: `InMemoryRetriever`, `JSONDocumentRetriever`, `RetrievalSystem` interface
   - Plug into any agent via custom `PromptService`
   - Swap retrieval backends without agent changes

2. **Memory Layers**: `Memory`, `JSONFileMemory`, extensible to Redis/Postgres
   - Inject via `memory` parameter
   - Validated keys prevent injection

3. **Clean Architecture**: SOLID principles throughout
   - **Single Responsibility**: Each module has one job
   - **Dependency Injection**: All components swappable
   - **Interface-based**: Protocols enable type safety and extensibility

4. **Security**: Input validators, path sanitization, resource limits
   - `security.py` provides reusable validators
   - Tool execution isolated
   - Comprehensive error handling

5. **Documentation**: Every function/class explains:
   - **Purpose**: What it does
   - **Reuse Patterns**: How to use it in different scenarios
   - **Security**: What to watch out for
   - **Examples**: Working code snippets

### Result
You can now:
- Build a RAG Q&A agent by injecting `PineconeRetriever` and `RAGPromptService`
- Add persistent memory by passing `JSONFileMemory` or `RedisMemory`
- Secure file tools with `PathValidator`
- Coordinate multi-agent workflows with `MultiAgentOrchestrator` or `Graph`
- Extend any component without touching core code

**No monolithic rewrites. Pure plugin architecture. Production-ready.**
