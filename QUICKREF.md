# NKit Quick Reference

## Common Patterns

### 1. Basic Agent
```python
from nkit import Agent

agent = Agent(llm=my_llm_function)
result = agent.run("Your task here")
```

### 2. Agent with Custom Tools
```python
@agent.tool("calculator", "Add two numbers")
def add(a: int, b: int) -> int:
    return a + b

result = agent.run("What is 5 + 3?")
```

### 3. Agent with Persistent Memory
```python
from nkit.memory import JSONFileMemory

memory = JSONFileMemory("./session.json")
agent = Agent(llm=my_llm, memory=memory)
```

### 4. RAG-Enabled Agent
```python
from nkit.retrieval import InMemoryRetriever
from nkit.prompt import ReActPromptService

# Setup retriever
retriever = InMemoryRetriever()
retriever.add_documents([
    {"content": "...", "metadata": {"source": "..."}}
])

# Custom prompt service
class RAGPromptService(ReActPromptService):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
    
    def build_agent_prompt(self, task, tools, history, memory):
        docs = self.retriever.retrieve(task, top_k=3)
        context = "\n".join([d["content"] for d in docs])
        base = super().build_agent_prompt(task, tools, history, memory)
        return f"Context:\n{context}\n\n{base}"

# Agent with RAG
agent = Agent(llm=my_llm, prompt_service=RAGPromptService(retriever))
```

### 5. Secure File Operations
```python
from nkit.security import PathValidator

validator = PathValidator(allowed_dirs=["./data", "/tmp"])

@agent.tool("safe_read", "Read file safely")
def read_file(file_path: str):
    safe_path = validator.validate_path(file_path)
    with open(safe_path) as f:
        return f.read()
```

### 6. Graph Orchestration
```python
from nkit.chain import Graph, Node, State

def step1(state: State):
    return {"result": "processed"}

def step2(state: State):
    data = state.get("result")
    return f"Final: {data}"

g = Graph()
g.add_node(Node("step1", step1))
g.add_node(Node("step2", step2))
g.add_edge("step1", "step2")
final_state = g.run(State())
```

### 7. Multi-Agent Orchestration
```python
from nkit.agents.orchestrator import MultiAgentOrchestrator, Role

planner = Agent(llm=planner_llm)
executor = Agent(llm=executor_llm)

orch = MultiAgentOrchestrator(
    [planner, executor],
    roles=[
        Role("planner", "Break down tasks"),
        Role("executor", "Execute tasks")
    ]
)

result = orch.run("Complex task here")
```

## Plugin Development

### Custom Memory Backend
```python
class MyMemory:
    def get(self, key, default=None): ...
    def set(self, key, value): ...
    def append(self, key, value): ...
    def clear(self): ...
    def to_dict(self): ...

agent = Agent(llm=my_llm, memory=MyMemory())
```

### Custom Retriever
```python
class MyRetriever:
    def retrieve(self, query, top_k=5, filters=None):
        # Return list of {"content": str, "metadata": dict, "score": float}
        ...
    
    def add_documents(self, documents):
        # documents: list of {"content": str, "metadata": dict}
        ...
```

### Custom Prompt Service
```python
from nkit.prompt import ReActPromptService

class MyPromptService(ReActPromptService):
    def build_agent_prompt(self, task, tools, history, memory):
        # Build and return prompt string
        ...
```

## Security Patterns

### Path Validation
```python
from nkit.security import PathValidator

validator = PathValidator(
    allowed_dirs=["/data", "/tmp"],
    allowed_extensions=[".txt", ".json"],
    max_path_length=4096
)

safe_path = validator.validate_path(user_input)
```

### String Validation
```python
from nkit.security import StringValidator

validator = StringValidator(
    max_length=1000,
    allowed_chars="alphanumeric_space_punct",
    forbidden_patterns=[r'<script', r'DROP\s+TABLE']
)

safe_string = validator.validate(user_input)
```

### Tool Input Validation
```python
from nkit.security import ToolInputValidator, PathValidator

validator = ToolInputValidator(
    param_validators={
        "file_path": PathValidator(allowed_dirs=["./data"]),
    },
    required=["file_path"]
)

safe_inputs = validator.validate("read_file", raw_inputs)
```

## Debugging

### Enable Debug Logging
```python
agent = Agent(llm=my_llm, log_level="DEBUG")
```

### Inspect Agent State
```python
# After running
print(agent.memory.to_dict())
print(len(agent.registry.tools))
```

### Custom Logger
```python
from nkit.utils import setup_logger

logger = setup_logger("my_component", "DEBUG")
logger.debug("Detailed info")
```

## Common Errors

### "Tool not found"
- Check tool name matches exactly
- Verify tool registered: `print(agent.registry.tools.keys())`

### "Max steps reached"
- Increase `max_steps` parameter
- Check LLM is returning valid JSON
- Enable debug logging to see prompts

### "Path validation failed"
- Add directory to `allowed_dirs` in PathValidator
- Check for typos in path
- Verify path is absolute or relative to CWD

### "Invalid LLM response"
- LLM must return JSON in markdown code block: \`\`\`json ... \`\`\`
- Check `response_parser` configuration
- Enable debug logging to see raw LLM output

## Performance Tips

1. **Use in-memory retriever for < 1000 docs**
2. **Enable async for I/O-bound tools**
3. **Limit `max_history` to reduce prompt tokens**
4. **Cache LLM responses** (implement in llm callable)
5. **Use JSONFileMemory only for small state** (< 1MB)

## Documentation Locations

- Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Summary: [SUMMARY.md](SUMMARY.md)
- Examples: `examples/` directory
- API docs: Inline docstrings in each module
