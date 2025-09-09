# NKit Summary

## What We Added

###  **Async Support**
- Full async/await support for both tools and LLM calls
- `run_async()` method for async execution
- `is_async_function()` utility to detect async functions
- `run_sync_or_async()` utility to handle both sync and async functions seamlessly

###  **Built-in Tools**
- **Web Search**: Search the web using DuckDuckGo API (both sync and async versions)
- **File Operations**: 
  - `read_file`: Read content from files
  - `write_file`: Write content to files
  - `list_files`: List files in directories
- **Time**: `get_current_time` - Get current date and time
- Auto-registration of built-in tools (can be disabled with `include_builtin_tools=False`)

###  **Comprehensive Logging**
- `setup_logger()` function for easy logger configuration
- Agent-specific loggers with configurable log levels
- Debug logging for tool execution and iterations
- Info/Warning/Error logging for key events
- Tool registration and execution logging

###  **Enhanced Error Handling**
- Better retry mechanisms for both LLM calls and tool execution
- Graceful error reporting with detailed messages
- Tool execution attempt logging
- Improved exception handling

###  **Improved Tool System**
- Enhanced `Tool` class with async support detection
- Better tool schema parsing
- Tool metadata including async status
- Improved tool registry with built-in tool management

###  **Better Agent Configuration**
- `log_level` parameter for controlling logging verbosity
- `include_builtin_tools` parameter to control built-in tool loading
- Enhanced initialization with better defaults

##  **Dependencies Added**
- `requests` - For web search functionality
- `aiohttp` - For async web search
- `asyncio` - For async support (built-in)
- `logging` - For comprehensive logging (built-in)

## 🎮 **Usage Examples**

### Basic Usage with Built-in Tools
```python
from nkit import Agent

agent = Agent(llm=your_llm_function, include_builtin_tools=True)
result = agent.run("Get the current time and save it to a file")
```

### Async Usage
```python
import asyncio
from nkit import Agent

async def main():
  agent = Agent(llm=async_llm_function, include_builtin_tools=True)
  result = await agent.run_async("Search for Python tutorials and save results")

asyncio.run(main())
```

### Custom Logging
```python
from nkit import Agent, setup_logger

logger = setup_logger("my_agent", "DEBUG")
agent = Agent(llm=your_llm, log_level="DEBUG")
```

### Custom Tools with Async
```python
@agent.tool("async_process")
async def async_process(data: str) -> str:
    await asyncio.sleep(1)  # Simulate async work
    return f"Processed: {data}"
```

##  **Verified Working Features**
- 1 File read/write operations work correctly
- 2 Time retrieval works correctly  
- 3 Async tool execution works
- 4 Comprehensive logging is functional
- 5 Error handling and retries work
- 6 Tool registration and discovery works
- 7 Both sync and async LLM calls supported

##  **Next Possible Enhancements**
- Memory/persistence system
- Multi-agent communication
- Plugin system
- Configuration file support
- Performance metrics and monitoring
- Additional built-in tools (database, API calls, etc.)
- Tool composition and chaining
- Custom prompt templates

Here is the sample Google Colab file you can try for experimenting with NKit:-
- https://colab.research.google.com/drive/1OKq398_9xKO7ZNEFF0MxuWoXmHZ0kFCi?usp=sharing

---

## Migration note

This project has been renamed to **NKit** (import as `nkit`). The old
implementation details remain under the internal package layout, but you
should update your imports to use `nkit` moving forward. Example:

```python
from nkit import Agent
```
#  NKit Summary

## What We Added

###  **Async Support**
- Full async/await support for both tools and LLM calls
- `run_async()` method for async execution
- `is_async_function()` utility to detect async functions
- `run_sync_or_async()` utility to handle both sync and async functions seamlessly

###  **Built-in Tools**
- **Web Search**: Search the web using DuckDuckGo API (both sync and async versions)
- **File Operations**: 
  - `read_file`: Read content from files
  - `write_file`: Write content to files
  - `list_files`: List files in directories
- **Time**: `get_current_time` - Get current date and time
- Auto-registration of built-in tools (can be disabled with `include_builtin_tools=False`)

###  **Comprehensive Logging**
- `setup_logger()` function for easy logger configuration
- Agent-specific loggers with configurable log levels
- Debug logging for tool execution and iterations
- Info/Warning/Error logging for key events
- Tool registration and execution logging

###  **Enhanced Error Handling**
- Better retry mechanisms for both LLM calls and tool execution
- Graceful error reporting with detailed messages
- Tool execution attempt logging
- Improved exception handling

###  **Improved Tool System**
- Enhanced `Tool` class with async support detection
- Better tool schema parsing
- Tool metadata including async status
- Improved tool registry with built-in tool management

###  **Better Agent Configuration**
- `log_level` parameter for controlling logging verbosity
- `include_builtin_tools` parameter to control built-in tool loading
- Enhanced initialization with better defaults

##  **Dependencies Added**
- `requests` - For web search functionality
- `aiohttp` - For async web search
- `asyncio` - For async support (built-in)
- `logging` - For comprehensive logging (built-in)

## 🎮 **Usage Examples**

### Basic Usage with Built-in Tools
```python
from nkit import Agent

agent = Agent(llm=your_llm_function, include_builtin_tools=True)
result = agent.run("Get the current time and save it to a file")
```

### Async Usage
```python
import asyncio
from nkit import Agent

async def main():
  agent = Agent(llm=async_llm_function, include_builtin_tools=True)
  result = await agent.run_async("Search for Python tutorials and save results")

asyncio.run(main())
```

### Custom Logging
```python
from nkit import Agent, setup_logger

logger = setup_logger("my_agent", "DEBUG")
agent = Agent(llm=your_llm, log_level="DEBUG")
```

### Custom Tools with Async
```python
@agent.tool("async_process")
async def async_process(data: str) -> str:
    await asyncio.sleep(1)  # Simulate async work
    return f"Processed: {data}"
```

##  **Verified Working Features**
- 1 File read/write operations work correctly
- 2 Time retrieval works correctly  
- 3 Async tool execution works
- 4 Comprehensive logging is functional
- 5 Error handling and retries work
- 6 Tool registration and discovery works
- 7 Both sync and async LLM calls supported

##  **Next Possible Enhancements**
- Memory/persistence system
- Multi-agent communication
- Plugin system
- Configuration file support
- Performance metrics and monitoring
- Additional built-in tools (database, API calls, etc.)
- Tool composition and chaining
- Custom prompt templates

Here is the sample Google Colab file you can try for experimenting with NBAgents:-
- https://colab.research.google.com/drive/1OKq398_9xKO7ZNEFF0MxuWoXmHZ0kFCi?usp=sharing

---

## Migration note

This project has been renamed to **NKit** (import as `nkit`). The old
implementation details remain under the internal package layout, but you
should update your imports to use `nkit` moving forward. Example:

```python
from nkit import Agent
```
