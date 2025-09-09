import os
import asyncio
import pytest
from nkit import Agent, setup_logger

# The test suite sometimes runs without the external 'groq' package available
# Provide a lightweight shim if it's missing so tests can run offline/CI.
try:
    from groq import Groq
except Exception:
    class Groq:
        def __init__(self, api_key: str | None = None):
            self.api_key = api_key

        class chat:
            class completions:
                @staticmethod
                def create(model, messages, temperature, max_tokens):
                    # Return a fake structure similar to the real SDK
                    class Resp:
                        class Choice:
                            class Message:
                                def __init__(self, content):
                                    self.content = content

                            def __init__(self, content):
                                self.message = Resp.Choice.Message(content)

                        def __init__(self, content):
                            self.choices = [Resp.Choice(content)]

                    # echo back the user content for tests
                    user_content = messages[0]["content"] if messages else ""
                    return Resp(user_content)

logger = setup_logger("test", "DEBUG")

def call_llm(prompt: str) -> str:
    """LLM call function that works offline for tests."""
    # For tests, return a simple JSON response that the agent can parse
    return '```json\n{"thought": "test response", "action": "", "action_input": "", "final_answer": "test completed"}\n```'

async def async_llm_call(prompt: str) -> str:
    """Async LLM call function."""
    return call_llm(prompt)

def test_builtin_tools():
    """Test agent with built-in tools."""
    logger.info("Testing agent with built-in tools")
    
    agent = Agent(call_llm, max_steps=5, include_builtin_tools=True)
    

    @agent.tool("calculate", "Perform basic mathematical calculations")
    def calculate(expression: str) -> float:
        """Calculate mathematical expression safely."""
        try:
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                return eval(expression)
            else:
                return "Invalid expression"
        except:
            return "Calculation error"
    
    result = agent.run("What's the current time? Then calculate 15 * 23 + 100")
    print(f"Result: {result}")

@pytest.mark.asyncio
async def test_async_agent():
    """Test async agent functionality."""
    logger.info("Testing async agent")
    
    agent = Agent(async_llm_call, max_steps=3, include_builtin_tools=True)

    @agent.tool("async_process", "Process data asynchronously")
    async def async_process(data: str) -> str:
        """Async processing simulation."""
        await asyncio.sleep(0.1) 
        return f"Processed: {data.upper()}"
    
    # Run async
    result = await agent.run_async("Process the text 'hello world' and tell me the current time")
    print(f"Async Result: {result}")

def test_file_operations():
    """Test file operation tools."""
    logger.info("Testing file operations")
    
    agent = Agent(call_llm, max_steps=5, include_builtin_tools=True)
    
    result = agent.run("""
    1. Write 'Hello, NKit!' to a file called 'test.txt'
    2. Read the content back from the file
    3. List the files in the current directory
    """)
    print(f"File Operations Result: {result}")


def test_memory_and_chain():
    """Test the tiny Memory store and Chain runner."""
    from nkit import Memory, Chain

    mem = Memory()
    mem.set("a", 1)
    assert mem.get("a") == 1
    mem.append("log", "first")
    mem.append("log", "second")
    assert mem.get("log") == ["first", "second"]

    def inc(x):
        return x + 1

    def mul2(x):
        return x * 2

    chain = Chain([inc, mul2])
    assert chain.run(3) == 8


def test_llm_adapter_and_prompt():
    from nkit import CallableLLMAdapter, PromptTemplate

    def echo(prompt: str) -> str:
        return f"ECHO: {prompt}"

    adapter = CallableLLMAdapter(echo)
    assert adapter("hello") == "ECHO: hello"

    template = PromptTemplate("Say hi to {name}.")
    assert template.format(name="Alice") == "Say hi to Alice."


def test_llm_chain():
    from nkit import LLMChain, CallableLLMAdapter, PromptTemplate

    def simple_llm(prompt: str) -> str:
        return f"LLM:{prompt}"

    adapter = CallableLLMAdapter(simple_llm)
    template = PromptTemplate("Ask: {q}")
    chain = LLMChain(adapter, template)
    assert chain.run(q="ping") == "LLM:Ask: ping"


def test_agent_memory_integration():
    from nkit import Agent, Memory

    # Fake LLM that always returns a final_answer in JSON markdown
    def fake_llm(prompt: str) -> str:
        return '```json{"thought":"done","action":"","action_input":"","final_answer":"42"}```'

    mem = Memory()
    agent = Agent(fake_llm, max_steps=1, include_builtin_tools=False, memory=mem)
    result = agent.run("Compute the ultimate answer")
    assert result == "42"
    assert mem.get("last_answer") == "42"




if __name__ == "__main__":
    test_builtin_tools()
    print("\n" + "="*50 + "\n")
    
    test_file_operations()
    print("\n" + "="*50 + "\n")
    
    asyncio.run(test_async_agent())
