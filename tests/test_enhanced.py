import os
import asyncio
from nanoagents import Agent, setup_logger
from groq import Groq

logger = setup_logger("test", "DEBUG")

def call_llm(prompt: str) -> str:
    """LLM call function."""
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="qwen-2.5-32b",
        messages=messages,
        temperature=0.7,
        max_tokens=2048
    )
    return response.choices[0].message.content

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
    1. Write 'Hello, nanoagents!' to a file called 'test.txt'
    2. Read the content back from the file
    3. List the files in the current directory
    """)
    print(f"File Operations Result: {result}")

if __name__ == "__main__":
    test_builtin_tools()
    print("\n" + "="*50 + "\n")
    
    test_file_operations()
    print("\n" + "="*50 + "\n")
    
    asyncio.run(test_async_agent())
