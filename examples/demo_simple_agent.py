import sys
import os

from nkit.agent import Agent

def mock_llm(prompt: str) -> str:
    """
    A simulated LLM function that returns structured JSON decisions.
    In reality, you would use OllamaLLM, OpenAILLM, etc.
    """
    # If the prompt shows the tool returned '12', output a final_answer
    if "observation: 12" in prompt:
        return '''```json
        {
          "thought": "The tool returned 12, so I can now give the final answer.",
          "final_answer": "The sum of 5 and 7 is 12."
        }
        ```'''
    # Otherwise, choose an action to execute our tool
    else:
        return '''```json
        {
          "thought": "I need to calculate the sum of 5 and 7.",
          "action": "add",
          "action_input": {"a": 5, "b": 7}
        }
        ```'''

def main():
    print("Initializing NKit Agent...")
    # Create the agent with our "LLM"
    agent = Agent(llm=mock_llm, log_level="ERROR")

    print("Registering tools via decorator...")
    @agent.tool("add", "Add two integers together")
    def add(a: int, b: int) -> int:
        print(f"\n[TOOL FIRED] Executing add({a}, {b})...")
        return a + b

    print("Task: 'Calculate 5 + 7'\n")
    
    # Run the agent! It will hit the mock_llm, see the action, execute 'add', and request again
    result = agent.run("Calculate 5 + 7")
    
    print(f"\nFinal Result: {result}")

if __name__ == "__main__":
    main()
