"""Example: RAG-enabled agent with custom memory and prompt service.

This demo shows how to build a sophisticated agent with:
- RAG retrieval for context augmentation
- Persistent file-based memory
- Custom prompt service that injects retrieved context
- Security validation on tool inputs

Use Case:
    Question-answering agent over a document corpus with session memory.
"""

import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from nkit.nbagents import Agent
from nkit.memory import JSONFileMemory
from nkit.retrieval import InMemoryRetriever
from nkit.prompt import ReActPromptService
from nkit.tools import ToolRegistry, Tool
from nkit.security import PathValidator, ToolInputValidator


class RAGPromptService(ReActPromptService):
    """Custom prompt service that injects retrieved context.
    
    Purpose:
        Extends ReActPromptService to include RAG context in prompts.
        Retrieves relevant documents and injects before task description.
    """
    
    def __init__(self, retriever, max_history=20, max_memory_size=5000, max_context_docs=3):
        super().__init__(max_history, max_memory_size)
        self.retriever = retriever
        self.max_context_docs = max_context_docs
    
    def build_agent_prompt(self, task, tools, history, memory=None):
        """Build prompt with RAG context injection."""
        # Retrieve relevant documents
        docs = self.retriever.retrieve(task, top_k=self.max_context_docs)
        
        # Format retrieved context
        if docs:
            context_text = "**Retrieved Context:**\n"
            for i, doc in enumerate(docs, 1):
                context_text += f"{i}. {doc['content']}\n"
            context_text += "\n"
        else:
            context_text = ""
        
        # Build base prompt
        base_prompt = super().build_agent_prompt(task, tools, history, memory)
        
        # Inject context after memory, before task
        if context_text:
            # Insert context right after memory section
            mem_end = base_prompt.find("You are an AI agent")
            if mem_end != -1:
                return base_prompt[:mem_end] + context_text + base_prompt[mem_end:]
        
        return base_prompt


def setup_knowledge_base(retriever):
    """Populate retriever with sample documents."""
    documents = [
        {
            "content": "Paris is the capital and largest city of France, located on the Seine River.",
            "metadata": {"source": "geography.txt", "topic": "france"}
        },
        {
            "content": "The Eiffel Tower is a wrought-iron lattice tower in Paris, built in 1889.",
            "metadata": {"source": "landmarks.txt", "topic": "france"}
        },
        {
            "content": "Python is a high-level programming language created by Guido van Rossum in 1991.",
            "metadata": {"source": "tech.txt", "topic": "programming"}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence focused on learning from data.",
            "metadata": {"source": "tech.txt", "topic": "ai"}
        },
        {
            "content": "The French Revolution was a period of social and political upheaval in France from 1789 to 1799.",
            "metadata": {"source": "history.txt", "topic": "france"}
        },
    ]
    retriever.add_documents(documents)
    print(f"Loaded {len(documents)} documents into knowledge base")


def mock_llm(prompt: str) -> str:
    """Mock LLM for demonstration (replace with real LLM)."""
    # Simple pattern matching for demo
    if "capital of France" in prompt.lower():
        return '''```json
{
  "thought": "The context mentions Paris is the capital of France",
  "final_answer": "The capital of France is Paris, located on the Seine River."
}
```'''
    elif "eiffel tower" in prompt.lower():
        return '''```json
{
  "thought": "I found information about the Eiffel Tower in the retrieved context",
  "final_answer": "The Eiffel Tower is a wrought-iron lattice tower in Paris, built in 1889."
}
```'''
    else:
        return '''```json
{
  "thought": "I need more information to answer this question",
  "action": "web_search",
  "action_input": {"query": "general knowledge"}
}
```'''


def main():
    print("=" * 60)
    print("RAG-Enabled Agent Demo")
    print("=" * 60)
    
    # 1. Setup retrieval system
    print("\n[1] Setting up knowledge base...")
    retriever = InMemoryRetriever()
    setup_knowledge_base(retriever)
    
    # 2. Setup memory (persistent JSON file)
    print("\n[2] Initializing persistent memory...")
    memory = JSONFileMemory("./demo_session.json")
    memory.set("session_start", "2025-12-28")
    
    # 3. Setup custom prompt service with RAG
    print("\n[3] Creating RAG prompt service...")
    prompt_service = RAGPromptService(retriever, max_context_docs=2)
    
    # 4. Setup tool registry with validation
    print("\n[4] Configuring tools with security validation...")
    registry = ToolRegistry(include_builtin=False)
    
    # Add a validated tool
    path_validator = PathValidator(allowed_dirs=[".", "/tmp"])
    def safe_read_file(file_path: str) -> str:
        """Read file with path validation."""
        validated_path = path_validator.validate_path(file_path)
        try:
            with open(validated_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    
    registry.register(Tool("safe_read_file", safe_read_file, "Read file safely"))
    
    # 5. Create agent with all plugins
    print("\n[5] Initializing agent with plugins...")
    agent = Agent(
        llm=mock_llm,
        registry=registry,
        memory=memory,
        prompt_service=prompt_service,
        max_steps=5,
        log_level="INFO"
    )
    
    # 6. Run queries
    print("\n" + "=" * 60)
    print("Running Queries")
    print("=" * 60)
    
    queries = [
        "What is the capital of France?",
        "Tell me about the Eiffel Tower",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 60)
        try:
            answer = agent.run(query)
            print(f"\n[Answer] {answer}\n")
        except Exception as e:
            print(f"\n[Error] {e}\n")
    
    # 7. Show memory persistence
    print("\n" + "=" * 60)
    print("Memory State")
    print("=" * 60)
    print(memory.to_dict())
    print(f"\nMemory persisted to: {memory.file_path}")


if __name__ == "__main__":
    main()
