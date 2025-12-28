"""Abstract interfaces for plugin architecture.

This module defines protocols and abstract base classes that enable a clean,
extensible plugin system following SOLID principles:

- **Single Responsibility**: Each interface has one clear purpose
- **Open/Closed**: Extend behavior by implementing interfaces, not modifying core
- **Liskov Substitution**: Any implementation of an interface can be swapped
- **Interface Segregation**: Small, focused interfaces instead of monolithic ones
- **Dependency Inversion**: Core components depend on abstractions, not concrete types

Security Considerations:
- All implementations should validate inputs before processing
- Memory stores should sanitize keys to prevent injection attacks
- Retrieval systems should validate queries and limit result sizes
- Prompt services should escape/sanitize user inputs in prompts
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol


class MemoryStore(Protocol):
    """Abstract interface for agent memory storage.
    
    Purpose:
        Provides persistent or in-memory state for agents across runs.
        Enables context retention, conversation history, and intermediate results.
    
    Reuse Patterns:
        - In-memory dict for testing and simple use cases
        - JSON file for session persistence
        - SQLite/PostgreSQL for production multi-agent systems
        - Redis for distributed agent coordination
        - Vector database for semantic memory retrieval
    
    Security:
        - Implementations MUST validate/sanitize keys (prevent path traversal, injection)
        - Consider encryption for sensitive data
        - Implement access control for multi-tenant scenarios
    
    Example:
        ```python
        memory = InMemoryStore()
        memory.set("user_name", "Alice")
        name = memory.get("user_name")
        ```
    """
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Retrieve a value by key.
        
        Args:
            key: Storage key (must be validated by implementation)
            default: Value to return if key doesn't exist
            
        Returns:
            Stored value or default
        """
        ...
    
    def set(self, key: str, value: Any) -> None:
        """Store a value under a key.
        
        Args:
            key: Storage key (must be validated)
            value: Any serializable value
        """
        ...
    
    def append(self, key: str, value: Any) -> None:
        """Append a value to a list under key.
        
        Args:
            key: Storage key
            value: Value to append
        """
        ...
    
    def clear(self) -> None:
        """Clear all stored data."""
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all data as a dictionary.
        
        Returns:
            Complete snapshot of stored data
        """
        ...


class PromptService(ABC):
    """Abstract interface for prompt construction.
    
    Purpose:
        Separates prompt engineering from agent logic, enabling:
        - A/B testing of prompt templates
        - Multi-language prompt support
        - Domain-specific prompt strategies
    
    Reuse Patterns:
        - ReActPromptService: ReAct-style reasoning prompts
        - ChainOfThoughtPromptService: Step-by-step reasoning
        - FewShotPromptService: Include example interactions
        - RAGPromptService: Inject retrieved context
        - MultilingualPromptService: Language-specific templates
    
    Security:
        - Sanitize user inputs to prevent prompt injection
        - Limit context size to prevent token exhaustion attacks
        - Validate tool/memory content before injection
    
    Example:
        ```python
        service = ReActPromptService()
        prompt = service.build_agent_prompt(
            task="Find current weather",
            tools=registry,
            history=steps,
            memory=memory
        )
        ```
    """
    
    @abstractmethod
    def build_agent_prompt(
        self,
        task: str,
        tools: "ToolRegistry",
        history: List["Step"],
        memory: Optional[MemoryStore] = None
    ) -> str:
        """Construct the prompt for the agent's next iteration.
        
        Args:
            task: User's task description
            tools: Available tools registry
            history: Previous reasoning steps
            memory: Optional memory store with context
            
        Returns:
            Formatted prompt string ready for LLM
        """
        pass


class ResponseParser(ABC):
    """Abstract interface for parsing LLM responses.
    
    Purpose:
        Decouples response format from agent logic, enabling:
        - JSON, XML, YAML, or custom parsers
        - Fallback/retry strategies
        - Format validation and correction
    
    Reuse Patterns:
        - JSONMarkdownParser: Extract JSON from markdown code blocks
        - XMLParser: Parse <action><tool>...</tool></action>
        - NaturalLanguageParser: Parse free-form text with regex
        - StructuredOutputParser: Enforce strict schemas
    
    Security:
        - Validate parsed output against schema
        - Reject malformed or suspicious responses
        - Limit recursion depth in nested structures
    
    Example:
        ```python
        parser = JSONMarkdownParser()
        result = parser.parse(llm_response)
        action = result.get("action")
        ```
    """
    
    @abstractmethod
    def parse(self, text: str) -> dict:
        """Parse LLM response into structured format.
        
        Args:
            text: Raw LLM response text
            
        Returns:
            Dictionary with at minimum:
            - "thought": reasoning text
            - "action": tool name (optional)
            - "action_input": tool parameters (optional)
            - "final_answer": completion response (optional)
        """
        pass


class RetrievalSystem(ABC):
    """Abstract interface for RAG and context retrieval.
    
    Purpose:
        Enables agents to query external knowledge bases, documents, or vector stores.
        Core component for RAG (Retrieval-Augmented Generation) pipelines.
    
    Reuse Patterns:
        - VectorStoreRetriever: Semantic search over embeddings
        - BM25Retriever: Traditional keyword search
        - HybridRetriever: Combine semantic + keyword
        - SQLRetriever: Query structured databases
        - GraphRetriever: Traverse knowledge graphs
        - WebRetriever: Search and scrape web content
    
    Security:
        - Validate/sanitize queries to prevent injection
        - Limit result count and size to prevent resource exhaustion
        - Implement access control for sensitive documents
        - Filter/redact PII from retrieved content
    
    Example:
        ```python
        retriever = VectorStoreRetriever(embeddings, index)
        docs = retriever.retrieve("How to reset password?", top_k=3)
        context = "\n".join([doc.content for doc in docs])
        ```
    """
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents/chunks for a query.
        
        Args:
            query: Search query (text or structured)
            top_k: Maximum number of results
            filters: Optional metadata filters (e.g., {"source": "manual.pdf"})
            
        Returns:
            List of retrieved items, each with:
            - "content": document text/chunk
            - "metadata": source, score, timestamp, etc.
            - "score": relevance score (optional)
        """
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the retrieval index.
        
        Args:
            documents: List of docs, each with "content" and "metadata"
        """
        pass


class ToolValidator(ABC):
    """Abstract interface for tool input validation.
    
    Purpose:
        Validates and sanitizes tool inputs before execution, preventing:
        - Path traversal attacks (file tools)
        - Command injection (shell tools)
        - Resource exhaustion (limit file sizes, timeouts)
    
    Reuse Patterns:
        - SchemaValidator: JSON schema validation
        - PathValidator: Sanitize file paths, check permissions
        - StringValidator: Max length, allowed characters
        - CommandValidator: Whitelist allowed commands/args
    
    Security:
        - Fail-safe: reject by default, allow by explicit rules
        - Log validation failures for audit
        - Rate limit tool calls per session
    
    Example:
        ```python
        validator = PathValidator(allowed_dirs=["/tmp", "/data"])
        validator.validate("read_file", {"path": "/etc/passwd"})  # raises
        ```
    """
    
    @abstractmethod
    def validate(self, tool_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize tool inputs.
        
        Args:
            tool_name: Name of tool being called
            inputs: Raw input dictionary from LLM/agent
            
        Returns:
            Sanitized inputs (may transform values)
            
        Raises:
            ValueError: If inputs are invalid/unsafe
        """
        pass


__all__ = [
    "MemoryStore",
    "PromptService",
    "ResponseParser",
    "RetrievalSystem",
    "ToolValidator",
]
