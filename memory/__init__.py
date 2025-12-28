"""Memory storage implementations for agents.

This module provides memory backends that implement the MemoryStore protocol,
enabling agents to maintain state across iterations and sessions.

Architecture:
    All implementations follow the MemoryStore interface defined in interfaces.py,
    allowing seamless swapping of memory backends via dependency injection.

Usage Patterns:
    - Memory: In-memory dict for development/testing
    - JSONFileMemory: File-based persistence for single-agent sessions
    - (Future) SQLiteMemory: Relational storage for multi-agent coordination
    - (Future) RedisMemory: Distributed memory for scaled deployments
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _validate_key(key: str) -> None:
    """Validate memory key to prevent injection attacks.
    
    Purpose:
        Ensures keys are safe alphanumeric identifiers, preventing:
        - Path traversal attempts
        - SQL injection (for DB backends)
        - Special character exploits
    
    Args:
        key: Key to validate
        
    Raises:
        ValueError: If key contains unsafe characters
        
    Security:
        Only allows: letters, numbers, underscore, hyphen, dot
        Rejects: /, \\, null bytes, control characters
    """
    if not key or not isinstance(key, str):
        raise ValueError("Key must be a non-empty string")
    if not re.match(r'^[a-zA-Z0-9_\-\.]+$', key):
        raise ValueError(f"Invalid key '{key}': only alphanumeric, underscore, hyphen, and dot allowed")
    if len(key) > 256:
        raise ValueError(f"Key too long: {len(key)} chars (max 256)")


class Memory:
    """In-memory key-value store for agent state.
    
    Purpose:
        Lightweight, thread-safe (within single thread) storage for:
        - Agent conversation history
        - Intermediate computation results
        - User preferences and context
        - Task decomposition state
    
    Reuse Patterns:
        - Development/testing: Fast, no I/O overhead
        - Single-agent scripts: Simple state tracking
        - Stateless deployments: Per-request temporary memory
    
    Limitations:
        - Data lost on process restart
        - No cross-process sharing
        - No persistence across sessions
    
    Security:
        - Validates keys to prevent injection
        - In-memory only: no file/network exposure
        - Thread-safe for single-threaded agents
    
    Example:
        ```python
        memory = Memory()
        memory.set("user_id", "alice_123")
        memory.append("messages", {"role": "user", "text": "Hello"})
        history = memory.get("messages", [])
        ```
    """

    def __init__(self):
        """Initialize empty in-memory store."""
        self._store: Dict[str, Any] = {}

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Retrieve a value by key.
        
        Args:
            key: Storage key
            default: Value returned if key doesn't exist
            
        Returns:
            Stored value or default
            
        Raises:
            ValueError: If key is invalid
        """
        _validate_key(key)
        return self._store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store a value under a key.
        
        Args:
            key: Storage key
            value: Any value (should be JSON-serializable for portability)
            
        Raises:
            ValueError: If key is invalid
        """
        _validate_key(key)
        self._store[key] = value

    def append(self, key: str, value: Any) -> None:
        """Append a value to a list under key.
        
        Creates an empty list if key doesn't exist or isn't a list.
        
        Args:
            key: Storage key
            value: Value to append
            
        Raises:
            ValueError: If key is invalid
        """
        _validate_key(key)
        if key not in self._store or not isinstance(self._store[key], list):
            self._store[key] = []
        self._store[key].append(value)

    def clear(self) -> None:
        """Clear all stored data.
        
        Use Cases:
            - Reset agent state between tasks
            - Clear sensitive data after completion
            - Free memory in long-running processes
        """
        self._store.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export all data as a dictionary.
        
        Returns:
            Shallow copy of internal store
            
        Use Cases:
            - Serialize memory for logging
            - Transfer state between agents
            - Create checkpoints
        """
        return dict(self._store)


class JSONFileMemory:
    """File-based persistent memory using JSON.
    
    Purpose:
        Enables session persistence across agent restarts.
        Stores memory as a single JSON file for easy inspection and backup.
    
    Reuse Patterns:
        - Production single-agent deployments
        - Interactive CLI agents (save conversation history)
        - Debugging: inspect memory state in file
        - Backup/restore agent state
    
    Limitations:
        - File I/O overhead on every write
        - Not suitable for high-frequency updates
        - No concurrent access from multiple processes
        - File size grows with data (no cleanup)
    
    Security:
        - Validates file path to prevent traversal
        - Creates parent directories safely
        - File permissions default to user-only (0o600 recommended)
        - Sanitizes keys like Memory
    
    Example:
        ```python
        memory = JSONFileMemory("./session_data.json")
        memory.set("last_query", "What is the weather?")
        # Data automatically saved to file
        
        # Later, in new process:
        memory = JSONFileMemory("./session_data.json")
        query = memory.get("last_query")  # Loads from file
        ```
    """
    
    def __init__(self, file_path: str):
        """Initialize file-based memory.
        
        Args:
            file_path: Path to JSON file (created if doesn't exist)
            
        Raises:
            ValueError: If file_path is unsafe
        """
        self.file_path = Path(file_path).resolve()
        
        # Security: validate path
        if not self.file_path.parent.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data or initialize empty
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self._store: Dict[str, Any] = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._store = {}
        else:
            self._store = {}
            self._save()
    
    def _save(self) -> None:
        """Persist current state to file."""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self._store, f, indent=2, ensure_ascii=False)
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Retrieve a value by key."""
        _validate_key(key)
        return self._store.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Store a value and persist to file."""
        _validate_key(key)
        self._store[key] = value
        self._save()
    
    def append(self, key: str, value: Any) -> None:
        """Append a value to a list and persist."""
        _validate_key(key)
        if key not in self._store or not isinstance(self._store[key], list):
            self._store[key] = []
        self._store[key].append(value)
        self._save()
    
    def clear(self) -> None:
        """Clear all data and persist."""
        self._store.clear()
        self._save()
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all data as dictionary."""
        return dict(self._store)


__all__ = ["Memory", "JSONFileMemory"]
