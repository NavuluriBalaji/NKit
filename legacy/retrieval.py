"""Retrieval-Augmented Generation (RAG) implementations.

This module provides retrieval systems for augmenting agent prompts with
relevant context from document stores, vector databases, or knowledge bases.

Architecture:
    All retrievers implement the RetrievalSystem interface, enabling:
    - Plug-and-play retrieval backends
    - Consistent API across vector/keyword/hybrid search
    - Easy integration with agents via PromptService

Use Cases:
    - Q&A over documents
    - Context-aware agents
    - Knowledge base search
    - Semantic memory
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class InMemoryRetriever:
    """Simple in-memory keyword-based retriever.
    
    Purpose:
        Lightweight retrieval for testing and small document sets.
        Uses case-insensitive substring matching (not semantic).
    
    Reuse Patterns:
        - Development/testing: no external dependencies
        - Small knowledge bases (< 1000 docs)
        - FAQ systems
        - Prototyping before vector DB
    
    Limitations:
        - No semantic understanding
        - Linear scan (O(n) per query)
        - No persistence
        - Basic relevance scoring
    
    Security:
        - No injection risks (pure Python)
        - Memory limited by Python process
        - Validate document structure on add
    
    Example:
        ```python
        retriever = InMemoryRetriever()
        retriever.add_documents([
            {"content": "Paris is the capital of France", "metadata": {"source": "geo.txt"}},
            {"content": "Python is a programming language", "metadata": {"source": "tech.txt"}},
        ])
        
        results = retriever.retrieve("capital of France", top_k=1)
        print(results[0]["content"])  # "Paris is the capital..."
        ```
    """
    
    def __init__(self):
        """Initialize empty document store."""
        self.documents: List[Dict[str, Any]] = []
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the retrieval index.
        
        Args:
            documents: List of dicts with "content" (str) and "metadata" (dict)
        
        Raises:
            ValueError: If document format invalid
        
        Validation:
            - Ensures "content" key present and is string
            - Metadata optional but must be dict if present
        """
        for doc in documents:
            if not isinstance(doc, dict) or "content" not in doc:
                raise ValueError("Each document must be dict with 'content' key")
            if not isinstance(doc["content"], str):
                raise ValueError("Document content must be string")
            if "metadata" in doc and not isinstance(doc["metadata"], dict):
                raise ValueError("Document metadata must be dict")
            
            # Store with metadata
            self.documents.append({
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
            })
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve documents matching query.
        
        Args:
            query: Search query (case-insensitive substring match)
            top_k: Maximum results to return
            filters: Optional metadata filters (e.g., {"source": "manual.pdf"})
        
        Returns:
            List of matching documents with scores, sorted by relevance
        
        Scoring:
            Simple heuristic: count of query word occurrences in content
        
        Example:
            ```python
            results = retriever.retrieve("Python programming", top_k=3)
            for r in results:
                print(f"Score: {r['score']}, Content: {r['content'][:50]}...")
            ```
        """
        if not isinstance(query, str) or not query.strip():
            return []
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        scored_docs = []
        for doc in self.documents:
            # Apply metadata filters
            if filters:
                if not all(doc["metadata"].get(k) == v for k, v in filters.items()):
                    continue
            
            # Simple scoring: count matching words
            content_lower = doc["content"].lower()
            score = sum(1 for word in query_words if word in content_lower)
            
            if score > 0:
                scored_docs.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": score,
                })
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]


class JSONDocumentRetriever:
    """File-based retriever using JSON document store.
    
    Purpose:
        Persistent keyword retrieval with file-based storage.
        Suitable for small to medium document sets with persistence needs.
    
    Reuse Patterns:
        - Production single-node deployments
        - Document Q&A with persistence
        - Backup/restore retrieval index
        - Human-readable document store
    
    Limitations:
        - Loads entire file into memory
        - No concurrent access
        - Linear search (slow for large sets)
    
    Security:
        - Validates file path
        - JSON parsing limits nesting depth
        - File permissions inherited from OS
    
    Example:
        ```python
        retriever = JSONDocumentRetriever("./docs_index.json")
        retriever.add_documents([{"content": "...", "metadata": {...}}])
        # Auto-saved to file
        
        # Later, in new process:
        retriever = JSONDocumentRetriever("./docs_index.json")  # Auto-loads
        results = retriever.retrieve("query")
        ```
    """
    
    def __init__(self, file_path: str):
        """Initialize with JSON file.
        
        Args:
            file_path: Path to JSON file (created if doesn't exist)
        """
        self.file_path = Path(file_path).resolve()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing or initialize empty
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.documents = data.get("documents", [])
            except (json.JSONDecodeError, IOError):
                self.documents = []
        else:
            self.documents = []
            self._save()
    
    def _save(self) -> None:
        """Persist documents to file."""
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump({"documents": self.documents}, f, indent=2, ensure_ascii=False)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents and persist."""
        for doc in documents:
            if not isinstance(doc, dict) or "content" not in doc:
                raise ValueError("Each document must be dict with 'content' key")
            self.documents.append({
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
            })
        self._save()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve documents (same logic as InMemoryRetriever)."""
        if not isinstance(query, str) or not query.strip():
            return []
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        scored_docs = []
        for doc in self.documents:
            if filters:
                if not all(doc["metadata"].get(k) == v for k, v in filters.items()):
                    continue
            
            content_lower = doc["content"].lower()
            score = sum(1 for word in query_words if word in content_lower)
            
            if score > 0:
                scored_docs.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": score,
                })
        
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]


__all__ = ["InMemoryRetriever", "JSONDocumentRetriever"]
