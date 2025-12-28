"""Knowledge base management with embedding support.

This module provides document storage, retrieval, and embedding generation:
- Document chunking and indexing
- Embedding generation (OpenAI, local models)
- Vector similarity search
- Integration with existing retrieval system

Architecture:
    - KnowledgeBase: Main interface for document management
    - EmbeddingProvider: Abstract embedding generation
    - VectorStore: Storage backend for embeddings
    - Chunker: Document splitting strategies
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re


@dataclass
class Document:
    """Represents a document in the knowledge base.
    
    Attributes:
        content: Document text
        metadata: Document metadata (source, title, etc.)
        id: Unique document identifier
        embedding: Vector embedding (if generated)
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if self.id is None:
            # Generate ID from content hash
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()
            self.id = content_hash[:16]


@dataclass
class Chunk:
    """Represents a chunk of a document.
    
    Attributes:
        content: Chunk text
        doc_id: Parent document ID
        chunk_index: Position in document
        metadata: Chunk-specific metadata
        embedding: Vector embedding
    """
    content: str
    doc_id: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


class Chunker(ABC):
    """Abstract document chunking strategy.
    
    Purpose:
        Split documents into smaller pieces for:
        - Better retrieval granularity
        - Embedding model token limits
        - Context window management
    """
    
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into chunks.
        
        Args:
            document: Document to chunk
        
        Returns:
            List of chunks with metadata
        """
        pass


class FixedSizeChunker(Chunker):
    """Split documents into fixed-size chunks with overlap.
    
    Purpose:
        Simple chunking strategy that maintains context across chunks.
    
    Args:
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks
    
    Example:
        ```python
        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
        doc = Document(content="..." * 10000)
        chunks = chunker.chunk(doc)
        ```
    """
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        """Initialize chunker.
        
        Args:
            chunk_size: Characters per chunk
            overlap: Overlapping characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into fixed-size chunks.
        
        Args:
            document: Document to chunk
        
        Returns:
            List of chunks with overlap
        """
        content = document.content
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = start + self.chunk_size
            chunk_text = content[start:end]
            
            chunks.append(Chunk(
                content=chunk_text,
                doc_id=document.id,
                chunk_index=chunk_index,
                metadata={**document.metadata, "chunk_size": len(chunk_text)}
            ))
            
            start = end - self.overlap
            chunk_index += 1
        
        return chunks


class SentenceChunker(Chunker):
    """Split documents by sentences with size limits.
    
    Purpose:
        Preserve sentence boundaries for better semantic coherence.
    
    Args:
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk
    
    Example:
        ```python
        chunker = SentenceChunker(max_chunk_size=1000)
        chunks = chunker.chunk(document)
        ```
    """
    
    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100):
        """Initialize sentence-based chunker.
        
        Args:
            max_chunk_size: Maximum chunk size
            min_chunk_size: Minimum chunk size
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document by sentences.
        
        Args:
            document: Document to chunk
        
        Returns:
            List of sentence-based chunks
        """
        # Simple sentence splitting (can be improved with nltk)
        sentences = re.split(r'[.!?]+', document.content)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds max size, create chunk
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    content=chunk_text,
                    doc_id=document.id,
                    chunk_index=chunk_index,
                    metadata={**document.metadata, "sentences": len(current_chunk)}
                ))
                current_chunk = []
                current_size = 0
                chunk_index += 1
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add final chunk if it meets min size
        if current_chunk and current_size >= self.min_chunk_size:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                content=chunk_text,
                doc_id=document.id,
                chunk_index=chunk_index,
                metadata={**document.metadata, "sentences": len(current_chunk)}
            ))
        
        return chunks


class EmbeddingProvider(ABC):
    """Abstract embedding generation.
    
    Purpose:
        Generate vector embeddings for semantic search.
        Supports multiple providers (OpenAI, local models, etc.).
    """
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query.
        
        Args:
            query: Query string
        
        Returns:
            Embedding vector
        """
        pass


class SimpleEmbeddingProvider(EmbeddingProvider):
    """Simple embedding using character frequencies (for testing).
    
    Purpose:
        Provides basic embeddings without external dependencies.
        Not suitable for production - use OpenAI or sentence-transformers.
    
    Note:
        This is a placeholder. For production, use:
        - OpenAIEmbeddingProvider (with openai library)
        - HuggingFaceEmbeddingProvider (with sentence-transformers)
    """
    
    def __init__(self, dimension: int = 128):
        """Initialize simple embedder.
        
        Args:
            dimension: Embedding vector size
        """
        self.dimension = dimension
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate simple embeddings.
        
        Args:
            texts: Texts to embed
        
        Returns:
            List of embedding vectors
        """
        return [self.embed_query(text) for text in texts]
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query.
        
        Args:
            query: Query text
        
        Returns:
            Embedding vector
        """
        # Simple character frequency embedding (not semantic!)
        embedding = [0.0] * self.dimension
        for char in query.lower():
            idx = ord(char) % self.dimension
            embedding[idx] += 1.0
        
        # Normalize
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding


class VectorStore:
    """In-memory vector storage with cosine similarity search.
    
    Purpose:
        Store and retrieve embeddings efficiently.
        For production, consider Pinecone, Weaviate, or Chroma.
    
    Features:
        - Cosine similarity search
        - Metadata filtering
        - Persistence to JSON
    
    Example:
        ```python
        store = VectorStore()
        store.add(chunk_id="c1", embedding=[0.1, 0.2, ...], metadata={...})
        results = store.search(query_embedding=[0.15, 0.25, ...], top_k=5)
        ```
    """
    
    def __init__(self):
        """Initialize vector store."""
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def add(
        self,
        chunk_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add vector to store.
        
        Args:
            chunk_id: Unique chunk identifier
            embedding: Vector embedding
            metadata: Associated metadata
        """
        self.vectors[chunk_id] = embedding
        self.metadata[chunk_id] = metadata or {}
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            metadata_filter: Filter by metadata (exact match)
        
        Returns:
            List of (chunk_id, similarity_score, metadata) tuples
        """
        results = []
        
        for chunk_id, vector in self.vectors.items():
            # Apply metadata filter
            if metadata_filter:
                chunk_meta = self.metadata.get(chunk_id, {})
                if not all(chunk_meta.get(k) == v for k, v in metadata_filter.items()):
                    continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, vector)
            results.append((chunk_id, similarity, self.metadata.get(chunk_id, {})))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
        
        Returns:
            Similarity score (0-1)
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def save(self, filepath: Path) -> None:
        """Save vector store to JSON.
        
        Args:
            filepath: Output file path
        """
        data = {
            "vectors": self.vectors,
            "metadata": self.metadata
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: Path) -> None:
        """Load vector store from JSON.
        
        Args:
            filepath: Input file path
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.vectors = data["vectors"]
        self.metadata = data["metadata"]


class KnowledgeBase:
    """Main knowledge base interface.
    
    Purpose:
        Unified API for document management, embedding, and retrieval.
        Integrates with existing retrieval.py system.
    
    Features:
        - Document ingestion with chunking
        - Automatic embedding generation
        - Semantic search
        - Metadata management
    
    Reuse Patterns:
        - RAG: Retrieve relevant context for LLM prompts
        - Q&A: Search knowledge base for answers
        - Documentation: Store and search code/docs
    
    Example:
        ```python
        kb = KnowledgeBase(
            chunker=SentenceChunker(),
            embedding_provider=SimpleEmbeddingProvider()
        )
        
        # Add document
        doc = Document(
            content="Python is a programming language...",
            metadata={"source": "wiki", "title": "Python"}
        )
        kb.add_document(doc)
        
        # Search
        results = kb.search("What is Python?", top_k=3)
        for chunk, score in results:
            print(f"Score: {score:.2f}")
            print(chunk.content)
        ```
    """
    
    def __init__(
        self,
        chunker: Optional[Chunker] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None
    ):
        """Initialize knowledge base.
        
        Args:
            chunker: Document chunking strategy
            embedding_provider: Embedding generation
            vector_store: Vector storage backend
        """
        self.chunker = chunker or FixedSizeChunker()
        self.embedding_provider = embedding_provider or SimpleEmbeddingProvider()
        self.vector_store = vector_store or VectorStore()
        
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}
    
    def add_document(self, document: Document) -> List[str]:
        """Add document to knowledge base.
        
        Args:
            document: Document to add
        
        Returns:
            List of chunk IDs created
        """
        # Store document
        self.documents[document.id] = document
        
        # Chunk document
        chunks = self.chunker.chunk(document)
        
        # Generate embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_provider.embed(chunk_texts)
        
        # Store chunks with embeddings
        chunk_ids = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            chunk_id = f"{document.id}_chunk_{chunk.chunk_index}"
            self.chunks[chunk_id] = chunk
            
            # Add to vector store
            self.vector_store.add(
                chunk_id=chunk_id,
                embedding=embedding,
                metadata={
                    **chunk.metadata,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index
                }
            )
            chunk_ids.append(chunk_id)
        
        return chunk_ids
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """Search knowledge base for relevant chunks.
        
        Args:
            query: Search query
            top_k: Number of results
            metadata_filter: Filter by metadata
        
        Returns:
            List of (chunk, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_provider.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter
        )
        
        # Return chunks with scores
        return [
            (self.chunks[chunk_id], score)
            for chunk_id, score, _ in results
            if chunk_id in self.chunks
        ]
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve document by ID.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Document or None if not found
        """
        return self.documents.get(doc_id)
    
    def save(self, directory: Path) -> None:
        """Save knowledge base to directory.
        
        Args:
            directory: Output directory
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save documents
        docs_file = directory / "documents.json"
        docs_data = {
            doc_id: {
                "content": doc.content,
                "metadata": doc.metadata
            }
            for doc_id, doc in self.documents.items()
        }
        with open(docs_file, 'w') as f:
            json.dump(docs_data, f, indent=2)
        
        # Save vector store
        self.vector_store.save(directory / "vectors.json")
    
    def load(self, directory: Path) -> None:
        """Load knowledge base from directory.
        
        Args:
            directory: Input directory
        """
        directory = Path(directory)
        
        # Load documents
        docs_file = directory / "documents.json"
        with open(docs_file, 'r') as f:
            docs_data = json.load(f)
        
        for doc_id, doc_dict in docs_data.items():
            doc = Document(
                content=doc_dict["content"],
                metadata=doc_dict["metadata"],
                id=doc_id
            )
            self.documents[doc_id] = doc
        
        # Load vector store
        self.vector_store.load(directory / "vectors.json")
        
        # Rebuild chunks from documents
        for doc in self.documents.values():
            chunks = self.chunker.chunk(doc)
            for chunk in chunks:
                chunk_id = f"{doc.id}_chunk_{chunk.chunk_index}"
                self.chunks[chunk_id] = chunk


__all__ = [
    "Document",
    "Chunk",
    "Chunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "EmbeddingProvider",
    "SimpleEmbeddingProvider",
    "VectorStore",
    "KnowledgeBase"
]
