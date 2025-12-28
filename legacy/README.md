# Legacy Code Archive

This folder contains deprecated implementations that have been replaced by newer modular components.

## Files

### llm_adapter.py
**Replaced by:** llms/ module
**Description:** Old LLM adapter interface. The new llms/ module provides:
- BaseLLM abstract class
- OpenAILLM, OllamaLLM, AnthropicLLM implementations
- Unified sync/async interface
- Streaming support

### retrieval.py
**Replaced by:** knowledge/ module
**Description:** Old RAG retrieval system. The new knowledge/ module provides:
- KnowledgeBase for document management
- Multiple chunking strategies
- Vector embeddings support
- Semantic search with VectorStore

### prompt.py
**Replaced by:** Prompt services in multiple modules
**Description:** Old prompt templates. New implementations:
- ReActPromptService in nbagents.py
- Custom prompt injection in KnowledgeBase
- Task/Crew-specific prompts in tasks/ and crews/

## Migration Guide

If you need to reference old implementations, files are preserved here for backward compatibility. However, prefer using the new modular components for new code.

## Removal Timeline

These files can be safely deleted after confirming no external code depends on them.
