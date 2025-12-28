"""LLM adapters for various providers.

This module provides unified interfaces to different LLM providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Ollama (local models)
- Azure OpenAI
- Custom endpoints

Architecture:
    All adapters implement BaseLLM interface for consistency.
    Supports streaming, retries, rate limiting, token counting.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, AsyncIterator

@dataclass
class LLMConfig:
    """Configuration for LLM providers.
    
    Attributes:
        model: Model identifier (e.g., "gpt-4", "claude-3-opus")
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum response tokens
        api_key: Provider API key
        base_url: Custom API endpoint
        timeout: Request timeout in seconds
        max_retries: Retry attempts on failure
        stream: Enable streaming responses
    """
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    stream: bool = False
    extra: Dict[str, Any] = None


class BaseLLM(ABC):
    """Base class for all LLM adapters.
    
    Purpose:
        Provides uniform interface across providers, enabling easy swapping
        and testing with different models.
    
    Reuse Pattern:
        ```python
        # Development with Ollama
        llm = OllamaLLM(model="llama2")
        
        # Production with OpenAI
        llm = OpenAILLM(model="gpt-4", api_key=key)
        
        # Both work with Agent
        agent = Agent(llm=llm.call)
        ```
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize LLM with configuration."""
        self.config = config
    
    @abstractmethod
    def call(self, prompt: str, **kwargs) -> str:
        """Synchronous LLM call.
        
        Args:
            prompt: Input prompt
            **kwargs: Override config parameters
        
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    async def acall(self, prompt: str, **kwargs) -> str:
        """Asynchronous LLM call.
        
        Args:
            prompt: Input prompt
            **kwargs: Override config parameters
        
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream LLM response.
        
        Args:
            prompt: Input prompt
            **kwargs: Override config parameters
        
        Yields:
            Text chunks as they're generated
        """
        pass
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation).
        
        Args:
            text: Text to count
        
        Returns:
            Approximate token count
        """
        return len(text) // 4


class OpenAILLM(BaseLLM):
    """OpenAI API adapter (GPT-4, GPT-3.5, etc.).
    
    Purpose:
        Production-ready adapter for OpenAI models with retry logic,
        error handling, and streaming support.
    
    Example:
        ```python
        llm = OpenAILLM(LLMConfig(
            model="gpt-4-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=2000
        ))
        
        response = llm.call("Explain quantum computing")
        ```
    
    Requires:
        ```bash
        pip install openai
        ```
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout
            )
            self.async_client = openai.AsyncOpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout
            )
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")
    
    def call(self, prompt: str, **kwargs) -> str:
        """Synchronous OpenAI call."""
        params = {
            "model": kwargs.get("model", self.config.model),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    async def acall(self, prompt: str, **kwargs) -> str:
        """Asynchronous OpenAI call."""
        params = {
            "model": kwargs.get("model", self.config.model),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = await self.async_client.chat.completions.create(**params)
        return response.choices[0].message.content
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream OpenAI response."""
        params = {
            "model": kwargs.get("model", self.config.model),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        
        stream = await self.async_client.chat.completions.create(**params)
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaLLM(BaseLLM):
    """Ollama local model adapter.
    
    Purpose:
        Run models locally without API costs. Great for development,
        privacy-sensitive applications, or offline use.
    
    Example:
        ```python
        llm = OllamaLLM(LLMConfig(
            model="llama2",
            base_url="http://localhost:11434"
        ))
        
        response = llm.call("Write a haiku")
        ```
    
    Requires:
        - Ollama running locally
        - Model pulled: `ollama pull llama2`
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "http://localhost:11434"
    
    def call(self, prompt: str, **kwargs) -> str:
        """Synchronous Ollama call."""
        import requests
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": kwargs.get("model", self.config.model),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens)
            }
        }
        
        response = requests.post(url, json=payload, timeout=self.config.timeout)
        response.raise_for_status()
        return response.json()["response"]
    
    async def acall(self, prompt: str, **kwargs) -> str:
        """Asynchronous Ollama call."""
        import aiohttp
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": kwargs.get("model", self.config.model),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens)
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=self.config.timeout) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["response"]
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream Ollama response."""
        import aiohttp
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": kwargs.get("model", self.config.model),
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens)
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.content:
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]


class AnthropicLLM(BaseLLM):
    """Anthropic Claude adapter.
    
    Purpose:
        Access Claude models (Sonnet, Opus) via Anthropic API.
    
    Example:
        ```python
        llm = AnthropicLLM(LLMConfig(
            model="claude-3-opus-20240229",
            api_key=os.getenv("ANTHROPIC_API_KEY")
        ))
        ```
    
    Requires:
        ```bash
        pip install anthropic
        ```
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=config.api_key)
            self.async_client = anthropic.AsyncAnthropic(api_key=config.api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install: pip install anthropic")
    
    def call(self, prompt: str, **kwargs) -> str:
        """Synchronous Claude call."""
        response = self.client.messages.create(
            model=kwargs.get("model", self.config.model),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    async def acall(self, prompt: str, **kwargs) -> str:
        """Asynchronous Claude call."""
        response = await self.async_client.messages.create(
            model=kwargs.get("model", self.config.model),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream Claude response."""
        async with self.async_client.messages.stream(
            model=kwargs.get("model", self.config.model),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            async for text in stream.text_stream:
                yield text


__all__ = [
    "BaseLLM",
    "LLMConfig",
    "OpenAILLM",
    "OllamaLLM",
    "AnthropicLLM",
]
