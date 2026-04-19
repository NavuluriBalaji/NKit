"""LLM Provider Abstractions for NKit.

Provides unified interfaces to:
1. OllamaLLM — Local models (llama3, mistral)
2. OpenAILLM — Cloud models (gpt-4o)
3. AnthropicLLM — Cloud models (claude-3-opus)
4. OpenRouterLLM — Universal passthrough

Architecture:
    All providers implement `BaseLLM` consisting of:
    - `complete(prompt: str) -> str`
    - `stream(prompt: str) -> Iterator[str]`
    - `health_check() -> bool`
"""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional
import urllib.request
import urllib.error

class BaseLLM(ABC):
    """Abstract Base Class for all NKit LLM Providers."""
    
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Process the prompt and return the complete string."""
        pass

    @abstractmethod
    def stream(self, prompt: str) -> Iterator[str]:
        """Process the prompt and yield streaming output chunks."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verify the provider is reachable and authenticated."""
        pass


class OllamaLLM(BaseLLM):
    """Local model provider via Ollama."""
    
    def __init__(self, model: str = "llama3", timeout: int = 30, base_url: str = "http://localhost:11434"):
        self.model = model
        self.timeout = timeout
        self.base_url = base_url

    def _post(self, endpoint: str, payload: dict) -> urllib.request.Request:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(f"{self.base_url}{endpoint}", data=data)
        req.add_header('Content-Type', 'application/json')
        return req

    def complete(self, prompt: str) -> str:
        req = self._post("/api/generate", {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode())
                return result.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama complete failed: {e}")

    def stream(self, prompt: str) -> Iterator[str]:
        req = self._post("/api/generate", {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                for line in response:
                    if line:
                        chunk = json.loads(line.decode())
                        yield chunk.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama stream failed: {e}")

    def health_check(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False


class OpenAILLM(BaseLLM):
    """OpenAI API Provider with exponential backoff retries."""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 4096, api_key: Optional[str] = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("api_key must be provided directly or via OPENAI_API_KEY environment variable")

    def _request_with_retry(self, payload: dict, stream: bool = False, max_retries: int = 3):
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(payload).encode('utf-8'))
        req.add_header('Content-Type', 'application/json')
        req.add_header('Authorization', f'Bearer {self.api_key}')
        
        for attempt in range(max_retries):
            try:
                return urllib.request.urlopen(req, timeout=30)
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < max_retries - 1: # Rate limited
                    time.sleep(2 ** attempt) # Exponential backoff
                    continue
                raise RuntimeError(f"OpenAI API error: {e.read().decode()}")
            except Exception as e:
                raise RuntimeError(f"OpenAI connection error: {e}")
        raise RuntimeError("OpenAI API exhausted max retries")

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        with self._request_with_retry(payload) as response:
            data = json.loads(response.read().decode())
            return data["choices"][0]["message"]["content"]

    def stream(self, prompt: str) -> Iterator[str]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        with self._request_with_retry(payload, stream=True) as response:
            for line in response:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]

    def health_check(self) -> bool:
        req = urllib.request.Request("https://api.openai.com/v1/models")
        req.add_header('Authorization', f'Bearer {self.api_key}')
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False


class AnthropicLLM(BaseLLM):
    """Anthropic API Provider."""
    
    def __init__(self, model: str = "claude-3-opus-20240229", temperature: float = 0.7, max_tokens: int = 4096, api_key: Optional[str] = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("api_key must be provided directly or via ANTHROPIC_API_KEY")

    def _post(self, payload: dict) -> urllib.request.Request:
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(payload).encode('utf-8'))
        req.add_header('Content-Type', 'application/json')
        req.add_header('x-api-key', self.api_key)
        req.add_header('anthropic-version', '2023-06-01')
        return req

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        try:
            with urllib.request.urlopen(self._post(payload), timeout=30) as response:
                data = json.loads(response.read().decode())
                return data["content"][0]["text"]
        except Exception as e:
            raise RuntimeError(f"Anthropic complete failed: {e}")

    def stream(self, prompt: str) -> Iterator[str]:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        try:
            with urllib.request.urlopen(self._post(payload), timeout=30) as response:
                for line in response:
                    line = line.decode().strip()
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta":
                            yield data["delta"].get("text", "")
        except Exception as e:
            raise RuntimeError(f"Anthropic stream failed: {e}")

    def health_check(self) -> bool:
        # Anthropic doesn't have a simple models endpoint that works universally unauthenticated, 
        # so we ping a fast cheap endpoint or rely on initial auth testing.
        payload = {"model": "claude-3-haiku-20240307", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
        try:
            with urllib.request.urlopen(self._post(payload), timeout=5) as response:
                return response.status == 200
        except Exception:
            return False


class OpenRouterLLM(BaseLLM):
    """OpenRouter Provider for universal model passthrough."""
    
    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 4096, api_key: Optional[str] = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("api_key must be provided directly or via OPENROUTER_API_KEY")

    def _post(self, payload: dict) -> urllib.request.Request:
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(payload).encode('utf-8'))
        req.add_header('Content-Type', 'application/json')
        req.add_header('Authorization', f'Bearer {self.api_key}')
        req.add_header('HTTP-Referer', 'https://nkit.ai')
        req.add_header('X-Title', 'NKit Framework')
        return req

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        try:
            with urllib.request.urlopen(self._post(payload), timeout=30) as response:
                data = json.loads(response.read().decode())
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"OpenRouter complete failed: {e}")

    def stream(self, prompt: str) -> Iterator[str]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        try:
            with urllib.request.urlopen(self._post(payload), timeout=30) as response:
                for line in response:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data = json.loads(line[6:])
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
        except Exception as e:
            raise RuntimeError(f"OpenRouter stream failed: {e}")

    def health_check(self) -> bool:
        req = urllib.request.Request("https://openrouter.ai/api/v1/auth/key")
        req.add_header('Authorization', f'Bearer {self.api_key}')
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

class GeminiLLM(BaseLLM):
    """Google Gemini API Provider."""
    
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.7, max_tokens: int = 4096, api_key: Optional[str] = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("api_key must be provided directly or via GEMINI_API_KEY")

    def complete(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'))
        req.add_header('Content-Type', 'application/json')
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
                return data["candidates"][0]["content"]["parts"][0].get("text", "")
        except Exception as e:
            raise RuntimeError(f"Gemini complete failed: {e}")

    def stream(self, prompt: str) -> Iterator[str]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens
            }
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'))
        req.add_header('Content-Type', 'application/json')
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                for line in response:
                    line = line.decode().strip()
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "candidates" in data and len(data["candidates"]) > 0:
                            parts = data["candidates"][0]["content"].get("parts", [])
                            if parts:
                                yield parts[0].get("text", "")
        except Exception as e:
            raise RuntimeError(f"Gemini stream failed: {e}")

    def health_check(self) -> bool:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

__all__ = ["BaseLLM", "OllamaLLM", "OpenAILLM", "AnthropicLLM", "OpenRouterLLM", "GeminiLLM"]
