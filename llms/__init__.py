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
    - `count_tokens(prompt: str) -> int` (production tracking)

Features:
    - Rate limiting (RateLimiter)
    - Token counting (TokenCounter)
    - Cost tracking
    - Exponential backoff on 429 errors
"""

import os
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional
import urllib.request
import urllib.error

from .rate_limiter import RateLimiter, TokenCounter, ProviderLimits

class BaseLLM(ABC):
    """Abstract Base Class for all NKit LLM Providers."""
    
    def __init__(self, enable_rate_limiting: bool = True, track_tokens: bool = True):
        """Initialize LLM provider.
        
        Args:
            enable_rate_limiting: Enable automatic rate limiting (default True)
            track_tokens: Track token usage for cost monitoring (default True)
        """
        self.enable_rate_limiting = enable_rate_limiting
        self.track_tokens = track_tokens
        self.rate_limiter: Optional[RateLimiter] = None
        self.total_tokens = 0  # For session tracking
        self.total_cost = 0.0  # For session tracking
    
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Process the prompt and return the complete string."""
        pass

    def __call__(self, prompt: str) -> str:
        """Makes the object callable, automatically routing to .complete()."""
        return self.complete(prompt)

    @abstractmethod
    def stream(self, prompt: str) -> Iterator[str]:
        """Process the prompt and yield streaming output chunks."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verify the provider is reachable and authenticated."""
        pass
    
    def count_tokens(self, prompt: str) -> int:
        """Estimate tokens for a prompt (can be overridden by providers).
        
        Returns:
            Estimated token count
        """
        return TokenCounter.count_tokens(prompt, model=getattr(self, 'model', 'gpt-4o'))
    
    def get_token_stats(self) -> Dict[str, Any]:
        """Get session token usage statistics.
        
        Returns:
            Dict with total_tokens and total_cost
        """
        return {
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "tokens_per_minute": self.rate_limiter.tokens_per_minute if self.rate_limiter else None,
        }


class OllamaLLM(BaseLLM):
    """Local model provider via Ollama (no rate limiting needed)."""
    
    def __init__(self, model: str = "llama3", timeout: int = 30, base_url: str = "http://localhost:11434",
                 enable_rate_limiting: bool = False, track_tokens: bool = True):
        super().__init__(enable_rate_limiting=False, track_tokens=track_tokens)  # Ollama is local, no rate limits
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
                response_text = result.get("response", "")
                
                # Track token usage if available
                if self.track_tokens and "eval_count" in result:
                    self.total_tokens += result.get("eval_count", 0) + result.get("prompt_eval_count", 0)
                
                return response_text
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
    """OpenAI API Provider with exponential backoff retries and rate limiting."""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.7, max_tokens: int = 4096, 
                 api_key: Optional[str] = None, enable_rate_limiting: bool = True, 
                 track_tokens: bool = True):
        super().__init__(enable_rate_limiting=enable_rate_limiting, track_tokens=track_tokens)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("api_key must be provided directly or via OPENAI_API_KEY environment variable")
        
        # Initialize rate limiter with OpenAI limits
        if self.enable_rate_limiting:
            limits = ProviderLimits.OPENAI
            self.rate_limiter = RateLimiter(
                tokens_per_minute=limits["tokens_per_minute"],
                requests_per_minute=limits["requests_per_minute"]
            )
        
        # Pricing for token tracking (as of 2025)
        self.input_price_per_1k = 0.005  # $0.005 per 1K input tokens
        self.output_price_per_1k = 0.015  # $0.015 per 1K output tokens

    def _request_with_retry(self, payload: dict, stream: bool = False, max_retries: int = 3):
        # Rate limiting
        if self.rate_limiter:
            estimated_tokens = TokenCounter.count_messages(payload.get("messages", []), self.model)
            self.rate_limiter.wait_if_needed(estimated_tokens, reason="OpenAI API call")
        
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(payload).encode('utf-8'))
        req.add_header('Content-Type', 'application/json')
        req.add_header('Authorization', f'Bearer {self.api_key}')
        
        for attempt in range(max_retries):
            try:
                response = urllib.request.urlopen(req, timeout=30)
                if self.rate_limiter:
                    self.rate_limiter.reset_backoff()
                return response
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < max_retries - 1:  # Rate limited
                    if self.rate_limiter:
                        self.rate_limiter.handle_rate_limit_error()
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
            result = data["choices"][0]["message"]["content"]
            
            # Track token usage
            if self.track_tokens and "usage" in data:
                input_tokens = data["usage"].get("prompt_tokens", 0)
                output_tokens = data["usage"].get("completion_tokens", 0)
                self.total_tokens += input_tokens + output_tokens
                cost = TokenCounter.estimate_cost(input_tokens, output_tokens, 
                                                  self.input_price_per_1k, self.output_price_per_1k)
                self.total_cost += cost
            
            return result

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
    """Anthropic API Provider with rate limiting."""
    
    def __init__(self, model: str = "claude-3-opus-20240229", temperature: float = 0.7, 
                 max_tokens: int = 4096, api_key: Optional[str] = None, 
                 enable_rate_limiting: bool = True, track_tokens: bool = True):
        super().__init__(enable_rate_limiting=enable_rate_limiting, track_tokens=track_tokens)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("api_key must be provided directly or via ANTHROPIC_API_KEY")
        
        # Initialize rate limiter with Anthropic limits
        if self.enable_rate_limiting:
            limits = ProviderLimits.ANTHROPIC
            self.rate_limiter = RateLimiter(
                tokens_per_minute=limits["tokens_per_minute"],
                requests_per_minute=limits["requests_per_minute"]
            )
        
        # Pricing for token tracking
        self.input_price_per_1k = 0.003  # $0.003 per 1K input tokens
        self.output_price_per_1k = 0.015  # $0.015 per 1K output tokens

    def _post(self, payload: dict) -> urllib.request.Request:
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(payload).encode('utf-8'))
        req.add_header('Content-Type', 'application/json')
        req.add_header('x-api-key', self.api_key)
        req.add_header('anthropic-version', '2023-06-01')
        return req

    def complete(self, prompt: str) -> str:
        # Rate limiting
        if self.rate_limiter:
            estimated_tokens = TokenCounter.count_tokens(prompt, self.model)
            self.rate_limiter.wait_if_needed(estimated_tokens, reason="Anthropic API call")
        
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        try:
            with urllib.request.urlopen(self._post(payload), timeout=30) as response:
                data = json.loads(response.read().decode())
                result = data["content"][0]["text"]
                
                # Track token usage
                if self.track_tokens and "usage" in data:
                    input_tokens = data["usage"].get("input_tokens", 0)
                    output_tokens = data["usage"].get("output_tokens", 0)
                    self.total_tokens += input_tokens + output_tokens
                    cost = TokenCounter.estimate_cost(input_tokens, output_tokens,
                                                      self.input_price_per_1k, self.output_price_per_1k)
                    self.total_cost += cost
                
                if self.rate_limiter:
                    self.rate_limiter.reset_backoff()
                
                return result
        except urllib.error.HTTPError as e:
            if e.code == 429 and self.rate_limiter:
                self.rate_limiter.handle_rate_limit_error()
            raise RuntimeError(f"Anthropic complete failed: {e}")
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
        payload = {"model": "claude-3-haiku-20240307", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
        try:
            with urllib.request.urlopen(self._post(payload), timeout=5) as response:
                return response.status == 200
        except Exception:
            return False


class OpenRouterLLM(BaseLLM):
    """OpenRouter Provider for universal model passthrough with rate limiting."""
    
    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 4096, 
                 api_key: Optional[str] = None, enable_rate_limiting: bool = True,
                 track_tokens: bool = True):
        super().__init__(enable_rate_limiting=enable_rate_limiting, track_tokens=track_tokens)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("api_key must be provided directly or via OPENROUTER_API_KEY")
        
        # Initialize rate limiter with OpenRouter limits
        if self.enable_rate_limiting:
            limits = ProviderLimits.OPENROUTER
            self.rate_limiter = RateLimiter(
                tokens_per_minute=limits["tokens_per_minute"],
                requests_per_minute=limits["requests_per_minute"]
            )
        
        # Default pricing (varies by model)
        self.input_price_per_1k = 0.005
        self.output_price_per_1k = 0.015

    def _post(self, payload: dict) -> urllib.request.Request:
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(payload).encode('utf-8'))
        req.add_header('Content-Type', 'application/json')
        req.add_header('Authorization', f'Bearer {self.api_key}')
        req.add_header('HTTP-Referer', 'https://nkit.ai')
        req.add_header('X-Title', 'NKit Framework')
        return req

    def complete(self, prompt: str) -> str:
        # Rate limiting
        if self.rate_limiter:
            estimated_tokens = TokenCounter.count_tokens(prompt, self.model)
            self.rate_limiter.wait_if_needed(estimated_tokens, reason="OpenRouter API call")
        
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
                result = data["choices"][0]["message"]["content"]
                
                # Track token usage
                if self.track_tokens and "usage" in data:
                    input_tokens = data["usage"].get("prompt_tokens", 0)
                    output_tokens = data["usage"].get("completion_tokens", 0)
                    self.total_tokens += input_tokens + output_tokens
                    cost = TokenCounter.estimate_cost(input_tokens, output_tokens,
                                                      self.input_price_per_1k, self.output_price_per_1k)
                    self.total_cost += cost
                
                if self.rate_limiter:
                    self.rate_limiter.reset_backoff()
                
                return result
        except urllib.error.HTTPError as e:
            if e.code == 429 and self.rate_limiter:
                self.rate_limiter.handle_rate_limit_error()
            raise RuntimeError(f"OpenRouter complete failed: {e}")
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
    """Google Gemini API Provider with rate limiting."""
    
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.7, max_tokens: int = 4096, 
                 api_key: Optional[str] = None, enable_rate_limiting: bool = True,
                 track_tokens: bool = True):
        super().__init__(enable_rate_limiting=enable_rate_limiting, track_tokens=track_tokens)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("api_key must be provided directly or via GEMINI_API_KEY")
        
        # Initialize rate limiter with Gemini limits
        if self.enable_rate_limiting:
            limits = ProviderLimits.GEMINI
            self.rate_limiter = RateLimiter(
                tokens_per_minute=limits["tokens_per_minute"],
                requests_per_minute=limits["requests_per_minute"]
            )
        
        # Pricing for token tracking
        self.input_price_per_1k = 0.000075  # Free tier or very cheap
        self.output_price_per_1k = 0.0003

    def complete(self, prompt: str) -> str:
        # Rate limiting
        if self.rate_limiter:
            estimated_tokens = TokenCounter.count_tokens(prompt, self.model)
            self.rate_limiter.wait_if_needed(estimated_tokens, reason="Gemini API call")
        
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
                result = data["candidates"][0]["content"]["parts"][0].get("text", "")
                
                # Track token usage if available
                if self.track_tokens and "usageMetadata" in data:
                    input_tokens = data["usageMetadata"].get("promptTokenCount", 0)
                    output_tokens = data["usageMetadata"].get("candidatesTokenCount", 0)
                    self.total_tokens += input_tokens + output_tokens
                    cost = TokenCounter.estimate_cost(input_tokens, output_tokens,
                                                      self.input_price_per_1k, self.output_price_per_1k)
                    self.total_cost += cost
                
                if self.rate_limiter:
                    self.rate_limiter.reset_backoff()
                
                return result
        except urllib.error.HTTPError as e:
            if e.code == 429 and self.rate_limiter:
                self.rate_limiter.handle_rate_limit_error()
            raise RuntimeError(f"Gemini complete failed: {e}")
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

class LMStudioLLM(BaseLLM):
    """LM Studio Provider utilizing native HTTP endpoints with rate limiting."""
    
    def __init__(self, model: str = "local-model", temperature: float = 0.7, max_tokens: int = 4096, 
                 base_url: str = "http://localhost:1234", enable_rate_limiting: bool = False,
                 track_tokens: bool = True):
        """Initialize the LM Studio client.
        
        Args:
            model: The name of the loaded model. LM Studio ignores this if only one model is loaded.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            base_url: The base URL and port where LM Studio server is running (default: http://localhost:1234).
            enable_rate_limiting: Enable rate limiting (default False, local unlimited)
            track_tokens: Track token usage (default True)
        """
        super().__init__(enable_rate_limiting=False, track_tokens=track_tokens)  # Local, no rate limits needed
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")

    def _post(self, payload: dict) -> urllib.request.Request:
        # We leverage the universally compatible OpenAI endpoint that LM Studio explicitly exposes natively
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=json.dumps(payload).encode('utf-8'))
        req.add_header('Content-Type', 'application/json')
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
            with urllib.request.urlopen(self._post(payload), timeout=120) as response:
                data = json.loads(response.read().decode())
                result = data["choices"][0]["message"]["content"]
                
                # Track token usage if available
                if self.track_tokens and "usage" in data:
                    input_tokens = data["usage"].get("prompt_tokens", 0)
                    output_tokens = data["usage"].get("completion_tokens", 0)
                    self.total_tokens += input_tokens + output_tokens
                
                return result
        except Exception as e:
            raise RuntimeError(f"LM Studio complete failed: Are you sure the server is bridging to {self.base_url}? Error: {e}")

    def stream(self, prompt: str) -> Iterator[str]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        try:
            with urllib.request.urlopen(self._post(payload), timeout=120) as response:
                for line in response:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data = json.loads(line[6:])
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
        except Exception as e:
            raise RuntimeError(f"LM Studio stream failed: {e}")

    def health_check(self) -> bool:
        # Ping the models endpoint to verify server is alive
        req = urllib.request.Request(f"{self.base_url}/v1/models")
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

__all__ = ["BaseLLM", "OllamaLLM", "OpenAILLM", "AnthropicLLM", "OpenRouterLLM", "GeminiLLM", "LMStudioLLM", 
           "RateLimiter", "TokenCounter", "ProviderLimits"]
