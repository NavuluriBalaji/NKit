"""Rate limiting and token counting for production LLM deployments.

Provides:
1. RateLimiter: Token/request rate limiting with sliding window
2. TokenCounter: Approximate token counting for cost tracking
3. Integration points for LLM providers

Design:
    - Thread-safe rate limiting
    - Exponential backoff on rate limit (429) errors
    - Token estimation for cost calculations
    - Per-provider configuration
"""

import time
import threading
from typing import Optional
from collections import deque


class TokenCounter:
    """Estimate token counts for cost tracking and rate limiting.
    
    Purpose:
        Most LLM APIs charge by token. This provides rough estimates
        before calling the API, plus exact counts from API responses.
    
    Patterns:
        - Estimate before API call: count_tokens(prompt)
        - Exact count after: extract from response metadata
        - Cost calculation: count * rate_per_token
    
    Accuracy:
        - ±15-20% error (acceptable for budgeting)
        - Exact counts available from API responses
        - Model-specific overrides supported
    """
    
    # Token estimation ratios (chars per token varies by model)
    # These are rough approximations for budget calculations
    CHARS_PER_TOKEN = {
        "gpt-4o": 4.0,
        "gpt-4": 4.0,
        "gpt-3.5-turbo": 4.0,
        "claude-3-opus": 3.5,
        "claude-3-sonnet": 3.5,
        "claude-3-haiku": 3.5,
        "llama3": 3.0,
        "mistral": 3.0,
        "gemini": 3.5,
    }
    
    @staticmethod
    def count_tokens(text: str, model: str = "gpt-4o") -> int:
        """Estimate token count for text.
        
        Args:
            text: Text to count
            model: Model name for estimation accuracy
            
        Returns:
            Estimated token count
            
        Note:
            For exact counts, use provider-specific token counters:
            - OpenAI: tiktoken.encoding_for_model(model).encode(text)
            - Anthropic: Use API response usage metadata
            - Local: Use provider's tokenizer
        """
        chars_per_token = TokenCounter.CHARS_PER_TOKEN.get(model, 3.5)
        # Account for whitespace/structure that takes tokens
        return max(1, int(len(text) / chars_per_token) + 5)
    
    @staticmethod
    def count_messages(messages: list, model: str = "gpt-4o") -> int:
        """Estimate token count for message list.
        
        Args:
            messages: List of message dicts (role, content)
            model: Model name
            
        Returns:
            Estimated token count (includes formatting overhead)
        """
        total = 0
        # Each message has ~4 token overhead for formatting
        total += len(messages) * 4
        
        # Count content tokens
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    total += TokenCounter.count_tokens(content, model)
        
        return total
    
    @staticmethod
    def estimate_cost(input_tokens: int, output_tokens: int, 
                     input_rate: float, output_rate: float) -> float:
        """Estimate API cost.
        
        Args:
            input_tokens: Prompt tokens
            output_tokens: Completion tokens
            input_rate: Cost per 1000 input tokens (e.g., 0.03 for GPT-4o)
            output_rate: Cost per 1000 output tokens (e.g., 0.06 for GPT-4o)
            
        Returns:
            Estimated cost in dollars
        """
        return (input_tokens / 1000 * input_rate) + (output_tokens / 1000 * output_rate)


class RateLimiter:
    """Token and request rate limiting with sliding window.
    
    Purpose:
        Prevents rate limit errors from LLM APIs by:
        - Tracking request/token rates
        - Delaying requests to stay under limits
        - Exponential backoff on rate limit (429) errors
        - Per-provider configuration
    
    Limits (typical):
        - OpenAI: 3,500 RPM, 200k TPM for GPT-4o
        - Anthropic: 50k TPM standard, 1M for batches
        - Local (Ollama): No rate limits
    
    Design:
        - Thread-safe with locks
        - Sliding window tracking (not fixed windows)
        - Graceful degradation if limits exceeded
        - Automatic backoff with jitter
    
    Example:
        ```python
        limiter = RateLimiter(
            tokens_per_minute=200000,
            requests_per_minute=3500
        )
        
        limiter.wait_if_needed(estimated_tokens=1500)
        # Makes LLM call
        limiter.record_usage(tokens=1500)
        ```
    """
    
    def __init__(self, tokens_per_minute: Optional[int] = None,
                 requests_per_minute: Optional[int] = None,
                 enable_backoff: bool = True):
        """Initialize rate limiter.
        
        Args:
            tokens_per_minute: Max tokens/min (None = unlimited)
            requests_per_minute: Max requests/min (None = unlimited)
            enable_backoff: Auto-backoff on 429 errors
        """
        self.tokens_per_minute = tokens_per_minute
        self.requests_per_minute = requests_per_minute
        self.enable_backoff = enable_backoff
        
        # Sliding window tracking (last 60 seconds)
        self.token_history = deque()  # (timestamp, token_count)
        self.request_history = deque()  # (timestamp, 1)
        
        self.lock = threading.Lock()
        self.backoff_until = 0  # Exponential backoff deadline
        self.backoff_multiplier = 1.0
    
    def _cleanup_old_entries(self, now: float) -> None:
        """Remove entries older than 60 seconds."""
        cutoff = now - 60
        
        while self.token_history and self.token_history[0][0] < cutoff:
            self.token_history.popleft()
        
        while self.request_history and self.request_history[0][0] < cutoff:
            self.request_history.popleft()
    
    def wait_if_needed(self, estimated_tokens: int = 0, reason: str = "") -> None:
        """Block until request can be sent within rate limits.
        
        Args:
            estimated_tokens: Tokens the request will likely use
            reason: Debug reason for logging
        """
        with self.lock:
            now = time.time()
            
            # Honor exponential backoff
            if now < self.backoff_until:
                wait_time = self.backoff_until - now
                time.sleep(wait_time)
                now = time.time()
            
            self._cleanup_old_entries(now)
            
            # Check token rate
            if self.tokens_per_minute:
                total_tokens = sum(count for _, count in self.token_history)
                if total_tokens + estimated_tokens > self.tokens_per_minute:
                    # Calculate wait time for sliding window
                    oldest_token_time = self.token_history[0][0] if self.token_history else now
                    wait_until = oldest_token_time + 60
                    wait_time = max(0, wait_until - now)
                    if wait_time > 0:
                        time.sleep(wait_time)
            
            # Check request rate
            if self.requests_per_minute:
                request_count = len(self.request_history)
                if request_count >= self.requests_per_minute:
                    oldest_request_time = self.request_history[0][0]
                    wait_until = oldest_request_time + 60
                    wait_time = max(0, wait_until - time.time())
                    if wait_time > 0:
                        time.sleep(wait_time)
    
    def record_usage(self, tokens: int = 0) -> None:
        """Record actual token/request usage.
        
        Args:
            tokens: Tokens actually used
        """
        with self.lock:
            now = time.time()
            if tokens > 0:
                self.token_history.append((now, tokens))
            self.request_history.append((now, 1))
    
    def handle_rate_limit_error(self) -> None:
        """Handle 429 (rate limit) error with exponential backoff.
        
        Implements exponential backoff:
        - 1st attempt: wait 1s
        - 2nd attempt: wait 2s
        - 3rd attempt: wait 4s
        - Max: 60s
        """
        if not self.enable_backoff:
            return
        
        with self.lock:
            # Exponential backoff: 2^attempt, max 60s
            wait_time = min(2 ** max(0, self.backoff_multiplier - 1), 60)
            self.backoff_until = time.time() + wait_time
            self.backoff_multiplier += 1
    
    def reset_backoff(self) -> None:
        """Reset backoff (call after successful request)."""
        with self.lock:
            self.backoff_multiplier = 1.0
            self.backoff_until = 0


class ProviderLimits:
    """Pre-configured rate limits for popular providers."""
    
    OPENAI = {"tokens_per_minute": 200000, "requests_per_minute": 3500}
    ANTHROPIC = {"tokens_per_minute": 50000, "requests_per_minute": 50}
    OPENROUTER = {"tokens_per_minute": 10000, "requests_per_minute": 100}
    OLLAMA = {"tokens_per_minute": None, "requests_per_minute": None}  # Local, no limits
    GEMINI = {"tokens_per_minute": 32000, "requests_per_minute": 100}
