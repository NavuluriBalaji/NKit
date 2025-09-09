import asyncio
from typing import Callable, Any
from .nbagents import is_async_function, run_sync_or_async


class LLMAdapter:
    """Abstract adapter for LLM callables."""

    def __call__(self, prompt: str) -> str:
        raise NotImplementedError()

    async def arun(self, prompt: str) -> str:
        raise NotImplementedError()


class CallableLLMAdapter(LLMAdapter):
    """Wraps a callable (sync or async) into a consistent adapter API."""

    def __init__(self, func: Callable[[str], Any]):
        self.func = func
        self._is_async = is_async_function(func)

    def __call__(self, prompt: str) -> str:
        if self._is_async:
            # run async function from sync context
            return asyncio.run(self.func(prompt))
        return self.func(prompt)

    async def arun(self, prompt: str) -> str:
        return await run_sync_or_async(self.func, prompt)
