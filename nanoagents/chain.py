from typing import Callable, List, Any, Optional


class Chain:
    """Simple chain runner that applies a sequence of callables.

    Each step receives the previous step's output as input. Useful to model
    small deterministic pipelines (LLM -> tool -> postprocess).
    """

    def __init__(self, steps: List[Callable[[Any], Any]]):
        self.steps = steps

    def run(self, input_data: Any) -> Any:
        data = input_data
        for step in self.steps:
            data = step(data)
        return data


class LLMChain:
    """Very small LLMChain: runs a prompt template through an LLM callable.

    This mirrors a tiny part of LangChain's LLMChain for testing and simple
    composition. The `llm` can be any callable that accepts a single `prompt`
    string and returns a string. Optional `prompt_formatter` should provide a
    `format(**kwargs)` method.
    """

    def __init__(self, llm: Callable[[str], str], prompt_formatter: Optional[object] = None):
        self.llm = llm
        self.prompt_formatter = prompt_formatter

    def run(self, **inputs) -> str:
        if self.prompt_formatter:
            prompt = self.prompt_formatter.format(**inputs)
        else:
            prompt = str(inputs)
        return self.llm(prompt)

    async def arun(self, **inputs) -> str:
        # support async callables by awaiting if needed
        if self.prompt_formatter:
            prompt = self.prompt_formatter.format(**inputs)
        else:
            prompt = str(inputs)
        result = self.llm(prompt)
        # if result is awaitable, await it
        if hasattr(result, "__await__"):
            return await result
        return result
