from typing import Callable, List, Any, Optional


class Chain:
    def __init__(self, steps: List[Callable[[Any], Any]]):
        self.steps = steps

    def run(self, input_data: Any) -> Any:
        data = input_data
        for step in self.steps:
            data = step(data)
        return data


class LLMChain:
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
        if self.prompt_formatter:
            prompt = self.prompt_formatter.format(**inputs)
        else:
            prompt = str(inputs)
        result = self.llm(prompt)
        if hasattr(result, "__await__"):
            return await result
        return result

__all__ = ["Chain", "LLMChain"]
