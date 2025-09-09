from typing import Dict


class PromptTemplate:
    """Very small prompt templating helper."""

    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs: Dict[str, str]) -> str:
        return self.template.format(**kwargs)
