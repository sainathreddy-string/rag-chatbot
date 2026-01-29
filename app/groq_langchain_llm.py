from typing import List, Optional
from langchain_core.language_models.llms import LLM
from pydantic import PrivateAttr

from app.groq_client import GroqLLM


class GroqLangChainLLM(LLM):
    _groq: GroqLLM = PrivateAttr()

    class Config:
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True

    def __init__(self):
        super().__init__()
        object.__setattr__(self, "_groq", GroqLLM())

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        return self._groq.invoke(prompt)
