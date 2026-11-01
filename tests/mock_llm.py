"""Scripted LLM used by the loop tests (no network, deterministic)."""

from __future__ import annotations

from elagent.llm import LLMResponse


class MockLLM:
    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.calls: list[list[dict]] = []
        self.model = "mock-model"

    def resolve_model(self) -> str:
        return self.model

    def chat(self, messages, **kwargs) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        text = self.script.pop(0) if self.script else '{"final":"done"}'
        return LLMResponse(text=text, prompt_tokens=12, completion_tokens=6, finish_reason="stop")

    def close(self) -> None:
        pass
