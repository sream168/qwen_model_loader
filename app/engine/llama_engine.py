from __future__ import annotations

from typing import Iterator

from app.engine.base import BaseEngine
from app.schemas import Message


class LlamaCppEngine(BaseEngine):
    def __init__(self, model_path: str):
        try:
            import llama_cpp
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError("llama-cpp-python is not installed") from exc

        try:
            self._llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=8,
                n_batch=512,
                verbose=False,
            )
        except ValueError as exc:
            msg = str(exc)
            if "Failed to load model from file" in msg:
                raise RuntimeError(
                    "Model load failed. This GGUF may require a newer llama.cpp/llama-cpp-python runtime "
                    f"(current llama-cpp-python={getattr(llama_cpp, '__version__', 'unknown')}). "
                    "Please upgrade llama-cpp-python and retry."
                ) from exc
            raise

    def _to_payload(self, messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def chat(self, messages: list[Message], **kwargs) -> str:
        response = self._llm.create_chat_completion(
            messages=self._to_payload(messages),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 512),
            top_p=kwargs.get("top_p", 0.9),
            stream=False,
        )
        return response["choices"][0]["message"]["content"]

    def stream_chat(self, messages: list[Message], **kwargs) -> Iterator[str]:
        chunks = self._llm.create_chat_completion(
            messages=self._to_payload(messages),
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 512),
            top_p=kwargs.get("top_p", 0.9),
            stream=True,
        )
        for chunk in chunks:
            delta = chunk["choices"][0].get("delta", {}).get("content")
            if delta:
                yield delta
