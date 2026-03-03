from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

from app.config_store import ConfigStore
from app.model_manager import ModelManager
from app.schemas import Message


class ChatService:
    def __init__(self, config_store: ConfigStore, model_manager: ModelManager):
        self.config_store = config_store
        self.model_manager = model_manager

    def now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def resolve_model(self, model: str | None) -> str:
        cfg = self.config_store.get()
        return model or cfg.default_model

    def _merge_options(self, options: dict | None, overrides: dict | None = None) -> dict:
        cfg = self.config_store.get()
        result = cfg.generation_defaults.model_dump()
        if options:
            result.update({k: v for k, v in options.items() if v is not None})
        if overrides:
            result.update({k: v for k, v in overrides.items() if v is not None})
        return result

    def chat(self, model: str | None, messages: list[Message], options: dict | None = None, overrides: dict | None = None) -> tuple[str, str]:
        target = self.resolve_model(model)
        params = self._merge_options(options, overrides)
        engine = self.model_manager.get_engine(target)
        text = engine.chat(messages, **params)
        return target, text

    def stream_chat(
        self,
        model: str | None,
        messages: list[Message],
        options: dict | None = None,
        overrides: dict | None = None,
    ) -> tuple[str, Iterator[str]]:
        target = self.resolve_model(model)
        params = self._merge_options(options, overrides)
        engine = self.model_manager.get_engine(target)
        return target, engine.stream_chat(messages, **params)
