from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from app.schemas import Message


class BaseEngine(ABC):
    @abstractmethod
    def chat(self, messages: list[Message], **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def stream_chat(self, messages: list[Message], **kwargs) -> Iterator[str]:
        raise NotImplementedError

    def stop(self) -> None:
        """停止引擎并释放资源，默认空实现。子类可按需覆盖。"""
