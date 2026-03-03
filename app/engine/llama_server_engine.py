"""llama-server 子进程引擎。

通过启动 llama-server 子进程并使用 HTTP 接口进行推理，
解耦于 llama-cpp-python 的发布节奏，支持最新 GGUF 架构（如 Qwen 3.5）。
"""

from __future__ import annotations

import json
import logging
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterator

import httpx

from app.engine.base import BaseEngine
from app.schemas import Message

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    """获取一个操作系统分配的空闲端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaServerEngine(BaseEngine):
    """通过 llama-server 子进程提供推理能力的引擎实现。

    启动流程：
    1. 自动分配空闲端口
    2. 以子进程方式启动 llama-server（日志输出到临时文件，避免管道死锁）
    3. 轮询 /health 端点等待服务就绪
    4. 通过 /v1/chat/completions 进行推理调用
    """

    def __init__(
        self,
        model_path: str,
        llama_server_path: str = "/home/admin/llama.cpp/build/bin/llama-server",
        n_ctx: int = 4096,
        n_threads: int = 8,
        startup_timeout: int = 60,
    ):
        self._port = _find_free_port()
        self._base_url = f"http://127.0.0.1:{self._port}"
        self._process: subprocess.Popen | None = None
        self._log_file: Path | None = None

        # 构建 llama-server 启动命令
        cmd = [
            llama_server_path,
            "--model", model_path,
            "--host", "127.0.0.1",
            "--port", str(self._port),
            "--ctx-size", str(n_ctx),
            "--threads", str(n_threads),
        ]

        # 使用临时日志文件代替 PIPE，避免缓冲区满导致死锁
        log_dir = Path(tempfile.gettempdir()) / "llama_server_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = log_dir / f"llama-server-{self._port}.log"
        log_handle = open(self._log_file, "w")

        logger.info("启动 llama-server: %s (日志: %s)", " ".join(cmd), self._log_file)
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        except FileNotFoundError:
            log_handle.close()
            raise RuntimeError(
                f"llama-server 二进制文件未找到: {llama_server_path}。"
                "请先编译 llama.cpp 或检查路径配置。"
            )
        finally:
            log_handle.close()

        # 轮询 /health 等待服务就绪
        self._wait_for_ready(startup_timeout)

    def _read_log_tail(self, max_chars: int = 500) -> str:
        """读取日志文件末尾内容，用于错误报告。"""
        if not self._log_file or not self._log_file.exists():
            return ""
        try:
            text = self._log_file.read_text(errors="replace")
            return text[-max_chars:] if len(text) > max_chars else text
        except OSError:
            return ""

    def _wait_for_ready(self, timeout: int) -> None:
        """轮询 llama-server 的 /health 端点，等待服务就绪。"""
        deadline = time.monotonic() + timeout
        health_url = f"{self._base_url}/health"

        while time.monotonic() < deadline:
            # 检查子进程是否意外退出
            if self._process and self._process.poll() is not None:
                log_tail = self._read_log_tail()
                raise RuntimeError(
                    f"llama-server 启动失败，退出码: {self._process.returncode}。"
                    f"日志: {log_tail}"
                )

            try:
                resp = httpx.get(health_url, timeout=2.0)
                if resp.status_code == 200:
                    logger.info("llama-server 已就绪，端口: %d", self._port)
                    return
            except httpx.ConnectError:
                pass
            except httpx.TimeoutException:
                pass

            time.sleep(0.5)

        # 超时，终止子进程
        log_tail = self._read_log_tail()
        self.stop()
        raise RuntimeError(
            f"llama-server 启动超时（{timeout}s）。日志: {log_tail}"
        )

    def _to_payload(self, messages: list[Message]) -> list[dict[str, str]]:
        """将 Message 列表转换为 OpenAI 兼容格式。"""
        return [{"role": m.role, "content": m.content} for m in messages]

    def chat(self, messages: list[Message], **kwargs) -> str:
        """非流式聊天推理。"""
        payload = {
            "messages": self._to_payload(messages),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": False,
        }

        try:
            resp = httpx.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.ConnectError:
            raise RuntimeError("llama-server 连接失败，子进程可能已退出")
        except httpx.TimeoutException:
            raise RuntimeError("llama-server 推理超时（120s）")
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"llama-server 返回了意外的响应格式: {exc}")

    def stream_chat(self, messages: list[Message], **kwargs) -> Iterator[str]:
        """流式聊天推理，通过 SSE 逐块返回内容。"""
        payload = {
            "messages": self._to_payload(messages),
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512),
            "top_p": kwargs.get("top_p", 0.9),
            "stream": True,
        }

        try:
            with httpx.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=120.0,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    # SSE 格式: "data: {...}" 或 "data: [DONE]"
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except httpx.ConnectError:
            raise RuntimeError("llama-server 连接失败，子进程可能已退出")
        except httpx.TimeoutException:
            raise RuntimeError("llama-server 流式推理超时（120s）")

    def stop(self) -> None:
        """优雅终止 llama-server 子进程。

        流程：SIGTERM -> 等待 5s -> SIGKILL（如仍存活）
        """
        if self._process is None or self._process.poll() is not None:
            self._process = None
            return

        logger.info("正在停止 llama-server (pid=%d)...", self._process.pid)
        try:
            self._process.send_signal(signal.SIGTERM)
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("llama-server 未响应 SIGTERM，发送 SIGKILL")
            self._process.kill()
            self._process.wait(timeout=3)
        except OSError:
            pass
        finally:
            self._process = None

    def __del__(self) -> None:
        """防御性析构，确保子进程被清理。"""
        try:
            self.stop()
        except Exception:
            pass
