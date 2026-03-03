from __future__ import annotations

from pathlib import Path

import pytest

from app.config_store import ConfigStore
from app.model_manager import ModelManager
from app.schemas import ConfigPatch, Message
from app.service import ChatService


class FakeEngine:
    """模拟引擎，返回固定响应用于快速测试。"""

    def __init__(self):
        self.stopped = False

    def chat(self, messages: list[Message], **kwargs):
        return "ok"

    def stream_chat(self, messages: list[Message], **kwargs):
        yield "o"
        yield "k"

    def stop(self):
        self.stopped = True


class FakeModelManager:
    def __init__(self):
        self.last_model = None
        self.engine = FakeEngine()

    def get_engine(self, model_name: str):
        self.last_model = model_name
        return self.engine


# ─── ConfigStore 测试 ───


def test_config_store_defaults_and_update(tmp_path: Path):
    store = ConfigStore(tmp_path / "config.json")
    cfg = store.get()

    assert cfg.default_model == "Qwen3.5-0.8B"
    assert "Qwen3.5-0.8B" in cfg.model_mapping
    # 验证新增字段的默认值
    assert cfg.engine_type == "llama_server"
    assert cfg.n_ctx == 4096
    assert cfg.n_threads == 8
    assert cfg.startup_timeout == 60

    updated = store.update(
        ConfigPatch(
            model_mapping={
                "Qwen3.5-0.8B": "Qwen3.5-0.8B-Q4_K_M.gguf",
                "custom": "custom.gguf",
            },
            default_model="custom",
            generation_defaults={"temperature": 0.2},
        )
    )
    assert updated.default_model == "custom"
    assert updated.generation_defaults.temperature == 0.2


def test_config_store_rejects_invalid_default_model(tmp_path: Path):
    store = ConfigStore(tmp_path / "config.json")
    with pytest.raises(ValueError):
        store.update(ConfigPatch(default_model="not-exist"))


def test_config_store_update_engine_type(tmp_path: Path):
    """验证可以通过 ConfigPatch 更新引擎类型和相关参数。"""
    store = ConfigStore(tmp_path / "config.json")
    updated = store.update(
        ConfigPatch(engine_type="llama_cpp", n_ctx=8192, n_threads=4)
    )
    assert updated.engine_type == "llama_cpp"
    assert updated.n_ctx == 8192
    assert updated.n_threads == 4


# ─── ChatService 测试 ───


def test_chat_service_uses_default_model_and_merges_options(tmp_path: Path):
    store = ConfigStore(tmp_path / "config.json")
    manager = FakeModelManager()
    service = ChatService(store, manager)

    model, text = service.chat(
        model=None,
        messages=[Message(role="user", content="hi")],
        options={"temperature": 0.4},
        overrides={"max_tokens": 128},
    )

    assert model == "Qwen3.5-0.8B"
    assert manager.last_model == "Qwen3.5-0.8B"
    assert text == "ok"


def test_chat_service_stream(tmp_path: Path):
    store = ConfigStore(tmp_path / "config.json")
    manager = FakeModelManager()
    service = ChatService(store, manager)

    model, chunks = service.stream_chat(
        model="Qwen3.5-0.8B",
        messages=[Message(role="user", content="hi")],
    )
    assert model == "Qwen3.5-0.8B"
    assert "".join(chunks) == "ok"


# ─── ModelManager 测试 ───


def test_model_manager_resolve_path_and_cache(tmp_path: Path, monkeypatch):
    model_file = tmp_path / "Qwen3.5-0.8B-Q4_K_M.gguf"
    model_file.write_text("fake")

    store = ConfigStore(tmp_path / "config.json")
    store.update(
        ConfigPatch(
            model_dir=str(tmp_path),
            model_mapping={"Qwen3.5-0.8B": model_file.name},
            default_model="Qwen3.5-0.8B",
        )
    )

    created = []

    class _FakeEngine:
        def __init__(self, **kwargs):
            created.append(kwargs.get("model_path"))

        def stop(self):
            pass

    monkeypatch.setattr(
        "app.model_manager.ModelManager._create_engine",
        lambda self, path: _FakeEngine(model_path=path),
    )

    manager = ModelManager(store)
    first = manager.get_engine("Qwen3.5-0.8B")
    second = manager.get_engine("Qwen3.5-0.8B")

    assert first is second
    assert len(created) == 1
    assert created[0] == str(model_file)


def test_model_manager_missing_file(tmp_path: Path, monkeypatch):
    store = ConfigStore(tmp_path / "config.json")
    store.update(
        ConfigPatch(
            model_dir=str(tmp_path),
            model_mapping={"Qwen3.5-0.8B": "missing.gguf"},
            default_model="Qwen3.5-0.8B",
        )
    )

    monkeypatch.setattr(
        "app.model_manager.ModelManager._create_engine",
        lambda self, path: None,
    )

    manager = ModelManager(store)
    with pytest.raises(FileNotFoundError):
        manager.get_engine("Qwen3.5-0.8B")


def test_model_manager_eviction_calls_stop(tmp_path: Path, monkeypatch):
    """验证缓存淘汰时会调用被淘汰引擎的 stop() 方法。"""
    model_a = tmp_path / "model_a.gguf"
    model_b = tmp_path / "model_b.gguf"
    model_a.write_text("fake")
    model_b.write_text("fake")

    store = ConfigStore(tmp_path / "config.json")
    store.update(
        ConfigPatch(
            model_dir=str(tmp_path),
            model_mapping={"model-a": model_a.name, "model-b": model_b.name},
            default_model="model-a",
            max_loaded_models=1,
        )
    )

    engines: list[FakeEngine] = []

    def _create(self, path: str):
        eng = FakeEngine()
        engines.append(eng)
        return eng

    monkeypatch.setattr(
        "app.model_manager.ModelManager._create_engine", _create,
    )

    manager = ModelManager(store)
    manager.get_engine("model-a")
    assert len(engines) == 1
    assert not engines[0].stopped

    # 加载第二个模型，应触发淘汰第一个
    manager.get_engine("model-b")
    assert len(engines) == 2
    assert engines[0].stopped  # model-a 的引擎已被停止
    assert not engines[1].stopped  # model-b 仍然活跃


def test_model_manager_shutdown(tmp_path: Path, monkeypatch):
    """验证 shutdown() 方法停止所有引擎并清空缓存。"""
    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake")

    store = ConfigStore(tmp_path / "config.json")
    store.update(
        ConfigPatch(
            model_dir=str(tmp_path),
            model_mapping={"test-model": model_file.name},
            default_model="test-model",
        )
    )

    engine = FakeEngine()
    monkeypatch.setattr(
        "app.model_manager.ModelManager._create_engine",
        lambda self, path: engine,
    )

    manager = ModelManager(store)
    manager.get_engine("test-model")
    assert not engine.stopped

    manager.shutdown()
    assert engine.stopped
    assert len(manager._cache) == 0


# ─── LlamaServerEngine 单元测试（Mock HTTP） ───


def test_llama_server_engine_chat(monkeypatch):
    """验证 LlamaServerEngine.chat() 正确调用 HTTP 接口。"""
    import app.engine.llama_server_engine as lse

    # Mock _find_free_port
    monkeypatch.setattr(lse, "_find_free_port", lambda: 9999)

    # 跳过子进程启动和健康检查
    class MockEngine(lse.LlamaServerEngine):
        def __init__(self):
            self._port = 9999
            self._base_url = "http://127.0.0.1:9999"
            self._process = None

    engine = MockEngine()

    # Mock httpx.post
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": "hello"}}
                ]
            }

    import httpx
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: FakeResponse())

    result = engine.chat([Message(role="user", content="hi")])
    assert result == "hello"


def test_llama_server_engine_stream_chat(monkeypatch):
    """验证 LlamaServerEngine.stream_chat() 正确解析 SSE 流。"""
    import app.engine.llama_server_engine as lse

    monkeypatch.setattr(lse, "_find_free_port", lambda: 9999)

    class MockEngine(lse.LlamaServerEngine):
        def __init__(self):
            self._port = 9999
            self._base_url = "http://127.0.0.1:9999"
            self._process = None

    engine = MockEngine()

    # 模拟 SSE 流
    sse_lines = [
        'data: {"choices":[{"delta":{"content":"hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        "data: [DONE]",
    ]

    class FakeStreamResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield from sse_lines

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    import httpx
    monkeypatch.setattr(
        httpx, "stream", lambda *args, **kwargs: FakeStreamResponse()
    )

    chunks = list(engine.stream_chat([Message(role="user", content="hi")]))
    assert "".join(chunks) == "hello"


def test_llama_server_engine_stop():
    """验证 stop() 在进程为 None 时不会报错。"""
    import app.engine.llama_server_engine as lse

    class MockEngine(lse.LlamaServerEngine):
        def __init__(self):
            self._port = 9999
            self._base_url = "http://127.0.0.1:9999"
            self._process = None

    engine = MockEngine()
    # 不应抛出异常
    engine.stop()
