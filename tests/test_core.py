from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config_store import ConfigStore
from app.model_manager import ModelManager
from app.schemas import ConfigPatch, ContentPart, Message
from app.service import ChatService


class FakeEngine:
    """模拟引擎，返回固定响应用于快速测试。"""

    def __init__(self):
        self.stopped = False

    def chat(self, messages: list[Message], **kwargs):
        return {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }

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

    model, choice = service.chat(
        model=None,
        messages=[Message(role="user", content="hi")],
        options={"temperature": 0.4},
        overrides={"max_tokens": 128},
    )

    assert model == "Qwen3.5-0.8B"
    assert manager.last_model == "Qwen3.5-0.8B"
    assert choice["message"]["content"] == "ok"


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
                    {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ]
            }

    import httpx
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: FakeResponse())

    result = engine.chat([Message(role="user", content="hi")])
    assert result["message"]["content"] == "hello"


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


# ─── HTTP API 路由集成测试 ───


@pytest.fixture()
def test_client(tmp_path: Path, monkeypatch):
    """构建一个使用 FakeEngine 的 FastAPI TestClient。"""
    from fastapi.testclient import TestClient

    from app.api import routes
    from app.main import create_app

    # 使用临时配置文件，避免影响真实配置
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(
        "app.model_manager.ModelManager._create_engine",
        lambda self, path: FakeEngine(),
    )

    # 创建模型文件以通过路径校验
    model_file = Path(tmp_path / "Qwen3.5-0.8B-Q4_K_M.gguf")
    model_file.write_text("fake")

    # 更新配置指向临时目录
    app = create_app(str(config_path))
    store: ConfigStore = None
    for dep_fn, override_fn in app.dependency_overrides.items():
        svc = override_fn()
        store = svc.config_store
        break

    store.update(ConfigPatch(model_dir=str(tmp_path)))

    return TestClient(app)


def test_route_health(test_client):
    """验证 /health 端点返回正常状态。"""
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_route_models(test_client):
    """验证 /models 和 /v1/models 端点返回模型列表。"""
    resp = test_client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1

    # /v1/models 应返回相同格式
    resp2 = test_client.get("/v1/models")
    assert resp2.status_code == 200
    assert resp2.json()["data"] == data["data"]


def test_route_tags(test_client):
    """验证 Ollama /api/tags 端点返回模型标签列表。"""
    resp = test_client.get("/api/tags")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert len(data["models"]) >= 1
    assert "name" in data["models"][0]


def test_route_openai_chat_non_streaming(test_client):
    """验证 OpenAI /v1/chat/completions 非流式响应。"""
    resp = test_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "ok"
    assert "id" in data


def test_route_openai_chat_streaming(test_client):
    """验证 OpenAI /v1/chat/completions 流式 SSE 响应。"""
    resp = test_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    lines = resp.text.strip().split("\n")
    # 过滤出 data: 行
    data_lines = [l for l in lines if l.startswith("data: ")]
    assert len(data_lines) >= 3  # 至少有内容块 + 结束块 + [DONE]

    # 最后一行应为 [DONE]
    assert data_lines[-1].strip() == "data: [DONE]"

    # 解析中间的内容块
    content_pieces = []
    for dl in data_lines:
        payload = dl[len("data: "):]
        if payload.strip() == "[DONE]":
            break
        chunk = json.loads(payload)
        delta_content = chunk["choices"][0]["delta"].get("content")
        if delta_content:
            content_pieces.append(delta_content)
    assert "".join(content_pieces) == "ok"


def test_route_ollama_chat_non_streaming(test_client):
    """验证 Ollama /api/chat 非流式响应。"""
    resp = test_client.post(
        "/api/chat",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is True
    assert data["message"]["content"] == "ok"


def test_route_ollama_generate_non_streaming(test_client):
    """验证 Ollama /api/generate 非流式响应。"""
    resp = test_client.post(
        "/api/generate",
        json={"prompt": "hello", "stream": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is True
    assert data["response"] == "ok"


def test_route_config_get_and_update(test_client):
    """验证 /config GET 和 POST 端点。"""
    # GET
    resp = test_client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "engine_type" in data

    # POST 更新
    resp = test_client.post("/config", json={"n_ctx": 8192})
    assert resp.status_code == 200
    assert resp.json()["n_ctx"] == 8192


# ─── Function Calling 参数透传测试 ───


def test_function_calling_passthrough(monkeypatch):
    """验证 tools 和 tool_choice 参数被正确传递到引擎层。"""
    import app.engine.llama_server_engine as lse

    monkeypatch.setattr(lse, "_find_free_port", lambda: 9999)

    class MockEngine(lse.LlamaServerEngine):
        def __init__(self):
            self._port = 9999
            self._base_url = "http://127.0.0.1:9999"
            self._process = None
            self._log_file = None

    engine = MockEngine()

    # 捕获实际发送的 payload
    captured_payload = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"北京"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            }

    import httpx

    def fake_post(*args, **kwargs):
        captured_payload.update(kwargs.get("json", {}))
        return FakeResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }]

    result = engine.chat(
        [Message(role="user", content="北京天气如何")],
        tools=tools,
        tool_choice="auto",
    )

    # 验证参数被透传到 HTTP 请求
    assert captured_payload["tools"] == tools
    assert captured_payload["tool_choice"] == "auto"

    # 验证响应中包含 tool_calls
    assert result["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert result["finish_reason"] == "tool_calls"


def test_function_calling_stream_passthrough(monkeypatch):
    """验证流式模式下 tools 参数被正确透传。"""
    import app.engine.llama_server_engine as lse

    monkeypatch.setattr(lse, "_find_free_port", lambda: 9999)

    class MockEngine(lse.LlamaServerEngine):
        def __init__(self):
            self._port = 9999
            self._base_url = "http://127.0.0.1:9999"
            self._process = None
            self._log_file = None

    engine = MockEngine()
    captured_payload = {}

    class FakeStreamResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
            yield "data: [DONE]"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    import httpx

    def fake_stream(*args, **kwargs):
        captured_payload.update(kwargs.get("json", {}))
        return FakeStreamResponse()

    monkeypatch.setattr(httpx, "stream", fake_stream)

    tools = [{"type": "function", "function": {"name": "test_fn"}}]
    list(engine.stream_chat(
        [Message(role="user", content="test")],
        tools=tools,
        tool_choice="auto",
    ))

    assert captured_payload["tools"] == tools
    assert captured_payload["tool_choice"] == "auto"


# ─── 多模态 Message 序列化测试 ───


def test_llama_server_to_payload_multimodal(monkeypatch):
    """验证 LlamaServerEngine._to_payload() 正确序列化多模态消息。"""
    import app.engine.llama_server_engine as lse

    monkeypatch.setattr(lse, "_find_free_port", lambda: 9999)

    class MockEngine(lse.LlamaServerEngine):
        def __init__(self):
            self._port = 9999
            self._base_url = "http://127.0.0.1:9999"
            self._process = None
            self._log_file = None

    engine = MockEngine()

    messages = [
        Message(
            role="user",
            content=[
                ContentPart(type="text", text="这张图片是什么？"),
                ContentPart(
                    type="image_url",
                    image_url={"url": "data:image/png;base64,iVBOR..."},
                ),
            ],
        )
    ]

    payload = engine._to_payload(messages)

    assert len(payload) == 1
    assert payload[0]["role"] == "user"
    parts = payload[0]["content"]
    assert isinstance(parts, list)
    assert len(parts) == 2
    assert parts[0] == {"type": "text", "text": "这张图片是什么？"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,iVBOR..."


def test_llama_server_to_payload_tool_calls(monkeypatch):
    """验证 _to_payload() 正确序列化 Function Calling 消息。"""
    import app.engine.llama_server_engine as lse

    monkeypatch.setattr(lse, "_find_free_port", lambda: 9999)

    class MockEngine(lse.LlamaServerEngine):
        def __init__(self):
            self._port = 9999
            self._base_url = "http://127.0.0.1:9999"
            self._process = None
            self._log_file = None

    engine = MockEngine()

    tool_call = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"上海"}'},
    }

    messages = [
        Message(role="user", content="上海天气"),
        Message(role="assistant", content=None, tool_calls=[tool_call]),
        Message(role="tool", content='{"temp": 25}', tool_call_id="call_abc"),
    ]

    payload = engine._to_payload(messages)

    assert payload[0]["content"] == "上海天气"
    # assistant 消息的 content 应为 None，带 tool_calls
    assert payload[1]["content"] is None
    assert payload[1]["tool_calls"] == [tool_call]
    # tool 消息应带 tool_call_id
    assert payload[2]["role"] == "tool"
    assert payload[2]["tool_call_id"] == "call_abc"
    assert payload[2]["content"] == '{"temp": 25}'


def test_llama_engine_to_payload_multimodal_text_only():
    """验证 LlamaCppEngine._to_payload() 从多模态消息中提取纯文本。"""
    from app.engine.llama_engine import LlamaCppEngine

    # 无法实例化（需要模型文件），直接测试类方法
    engine = LlamaCppEngine.__new__(LlamaCppEngine)

    messages = [
        Message(
            role="user",
            content=[
                ContentPart(type="text", text="描述图片"),
                ContentPart(type="image_url", image_url={"url": "http://example.com/img.png"}),
            ],
        )
    ]

    payload = engine._to_payload(messages)

    # llama-cpp-python 不支持多模态，应只提取文本
    assert payload[0]["content"] == "描述图片"


# ─── content=None 处理测试 ───


def test_llama_server_to_payload_content_none(monkeypatch):
    """验证 LlamaServerEngine._to_payload() 正确处理 content=None。"""
    import app.engine.llama_server_engine as lse

    monkeypatch.setattr(lse, "_find_free_port", lambda: 9999)

    class MockEngine(lse.LlamaServerEngine):
        def __init__(self):
            self._port = 9999
            self._base_url = "http://127.0.0.1:9999"
            self._process = None
            self._log_file = None

    engine = MockEngine()

    messages = [
        Message(role="assistant", content=None, tool_calls=[{
            "id": "call_1", "type": "function",
            "function": {"name": "fn", "arguments": "{}"},
        }]),
    ]

    payload = engine._to_payload(messages)
    assert payload[0]["content"] is None
    assert payload[0]["tool_calls"] is not None


def test_llama_engine_to_payload_content_none():
    """验证 LlamaCppEngine._to_payload() 正确处理 content=None（不崩溃）。"""
    from app.engine.llama_engine import LlamaCppEngine

    engine = LlamaCppEngine.__new__(LlamaCppEngine)

    messages = [
        Message(role="assistant", content=None),
    ]

    payload = engine._to_payload(messages)
    assert payload[0]["role"] == "assistant"
    assert payload[0]["content"] == ""


# ─── 配置持久化测试 ───


def test_config_persistence_write_then_reload(tmp_path: Path):
    """验证配置更新后写入磁盘，重新加载能恢复修改。"""
    config_path = tmp_path / "config.json"

    # 第一个 ConfigStore 实例：修改配置
    store1 = ConfigStore(config_path)
    store1.update(ConfigPatch(
        engine_type="llama_cpp",
        n_ctx=16384,
        n_threads=16,
        mmproj_path="/tmp/test.gguf",
    ))

    # 第二个 ConfigStore 实例：从磁盘重新加载
    store2 = ConfigStore(config_path)
    cfg = store2.get()

    assert cfg.engine_type == "llama_cpp"
    assert cfg.n_ctx == 16384
    assert cfg.n_threads == 16
    assert cfg.mmproj_path == "/tmp/test.gguf"


def test_config_atomic_write_creates_valid_json(tmp_path: Path):
    """验证原子写入后文件是有效的 JSON。"""
    import orjson

    config_path = tmp_path / "config.json"
    store = ConfigStore(config_path)
    store.update(ConfigPatch(
        model_mapping={"Qwen3.5-0.8B": "Qwen3.5-0.8B-Q4_K_M.gguf", "test": "test.gguf"},
        default_model="Qwen3.5-0.8B",
    ))

    # 直接读取磁盘文件并验证 JSON 合法性
    raw = config_path.read_bytes()
    data = orjson.loads(raw)
    assert "test" in data["model_mapping"]
    assert data["default_model"] == "Qwen3.5-0.8B"


# ─── ChatService 兼容性测试 ───


def test_chat_service_wraps_string_result(tmp_path: Path):
    """验证 ChatService 对旧引擎返回纯字符串时进行包装。"""

    class StringEngine:
        def chat(self, messages, **kwargs):
            return "plain text response"

        def stream_chat(self, messages, **kwargs):
            yield "hi"

        def stop(self):
            pass

    class StringModelManager:
        def get_engine(self, model_name):
            return StringEngine()

    store = ConfigStore(tmp_path / "config.json")
    service = ChatService(store, StringModelManager())

    model, choice = service.chat(None, [Message(role="user", content="hi")])
    assert isinstance(choice, dict)
    assert choice["message"]["content"] == "plain text response"
    assert choice["finish_reason"] == "stop"
