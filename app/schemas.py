from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ContentPart(BaseModel):
    """多模态内容块：文本或图片。"""
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: dict[str, str] | None = None


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    # 纯文本字符串或多模态内容块列表（OpenAI vision 格式）
    content: str | list[ContentPart] | None = None
    # Function Calling 相关字段
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class OpenAIChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    # Function Calling 支持
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


class AnthropicTextBlock(BaseModel):
    type: Literal["text"]
    text: str


class AnthropicImageSource(BaseModel):
    type: Literal["base64"]
    media_type: str
    data: str


class AnthropicImageBlock(BaseModel):
    type: Literal["image"]
    source: AnthropicImageSource


class AnthropicToolUseBlock(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class AnthropicToolResultBlock(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[AnthropicTextBlock] | None = None
    is_error: bool = False


AnthropicContentBlock = (
    AnthropicTextBlock
    | AnthropicImageBlock
    | AnthropicToolUseBlock
    | AnthropicToolResultBlock
)


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


class AnthropicTool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)


class AnthropicMessageRequest(BaseModel):
    model: str | None = None
    messages: list[AnthropicMessage]
    system: str | list[AnthropicTextBlock] | None = None
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    tools: list[AnthropicTool] | None = None


class OllamaChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    stream: bool = True
    options: dict[str, Any] | None = None


class OllamaGenerateRequest(BaseModel):
    model: str | None = None
    prompt: str
    system: str | None = None
    stream: bool = True
    options: dict[str, Any] | None = None


class GenerationDefaults(BaseModel):
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    top_p: float = Field(default=0.9, ge=0, le=1)


class ServerConfig(BaseModel):
    model_dir: str = "/home/admin/Downloads"
    model_mapping: dict[str, str] = {
        "Qwen3.5-0.8B": "Qwen3.5-0.8B-Q4_K_M.gguf",
    }
    default_model: str = "Qwen3.5-0.8B"
    generation_defaults: GenerationDefaults = GenerationDefaults()
    max_loaded_models: int = Field(default=1, ge=1, le=4)
    # 引擎类型：llama_server 使用子进程，llama_cpp 使用 Python 绑定
    engine_type: Literal["llama_server", "llama_cpp"] = "llama_server"
    llama_server_path: str = "/home/admin/llama.cpp/build/bin/llama-server"
    n_ctx: int = Field(default=4096, ge=512, le=131072)
    n_threads: int = Field(default=8, ge=1, le=128)
    startup_timeout: int = Field(default=60, ge=5, le=300)
    # 多模态视觉投影文件路径，留空则不启用视觉能力
    mmproj_path: str = ""


class PartialGenerationDefaults(BaseModel):
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    top_p: float | None = Field(default=None, ge=0, le=1)


class ConfigPatch(BaseModel):
    model_dir: str | None = None
    model_mapping: dict[str, str] | None = None
    default_model: str | None = None
    generation_defaults: PartialGenerationDefaults | None = None
    max_loaded_models: int | None = Field(default=None, ge=1, le=4)
    engine_type: Literal["llama_server", "llama_cpp"] | None = None
    llama_server_path: str | None = None
    n_ctx: int | None = Field(default=None, ge=512, le=131072)
    n_threads: int | None = Field(default=None, ge=1, le=128)
    startup_timeout: int | None = Field(default=None, ge=5, le=300)
    mmproj_path: str | None = None


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "local"


class TagsModel(BaseModel):
    name: str
    model: str
    modified_at: datetime
    size: int = 0
    digest: str = "local"
    details: dict[str, Any] = {
        "family": "qwen",
        "format": "gguf",
        "quantization_level": "Q4_K_M",
    }
