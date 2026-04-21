from __future__ import annotations

import json
from collections.abc import Iterator

from app.schemas import (
    AnthropicImageBlock,
    AnthropicMessage,
    AnthropicMessageRequest,
    AnthropicTextBlock,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    ContentPart,
    Message,
)


def _system_to_text(system: str | list[AnthropicTextBlock] | None) -> str | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system
    return "\n\n".join(block.text for block in system if block.text)


def _tool_result_to_text(block: AnthropicToolResultBlock) -> str:
    if isinstance(block.content, str):
        return block.content
    if isinstance(block.content, list):
        return "\n".join(item.text for item in block.content if item.text)
    return ""


def _append_user_content(
    normalized: list[Message],
    content_parts: list[ContentPart],
) -> None:
    if not content_parts:
        return
    if len(content_parts) == 1 and content_parts[0].type == "text":
        normalized.append(Message(role="user", content=content_parts[0].text or ""))
        return
    normalized.append(Message(role="user", content=content_parts))


def _normalize_message(message: AnthropicMessage) -> list[Message]:
    if isinstance(message.content, str):
        return [Message(role=message.role, content=message.content)]

    if message.role == "assistant":
        parts: list[ContentPart] = []
        tool_calls: list[dict] = []
        for block in message.content:
            if isinstance(block, AnthropicTextBlock):
                parts.append(ContentPart(type="text", text=block.text))
            elif isinstance(block, AnthropicToolUseBlock):
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input, ensure_ascii=False),
                        },
                    }
                )
        content: str | list[ContentPart] | None
        if not parts:
            content = None
        elif len(parts) == 1:
            content = parts[0].text or ""
        else:
            content = parts
        return [Message(role="assistant", content=content, tool_calls=tool_calls or None)]

    normalized: list[Message] = []
    pending_parts: list[ContentPart] = []
    for block in message.content:
        if isinstance(block, AnthropicTextBlock):
            pending_parts.append(ContentPart(type="text", text=block.text))
        elif isinstance(block, AnthropicImageBlock):
            pending_parts.append(
                ContentPart(
                    type="image_url",
                    image_url={
                        "url": (
                            f"data:{block.source.media_type};base64,{block.source.data}"
                        )
                    },
                )
            )
        elif isinstance(block, AnthropicToolResultBlock):
            _append_user_content(normalized, pending_parts)
            pending_parts = []
            normalized.append(
                Message(
                    role="tool",
                    content=_tool_result_to_text(block),
                    tool_call_id=block.tool_use_id,
                )
            )
    _append_user_content(normalized, pending_parts)
    return normalized


def request_to_messages(req: AnthropicMessageRequest) -> list[Message]:
    normalized: list[Message] = []
    system_text = _system_to_text(req.system)
    if system_text:
        normalized.append(Message(role="system", content=system_text))
    for message in req.messages:
        normalized.extend(_normalize_message(message))
    return normalized


def request_to_overrides(req: AnthropicMessageRequest) -> dict:
    overrides = {
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "top_p": req.top_p,
    }
    if req.tools:
        overrides["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in req.tools
        ]
        overrides["tool_choice"] = "auto"
    return overrides


def _finish_reason_to_stop_reason(finish_reason: str | None) -> str | None:
    mapping = {
        None: None,
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    return mapping.get(finish_reason, "end_turn")


def choice_to_message_response(request_id: str, model: str, choice: dict) -> dict:
    message = choice.get("message", {})
    content: list[dict] = []
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function", {})
        arguments = function.get("arguments") or "{}"
        try:
            tool_input = json.loads(arguments)
        except json.JSONDecodeError:
            tool_input = {"raw_arguments": arguments}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id", ""),
                "name": function.get("name", ""),
                "input": tool_input,
            }
        )
    return {
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _finish_reason_to_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def stream_events(
    request_id: str,
    model: str,
    chunks: Iterator[str],
) -> Iterator[str]:
    message_start = {
        "type": "message_start",
        "message": {
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    yield "event: message_start\n"
    yield f"data: {json.dumps(message_start, ensure_ascii=False)}\n\n"

    content_block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    yield "event: content_block_start\n"
    yield f"data: {json.dumps(content_block_start, ensure_ascii=False)}\n\n"

    for piece in chunks:
        payload = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }
        yield "event: content_block_delta\n"
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    content_block_stop = {"type": "content_block_stop", "index": 0}
    yield "event: content_block_stop\n"
    yield f"data: {json.dumps(content_block_stop, ensure_ascii=False)}\n\n"

    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 0},
    }
    yield "event: message_delta\n"
    yield f"data: {json.dumps(message_delta, ensure_ascii=False)}\n\n"

    message_stop = {"type": "message_stop"}
    yield "event: message_stop\n"
    yield f"data: {json.dumps(message_stop, ensure_ascii=False)}\n\n"
