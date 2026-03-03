from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.schemas import (
    ConfigPatch,
    HealthResponse,
    ModelCard,
    OllamaChatRequest,
    OllamaGenerateRequest,
    OpenAIChatRequest,
    TagsModel,
)
from app.service import ChatService

router = APIRouter()


def get_service() -> ChatService:
    raise RuntimeError("service dependency was not wired")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc))


@router.get("/models")
def models(service: ChatService = Depends(get_service)) -> dict:
    cfg = service.config_store.get()
    data = [ModelCard(id=name).model_dump() for name in cfg.model_mapping]
    return {"object": "list", "data": data}


@router.get("/v1/models")
def v1_models(service: ChatService = Depends(get_service)) -> dict:
    return models(service)


@router.get("/api/tags")
def tags(service: ChatService = Depends(get_service)) -> dict:
    cfg = service.config_store.get()
    models_data = [
        TagsModel(name=name, model=name, modified_at=datetime.now(timezone.utc)).model_dump()
        for name in cfg.model_mapping
    ]
    return {"models": models_data}


@router.get("/config")
def get_config(service: ChatService = Depends(get_service)) -> dict:
    return service.config_store.get().model_dump()


@router.post("/config")
def patch_config(patch: ConfigPatch, service: ChatService = Depends(get_service)) -> dict:
    try:
        cfg = service.config_store.update(patch)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return cfg.model_dump()


@router.post("/v1/chat/completions")
def openai_chat(req: OpenAIChatRequest, service: ChatService = Depends(get_service)):
    request_id = f"chatcmpl-{uuid4().hex[:20]}"

    options = {
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "top_p": req.top_p,
    }

    if req.stream:
        model, generator = service.stream_chat(req.model, req.messages, overrides=options)

        def stream():
            for piece in generator:
                payload = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            final = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(datetime.now(timezone.utc).timestamp()),
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    model, text = service.chat(req.model, req.messages, overrides=options)
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }


@router.post("/api/chat")
def ollama_chat(req: OllamaChatRequest, service: ChatService = Depends(get_service)):
    if req.stream:
        model, generator = service.stream_chat(req.model, req.messages, options=req.options)

        def stream():
            for piece in generator:
                payload = {
                    "model": model,
                    "created_at": service.now(),
                    "message": {"role": "assistant", "content": piece},
                    "done": False,
                }
                yield json.dumps(payload, ensure_ascii=False) + "\n"
            yield json.dumps(
                {
                    "model": model,
                    "created_at": service.now(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                },
                ensure_ascii=False,
            ) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    model, text = service.chat(req.model, req.messages, options=req.options)
    return JSONResponse(
        {
            "model": model,
            "created_at": service.now(),
            "message": {"role": "assistant", "content": text},
            "done": True,
            "done_reason": "stop",
        }
    )


@router.post("/api/generate")
def ollama_generate(req: OllamaGenerateRequest, service: ChatService = Depends(get_service)):
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})

    from app.schemas import Message

    msg_objs = [Message(**m) for m in messages]

    if req.stream:
        model, generator = service.stream_chat(req.model, msg_objs, options=req.options)

        def stream():
            for piece in generator:
                payload = {
                    "model": model,
                    "created_at": service.now(),
                    "response": piece,
                    "done": False,
                }
                yield json.dumps(payload, ensure_ascii=False) + "\n"
            yield json.dumps(
                {
                    "model": model,
                    "created_at": service.now(),
                    "response": "",
                    "done": True,
                    "done_reason": "stop",
                },
                ensure_ascii=False,
            ) + "\n"

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    model, text = service.chat(req.model, msg_objs, options=req.options)
    return {
        "model": model,
        "created_at": service.now(),
        "response": text,
        "done": True,
        "done_reason": "stop",
    }
