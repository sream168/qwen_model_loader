# Anthropic Compatibility Design

**Date:** 2026-04-21

## Goal

Add a minimal Anthropic-compatible Messages API to the existing service so Claude Code can connect, send basic chat requests, and use streaming without immediate protocol errors.

## Scope

In scope:
- Add `POST /v1/messages`
- Accept Anthropic-style request bodies for text, `system`, images, and basic `tools`
- Return Anthropic-style non-streaming responses
- Return Anthropic-style streaming SSE events
- Reuse the existing internal `Message` model and engine abstraction
- Add tests for request conversion, non-streaming, and streaming behavior

Out of scope for this pass:
- Full Anthropic protocol parity
- Complete `tool_use` / `tool_result` execution semantics
- Admin endpoints, token counting, or batch APIs

## Design

Keep the compatibility layer at the HTTP boundary. The internal service and engine interfaces already operate on an OpenAI-like normalized message format, so the new Anthropic endpoint should translate requests into that format and translate responses back into Anthropic shapes.

Add a small adapter module responsible for:
- converting Anthropic `system` + `messages` content blocks into internal `Message` objects
- mapping Anthropic `tools` into the existing OpenAI-style tool schema expected by the engine layer
- extracting assistant text from engine responses and wrapping it into Anthropic `content` blocks
- formatting Anthropic streaming events

This keeps engine behavior unchanged and reduces regression risk for existing OpenAI and Ollama routes.

## Error Handling

- Accept `x-api-key` and `anthropic-version` headers but do not authenticate them in this local compatibility layer
- Reject malformed Anthropic payloads through Pydantic validation
- Preserve existing service and runtime error handling from the FastAPI app

## Testing

- Add unit tests for Anthropic request conversion
- Add route tests for `/v1/messages` non-streaming responses
- Add route tests for `/v1/messages` streaming SSE event flow
- Verify existing OpenAI and Ollama tests still pass

## Follow-Up Optimization Candidates

- Split protocol adapters out of `app/api/routes.py` to keep route handlers thin
- Unify streaming event generation behind dedicated formatter helpers
- Replace raw `dict` tool payloads with typed models if Anthropic support expands
