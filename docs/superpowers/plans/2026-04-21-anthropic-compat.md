# Anthropic Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal Anthropic-compatible `/v1/messages` endpoint that Claude Code can use for basic chat and streaming without protocol errors.

**Architecture:** Keep compatibility logic at the API layer. Convert Anthropic requests into the existing internal `Message` model, call the current `ChatService`, then format responses back into Anthropic message and SSE shapes.

**Tech Stack:** FastAPI, Pydantic v2, pytest, httpx

---

### Task 1: Add failing Anthropic compatibility tests

**Files:**
- Modify: `tests/test_core.py`
- Test: `tests/test_core.py`

- [ ] **Step 1: Write the failing tests**

Add tests for:
- Anthropic request conversion with `system`, text, and image blocks
- `/v1/messages` non-streaming response shape
- `/v1/messages` streaming SSE event flow

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run: `.venv/bin/python -m pytest -q tests/test_core.py -k anthropic`
Expected: FAIL because Anthropic schemas, adapters, and route do not exist yet

### Task 2: Implement Anthropic request/response adapters

**Files:**
- Create: `app/anthropic.py`
- Modify: `app/schemas.py`

- [ ] **Step 1: Add typed request models for Anthropic payloads**

Model:
- top-level `model`, `messages`, `system`, `max_tokens`, `temperature`, `top_p`, `stream`, `tools`
- content blocks for `text` and image input

- [ ] **Step 2: Add adapter helpers**

Implement helpers to:
- normalize top-level `system` into internal system messages
- map Anthropic messages into internal `Message`
- map Anthropic tools into existing OpenAI-style tool definitions
- build Anthropic non-streaming and streaming response payloads

- [ ] **Step 3: Run targeted tests**

Run: `.venv/bin/python -m pytest -q tests/test_core.py -k anthropic`
Expected: some failures remain until route wiring is added

### Task 3: Wire the new route

**Files:**
- Modify: `app/api/routes.py`

- [ ] **Step 1: Add `POST /v1/messages`**

Use the adapter helpers to:
- parse Anthropic requests
- pass normalized messages into `ChatService`
- stream Anthropic SSE events when `stream=true`
- return Anthropic message objects otherwise

- [ ] **Step 2: Run targeted tests again**

Run: `.venv/bin/python -m pytest -q tests/test_core.py -k anthropic`
Expected: PASS

### Task 4: Regression verification and docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document Anthropic compatibility**

Add the new compatibility surface and note that this pass targets basic Claude Code connectivity and debugging.

- [ ] **Step 2: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS with 0 failures

- [ ] **Step 3: Review optimization opportunities**

Check whether any duplicated route formatting or schema looseness should be called out in the final summary.
