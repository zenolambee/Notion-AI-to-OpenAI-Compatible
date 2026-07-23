"""
OpenAI-compatible API for Arena.ai (Chatbot Arena).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from notionchat.arena_client import (
    ArenaHttpClient,
    ArenaStreamChunk,
    get_arena_models,
    sync_arena_models_to_catalog,
)
from notionchat.config import Settings, load_account_from_env, load_settings
from notionchat.exceptions import NotionChatError
from notionchat.model_catalog import (
    catalog_is_fresh,
    closest_model_names,
    load_catalog,
    openai_model_list,
    resolve_model_id,
)

log = logging.getLogger(__name__)

# Default arena models
DEFAULT_ARENA_MODELS = [
    {"id": "arena-gpt-4o", "object": "model", "created": 1700000000, "owned_by": "openai", "description": "GPT-4o via Arena.ai"},
    {"id": "arena-claude-3-5-sonnet", "object": "model", "created": 1700000000, "owned_by": "anthropic", "description": "Claude 3.5 Sonnet via Arena.ai"},
    {"id": "arena-gemini-1.5-pro", "object": "model", "created": 1700000000, "owned_by": "google", "description": "Gemini 1.5 Pro via Arena.ai"},
    {"id": "arena-claude-3-opus", "object": "model", "created": 1700000000, "owned_by": "anthropic", "description": "Claude 3 Opus via Arena.ai"},
    {"id": "arena-gpt-4-turbo", "object": "model", "created": 1700000000, "owned_by": "openai", "description": "GPT-4 Turbo via Arena.ai"},
    {"id": "arena-gpt-4", "object": "model", "created": 1700000000, "owned_by": "openai", "description": "GPT-4 via Arena.ai"},
    {"id": "arena-claude-3-sonnet", "object": "model", "created": 1700000000, "owned_by": "anthropic", "description": "Claude 3 Sonnet via Arena.ai"},
    {"id": "arena-claude-3-haiku", "object": "model", "created": 1700000000, "owned_by": "anthropic", "description": "Claude 3 Haiku via Arena.ai"},
    {"id": "arena-gemini-1.5-flash", "object": "model", "created": 1700000000, "owned_by": "google", "description": "Gemini 1.5 Flash via Arena.ai"},
    {"id": "arena-llama-3-70b", "object": "model", "created": 1700000000, "owned_by": "meta", "description": "Llama 3 70B via Arena.ai"},
    {"id": "arena-llama-3-8b", "object": "model", "created": 1700000000, "owned_by": "meta", "description": "Llama 3 8B via Arena.ai"},
    {"id": "arena-mixtral-8x7b", "object": "model", "created": 1700000000, "owned_by": "mistral", "description": "Mixtral 8x7B via Arena.ai"},
]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str | list[Any] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = "arena-gpt-4o"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None


def _chunk(
    *,
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> str:
    """Create an SSE chunk."""
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create FastAPI application."""
    settings = settings or load_settings()
    app = FastAPI(
        title="ArenaChat",
        description="OpenAI-compatible API for Arena.ai (Chatbot Arena)",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    def verify_key(authorization: str | None = Header(default=None)) -> None:
        """Verify API key from Authorization header."""
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        token = authorization.removeprefix("Bearer ").strip()
        if token != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(_: None = Depends(verify_key)) -> dict[str, Any]:
        """List available models.

        Tries the local catalog first (fast). If stale or missing, auto-syncs
        from Arena.ai on the fly.
        """
        catalog = load_catalog()

        if not catalog_is_fresh(catalog):
            try:
                account = load_account_from_env(settings)
                catalog = await sync_arena_models_to_catalog(account)
                log.info("Auto-synced model catalog (%d models)", len(catalog.get("models", [])))
            except Exception as e:
                log.warning("Could not sync models from arena.ai: %s", e)

        models = openai_model_list(catalog)

        # Fallback to hardcoded defaults if the catalog is truly empty
        if not models:
            log.warning("Model catalog empty — falling back to hardcoded defaults")
            models = DEFAULT_ARENA_MODELS.copy()

        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        _: None = Depends(verify_key),
    ) -> Any:
        """Handle chat completion requests."""
        try:
            account = load_account_from_env(settings)
            client = ArenaHttpClient(account)

            # Convert messages
            messages = [
                {"role": m.role, "content": m.content or ""}
                for m in req.messages
            ]

            # ── Model resolution ────────────────────────────────────
            requested_model = req.model
            arena_model_id = resolve_model_id(requested_model)

            if arena_model_id is None:
                # Auto-sync and retry once
                catalog = load_catalog()
                try:
                    catalog = await sync_arena_models_to_catalog(account)
                except Exception:
                    pass
                arena_model_id = resolve_model_id(requested_model, catalog=catalog)

            if arena_model_id is None:
                await client.close()
                suggestions = closest_model_names(requested_model)
                hint = ""
                if suggestions:
                    hint = f" Closest matches: {', '.join(suggestions[:10])}"
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Model '{requested_model}' not found in Arena model catalog. "
                        f"Run `python -m notionchat sync-models` to refresh, "
                        f"or pass a raw Arena model UUID.{hint}"
                    ),
                )

            log.info(
                "chat stream=%s model=%s→%s msgs=%d",
                req.stream,
                requested_model,
                arena_model_id,
                len(messages),
            )

            if req.stream:
                return StreamingResponse(
                    _stream_response(client, ChatCompletionRequest(
                        model=arena_model_id,
                        messages=req.messages,
                        stream=True,
                        temperature=req.temperature,
                        max_tokens=req.max_tokens,
                        top_p=req.top_p,
                        stop=req.stop,
                    ), settings),
                    media_type="text/event-stream",
                )

            # Non-streaming
            result = await client.chat_completion(
                model=arena_model_id,
                messages=messages,
                temperature=req.temperature or 1.0,
                max_tokens=req.max_tokens,
            )
            await client.close()

            return _format_response(result, requested_model)

        except HTTPException:
            raise
        except NotionChatError as e:
            log.error("Chat completion error: %s", e)
            raise HTTPException(status_code=e.status_code, detail=str(e)) from e
        except Exception as e:
            log.error("Unexpected error: %s", e)
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app


async def _stream_response(
    client: ArenaHttpClient,
    req: ChatCompletionRequest,
    settings: Settings,
) -> AsyncIterator[str]:
    """Stream chat completion response."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    try:
        messages = [
            {"role": m.role, "content": m.content or ""}
            for m in req.messages
        ]

        async for chunk in client.chat_completion_stream(
            model=req.model,
            messages=messages,
            temperature=req.temperature or 1.0,
            max_tokens=req.max_tokens,
        ):
            if chunk.content:
                yield _chunk(
                    completion_id=completion_id,
                    created=created,
                    model=req.model,
                    delta={"content": chunk.content},
                )
            if chunk.done:
                yield _chunk(
                    completion_id=completion_id,
                    created=created,
                    model=req.model,
                    delta={},
                    finish_reason=chunk.finish_reason or "stop",
                )

        yield "data: [DONE]\n\n"

    except NotionChatError as e:
        err = {"error": {"message": str(e), "type": "arena_error", "code": e.status_code}}
        yield f"data: {json.dumps(err)}\n\n"
    finally:
        await client.close()


def _format_response(data: dict[str, Any], model: str) -> dict[str, Any]:
    """Format arena.ai response to OpenAI format."""
    content = ""
    finish_reason = "stop"

    # Try different content extraction methods
    if "choices" in data and data["choices"]:
        content = data["choices"][0].get("message", {}).get("content", "")
        finish_reason = data["choices"][0].get("finish_reason", "stop")
    elif "text" in data:
        content = data["text"]
    elif "message" in data:
        content = data["message"].get("content", "")
    elif "content" in data:
        content = data["content"]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": data.get("usage", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }),
    }
