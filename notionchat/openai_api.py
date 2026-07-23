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
    load_model_catalog,
    openai_models,
    resolve_model_id,
)
from notionchat.config import Settings, load_account_from_env, load_settings
from notionchat.exceptions import NotionChatError

log = logging.getLogger(__name__)

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str | list[Any] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = ""
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

    async def _catalog(client: ArenaHttpClient) -> list[Any]:
        """Load an explicit local catalog first, otherwise query Arena live.

        This priority is important when Arena's public endpoint returns survey or
        metadata records rather than the account's callable model UUIDs.
        """
        local_models = load_model_catalog(settings.model_catalog_path)
        return local_models or await client.get_models()

    @app.get("/v1/models")
    async def list_models(_: None = Depends(verify_key)) -> dict[str, Any]:
        """List callable raw Arena model ids, not obsolete friendly aliases."""
        account = load_account_from_env(settings)
        client = ArenaHttpClient(account)
        try:
            models = await _catalog(client)
        finally:
            await client.close()
        if models:
            log.info("Loaded %d Arena models", len(models))
        else:
            log.warning("No Arena models found (live endpoint and %s were empty)", settings.model_catalog_path)
        return {"object": "list", "data": openai_models(models)}

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

            log.info(
                "chat stream=%s model=%s msgs=%d",
                req.stream,
                req.model,
                len(messages),
            )

            models = await _catalog(client)
            # Do not send deprecated aliases to Arena. This turns the opaque
            # upstream 404 into an actionable OpenAI-compatible 400 response.
            requested_model = req.model or settings.default_model
            req.model = resolve_model_id(requested_model, models)

            if req.stream:
                return StreamingResponse(
                    _stream_response(client, req, settings),
                    media_type="text/event-stream",
                )

            # Non-streaming
            try:
                result = await client.chat_completion(
                    model=req.model,
                    messages=messages,
                    temperature=req.temperature if req.temperature is not None else 1.0,
                    max_tokens=req.max_tokens,
                    top_p=req.top_p,
                    stop=req.stop,
                )
            finally:
                await client.close()

            return _format_response(result, req.model)

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
