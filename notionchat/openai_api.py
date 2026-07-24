"""
OpenAI-compatible API for Arena.ai (Chatbot Arena).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from notionchat.arena_client import ArenaHttpClient
from notionchat.config import Settings, load_account_from_env, load_settings
from notionchat.exceptions import NotionChatError

log = logging.getLogger(__name__)

# Models are discovered from the logged-in Arena direct-mode page.  Do not
# advertise a stale hard-coded list: those names cannot be sent to Arena.


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


def _new_arena_client(account, settings: Settings) -> ArenaHttpClient:
    """Construct a client with settings shared by model and chat endpoints."""
    return ArenaHttpClient(
        account,
        base_url=settings.base_url,
        recaptcha_v3_token=settings.recaptcha_v3_token,
    )


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
        """List models enabled for the authenticated Arena direct-mode session."""
        account = load_account_from_env(settings)
        client = _new_arena_client(account, settings)
        try:
            real_models = await client.get_models()
        finally:
            await client.close()

        models = [
            {
                "id": model.id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": model.provider or "arena",
                "description": model.description,
            }
            for model in real_models
        ]
        log.info("Fetched %d models from arena.ai", len(models))
        return {"object": "list", "data": models}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        _: None = Depends(verify_key),
    ) -> Any:
        """Handle chat completion requests."""
        client: ArenaHttpClient | None = None
        try:
            account = load_account_from_env(settings)
            client = _new_arena_client(account, settings)

            messages = [
                {"role": message.role, "content": message.content or ""}
                for message in req.messages
            ]
            log.info("chat stream=%s model=%s msgs=%d", req.stream, req.model, len(messages))

            if req.stream:
                # The generator owns and closes this client after the response.
                return StreamingResponse(
                    _stream_response(client, req),
                    media_type="text/event-stream",
                )

            result = await client.chat_completion(
                model=req.model,
                messages=messages,
                temperature=req.temperature or 1.0,
                max_tokens=req.max_tokens,
                top_p=req.top_p,
                stop=req.stop,
            )
            return _format_response(result, req.model)

        except NotionChatError as e:
            log.error("Chat completion error: %s", e)
            raise HTTPException(status_code=e.status_code, detail=str(e)) from e
        except Exception as e:
            log.exception("Unexpected error during chat completion")
            raise HTTPException(status_code=500, detail="Internal server error") from e
        finally:
            # StreamingResponse consumes the generator after this route returns,
            # so its client is closed by _stream_response instead.
            if client is not None and not req.stream:
                await client.close()

    return app


async def _stream_response(
    client: ArenaHttpClient,
    req: ChatCompletionRequest,
) -> AsyncIterator[str]:
    """Stream chat completion response."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    sent_finish = False
    try:
        messages = [
            {"role": message.role, "content": message.content or ""}
            for message in req.messages
        ]

        async for chunk in client.chat_completion_stream(
            model=req.model,
            messages=messages,
            temperature=req.temperature or 1.0,
            max_tokens=req.max_tokens,
            top_p=req.top_p,
            stop=req.stop,
        ):
            if chunk.content:
                yield _chunk(
                    completion_id=completion_id,
                    created=created,
                    model=req.model,
                    delta={"content": chunk.content},
                )
            if chunk.done and not sent_finish:
                sent_finish = True
                yield _chunk(
                    completion_id=completion_id,
                    created=created,
                    model=req.model,
                    delta={},
                    finish_reason=chunk.finish_reason or "stop",
                )

        if not sent_finish:
            yield _chunk(
                completion_id=completion_id,
                created=created,
                model=req.model,
                delta={},
                finish_reason="stop",
            )
        yield "data: [DONE]\n\n"

    except NotionChatError as e:
        err = {"error": {"message": str(e), "type": "arena_error", "code": e.status_code}}
        yield f"data: {json.dumps(err)}\n\n"
        yield "data: [DONE]\n\n"
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
