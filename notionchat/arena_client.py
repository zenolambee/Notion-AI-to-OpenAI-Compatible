"""
Arena.ai (Chatbot Arena) API client for OpenAI-compatible chat completions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx

from notionchat.account import ArenaAccount

log = logging.getLogger(__name__)

# Arena.ai API endpoints
ARENA_API_BASE = "https://arena.ai"
ARENA_API_CHAT = f"{ARENA_API_BASE}/api/chat"
ARENA_API_MODELS = f"{ARENA_API_BASE}/api/models"
ARENA_DIRECT_MODE_URL = f"{ARENA_API_BASE}/?mode=direct"

# Default timeout for requests (seconds)
DEFAULT_TIMEOUT = 120.0


@dataclass
class ArenaModel:
    """Represents an Arena.ai model."""
    id: str
    name: str
    provider: str = ""
    description: str = ""
    supports_streaming: bool = True


@dataclass
class ArenaStreamChunk:
    """Represents a streaming response chunk."""
    content: str = ""
    done: bool = False
    model: str = ""
    finish_reason: str | None = None
    raw: dict[str, Any] = None

    def __post_init__(self):
        if self.raw is None:
            self.raw = {}


class ArenaHttpClient:
    """HTTP client for arena.ai API."""

    def __init__(
        self,
        account: ArenaAccount,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._account = account
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_headers(self) -> dict[str, str]:
        """Build request headers with cookie authentication."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Origin": ARENA_API_BASE,
            "Referer": ARENA_DIRECT_MODE_URL,
            "User-Agent": self._account.user_agent,
            "Cookie": self._get_cookie_header(),
        }

    def _get_cookie_header(self) -> str:
        """Build cookie header from account."""
        if self._account.full_cookie:
            return self._account.full_cookie.strip().rstrip(";")
        return f"arena-auth-prod-v1={self._account.token_v2}"

    async def get_models(self) -> list[ArenaModel]:
        """Fetch available models from arena.ai."""
        client = await self._get_client()
        headers = self._build_headers()

        try:
            resp = await client.get(ARENA_API_MODELS, headers=headers)
            if resp.status_code != 200:
                log.warning(
                    "Failed to fetch models: %s - %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return []

            data = resp.json()
            return self._parse_models_response(data)
        except httpx.HTTPError as e:
            log.error("HTTP error fetching models: %s", e)
            return []
        except Exception as e:
            log.error("Error fetching models: %s", e)
            return []

    def _parse_models_response(self, data: dict[str, Any]) -> list[ArenaModel]:
        """Parse arena.ai models response."""
        models: list[ArenaModel] = []

        # Try different response formats
        raw_models = (
            data.get("models", [])
            or data.get("data", [])
            or data.get("result", [])
        )

        for m in raw_models:
            if isinstance(m, dict):
                models.append(ArenaModel(
                    id=m.get("id", ""),
                    name=m.get("name", m.get("id", "")),
                    provider=m.get("provider", ""),
                    description=m.get("description", ""),
                    supports_streaming=m.get("streaming", True),
                ))
            elif isinstance(m, str):
                models.append(ArenaModel(id=m, name=m))

        return models

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """Send chat completion request (non-streaming)."""
        client = await self._get_client()
        headers = self._build_headers()

        payload = self._build_chat_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            stream=False,
        )

        try:
            resp = await client.post(ARENA_API_CHAT, headers=headers, json=payload)
            if resp.status_code != 200:
                log.error(
                    "Chat completion failed: %s - %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return {"choices": [{"message": {"content": ""}}]}
            return resp.json()
        except httpx.HTTPError as e:
            log.error("HTTP error in chat completion: %s", e)
            return {"choices": [{"message": {"content": ""}}]}

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
    ) -> AsyncIterator[ArenaStreamChunk]:
        """Send chat completion request (streaming)."""
        client = await self._get_client()
        headers = self._build_headers()
        headers["Accept"] = "text/event-stream"

        payload = self._build_chat_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            stream=True,
        )

        try:
            async with client.stream(
                "POST",
                ARENA_API_CHAT,
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    log.error("Streaming failed: %s - %s", resp.status_code, text[:200])
                    return

                async for line in resp.aiter_lines():
                    if line.strip():
                        chunk = self._parse_stream_chunk(line, model)
                        if chunk:
                            yield chunk
        except httpx.HTTPError as e:
            log.error("HTTP error in streaming: %s", e)
        except Exception as e:
            log.error("Error in streaming: %s", e)

    def _build_chat_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        top_p: float | None,
        stop: str | list[str] | None,
        stream: bool,
    ) -> dict[str, Any]:
        """Build arena.ai chat API payload."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens
        if top_p:
            payload["top_p"] = top_p
        if stop:
            payload["stop"] = stop
        payload["stream"] = stream

        return payload

    def _parse_stream_chunk(self, line: str, model: str) -> ArenaStreamChunk | None:
        """Parse a streaming response line."""
        try:
            # Handle SSE format: "data: {...}"
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]" or line == "data: [DONE]":
                return ArenaStreamChunk(content="", done=True, model=model)

            data = json.loads(line)
            return self._parse_sse_chunk(data, model)
        except json.JSONDecodeError:
            return None

    def _parse_sse_chunk(self, data: dict[str, Any], model: str) -> ArenaStreamChunk | None:
        """Parse SSE chunk data."""
        content = ""
        done = data.get("done", False) or data.get("finish", False)
        finish_reason = data.get("finish_reason") or data.get("stop_reason")

        # Try OpenAI-like format first
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            delta = choice.get("delta", {})
            content = delta.get("content", "")
            done = done or choice.get("finish_reason") is not None
            finish_reason = choice.get("finish_reason") or finish_reason

        # Try arena-specific formats
        elif "text" in data:
            content = data.get("text", "")
        elif "message" in data:
            content = data.get("message", {}).get("content", "")
        elif "content" in data:
            content = data.get("content", "")

        if not content and not done:
            return None

        return ArenaStreamChunk(
            content=content,
            done=done,
            model=model,
            finish_reason=finish_reason,
            raw=data,
        )


async def get_arena_models(account: ArenaAccount) -> list[dict[str, Any]]:
    """Get list of available arena.ai models in OpenAI format."""
    client = ArenaHttpClient(account)
    try:
        models = await client.get_models()
        await client.close()
        return [
            {
                "id": m.id,
                "object": "model",
                "created": 1700000000,
                "owned_by": m.provider or "arena",
                "description": m.description,
            }
            for m in models
        ]
    except Exception as e:
        log.error("Error getting arena models: %s", e)
        await client.close()
        return []


async def sync_arena_models_to_catalog(account: ArenaAccount) -> dict[str, Any]:
    """Fetch the raw model list from Arena and persist it to the local catalog.

    Returns the saved catalog dict (with ``models`` and ``synced_at`` keys).
    """
    from notionchat.model_catalog import save_catalog  # noqa: PLC0415

    client = ArenaHttpClient(account)
    try:
        resp_models = await client.get_models()
    finally:
        await client.close()

    # Convert ArenaModel dataclasses back to plain dicts for JSON serialisation
    raw_models: list[dict[str, Any]] = []
    for m in resp_models:
        entry: dict[str, Any] = {"id": m.id}
        if m.name:
            entry["name"] = m.name
        if m.provider:
            entry["provider"] = m.provider
        if m.description:
            entry["description"] = m.description
        entry["streaming"] = m.supports_streaming
        raw_models.append(entry)

    catalog: dict[str, Any] = {
        "synced_at": __import__("time").time(),
        "models": raw_models,
    }
    save_catalog(catalog)
    return catalog
