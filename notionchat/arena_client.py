"""Arena.ai (Chatbot Arena) HTTP client and model-catalog helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from notionchat.account import ArenaAccount
from notionchat.exceptions import NotionChatError

log = logging.getLogger(__name__)

ARENA_API_BASE = "https://arena.ai"
ARENA_API_CHAT = f"{ARENA_API_BASE}/api/chat"
ARENA_API_MODELS = f"{ARENA_API_BASE}/api/models"
ARENA_DIRECT_MODE_URL = f"{ARENA_API_BASE}/?mode=direct"
DEFAULT_TIMEOUT = 120.0


@dataclass(frozen=True)
class ArenaModel:
    """A callable Arena model.

    ``id`` is deliberately the value Arena expects in a chat request (normally an
    Arena UUID).  ``name`` is display-only and is never sent as a model id.
    """

    id: str
    name: str
    provider: str = ""
    description: str = ""
    supports_streaming: bool = True


@dataclass
class ArenaStreamChunk:
    content: str = ""
    done: bool = False
    model: str = ""
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.raw is None:
            self.raw = {}


def _first_text(item: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _model_from_item(item: Any) -> ArenaModel | None:
    """Convert the several catalog shapes used by Arena into one safe format."""
    if isinstance(item, str):
        return ArenaModel(id=item.strip(), name=item.strip()) if item.strip() else None
    if not isinstance(item, dict):
        return None

    # API catalogues have used each of these spellings.  Prefer explicit model
    # identifiers over a generic object id.
    model_id = _first_text(item, "model_id", "modelId", "uuid", "id")
    if not model_id:
        return None
    return ArenaModel(
        id=model_id,
        name=_first_text(item, "name", "display_name", "displayName", "title", "model") or model_id,
        provider=_first_text(item, "provider", "owner", "owned_by"),
        description=_first_text(item, "description", "summary"),
        supports_streaming=bool(item.get("streaming", item.get("supports_streaming", True))),
    )


def parse_model_catalog(data: Any) -> list[ArenaModel]:
    """Parse an Arena API response or a local ``models.json`` file.

    Local files may be an OpenAI ``{\"data\": [...]}`` response, an Arena
    ``{\"models\": [...]}`` response, or a plain array.  Invalid entries and
    duplicate ids are omitted so survey/metadata objects never become models.
    """
    raw_models: Any = data
    if isinstance(data, dict):
        raw_models = data.get("models") or data.get("data") or data.get("result") or []
        # Some endpoints wrap the usable list one level further down.
        if isinstance(raw_models, dict):
            raw_models = raw_models.get("models") or raw_models.get("data") or []
    if not isinstance(raw_models, list):
        return []

    models: list[ArenaModel] = []
    seen: set[str] = set()
    for item in raw_models:
        model = _model_from_item(item)
        if model is None or model.id in seen:
            continue
        seen.add(model.id)
        models.append(model)
    return models


def load_model_catalog(path: Path) -> list[ArenaModel]:
    """Load a user-maintained catalog; a missing file is intentionally OK."""
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as file:
            return parse_model_catalog(json.load(file))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not load Arena model catalog %s: %s", path, exc)
        return []


def openai_models(models: list[ArenaModel]) -> list[dict[str, Any]]:
    return [
        {
            "id": model.id,
            "object": "model",
            "created": 1700000000,
            "owned_by": model.provider or "arena",
            "description": model.description or model.name,
        }
        for model in models
    ]


def resolve_model_id(requested: str, models: list[ArenaModel]) -> str:
    """Resolve a raw id or a display name, without inventing legacy model ids."""
    wanted = requested.strip()
    if not wanted:
        raise NotionChatError("A non-empty model id is required.", status_code=400)

    # Exact ids remain the canonical and unambiguous API contract.
    for model in models:
        if model.id == wanted:
            return model.id

    # Names are convenient for manually-maintained catalogues.  Only accept an
    # unambiguous case-insensitive match; aliases such as arena-gpt-4o are not
    # fabricated because Arena changes its model UUIDs over time.
    matches = [model.id for model in models if model.name.casefold() == wanted.casefold()]
    if len(matches) == 1:
        return matches[0]

    available = ", ".join(f"{m.name} ({m.id})" for m in models[:10])
    suffix = f" Available: {available}" if available else ""
    raise NotionChatError(
        f"Model {wanted!r} is not in the Arena model catalog. Use a raw Arena model id returned by "
        f"GET /v1/models, or add it to models.json.{suffix}",
        status_code=400,
    )


class ArenaHttpClient:
    """HTTP client for Arena's browser-authenticated API."""

    def __init__(self, account: ArenaAccount, timeout: float = DEFAULT_TIMEOUT) -> None:
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
        return {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Origin": ARENA_API_BASE,
            "Referer": ARENA_DIRECT_MODE_URL,
            "User-Agent": self._account.user_agent,
            "Cookie": self._get_cookie_header(),
        }

    def _get_cookie_header(self) -> str:
        if self._account.full_cookie:
            return self._account.full_cookie.strip().rstrip(";")
        return f"arena-auth-prod-v1={self._account.token_v2}"

    async def get_models(self) -> list[ArenaModel]:
        client = await self._get_client()
        try:
            response = await client.get(ARENA_API_MODELS, headers=self._build_headers())
            if response.status_code != 200:
                log.warning("Failed to fetch models: %s - %s", response.status_code, response.text[:200])
                return []
            return parse_model_catalog(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("Could not fetch Arena models: %s", exc)
            return []

    async def chat_completion(self, *, model: str, messages: list[dict[str, Any]], temperature: float = 1.0,
                              max_tokens: int | None = None, top_p: float | None = None,
                              stop: str | list[str] | None = None) -> dict[str, Any]:
        client = await self._get_client()
        response = await client.post(
            ARENA_API_CHAT,
            headers=self._build_headers(),
            json=self._build_chat_payload(model, messages, temperature, max_tokens, top_p, stop, False),
        )
        if response.status_code != 200:
            raise NotionChatError(f"Arena chat request failed ({response.status_code}): {response.text[:500]}", status_code=502)
        try:
            return response.json()
        except ValueError as exc:
            raise NotionChatError("Arena returned an invalid chat response.", status_code=502) from exc

    async def chat_completion_stream(self, *, model: str, messages: list[dict[str, Any]], temperature: float = 1.0,
                                     max_tokens: int | None = None, top_p: float | None = None,
                                     stop: str | list[str] | None = None) -> AsyncIterator[ArenaStreamChunk]:
        client = await self._get_client()
        headers = self._build_headers()
        headers["Accept"] = "text/event-stream"
        async with client.stream("POST", ARENA_API_CHAT, headers=headers,
                                 json=self._build_chat_payload(model, messages, temperature, max_tokens, top_p, stop, True)) as response:
            if response.status_code != 200:
                text = (await response.aread()).decode(errors="replace")
                raise NotionChatError(f"Arena streaming request failed ({response.status_code}): {text[:500]}", status_code=502)
            async for line in response.aiter_lines():
                chunk = self._parse_stream_chunk(line, model)
                if chunk:
                    yield chunk

    @staticmethod
    def _build_chat_payload(model: str, messages: list[dict[str, Any]], temperature: float,
                            max_tokens: int | None, top_p: float | None, stop: str | list[str] | None,
                            stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, "stream": stream}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if stop is not None:
            payload["stop"] = stop
        return payload

    @staticmethod
    def _parse_stream_chunk(line: str, model: str) -> ArenaStreamChunk | None:
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if not line:
            return None
        if line == "[DONE]":
            return ArenaStreamChunk(done=True, model=model)
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None
        done = bool(data.get("done") or data.get("finish"))
        finish_reason = data.get("finish_reason") or data.get("stop_reason")
        content = ""
        if data.get("choices"):
            choice = data["choices"][0]
            delta = choice.get("delta") or {}
            content = delta.get("content", "")
            done = done or choice.get("finish_reason") is not None
            finish_reason = choice.get("finish_reason") or finish_reason
        elif "text" in data:
            content = data.get("text", "")
        elif isinstance(data.get("message"), dict):
            content = data["message"].get("content", "")
        elif "content" in data:
            content = data.get("content", "")
        return ArenaStreamChunk(content=content, done=done, model=model, finish_reason=finish_reason, raw=data) if content or done else None


async def get_arena_models(account: ArenaAccount) -> list[dict[str, Any]]:
    client = ArenaHttpClient(account)
    try:
        return openai_models(await client.get_models())
    finally:
        await client.close()
