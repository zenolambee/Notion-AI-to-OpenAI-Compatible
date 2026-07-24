"""Arena.ai transport used by the OpenAI-compatible API.

Arena's web UI does not expose an OpenAI ``/api/chat`` endpoint.  Direct-mode
chat is created through its streaming ``nextjs-api`` endpoints and the response
uses Arena event records (for example ``a0:"text"`` and
``ad:{"finishReason":"stop"}``).  This module translates that protocol without
pretending an upstream failure is an empty assistant response.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from notionchat.account import ArenaAccount
from notionchat.exceptions import NotionChatError

log = logging.getLogger(__name__)

ARENA_API_BASE = "https://arena.ai"
ARENA_DIRECT_MODE_URL = f"{ARENA_API_BASE}/?mode=direct"
ARENA_CREATE_EVALUATION_PATH = "/nextjs-api/stream/create-evaluation"
DEFAULT_TIMEOUT = 120.0


class ArenaUpstreamError(NotionChatError):
    """An Arena response that cannot be represented as a successful completion."""


@dataclass(slots=True)
class ArenaModel:
    """One model exposed by Arena's direct-mode model picker."""

    # ``id`` is the public name returned to OpenAI clients. ``upstream_id`` is
    # the opaque Arena model id that must be sent in create-evaluation.
    id: str
    name: str
    upstream_id: str = ""
    provider: str = ""
    description: str = ""
    supports_streaming: bool = True
    modality: str = "chat"


@dataclass(slots=True)
class ArenaStreamChunk:
    """A parsed Arena stream record."""

    content: str = ""
    done: bool = False
    model: str = ""
    finish_reason: str | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class ArenaHttpClient:
    """HTTP client for the supported Arena direct-mode web protocol.

    A valid logged-in browser session is required.  Arena may also require a
    *fresh, user-obtained* reCAPTCHA token; this client accepts one but never
    attempts to solve or bypass a challenge.
    """

    def __init__(
        self,
        account: ArenaAccount,
        *,
        base_url: str = ARENA_API_BASE,
        recaptcha_v3_token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._account = account
        self._base_url = base_url.rstrip("/") or ARENA_API_BASE
        self._recaptcha_v3_token = (recaptcha_v3_token or "").strip()
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_headers(self, *, stream: bool = True) -> dict[str, str]:
        """Build browser-like headers needed by the Arena web endpoint."""
        return {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
            "Origin": self._base_url,
            "Referer": f"{self._base_url}/?mode=direct",
            "User-Agent": self._account.user_agent,
            "Cookie": self._get_cookie_header(),
        }

    def _get_cookie_header(self) -> str:
        if self._account.full_cookie:
            return self._account.full_cookie.strip().rstrip(";")
        return f"arena-auth-prod-v1={self._account.token_v2}"

    async def get_models(self) -> list[ArenaModel]:
        """Read the direct-mode model list embedded in Arena's web page.

        The old ``/api/models`` URL is not an Arena model API.  Returning an
        empty list on a page/authorization failure is intentional: inventing a
        legacy model name makes a later chat request fail opaquely.
        """
        client = await self._get_client()
        try:
            response = await client.get(
                f"{self._base_url}/?mode=direct",
                headers=self._build_headers(stream=False),
            )
        except httpx.HTTPError as exc:
            log.warning("Could not load Arena direct-mode page: %s", exc)
            return []

        if response.status_code != 200:
            log.warning(
                "Could not load Arena direct-mode page: HTTP %s: %s",
                response.status_code,
                _safe_body_preview(response.text),
            )
            return []

        try:
            models = self._parse_models_page(response.text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            log.warning("Could not parse Arena model list: %s", exc)
            return []

        if not models:
            log.warning("Arena direct-mode page did not contain an enabled model list")
        return models

    @staticmethod
    def _parse_models_page(page: str) -> list[ArenaModel]:
        """Extract ``initialModels`` from current and escaped Next.js payloads."""
        raw_models = _extract_initial_models(page)
        models: list[ArenaModel] = []
        seen: set[str] = set()

        for item in raw_models:
            if not isinstance(item, dict) or item.get("isDisabled"):
                continue
            upstream_id = str(item.get("id") or item.get("modelId") or "").strip()
            public_name = str(
                item.get("publicName") or item.get("name") or item.get("displayName") or ""
            ).strip()
            if not upstream_id or not public_name or public_name in seen:
                continue

            capabilities = item.get("capabilities")
            output_capabilities = (
                capabilities.get("outputCapabilities", {}) if isinstance(capabilities, dict) else {}
            )
            modality = "image" if output_capabilities.get("image") else (
                "search" if output_capabilities.get("search") else "chat"
            )
            seen.add(public_name)
            models.append(
                ArenaModel(
                    id=public_name,
                    name=public_name,
                    upstream_id=upstream_id,
                    provider=str(item.get("organization") or item.get("provider") or "arena"),
                    description=str(item.get("description") or ""),
                    supports_streaming=True,
                    modality=modality,
                )
            )

        return models

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """Collect Arena's event stream for a non-streaming OpenAI request."""
        del temperature, max_tokens, top_p, stop  # Direct mode does not accept these controls.
        parts: list[str] = []
        finish_reason = "stop"
        async for chunk in self.chat_completion_stream(model=model, messages=messages):
            if chunk.content:
                parts.append(chunk.content)
            if chunk.done and chunk.finish_reason:
                finish_reason = _normalize_finish_reason(chunk.finish_reason)

        content = "".join(parts)
        if not content:
            raise ArenaUpstreamError(
                "Arena completed the request without assistant text.", status_code=502
            )
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ]
        }

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
    ) -> AsyncIterator[ArenaStreamChunk]:
        """Create one Arena direct-mode evaluation and yield its event records."""
        del temperature, max_tokens, top_p, stop  # See chat_completion().
        arena_model = await self._resolve_model(model)
        prompt = _messages_to_prompt(messages)
        payload = self._build_evaluation_payload(arena_model, prompt)
        url = f"{self._base_url}{ARENA_CREATE_EVALUATION_PATH}"
        client = await self._get_client()
        received_event = False
        received_text = False

        try:
            async with client.stream(
                "POST", url, headers=self._build_headers(), json=payload
            ) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    raise _upstream_http_error(response.status_code, body)

                async for line in response.aiter_lines():
                    chunk = self._parse_stream_chunk(line, model)
                    if chunk is None:
                        continue
                    received_event = True
                    if chunk.error:
                        raise ArenaUpstreamError(chunk.error, status_code=502)
                    if chunk.content:
                        received_text = True
                    yield chunk
        except ArenaUpstreamError:
            raise
        except httpx.HTTPError as exc:
            raise ArenaUpstreamError(
                f"Arena transport error: {exc}", status_code=502
            ) from exc

        if not received_event:
            raise ArenaUpstreamError(
                "Arena returned no stream events. Refresh the browser cookie and verify that "
                "Arena direct mode is available in that browser session.",
                status_code=502,
            )
        if not received_text:
            raise ArenaUpstreamError(
                "Arena returned stream metadata but no assistant text.", status_code=502
            )

    async def _resolve_model(self, requested: str) -> ArenaModel:
        models = await self.get_models()
        if not models:
            raise ArenaUpstreamError(
                "Could not discover Arena's current direct-mode models. Run GET /v1/models "
                "after refreshing your Arena browser cookie.",
                status_code=502,
            )

        normal_requested = _normalise_model_name(requested)
        for model in models:
            if requested == model.id or requested == model.upstream_id:
                return model
            if normal_requested == _normalise_model_name(model.id):
                return model

        available = ", ".join(model.id for model in models[:12])
        if len(models) > 12:
            available += ", ..."
        raise ArenaUpstreamError(
            f"Arena model {requested!r} is not available for this session. "
            f"Use a model id from GET /v1/models (for example: {available}).",
            status_code=400,
        )

    def _build_evaluation_payload(self, model: ArenaModel, prompt: str) -> dict[str, Any]:
        """Build the request shape used by the Arena direct-mode web UI."""
        session_id = str(uuid.uuid4())
        payload: dict[str, Any] = {
            "id": session_id,
            "mode": "direct",
            "modelAId": model.upstream_id,
            "userMessageId": str(uuid.uuid4()),
            "modelAMessageId": str(uuid.uuid4()),
            # Arena's endpoint expects both message ids even in direct mode.
            "modelBMessageId": str(uuid.uuid4()),
            "userMessage": {
                "content": prompt,
                "experimental_attachments": [],
                "metadata": {},
            },
            "modality": model.modality,
        }
        if self._recaptcha_v3_token:
            payload["recaptchaV3Token"] = self._recaptcha_v3_token
        return payload

    def _parse_stream_chunk(self, line: str, model: str) -> ArenaStreamChunk | None:
        """Parse Arena records and conventional SSE/OpenAI records defensively."""
        record = line.strip()
        if not record or record.startswith(("event:", "id:", ":")):
            return None
        if record.startswith("data:"):
            record = record[5:].strip()
        if not record:
            return None
        if record == "[DONE]":
            return ArenaStreamChunk(done=True, model=model)

        # Arena current direct-mode protocol: a0 text, ag reasoning, a3 error,
        # ad metadata.  The payload following the prefix is JSON.
        if len(record) >= 3 and record[:2] in {"a0", "a2", "a3", "ad"} and record[2] == ":":
            kind, raw_value = record[:2], record[3:]
            value = _decode_arena_record(raw_value)
            if kind == "a0":
                return ArenaStreamChunk(content=_as_text(value), model=model, raw={"event": kind})
            if kind == "a3":
                return ArenaStreamChunk(
                    model=model,
                    error=f"Arena error: {_as_text(value) or 'unknown upstream error'}",
                    raw={"event": kind},
                )
            if kind == "ad":
                metadata = value if isinstance(value, dict) else {}
                return ArenaStreamChunk(
                    done=True,
                    model=model,
                    finish_reason=(
                        metadata.get("finishReason")
                        or metadata.get("finish_reason")
                        or metadata.get("stopReason")
                    ),
                    raw=metadata,
                )
            # ``ag`` is a reasoning event (letter g, not a numeric event).  It
            # is handled below so reasoning is not incorrectly emitted as final
            # assistant content.

        if record.startswith("ag:"):
            return ArenaStreamChunk(model=model, raw={"event": "ag"})

        # Some deployments wrap the same response in OpenAI-style SSE.
        try:
            data = json.loads(record)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        if data.get("error"):
            return ArenaStreamChunk(model=model, error=f"Arena error: {_as_text(data['error'])}")
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice.get("delta") or choice.get("message") or {}
            content = _as_text(delta.get("content")) if isinstance(delta, dict) else ""
            return ArenaStreamChunk(
                content=content,
                done=choice.get("finish_reason") is not None,
                model=model,
                finish_reason=choice.get("finish_reason"),
                raw=data,
            )
        if data.get("text") is not None:
            return ArenaStreamChunk(content=_as_text(data["text"]), model=model, raw=data)
        return None


def _extract_initial_models(page: str) -> list[Any]:
    """Find the model array in either raw or JSON-escaped Next.js page data."""
    # Next.js flight data stores JSON quotes as ``\\\"`` inside a JavaScript
    # string. One unescape pass is sufficient for the model-array payload.
    for candidate in (page, page.replace('\\\"', '"')):
        marker = '"initialModels":'
        start = candidate.find(marker)
        if start < 0:
            continue
        array_start = candidate.find("[", start + len(marker))
        if array_start < 0:
            continue
        encoded = _balanced_json_array(candidate, array_start)
        if encoded is None:
            continue
        decoded = json.loads(encoded)
        if isinstance(decoded, list):
            return decoded
    return []


def _balanced_json_array(source: str, start: int) -> str | None:
    """Return one JSON array, respecting strings and escaped quote characters."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(source)):
        char = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    return None


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Convert OpenAI history into the single user text accepted by direct mode."""
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role") or "user").strip().lower()
        content = _as_text(message.get("content"))
        if not content:
            continue
        if role == "user" and len(messages) == 1:
            rendered.append(content)
        elif role == "system":
            rendered.append(f"System instructions:\n{content}")
        elif role == "assistant":
            rendered.append(f"Assistant:\n{content}")
        else:
            rendered.append(f"User:\n{content}")
    prompt = "\n\n".join(rendered).strip()
    if not prompt:
        raise ArenaUpstreamError("No text content was supplied in messages.", status_code=400)
    return prompt


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and item.get("type") in ("text", "input_text"):
                pieces.append(_as_text(item.get("text") or item.get("content")))
        return "".join(pieces)
    if isinstance(value, dict):
        return str(value.get("message") or value.get("detail") or value.get("error") or "")
    return str(value)


def _decode_arena_record(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Preserve a useful error rather than discarding a server-format change.
        return value.strip().strip('"')


def _normalise_model_name(value: str) -> str:
    value = value.strip().lower()
    if value.startswith("arena-"):
        value = value.removeprefix("arena-")
    return re.sub(r"[^a-z0-9]+", "", value)


def _normalize_finish_reason(value: str) -> str:
    normalised = value.lower().replace("_", "-")
    if normalised in {"length", "max-tokens", "max-tokens-reached"}:
        return "length"
    if normalised in {"content-filter", "content-filtered"}:
        return "content_filter"
    return "stop"


def _safe_body_preview(body: str, limit: int = 300) -> str:
    # Do not let a giant Cloudflare HTML page flood application logs/errors.
    return " ".join(body.split())[:limit]


def _upstream_http_error(status_code: int, body: str) -> ArenaUpstreamError:
    preview = _safe_body_preview(body)
    lower = preview.lower()
    if "recaptcha" in lower:
        message = (
            "Arena rejected the request because its browser reCAPTCHA check was not accepted. "
            "Open Arena in your normal browser, complete any challenge, then retry with a fresh session; "
            "this proxy does not bypass reCAPTCHA."
        )
    elif status_code in (401, 403):
        message = (
            f"Arena authentication was rejected (HTTP {status_code}). Refresh ARENA_COOKIE "
            "from a logged-in Arena browser session."
        )
    else:
        message = f"Arena request failed with HTTP {status_code}"
        if preview:
            message += f": {preview}"
    client_status = status_code if status_code in (400, 401, 403, 429) else 502
    return ArenaUpstreamError(message, status_code=client_status)


async def get_arena_models(account: ArenaAccount) -> list[dict[str, Any]]:
    """Get current Arena models in OpenAI's ``/v1/models`` representation."""
    client = ArenaHttpClient(account)
    try:
        models = await client.get_models()
        return [
            {
                "id": model.id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": model.provider or "arena",
                "description": model.description,
            }
            for model in models
        ]
    finally:
        await client.close()
