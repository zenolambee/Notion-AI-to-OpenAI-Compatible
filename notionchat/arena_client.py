"""
Arena.ai (Chatbot Arena) API client for OpenAI-compatible chat completions.

This client speaks the real arena.ai protocol used by the browser:

    POST https://arena.ai/nextjs-api/stream/create-evaluation-session-message
    POST https://arena.ai/nextjs-api/stream/post-to-evaluation/{sessionId}
    PUT  https://arena.ai/nextjs-api/stream/retry-evaluation-session-message/{sessionId}/messages/{messageId}

The payload uses arena.ai's evaluation-session shape (id, mode="direct",
modelAId (UUID), userMessage, modality, recaptchaV3Token, ...). Responses
stream back as Vercel-AI-SDK NDJSON lines like:

    a0:"Hello"           # text delta for model A
    ag:"reasoning..."    # reasoning delta
    ad:{"finishReason":"stop"}
    a2:[{"type":"image",...}]
    a3:"error message"

References for the wire format: CloudWaddie/LMArenaBridge on GitHub.

Requires the caller to supply, via ArenaAccount / cookies / env:
  - arena-auth-prod-v1 cookie (from your logged-in browser)
  - optionally cf_clearance / __cf_bm / _cfuvid / provisional_user_id
  - optionally ARENA_RECAPTCHA_TOKEN (a fresh grecaptcha/enterprise v3 token
    minted from arena.ai; short-lived, ~2 min)
  - optionally a preloaded models list at ARENACHAT_MODELS_FILE (JSON array
    from arena.ai's client-side model catalog) — needed to translate a
    human "publicName" like "claude-opus-4-5-20251101" into arena's
    internal UUID modelAId.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from notionchat.account import ArenaAccount
from notionchat.exceptions import NotionChatError
from notionchat.recaptcha import RecaptchaTokenManager

log = logging.getLogger(__name__)

# Arena.ai origin & endpoints
ARENA_ORIGIN = "https://arena.ai"
ARENA_DIRECT_MODE_URL = f"{ARENA_ORIGIN}/?mode=direct"

# Next.js API paths used by the browser client.
STREAM_CREATE_EVALUATION_PATH = (
    "/nextjs-api/stream/create-evaluation-session-message"
)
STREAM_POST_TO_EVALUATION_PATH = "/nextjs-api/stream/post-to-evaluation"
STREAM_RETRY_MESSAGE_PATH = (
    "/nextjs-api/stream/retry-evaluation-session-message"
)

DEFAULT_TIMEOUT = 300.0
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Global reCAPTCHA manager (one per account/process)
# ---------------------------------------------------------------------------


def _auto_recaptcha_enabled() -> bool:
    raw = os.getenv("ARENA_RECAPTCHA_AUTO", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


_recaptcha_manager: RecaptchaTokenManager | None = None
_recaptcha_lock = asyncio.Lock()


async def _get_recaptcha_manager(account: ArenaAccount) -> RecaptchaTokenManager:
    global _recaptcha_manager
    if _recaptcha_manager is not None:
        return _recaptcha_manager
    async with _recaptcha_lock:
        if _recaptcha_manager is None:
            _recaptcha_manager = RecaptchaTokenManager(account)
        return _recaptcha_manager


async def _acquire_recaptcha_token(
    account: ArenaAccount, *, force: bool = False
) -> str:
    """Return a fresh grecaptcha v3 token.

    Precedence:
      1) If ARENA_RECAPTCHA_TOKEN env is set, use it as-is (manual mode).
      2) If ARENA_RECAPTCHA_AUTO is not disabled, mint via Playwright.
      3) Otherwise return "" and let the upstream 403 surface.
    """
    manual = os.getenv("ARENA_RECAPTCHA_TOKEN", "").strip()
    if manual and not force:
        return manual
    if not _auto_recaptcha_enabled():
        return manual  # possibly empty
    try:
        mgr = await _get_recaptcha_manager(account)
        return await mgr.get_token(force=force)
    except NotionChatError:
        raise
    except Exception as e:
        log.error("reCAPTCHA auto mint failed: %s", e)
        return manual  # fall back to manual (maybe empty)


async def shutdown_recaptcha() -> None:
    """Close the shared Playwright browser (call on app shutdown)."""
    global _recaptcha_manager
    if _recaptcha_manager is not None:
        try:
            await _recaptcha_manager.close()
        finally:
            _recaptcha_manager = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ArenaModel:
    """Represents an Arena.ai model."""

    id: str  # arena UUID
    name: str  # publicName
    provider: str = ""
    description: str = ""
    supports_streaming: bool = True
    capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArenaStreamChunk:
    """Represents a streaming response chunk."""

    content: str = ""
    reasoning: str = ""
    done: bool = False
    model: str = ""
    finish_reason: str | None = None
    raw: dict[str, Any] | None = None

    def __post_init__(self):
        if self.raw is None:
            self.raw = {}


# ---------------------------------------------------------------------------
# UUID v7 helper (arena.ai uses UUIDv7 for its message/session IDs)
# ---------------------------------------------------------------------------


def _uuid7() -> str:
    """Generate a UUIDv7 string (time-ordered). Falls back to UUIDv4."""
    try:
        # Python 3.14+ has uuid.uuid7
        return str(uuid.uuid7())  # type: ignore[attr-defined]
    except AttributeError:
        pass
    # Manual UUIDv7 per draft-ietf-uuidrev-rfc4122bis
    ts_ms = int(time.time() * 1000)
    rand_a = int.from_bytes(os.urandom(2), "big") & 0x0FFF
    rand_b = int.from_bytes(os.urandom(8), "big") & 0x3FFFFFFFFFFFFFFF
    val = (
        (ts_ms & 0xFFFFFFFFFFFF) << 80
        | (0x7 << 76)
        | (rand_a << 64)
        | (0b10 << 62)
        | rand_b
    )
    return str(uuid.UUID(int=val))


# ---------------------------------------------------------------------------
# Local model catalog helpers
# ---------------------------------------------------------------------------


def _default_models_path() -> Path:
    raw = os.getenv("ARENACHAT_MODELS_FILE", "").strip()
    if raw:
        return Path(raw).expanduser()
    home = os.getenv("ARENACHAT_HOME", "").strip()
    if home:
        return Path(home).expanduser() / "models.json"
    return Path("models.json")


def load_local_models() -> list[dict[str, Any]]:
    """Load a cached models.json list scraped from arena.ai (best-effort)."""
    path = _default_models_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except Exception as e:
        log.warning("Failed to read %s: %s", path, e)
    return []


def _model_entry_to_dataclass(m: dict[str, Any]) -> ArenaModel:
    return ArenaModel(
        id=str(m.get("id") or ""),
        name=str(m.get("publicName") or m.get("name") or m.get("id") or ""),
        provider=str(m.get("organization") or m.get("provider") or ""),
        description=str(m.get("description") or ""),
        supports_streaming=True,
        capabilities=dict(m.get("capabilities") or {}),
    )


# Alias substrings → (organization filter, publicName substring filter).
# The best (by 'rank' then publicName length descending) matching entry
# is picked, so "gpt-4o" resolves to the newest GPT-4o snapshot.
_MODEL_ALIASES: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    # alias, org substrings, publicName substrings
    ("arena-gpt-4o", ("openai",), ("gpt-4o",)),
    ("gpt-4o", ("openai",), ("gpt-4o",)),
    ("gpt-4o-mini", ("openai",), ("gpt-4o", "mini")),
    ("gpt-4-mini", ("openai",), ("gpt-4", "mini")),
    ("gpt-5-mini", ("openai",), ("gpt-5", "mini")),
    ("gpt-5-nano", ("openai",), ("gpt-5", "nano")),
    ("gpt-4", ("openai",), ("gpt-4",)),
    ("gpt-5", ("openai",), ("gpt-5",)),
    ("gpt-4-turbo", ("openai",), ("gpt-4-turbo", "gpt-4-1106")),
    ("gpt-4.1", ("openai",), ("gpt-4.1",)),
    ("o1", ("openai",), ("o1",)),
    ("o3", ("openai",), ("o3",)),
    ("o4", ("openai",), ("o4",)),

    ("arena-claude-3-5-sonnet", ("anthropic",), ("claude-3-5-sonnet",)),
    ("arena-claude-3-opus", ("anthropic",), ("claude-3-opus",)),
    ("arena-claude-3-sonnet", ("anthropic",), ("claude-3-sonnet",)),
    ("arena-claude-3-haiku", ("anthropic",), ("claude-3-haiku",)),
    ("claude-opus", ("anthropic",), ("claude-opus",)),
    ("claude-sonnet", ("anthropic",), ("claude-sonnet",)),
    ("claude-haiku", ("anthropic",), ("claude-haiku",)),
    ("claude-3-5-sonnet", ("anthropic",), ("claude-3-5-sonnet",)),
    ("claude-3-opus", ("anthropic",), ("claude-3-opus",)),
    ("claude-3-haiku", ("anthropic",), ("claude-3-haiku",)),
    ("claude", ("anthropic",), ("claude",)),

    ("arena-gemini-1.5-pro", ("google",), ("gemini-1.5-pro",)),
    ("arena-gemini-1.5-flash", ("google",), ("gemini-1.5-flash",)),
    ("gemini-2.5-pro", ("google",), ("gemini-2.5-pro",)),
    ("gemini-2.5-flash", ("google",), ("gemini-2.5-flash",)),
    ("gemini-pro", ("google",), ("gemini", "pro")),
    ("gemini-flash", ("google",), ("gemini", "flash")),
    ("gemini", ("google",), ("gemini",)),

    ("arena-llama-3-70b", ("meta",), ("llama-3", "70b")),
    ("arena-llama-3-8b", ("meta",), ("llama-3", "8b")),
    ("llama", ("meta",), ("llama",)),

    ("arena-mixtral-8x7b", ("mistral",), ("mixtral-8x7b",)),
    ("mistral", ("mistral",), ("mistral",)),
    ("mixtral", ("mistral",), ("mixtral",)),

    ("grok", ("xai",), ("grok",)),
    ("deepseek", ("deepseek",), ("deepseek",)),
    ("qwen", ("alibaba", "qwen"), ("qwen",)),
]


_SMALL_VARIANT_MARKERS = ("mini", "nano", "lite", "small", "tiny", "flash-lite")


def _extract_date_score(name: str) -> int:
    """Return YYYYMMDD from a name like 'gpt-4o-2024-11-20' or 0."""
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", name)
    if m:
        try:
            return int(m.group(1) + m.group(2) + m.group(3))
        except Exception:
            return 0
    return 0


def _score_alias_match(
    requested_lower: str,
    entry: dict[str, Any],
    org_subs: tuple[str, ...],
    name_subs: tuple[str, ...],
) -> int | None:
    """Return a match score (higher = better) or None if doesn't match."""
    org = str(entry.get("organization") or "").lower()
    pub = str(entry.get("publicName") or entry.get("name") or "").lower()
    if not org or not pub:
        return None
    if not any(o in org for o in org_subs):
        return None
    if not all(s in pub for s in name_subs):
        return None

    score = 0

    # Prefer newer dated snapshots (gpt-4o-2024-11-20 > gpt-4o-2024-05-13).
    score += _extract_date_score(pub) * 10

    # Prefer arena's rank when available (lower rank number = better model).
    rank = entry.get("rank")
    try:
        score += -int(rank) * 100_000_000
    except Exception:
        pass

    # If the user didn't ask for a small variant, penalize mini/nano/lite.
    for marker in _SMALL_VARIANT_MARKERS:
        if marker in pub and marker not in requested_lower:
            score -= 10_000_000_000

    # Tiebreaker: shorter names are usually the "canonical" alias.
    score -= len(pub)

    return score


def _resolve_alias(
    requested: str, models: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Try common aliases (arena-gpt-4o, claude, gemini, ...) → best entry."""
    req = requested.lower().strip()
    for alias, org_subs, name_subs in _MODEL_ALIASES:
        if req != alias:
            continue
        best: tuple[int, dict[str, Any]] | None = None
        for m in models:
            if not isinstance(m, dict):
                continue
            score = _score_alias_match(req, m, org_subs, name_subs)
            if score is None:
                continue
            if best is None or score > best[0]:
                best = (score, m)
        if best is not None:
            log.info(
                "resolved alias %r → %s (org=%s, score=%d)",
                requested,
                best[1].get("publicName"),
                best[1].get("organization"),
                best[0],
            )
            return best[1]
    return None


def _resolve_model(
    requested: str, models: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    """Return (model_uuid, raw_entry) for the requested public name or UUID.

    Resolution order:
      1) exact publicName / id match
      2) case-insensitive publicName match
      3) alias table (arena-gpt-4o, claude, gemini, llama, mistral, grok…)
      4) raw UUID passthrough
    Raises NotionChatError with helpful suggestions if none match.
    """
    if not requested:
        raise NotionChatError("Empty model name", status_code=400)

    # 1) exact match
    for m in models:
        if not isinstance(m, dict):
            continue
        pub = str(m.get("publicName") or m.get("name") or "")
        mid = str(m.get("id") or "")
        if requested == pub or requested == mid:
            if not mid:
                raise NotionChatError(
                    f"Model {requested!r} has no arena UUID id",
                    status_code=502,
                )
            return mid, m

    # 2) case-insensitive publicName match
    req_lower = requested.lower()
    for m in models:
        if not isinstance(m, dict):
            continue
        pub = str(m.get("publicName") or m.get("name") or "").lower()
        if pub == req_lower:
            mid = str(m.get("id") or "")
            if mid:
                return mid, m

    # 3) alias table
    aliased = _resolve_alias(requested, models)
    if aliased is not None:
        mid = str(aliased.get("id") or "")
        if mid:
            return mid, aliased

    # 4) UUID passthrough
    try:
        uuid.UUID(requested)
        return requested, {"publicName": requested}
    except Exception:
        pass

    # Build helpful hint: suggest names that share tokens with the request.
    available = sorted(
        {
            str(m.get("publicName") or "")
            for m in models
            if isinstance(m, dict) and m.get("publicName")
        }
    )
    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", req_lower) if len(t) >= 2]
    scored: list[tuple[int, str]] = []
    for name in available:
        low = name.lower()
        score = sum(1 for t in tokens if t in low)
        if score:
            scored.append((score, name))
    scored.sort(reverse=True)
    suggestions = [n for _, n in scored[:15]]
    hint = ""
    if suggestions:
        hint = " Closest matches: " + ", ".join(suggestions)
    elif available:
        hint = " Some available: " + ", ".join(available[:15])

    raise NotionChatError(
        f"Model {requested!r} not found in local arena model catalog. "
        f"Try `notionchat sync-models` to refresh {_default_models_path()}, "
        f"or send a raw arena UUID." + hint,
        status_code=404,
    )


def _detect_modality(capabilities: dict[str, Any]) -> str:
    outputs = (capabilities or {}).get("outputCapabilities") or {}
    if outputs.get("image"):
        return "image"
    if outputs.get("search"):
        return "search"
    return "chat"


# ---------------------------------------------------------------------------
# Message flattening
# ---------------------------------------------------------------------------


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("type")
                if t in (None, "text", "input_text", "output_text"):
                    txt = item.get("text") or item.get("content") or ""
                    if isinstance(txt, str):
                        parts.append(txt)
                # image_url / image parts are dropped here; the caller
                # would need to build experimental_attachments separately.
        return "".join(parts)
    return str(content)


def _flatten_messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Squash OpenAI-style multi-turn messages into a single prompt.

    Arena.ai's evaluation-session endpoint models one user turn at a time
    against a running session. For a single call we concatenate any prior
    turns into the user prompt with role labels so the model sees the
    conversation context.
    """
    if not messages:
        return ""

    # Split system prompt(s) out
    system_parts = [
        _content_to_text(m.get("content"))
        for m in messages
        if m.get("role") == "system"
    ]
    system_text = "\n\n".join(p for p in system_parts if p).strip()

    convo: list[str] = []
    for m in messages:
        role = str(m.get("role") or "")
        if role == "system":
            continue
        text = _content_to_text(m.get("content"))
        if not text:
            continue
        if role == "user":
            convo.append(f"User: {text}")
        elif role == "assistant":
            convo.append(f"Assistant: {text}")
        elif role == "tool":
            convo.append(f"Tool: {text}")
        else:
            convo.append(text)

    # If the last message is a user message and it's the only turn, don't
    # prepend a "User: " label — send the raw content for the best UX.
    if len(messages) == 1 and messages[0].get("role") == "user":
        body = _content_to_text(messages[0].get("content"))
    elif (
        len(convo) >= 1
        and messages[-1].get("role") == "user"
        and len([m for m in messages if m.get("role") in ("user", "assistant")])
        == 1
    ):
        body = _content_to_text(messages[-1].get("content"))
    else:
        body = "\n\n".join(convo)

    if system_text:
        return f"{system_text}\n\n{body}"
    return body


# ---------------------------------------------------------------------------
# NDJSON stream parsing (Vercel AI SDK style used by arena.ai)
# ---------------------------------------------------------------------------


def _parse_vercel_line(line: str, model: str) -> ArenaStreamChunk | None:
    """Parse one line of arena.ai's NDJSON stream body.

    Recognised prefixes (as observed by LMArenaBridge):
      a0:"..."           -> text delta (model A)
      ag:"..."           -> reasoning delta
      ad:{...}           -> metadata / finishReason
      a2:[{...}]         -> image attachments
      a3:"..."           -> upstream error
      ac:{...}           -> citations (search modality)
      {"choices":[...]}  -> OpenAI-style chunk fallback
    """
    line = line.strip()
    if not line:
        return None
    if line.startswith("data:"):
        line = line[5:].lstrip()
    if not line or line == "[DONE]":
        if line == "[DONE]":
            return ArenaStreamChunk(done=True, model=model, finish_reason="stop")
        return None

    # a0: / ag: / ad: / a3: prefixes
    for prefix, kind in (
        ("a0:", "text"),
        ("ag:", "reasoning"),
        ("ad:", "meta"),
        ("a3:", "error"),
    ):
        if line.startswith(prefix):
            payload = line[len(prefix):]
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                return None
            if kind == "text" and isinstance(obj, str):
                return ArenaStreamChunk(content=obj, model=model)
            if kind == "reasoning" and isinstance(obj, str):
                return ArenaStreamChunk(reasoning=obj, model=model)
            if kind == "meta" and isinstance(obj, dict):
                fr = obj.get("finishReason") or "stop"
                return ArenaStreamChunk(
                    done=True, model=model, finish_reason=str(fr), raw=obj
                )
            if kind == "error":
                msg = obj if isinstance(obj, str) else json.dumps(obj)
                raise NotionChatError(
                    f"arena.ai upstream error: {msg}", status_code=502
                )
            return None

    # OpenAI-style JSON chunk (some endpoints/proxies emit this)
    if line.startswith("{"):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict) and obj.get("choices"):
            choice = obj["choices"][0]
            delta = choice.get("delta") or {}
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or ""
            fr = choice.get("finish_reason")
            if content or reasoning or fr:
                return ArenaStreamChunk(
                    content=content,
                    reasoning=reasoning,
                    done=fr is not None,
                    model=model,
                    finish_reason=fr,
                    raw=obj,
                )
    return None


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class ArenaHttpClient:
    """HTTP client for arena.ai's nextjs-api chat endpoints."""

    def __init__(
        self,
        account: ArenaAccount,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._account = account
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        # Cache session per (model_uuid, conversation_key) so follow-up
        # messages use post-to-evaluation instead of creating a new session.
        # For now we don't persist across requests — each client is short-lived.
        self._sessions: dict[str, str] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=False
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ---- headers / cookies -------------------------------------------------

    def _cookie_header(self) -> str:
        if self._account.full_cookie:
            return self._account.full_cookie.strip().rstrip(";")
        parts = [f"arena-auth-prod-v1={self._account.token_v2}"]
        for env_name, cookie_name in (
            ("ARENA_CF_CLEARANCE", "cf_clearance"),
            ("ARENA_CF_BM", "__cf_bm"),
            ("ARENA_CFUVID", "_cfuvid"),
            ("ARENA_PROVISIONAL_USER_ID", "provisional_user_id"),
        ):
            v = os.getenv(env_name, "").strip()
            if v:
                parts.append(f"{cookie_name}={v}")
        return "; ".join(parts)

    def _build_headers(
        self,
        *,
        streaming: bool,
        recaptcha_token: str = "",
    ) -> dict[str, str]:
        ua = self._account.user_agent or DEFAULT_USER_AGENT
        headers: dict[str, str] = {
            # Next.js server actions expect text/plain
            "Content-Type": "text/plain;charset=UTF-8",
            "Accept": "*/*",
            "Origin": ARENA_ORIGIN,
            "Referer": ARENA_DIRECT_MODE_URL,
            "User-Agent": ua,
            "Cookie": self._cookie_header(),
        }
        recaptcha_action = os.getenv(
            "ARENA_RECAPTCHA_ACTION", "chat_submit"
        ).strip()
        if recaptcha_token:
            headers["X-Recaptcha-Token"] = recaptcha_token
            headers["X-Recaptcha-Action"] = recaptcha_action
        return headers

    # ---- models ------------------------------------------------------------

    async def get_models(self) -> list[ArenaModel]:
        """Return the locally cached models list.

        arena.ai does not expose a stable public JSON models endpoint —
        the client-side catalog is embedded in the Next.js bundle. Users
        should populate `models.json` (or `$ARENACHAT_MODELS_FILE`) with a
        JSON list matching arena's format (`[{id, publicName, organization,
        capabilities}, ...]`). Otherwise return an empty list and the
        openai_api layer falls back to defaults.
        """
        raw = load_local_models()
        return [_model_entry_to_dataclass(m) for m in raw if isinstance(m, dict)]

    # ---- payloads ----------------------------------------------------------

    def _build_create_payload(
        self,
        *,
        model_uuid: str,
        prompt: str,
        modality: str,
        recaptcha_token: str,
        experimental_attachments: list | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build the create-evaluation-session-message payload.

        Returns (session_id, payload).
        """
        session_id = _uuid7()
        payload: dict[str, Any] = {
            "id": session_id,
            "mode": "direct",
            "modelAId": model_uuid,
            "userMessageId": _uuid7(),
            "modelAMessageId": _uuid7(),
            "modelBMessageId": _uuid7(),
            "userMessage": {
                "content": prompt,
                "experimental_attachments": experimental_attachments or [],
                "metadata": {},
            },
            "modality": modality,
            "recaptchaV3Token": recaptcha_token,
        }
        return session_id, payload

    # ---- core: send one turn ----------------------------------------------

    async def _open_arena_stream(
        self,
        *,
        model_uuid: str,
        model_public: str,
        prompt: str,
        modality: str,
        force_fresh_recaptcha: bool = False,
    ):
        """Open the HTTP stream to arena.ai and return the response context.

        Also returns the resolved session_id and the recaptcha_token used,
        so callers can retry with a fresh token on 403.
        """
        client = await self._get_client()

        recaptcha_token = await _acquire_recaptcha_token(
            self._account, force=force_fresh_recaptcha
        )
        if not recaptcha_token:
            log.warning(
                "No reCAPTCHA token available (auto=%s, manual=%s). "
                "arena.ai will very likely return 403.",
                _auto_recaptcha_enabled(),
                bool(os.getenv("ARENA_RECAPTCHA_TOKEN", "").strip()),
            )

        headers = self._build_headers(
            streaming=True, recaptcha_token=recaptcha_token
        )
        session_id, payload = self._build_create_payload(
            model_uuid=model_uuid,
            prompt=prompt,
            modality=modality,
            recaptcha_token=recaptcha_token,
        )
        url = f"{ARENA_ORIGIN}{STREAM_CREATE_EVALUATION_PATH}"
        body = json.dumps(payload, ensure_ascii=False)

        log.info(
            "arena POST %s model=%s(%s) modality=%s prompt_len=%d "
            "recaptcha=%s(len=%d) force_fresh=%s",
            url,
            model_public,
            model_uuid,
            modality,
            len(prompt),
            "yes" if recaptcha_token else "no",
            len(recaptcha_token),
            force_fresh_recaptcha,
        )

        return (
            client.stream("POST", url, headers=headers, content=body),
            session_id,
            recaptcha_token,
        )

    def _is_recaptcha_error(self, status_code: int, body: str) -> bool:
        if status_code != 403:
            return False
        lower = body.lower()
        return (
            "recaptcha" in lower
            or "captcha" in lower
            or "grecaptcha" in lower
        )

    async def _stream_arena(
        self,
        *,
        model_uuid: str,
        model_public: str,
        prompt: str,
        modality: str,
    ) -> AsyncIterator[ArenaStreamChunk]:
        """Open a POST stream to arena.ai and yield parsed chunks.

        On a 403 that looks like a reCAPTCHA validation failure, invalidates
        the cached token and retries once with a freshly minted one.
        """
        attempts = 0
        force_fresh = False
        max_attempts = 2

        while True:
            attempts += 1
            try:
                (
                    stream_ctx,
                    session_id,
                    _token_used,
                ) = await self._open_arena_stream(
                    model_uuid=model_uuid,
                    model_public=model_public,
                    prompt=prompt,
                    modality=modality,
                    force_fresh_recaptcha=force_fresh,
                )
            except httpx.HTTPError as e:
                raise NotionChatError(
                    f"arena.ai network error: {e}", status_code=502
                ) from e

            try:
                async with stream_ctx as resp:
                    log.info(
                        "arena stream status=%s ct=%s",
                        resp.status_code,
                        resp.headers.get("content-type"),
                    )
                    if resp.status_code >= 300:
                        text_bytes = await resp.aread()
                        text = text_bytes.decode("utf-8", errors="replace")
                        log.error(
                            "arena stream error %s: %s",
                            resp.status_code,
                            text[:800],
                        )
                        # Retry once with a fresh recaptcha token on
                        # 403-recaptcha responses.
                        if (
                            self._is_recaptcha_error(resp.status_code, text)
                            and attempts < max_attempts
                            and _auto_recaptcha_enabled()
                        ):
                            log.info(
                                "arena: 403 recaptcha — invalidating token "
                                "and retrying with a fresh mint."
                            )
                            # Nudge the manager to remint.
                            try:
                                mgr = await _get_recaptcha_manager(self._account)
                                mgr.invalidate()
                            except Exception:
                                pass
                            force_fresh = True
                            continue
                        raise NotionChatError(
                            f"arena.ai HTTP {resp.status_code}: {text[:500]}",
                            status_code=502,
                        )

                    self._sessions[model_public] = session_id
                    sampled: list[str] = []
                    any_content = False
                    async for raw_line in resp.aiter_lines():
                        if raw_line is None:
                            continue
                        if not raw_line.strip():
                            continue
                        if len(sampled) < 5:
                            sampled.append(raw_line[:200])
                        chunk = _parse_vercel_line(raw_line, model_public)
                        if chunk is None:
                            continue
                        if chunk.content or chunk.reasoning:
                            any_content = True
                        yield chunk
                        if chunk.done:
                            break

                    if not any_content:
                        log.warning(
                            "arena stream 200 but 0 content chunks. "
                            "First lines: %r",
                            sampled,
                        )
                    return
            except httpx.HTTPError as e:
                raise NotionChatError(
                    f"arena.ai network error: {e}", status_code=502
                ) from e

    # ---- public: OpenAI-shaped API ----------------------------------------

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 1.0,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming chat completion: buffer the stream into one string."""
        del temperature, max_tokens, top_p, stop  # arena ignores these
        catalog = load_local_models()
        model_uuid, entry = _resolve_model(model, catalog)
        modality = _detect_modality(entry.get("capabilities") or {})
        prompt = _flatten_messages_to_prompt(messages)

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = "stop"
        async for chunk in self._stream_arena(
            model_uuid=model_uuid,
            model_public=model,
            prompt=prompt,
            modality=modality,
        ):
            if chunk.content:
                content_parts.append(chunk.content)
            if chunk.reasoning:
                reasoning_parts.append(chunk.reasoning)
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason

        content = "".join(content_parts)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)

        return {
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
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
        """Streaming variant. Delegates to _stream_arena."""
        del temperature, max_tokens, top_p, stop
        catalog = load_local_models()
        model_uuid, entry = _resolve_model(model, catalog)
        modality = _detect_modality(entry.get("capabilities") or {})
        prompt = _flatten_messages_to_prompt(messages)
        async for chunk in self._stream_arena(
            model_uuid=model_uuid,
            model_public=model,
            prompt=prompt,
            modality=modality,
        ):
            yield chunk


# ---------------------------------------------------------------------------
# Convenience for openai_api.list_models fallback
# ---------------------------------------------------------------------------


async def get_arena_models(account: ArenaAccount) -> list[dict[str, Any]]:
    """Return OpenAI-shaped model dicts from the local catalog."""
    client = ArenaHttpClient(account)
    try:
        models = await client.get_models()
    finally:
        await client.close()
    return [
        {
            "id": m.name or m.id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": m.provider or "arena",
            "description": m.description,
        }
        for m in models
        if (m.name or m.id)
    ]
