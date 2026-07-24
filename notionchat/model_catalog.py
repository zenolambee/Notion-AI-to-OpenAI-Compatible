"""
Local model catalog for Arena.ai models.

Arena.ai's /api/models endpoint returns 403, so we maintain a hardcoded
catalog of known Arena.ai model IDs (from the leaderboard + API error hints)
and resolve user-friendly aliases to them.
"""

from __future__ import annotations

import json
import logging
import time
from difflib import get_close_matches
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CATALOG_FILENAME = "models.json"
DEFAULT_CATALOG_TTL = 86400  # 24 hours

# ── Known Arena.ai model IDs (scraped from leaderboard + API responses) ──
# Arena.ai uses human-readable model names, NOT UUIDs.
KNOWN_ARENA_MODELS: list[dict[str, str]] = [
    # OpenAI
    {"id": "gpt-4o", "name": "GPT-4o", "provider": "OpenAI"},
    {"id": "gpt-4o-2024-11-20", "name": "GPT-4o (2024-11-20)", "provider": "OpenAI"},
    {"id": "gpt-4-turbo-2024-04-09", "name": "GPT-4 Turbo", "provider": "OpenAI"},
    {"id": "gpt-4.1-2025-04-14", "name": "GPT-4.1", "provider": "OpenAI"},
    {"id": "gpt-4.1-mini-2025-04-14", "name": "GPT-4.1 Mini", "provider": "OpenAI"},
    {"id": "gpt-4.1-nano-2025-04-14", "name": "GPT-4.1 Nano", "provider": "OpenAI"},
    {"id": "gpt-5.2", "name": "GPT-5.2", "provider": "OpenAI"},
    {"id": "gpt-5.4", "name": "GPT-5.4", "provider": "OpenAI"},
    {"id": "gpt-5.4-high", "name": "GPT-5.4 High", "provider": "OpenAI"},
    {"id": "gpt-5.5-high", "name": "GPT-5.5 High", "provider": "OpenAI"},
    {"id": "gpt-5.6-terra", "name": "GPT-5.6 Terra", "provider": "OpenAI"},
    {"id": "gpt-5.6-sol", "name": "GPT-5.6 Sol", "provider": "OpenAI"},
    {"id": "o3-2025-04-16", "name": "o3", "provider": "OpenAI"},
    {"id": "o3-mini-2025-01-31", "name": "o3-mini", "provider": "OpenAI"},
    {"id": "o3-mini-2025-01-31-high", "name": "o3-mini (high)", "provider": "OpenAI"},
    {"id": "o4-mini-2025-04-16", "name": "o4-mini", "provider": "OpenAI"},
    {"id": "o4-mini-2025-04-16-high", "name": "o4-mini (high)", "provider": "OpenAI"},
    # Anthropic
    {"id": "claude-opus-4-7", "name": "Claude Opus 4.7", "provider": "Anthropic"},
    {"id": "claude-opus-4-7-thinking", "name": "Claude Opus 4.7 Thinking", "provider": "Anthropic"},
    {"id": "claude-opus-4-7-search", "name": "Claude Opus 4.7 Search", "provider": "Anthropic"},
    {"id": "claude-opus-4-6", "name": "Claude Opus 4.6", "provider": "Anthropic"},
    {"id": "claude-opus-4-6-thinking", "name": "Claude Opus 4.6 Thinking", "provider": "Anthropic"},
    {"id": "claude-opus-4-6-search", "name": "Claude Opus 4.6 Search", "provider": "Anthropic"},
    {"id": "claude-opus-4-5-20251101", "name": "Claude Opus 4.5", "provider": "Anthropic"},
    {"id": "claude-opus-4-5-20251101-thinking-32k", "name": "Claude Opus 4.5 Thinking", "provider": "Anthropic"},
    {"id": "claude-opus-4-20250514", "name": "Claude Opus 4", "provider": "Anthropic"},
    {"id": "claude-opus-4-20250514-thinking-16k", "name": "Claude Opus 4 Thinking", "provider": "Anthropic"},
    {"id": "claude-opus-4-1-20250805", "name": "Claude Opus 4.1", "provider": "Anthropic"},
    {"id": "claude-sonnet-4-5-20250929", "name": "Claude Sonnet 4.5", "provider": "Anthropic"},
    {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "provider": "Anthropic"},
    {"id": "claude-3-7-sonnet-20250219", "name": "Claude 3.7 Sonnet", "provider": "Anthropic"},
    {"id": "claude-3-7-sonnet-20250219-thinking-32k", "name": "Claude 3.7 Sonnet Thinking", "provider": "Anthropic"},
    {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "provider": "Anthropic"},
    {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "provider": "Anthropic"},
    {"id": "fable-5", "name": "Claude Fable 5", "provider": "Anthropic"},
    # Google
    {"id": "gemini-2.5", "name": "Gemini 2.5", "provider": "Google"},
    {"id": "gemini-2.5-pro-preview-05-06", "name": "Gemini 2.5 Pro", "provider": "Google"},
    {"id": "gemini-2.5-flash-preview-05-20", "name": "Gemini 2.5 Flash", "provider": "Google"},
    {"id": "gemini-3.1-pro", "name": "Gemini 3.1 Pro", "provider": "Google"},
    {"id": "gemini-3.6-flash", "name": "Gemini 3.6 Flash", "provider": "Google"},
    # xAI
    {"id": "grok-4.5", "name": "Grok 4.5", "provider": "xAI"},
    {"id": "grok-4.3", "name": "Grok 4.3", "provider": "xAI"},
    # Meta
    {"id": "llama-4-maverick-instruct-basic", "name": "Llama 4 Maverick", "provider": "Meta"},
    {"id": "llama-3.3-70b-instruct", "name": "Llama 3.3 70B", "provider": "Meta"},
    # DeepSeek
    {"id": "deepseek-r1", "name": "DeepSeek R1", "provider": "DeepSeek"},
    {"id": "deepseek-v4-pro", "name": "DeepSeek V4 Pro", "provider": "DeepSeek"},
    # Moonshot / Kimi
    {"id": "kimi-k2-0711-preview", "name": "Kimi K2", "provider": "Moonshot"},
    {"id": "kimi-k2-0905-preview", "name": "Kimi K2 (0905)", "provider": "Moonshot"},
    {"id": "kimi-k2-thinking-turbo", "name": "Kimi K2 Thinking", "provider": "Moonshot"},
    {"id": "kimi-k2.5", "name": "Kimi K2.5", "provider": "Moonshot"},
    {"id": "kimi-k2.5-instant", "name": "Kimi K2.5 Instant", "provider": "Moonshot"},
    {"id": "kimi-k2.6", "name": "Kimi K2.6", "provider": "Moonshot"},
    # Alibaba / Qwen
    {"id": "qwen3-235b-a22b", "name": "Qwen3 235B", "provider": "Alibaba"},
    {"id": "qwen3.7-max", "name": "Qwen3.7 Max", "provider": "Alibaba"},
    # Mistral
    {"id": "mixtral-8x7b-instruct-v0.1", "name": "Mixtral 8x7B", "provider": "Mistral"},
    # MiniMax
    {"id": "minimax-m2.5", "name": "MiniMax M2.5", "provider": "MiniMax"},
    # Zhipu / GLM
    {"id": "glm-5.2", "name": "GLM 5.2", "provider": "Zhipu"},
]


def _catalog_path() -> Path:
    """Return the path to the local models.json catalog."""
    from notionchat.config import _resolve_home  # noqa: PLC0415
    home = _resolve_home()
    if home is not None:
        return home / CATALOG_FILENAME
    return Path(CATALOG_FILENAME)


def load_catalog() -> dict[str, Any]:
    """Load the local models.json catalog from disk.

    Falls back to KNOWN_ARENA_MODELS if no file exists.
    """
    path = _catalog_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("models"):
                return data
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read %s: %s", path, exc)

    # Return hardcoded catalog as fallback
    return {
        "synced_at": time.time(),
        "source": "hardcoded",
        "models": [dict(m) for m in KNOWN_ARENA_MODELS],
    }


def save_catalog(catalog: dict[str, Any]) -> None:
    """Persist the catalog dict to models.json."""
    path = _catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Wrote model catalog to %s (%d entries)", path, len(catalog.get("models", [])))


def catalog_is_fresh(catalog: dict[str, Any], ttl: int = DEFAULT_CATALOG_TTL) -> bool:
    """Return True if the catalog was saved within *ttl* seconds."""
    ts = catalog.get("synced_at", 0)
    return bool(ts) and (time.time() - ts) < ttl


# ── parsing helpers ─────────────────────────────────────────────────

def _norm(s: str) -> str:
    """Lower-case, hyphen-normalised, stripped string."""
    return s.strip().lower().replace("_", "-").replace(" ", "-")


def _build_alias_map(catalog: dict[str, Any]) -> dict[str, str]:
    """Build a ``{ friendly_alias → arena_model_id }`` map from the catalog."""
    alias_map: dict[str, str] = {}
    for entry in catalog.get("models") or []:
        if not isinstance(entry, dict):
            continue
        model_id: str = entry.get("id", "")
        if not model_id:
            continue
        name = entry.get("name") or entry.get("displayName") or ""
        description = entry.get("description") or ""

        # Primary aliases
        aliases: set[str] = set()
        if name:
            aliases.add(_norm(name))
        if description:
            aliases.add(_norm(description))
        # The ID itself is always an alias
        aliases.add(model_id)
        aliases.add(_norm(model_id))

        # Common shorthand patterns
        if "gpt-4o" in model_id:
            aliases.update({"gpt-4o", "arena-gpt-4o", "4o"})
        if "gpt-5.4" in model_id:
            aliases.update({"gpt-5.4", "arena-gpt-5.4"})
        if "gpt-5.5" in model_id:
            aliases.update({"gpt-5.5", "arena-gpt-5.5"})
        if "gpt-5.6-terra" in model_id:
            aliases.update({"gpt-5.6-terra", "gpt-5.6"})
        if "gpt-5.6-sol" in model_id:
            aliases.update({"gpt-5.6-sol"})
        if "gpt-5.2" in model_id:
            aliases.update({"gpt-5.2", "arena-gpt-5.2"})
        if "o3" in model_id and "mini" not in model_id:
            aliases.update({"o3", "arena-o3"})
        if "o3-mini" in model_id:
            aliases.update({"o3-mini", "arena-o3-mini"})
        if "o4-mini" in model_id:
            aliases.update({"o4-mini", "arena-o4-mini"})
        if "claude-opus-4-7" in model_id and "thinking" not in model_id and "search" not in model_id:
            aliases.update({"claude-opus-4-7", "opus-4.7", "arena-claude-opus-4-7"})
        if "claude-opus-4-6" in model_id and "thinking" not in model_id and "search" not in model_id:
            aliases.update({"claude-opus-4-6", "opus-4.6", "arena-claude-opus-4-6"})
        if "claude-opus-4-5" in model_id and "thinking" not in model_id:
            aliases.update({"claude-opus-4-5", "opus-4.5", "arena-claude-opus-4-5"})
        if "claude-sonnet-4-5" in model_id:
            aliases.update({"claude-sonnet-4-5", "sonnet-4.5", "arena-claude-sonnet-4-5"})
        if "claude-sonnet-4" in model_id and "4-5" not in model_id:
            aliases.update({"claude-sonnet-4", "sonnet-4", "arena-claude-sonnet-4"})
        if "claude-3-5-sonnet" in model_id:
            aliases.update({"claude-3-5-sonnet", "arena-claude-3-5-sonnet", "claude-3.5-sonnet"})
        if "claude-3-7-sonnet" in model_id and "thinking" not in model_id:
            aliases.update({"claude-3-7-sonnet", "arena-claude-3-7-sonnet", "claude-3.7-sonnet"})
        if "haiku-4-5" in model_id or "haiku-4.5" in model_id:
            aliases.update({"haiku-4.5", "haiku-4-5", "arena-claude-haiku-4.5"})
        if "fable-5" in model_id:
            aliases.update({"fable-5", "arena-fable-5"})
        if "gemini-2.5-pro" in model_id or model_id == "gemini-2.5":
            aliases.update({"gemini-2.5-pro", "arena-gemini-2.5-pro"})
        if "gemini-2.5-flash" in model_id:
            aliases.update({"gemini-2.5-flash", "arena-gemini-2.5-flash"})
        if "gemini-3.1-pro" in model_id:
            aliases.update({"gemini-3.1-pro", "arena-gemini-3.1-pro"})
        if "gemini-3.6-flash" in model_id:
            aliases.update({"gemini-3.6-flash", "arena-gemini-3.6-flash"})
        if "grok-4.5" in model_id:
            aliases.update({"grok-4.5", "spacexai-4.5", "arena-grok-4.5"})
        if "grok-4.3" in model_id:
            aliases.update({"grok-4.3", "arena-grok-4.3"})
        if "llama-4" in model_id:
            aliases.update({"llama-4", "arena-llama-4"})
        if "llama-3.3" in model_id or "llama-3-70b" in model_id:
            aliases.update({"llama-3-70b", "arena-llama-3-70b", "llama-3.3-70b"})
        if "llama-3" in model_id and "8b" in model_id:
            aliases.update({"llama-3-8b", "arena-llama-3-8b"})
        if "deepseek-r1" in model_id:
            aliases.update({"deepseek-r1", "arena-deepseek-r1"})
        if "deepseek-v4" in model_id:
            aliases.update({"deepseek-v4-pro", "arena-deepseek-v4"})
        if "kimi-k2.6" in model_id or "kimi-k2-0905" in model_id:
            aliases.update({"kimi-k2.6", "kimi-k2", "arena-kimi-k2"})
        if "kimi-k2.5" in model_id:
            aliases.update({"kimi-k2.5", "arena-kimi-k2.5"})
        if "glm" in model_id:
            aliases.update({"glm-5.2", "arena-glm-5.2"})
        if "mixtral" in model_id:
            aliases.update({"mixtral-8x7b", "arena-mixtral-8x7b"})
        if "minimax" in model_id:
            aliases.update({"minimax-m2.5", "arena-minimax-m2.5"})
        if "qwen3.7" in model_id:
            aliases.update({"qwen3.7-max", "arena-qwen3.7-max"})
        if "qwen3" in model_id:
            aliases.update({"qwen3", "arena-qwen3"})

        for alias in aliases:
            if alias and alias not in alias_map:
                alias_map[alias] = model_id

    return alias_map


# ── resolution ──────────────────────────────────────────────────────

def _extract_provider_name(entry: dict[str, Any]) -> str:
    """Best-effort provider name from a model entry."""
    return (
        entry.get("provider")
        or entry.get("organization")
        or entry.get("owned_by")
        or "arena"
    )


def resolve_model_id(
    requested_model: str,
    *,
    catalog: dict[str, Any] | None = None,
) -> str | None:
    """Resolve a user-supplied model name to an Arena.ai model ID.

    Resolution order:
      1. Exact match in the alias map (case-insensitive).
      2. Substring containment.
      3. Fuzzy match via ``difflib.get_close_matches``.
      4. Return *None* if nothing matches.
    """
    if not requested_model:
        return None

    catalog = catalog if catalog is not None else load_catalog()
    alias_map = _build_alias_map(catalog)
    req_norm = _norm(requested_model)

    # 1. Exact alias hit
    if req_norm in alias_map:
        return alias_map[req_norm]

    # 2. Substring match (e.g. "gpt-4o" inside "gpt-4o-2024-11-20")
    for alias, mid in alias_map.items():
        if req_norm in alias or alias in req_norm:
            return mid

    # 3. Fuzzy match
    close = get_close_matches(req_norm, list(alias_map.keys()), n=1, cutoff=0.5)
    if close:
        return alias_map[close[0]]

    return None


def closest_model_names(
    requested_model: str,
    *,
    catalog: dict[str, Any] | None = None,
    n: int = 5,
) -> list[str]:
    """Return up to *n* catalog model names closest to *requested_model*."""
    catalog = catalog if catalog is not None else load_catalog()
    names: list[str] = []
    for entry in catalog.get("models") or []:
        if not isinstance(entry, dict):
            continue
        display = entry.get("name") or entry.get("id", "")
        if display:
            names.append(display)
    if not names:
        return []
    return get_close_matches(requested_model, names, n=n, cutoff=0.1)


# ── public model list (OpenAI /v1/models format) ───────────────────

def openai_model_list(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Build an OpenAI-compatible model list from the catalog."""
    import time as _time  # noqa: PLC0415

    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in catalog.get("models") or []:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id", "")
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        models.append({
            "id": model_id,
            "object": "model",
            "created": int(_time.time()),
            "owned_by": _extract_provider_name(entry),
        })
    models.sort(key=lambda m: m["id"].lower())
    return models
