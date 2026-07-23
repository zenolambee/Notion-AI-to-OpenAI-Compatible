"""
Local model catalog for Arena.ai models.

Fetched from the Arena API, cached to ``models.json`` so the server
can map friendly aliases → real Arena model UUIDs at request time.
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


# ── data helpers ────────────────────────────────────────────────────

def _catalog_path() -> Path:
    """Return the path to the local models.json catalog."""
    from notionchat.config import _resolve_home  # noqa: PLC0415
    home = _resolve_home()
    if home is not None:
        return home / CATALOG_FILENAME
    return Path(CATALOG_FILENAME)


def load_catalog() -> dict[str, Any]:
    """Load the local models.json catalog from disk.

    Returns the raw JSON dict (with ``models`` key) or an empty dict.
    """
    path = _catalog_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read %s: %s", path, exc)
        return {}


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
    """Build a ``{ friendly_alias → arena_model_id }`` map from the catalog.

    Each raw Arena model entry is expected to have at least ``id`` (the UUID)
    and optionally ``name`` / ``displayName`` / ``description``.
    """
    alias_map: dict[str, str] = {}
    for entry in catalog.get("models") or []:
        if not isinstance(entry, dict):
            continue
        model_id: str = entry.get("id", "")
        if not model_id:
            continue
        name = entry.get("name") or entry.get("displayName") or entry.get("description") or ""
        description = entry.get("description") or ""

        # Primary aliases
        aliases: set[str] = set()
        if name:
            aliases.add(_norm(name))
        if description:
            aliases.add(_norm(description))
        # The UUID itself is always an alias
        aliases.add(model_id)

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
    """Resolve a user-supplied model name to an Arena model UUID.

    Resolution order:
      1. Exact match in the alias map (case-insensitive).
      2. Substring containment: if the normalised request appears inside a
         catalog entry's normalised name (or vice-versa).
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

    # 2. Substring match (e.g. "gpt-4o" inside "GPT-4o (2024-08-06)")
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
    """Return up to *n* catalog model names closest to *requested_model*.

    Used for error messages so the user knows what they *can* pick.
    """
    catalog = catalog if catalog is not None else load_catalog()
    names: list[str] = []
    for entry in catalog.get("models") or []:
        if not isinstance(entry, dict):
            continue
        display = entry.get("name") or entry.get("displayName") or entry.get("id", "")
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
