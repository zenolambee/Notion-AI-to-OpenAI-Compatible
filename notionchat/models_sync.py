"""
Auto-sync arena.ai's model catalog into models.json.

Arena.ai doesn't expose a stable JSON `/api/models` endpoint — the catalog
is either embedded in the Next.js bundle or fetched via internal
`/nextjs-api/*` calls. We reuse the headless browser already used for
reCAPTCHA minting and:

  1. Attach a response listener that inspects every JSON response.
  2. Any response body that looks like arena's model list — an array of
     objects with (at minimum) `id` (UUID) and `publicName` — is captured.
  3. If nothing turns up passively, we probe React state / window
     globals for embedded arrays with the same shape.
  4. Merge everything, deduplicate by id, write to `models.json`
     (or `$ARENACHAT_MODELS_FILE`).

Run manually:
    notionchat sync-models
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from notionchat.account import ArenaAccount
from notionchat.arena_client import _default_models_path
from notionchat.exceptions import NotionChatError
from notionchat.recaptcha import RecaptchaTokenManager

log = logging.getLogger(__name__)

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


_JUNK_NAME_MARKERS = (
    '"everyone" survey',
    "everyone survey",
    "survey - (",
    "(Arena =",
    "(Age =",
    "(Geo =",
)


def _looks_like_model_entry(obj: Any) -> bool:
    """Return True only for entries that look like real chat/image models.

    Arena.ai's client bundle also carries analytics/survey rows keyed by
    UUID+publicName that would otherwise match. Distinguish real models by:
      * excluding names matching known analytics/survey patterns
      * requiring `organization` when present
      * requiring at least one output capability when `capabilities` is
        present (mirroring LMArenaBridge's filter)
    We stay a bit lenient when arena's bundle omits capabilities/organization
    entirely (some rare model entries), rejecting only clear analytics junk.
    """
    if not isinstance(obj, dict):
        return False
    mid = obj.get("id")
    pub = obj.get("publicName") or obj.get("public_name") or obj.get("name")
    if not isinstance(mid, str) or not isinstance(pub, str) or not pub:
        return False
    if not UUID_RE.match(mid):
        return False

    # Hard reject analytics / survey buckets that share the (id, publicName)
    # shape but aren't real models.
    low = pub.lower()
    if any(marker.lower() in low for marker in _JUNK_NAME_MARKERS):
        return False

    org = str(obj.get("organization") or "").strip()
    caps = obj.get("capabilities")
    outs = caps.get("outputCapabilities") if isinstance(caps, dict) else None

    if isinstance(outs, dict):
        # If caps are present, require at least one usable output.
        if not (outs.get("text") or outs.get("search") or outs.get("image")):
            return False
        # ...and if caps are present but organization is empty, still reject.
        if not org:
            return False
        return True

    # No capabilities info at all: require a plausible organization and a
    # reasonably short name (real model names are almost never >80 chars).
    if not org:
        return False
    if len(pub) > 80:
        return False
    return True


def _extract_model_entries(node: Any, out: list[dict]) -> None:
    """Recursively walk `node` collecting anything that looks like a model."""
    if isinstance(node, dict):
        if _looks_like_model_entry(node):
            out.append(node)
            return
        for v in node.values():
            _extract_model_entries(v, out)
    elif isinstance(node, list):
        # Fast path: if this list itself is a homogeneous list of models,
        # take it whole.
        if node and all(_looks_like_model_entry(x) for x in node):
            out.extend(node)
            return
        for v in node:
            _extract_model_entries(v, out)


def _normalize_entry(m: dict[str, Any]) -> dict[str, Any]:
    """Trim to fields we actually use in arena_client._resolve_model."""
    return {
        "id": m.get("id", ""),
        "publicName": m.get("publicName")
        or m.get("public_name")
        or m.get("name", ""),
        "organization": m.get("organization")
        or m.get("provider")
        or "",
        "description": m.get("description", ""),
        "capabilities": m.get("capabilities") or {},
    }


def _dedupe(entries: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for m in entries:
        mid = str(m.get("id") or "")
        if not mid or mid in seen:
            continue
        seen.add(mid)
        out.append(_normalize_entry(m))
    return out


async def sync_models(
    account: ArenaAccount,
    *,
    output_path: Path | None = None,
    dwell_seconds: float = 8.0,
    additional_urls: list[str] | None = None,
) -> Path:
    """Populate models.json by driving a headless browser at arena.ai.

    Returns the path written.
    """
    dest = Path(output_path) if output_path else _default_models_path()

    mgr = RecaptchaTokenManager(account)
    captured: list[dict] = []
    tried_urls: list[str] = []

    try:
        await mgr._ensure_started()
        page = mgr._page
        assert page is not None

        async def on_response(response) -> None:
            try:
                url = response.url or ""
                ct = (response.headers or {}).get("content-type", "")
                if "json" not in ct.lower():
                    return
                # Skip huge bodies (>5MB) — models catalog is small.
                try:
                    body = await response.json()
                except Exception:
                    return
                before = len(captured)
                _extract_model_entries(body, captured)
                gained = len(captured) - before
                if gained:
                    tried_urls.append(url)
                    log.info(
                        "sync-models: +%d entries from %s",
                        gained,
                        url[:120],
                    )
            except Exception:
                return

        page.on("response", on_response)

        # Try a few URLs likely to trigger the model catalog fetch.
        probe_urls = [
            "https://arena.ai/?mode=direct",
            "https://arena.ai/",
            "https://arena.ai/leaderboard",
        ]
        if additional_urls:
            probe_urls.extend(additional_urls)

        for url in probe_urls:
            try:
                log.info("sync-models: navigating %s", url)
                await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            except Exception as e:
                log.warning("sync-models: goto %s failed: %s", url, e)
                continue
            # Let async chunks / model fetches settle.
            try:
                await page.wait_for_load_state("networkidle", timeout=15_000)
            except Exception:
                pass
            await asyncio.sleep(dwell_seconds)
            if captured:
                # Even one non-trivial hit is often enough.
                if len(_dedupe(captured)) >= 5:
                    break

        # Fallback: scrape embedded arrays out of the page's window/React state.
        if len(_dedupe(captured)) < 5:
            log.info(
                "sync-models: passive capture found %d — probing page globals",
                len(_dedupe(captured)),
            )
            try:
                inline = await page.evaluate(
                    """
                    () => {
                        const out = [];
                        const seen = new WeakSet();
                        const uuidRe = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
                        function looksLikeModel(x) {
                            return x && typeof x === 'object' && typeof x.id === 'string'
                                && uuidRe.test(x.id)
                                && (typeof x.publicName === 'string' || typeof x.name === 'string');
                        }
                        function walk(v, depth) {
                            if (depth > 8 || v === null || v === undefined) return;
                            if (Array.isArray(v)) {
                                if (v.length && v.every(looksLikeModel)) { out.push(...v); return; }
                                for (const it of v) walk(it, depth + 1);
                                return;
                            }
                            if (typeof v !== 'object') return;
                            if (seen.has(v)) return;
                            seen.add(v);
                            if (looksLikeModel(v)) { out.push(v); return; }
                            for (const k of Object.keys(v)) {
                                try { walk(v[k], depth + 1); } catch {}
                            }
                        }
                        try { walk(window.__NEXT_DATA__, 0); } catch {}
                        try {
                            const root = document.getElementById('__next');
                            const key = root && Object.keys(root).find(k => k.startsWith('__reactContainer'));
                            if (key) walk(root[key], 0);
                        } catch {}
                        try { walk(window, 0); } catch {}
                        return out;
                    }
                    """
                )
                if isinstance(inline, list):
                    before = len(captured)
                    _extract_model_entries(inline, captured)
                    log.info(
                        "sync-models: +%d entries from page globals",
                        len(captured) - before,
                    )
            except Exception as e:
                log.warning("sync-models: page-globals probe failed: %s", e)

    finally:
        await mgr.close()

    models = _dedupe(captured)
    if not models:
        raise NotionChatError(
            "sync-models: could not find any arena.ai model entries. "
            "Your cookie may be expired, or Cloudflare blocked the page. "
            "Try re-copying ARENA_COOKIE from a logged-in browser, and "
            "re-run with ARENA_RECAPTCHA_HEADLESS=0 to watch the page.",
            status_code=502,
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(models, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(dest)
    log.info(
        "sync-models: wrote %d models to %s (sources: %s)",
        len(models),
        dest,
        [u[:80] for u in tried_urls[:5]],
    )
    return dest
