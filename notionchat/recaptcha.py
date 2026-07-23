"""
Automatic reCAPTCHA v3 (Enterprise) token minter for arena.ai.

Arena.ai's chat endpoint requires a fresh `recaptchaV3Token` on every request
(effective TTL ~2 minutes). This module runs a persistent headless Chromium
via Playwright, keeps a warm arena.ai tab loaded with the user's cookies,
and mints a token on demand by calling
`grecaptcha.enterprise.execute(sitekey, {action})` inside the page.

Design notes
------------
* One Playwright browser per RecaptchaTokenManager, reused across requests.
* One long-lived page kept on `https://arena.ai/?mode=direct` so that
  the grecaptcha bundle is already loaded (minting takes ~200ms after warmup
  vs 3-8s if we opened a fresh page each time).
* Tokens are cached for ~90s (real Google TTL is ~115s; leave slack).
* An asyncio.Lock ensures only one mint happens concurrently.
* Cookies are injected from ArenaAccount.full_cookie / ARENA_COOKIE plus
  the individual `ARENA_CF_*` env vars, so the headless tab reuses the
  same authenticated session Cloudflare/Arena already trust.
* On mint failure the cached token is invalidated so the next call retries
  from scratch instead of returning a stale value.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from notionchat.account import ArenaAccount
from notionchat.exceptions import NotionChatError

log = logging.getLogger(__name__)

# Public arena.ai sitekey (reCAPTCHA Enterprise v3). Discovered by
# CloudWaddie/LMArenaBridge; verified live against arena.ai bundles.
ARENA_RECAPTCHA_SITEKEY = "6Led_uYrAAAAAIP_9E8Ais_67Z6Vp4vdf40p8SQU"
ARENA_RECAPTCHA_ACTION = "chat_submit"

# How long we trust a minted token before re-minting. Google issues
# 115-120s tokens; keep a safe margin.
TOKEN_TTL_SECONDS = 90.0

# How long we wait for the initial grecaptcha bundle to load on the
# warm page (Cloudflare/turnstile can add a lot of latency the first time).
INITIAL_LOAD_TIMEOUT_MS = 60_000

# Per-mint timeout for grecaptcha.execute().
EXECUTE_TIMEOUT_MS = 20_000


def _parse_cookie_string(raw: str) -> list[dict[str, str]]:
    """Turn a `document.cookie` style string into Playwright cookies."""
    out: list[dict[str, str]] = []
    if not raw:
        return out
    for part in raw.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, _, value = part.partition("=")
        name = name.strip()
        value = value.strip()
        if not name:
            continue
        out.append(
            {
                "name": name,
                "value": value,
                "domain": ".arena.ai",
                "path": "/",
            }
        )
    return out


def _collect_cookies_from_env(account: ArenaAccount) -> list[dict[str, str]]:
    cookies: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(name: str, value: str) -> None:
        v = str(value or "").strip()
        if not name or not v or name in seen:
            return
        cookies.append(
            {"name": name, "value": v, "domain": ".arena.ai", "path": "/"}
        )
        seen.add(name)

    # Prefer the raw document.cookie style string if provided.
    for parsed in _parse_cookie_string(account.full_cookie):
        if parsed["name"] not in seen:
            cookies.append(parsed)
            seen.add(parsed["name"])

    # Then fall back to individual pieces.
    _add("arena-auth-prod-v1", account.token_v2)
    _add("cf_clearance", os.getenv("ARENA_CF_CLEARANCE", ""))
    _add("__cf_bm", os.getenv("ARENA_CF_BM", ""))
    _add("_cfuvid", os.getenv("ARENA_CFUVID", ""))
    _add("provisional_user_id", os.getenv("ARENA_PROVISIONAL_USER_ID", ""))
    return cookies


@dataclass
class MintResult:
    token: str
    minted_at: float


class RecaptchaTokenManager:
    """Persistent Playwright-backed grecaptcha v3 token minter."""

    def __init__(
        self,
        account: ArenaAccount,
        *,
        sitekey: str = ARENA_RECAPTCHA_SITEKEY,
        action: str = ARENA_RECAPTCHA_ACTION,
        headless: bool = True,
        arena_url: str = "https://arena.ai/?mode=direct",
        token_ttl: float = TOKEN_TTL_SECONDS,
    ) -> None:
        self.account = account
        self.sitekey = os.getenv("ARENA_RECAPTCHA_SITEKEY", sitekey)
        self.action = os.getenv("ARENA_RECAPTCHA_ACTION", action)
        env_headless = os.getenv("ARENA_RECAPTCHA_HEADLESS", "").strip().lower()
        if env_headless in ("0", "false", "no"):
            headless = False
        self.headless = headless
        self.arena_url = arena_url
        self.token_ttl = token_ttl

        self._token: str = ""
        self._minted_at: float = 0.0
        self._lock = asyncio.Lock()

        # Playwright objects live for the life of the manager.
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._closed = False

    # --------------------------------------------------------------- lifecycle

    async def _ensure_started(self) -> None:
        if self._page is not None or self._closed:
            return
        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise NotionChatError(
                "Playwright is not installed. Install it with:\n"
                "    pip install playwright\n"
                "    playwright install chromium\n"
                "Or disable auto reCAPTCHA by setting ARENA_RECAPTCHA_AUTO=0 "
                "and supplying ARENA_RECAPTCHA_TOKEN manually.",
                status_code=500,
            ) from e

        log.info(
            "reCAPTCHA: starting Playwright chromium (headless=%s)",
            self.headless,
        )
        self._playwright = await async_playwright().start()
        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-features=IsolateOrigins,site-per-process",
        ]
        if os.getenv("ARENA_RECAPTCHA_NO_SANDBOX", "").strip() in (
            "1",
            "true",
            "yes",
        ):
            launch_args.append("--no-sandbox")

        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=launch_args,
        )

        user_agent = (
            self.account.user_agent
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        )

        self._context = await self._browser.new_context(
            user_agent=user_agent,
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )

        cookies = _collect_cookies_from_env(self.account)
        if cookies:
            try:
                await self._context.add_cookies(cookies)
                log.info(
                    "reCAPTCHA: injected %d cookies (names=%s)",
                    len(cookies),
                    [c["name"] for c in cookies],
                )
            except Exception as e:
                log.warning("reCAPTCHA: failed to inject cookies: %s", e)

        self._page = await self._context.new_page()

        # Hide the obvious `navigator.webdriver` fingerprint before any script
        # on the page runs; helps a bit against basic bot checks.
        try:
            await self._page.add_init_script(
                "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
            )
        except Exception:
            pass

        log.info("reCAPTCHA: warming page %s", self.arena_url)
        try:
            await self._page.goto(
                self.arena_url,
                wait_until="domcontentloaded",
                timeout=INITIAL_LOAD_TIMEOUT_MS,
            )
        except Exception as e:
            log.warning("reCAPTCHA: initial navigation soft error: %s", e)

        # Wait for grecaptcha.enterprise.execute to become available.
        try:
            await self._page.wait_for_function(
                "() => !!(window.grecaptcha && window.grecaptcha.enterprise "
                "&& typeof window.grecaptcha.enterprise.execute === 'function')",
                timeout=INITIAL_LOAD_TIMEOUT_MS,
            )
            log.info("reCAPTCHA: grecaptcha.enterprise ready.")
        except Exception as e:
            log.warning(
                "reCAPTCHA: grecaptcha didn't appear during warmup "
                "(will still try on-demand): %s",
                e,
            )

    async def close(self) -> None:
        self._closed = True
        for attr in ("_page", "_context", "_browser"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    await obj.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    # ------------------------------------------------------------------ mint

    async def _mint_once(self) -> str:
        await self._ensure_started()
        assert self._page is not None

        # If for some reason grecaptcha isn't there yet, wait a bit more.
        try:
            await self._page.wait_for_function(
                "() => !!(window.grecaptcha && window.grecaptcha.enterprise "
                "&& typeof window.grecaptcha.enterprise.execute === 'function')",
                timeout=EXECUTE_TIMEOUT_MS,
            )
        except Exception as e:
            raise NotionChatError(
                f"reCAPTCHA: grecaptcha.enterprise never loaded on {self.arena_url}. "
                f"The warm page may be blocked by Cloudflare. Details: {e}",
                status_code=502,
            ) from e

        script = f"""
            async () => {{
                await new Promise((r) => grecaptcha.enterprise.ready(r));
                return await grecaptcha.enterprise.execute(
                    {self.sitekey!r},
                    {{ action: {self.action!r} }}
                );
            }}
        """
        try:
            token = await self._page.evaluate(script)
        except Exception as e:
            raise NotionChatError(
                f"reCAPTCHA: grecaptcha.enterprise.execute failed: {e}",
                status_code=502,
            ) from e

        token = str(token or "").strip()
        if not token:
            raise NotionChatError(
                "reCAPTCHA: grecaptcha.enterprise.execute returned empty token.",
                status_code=502,
            )
        return token

    async def get_token(self, *, force: bool = False) -> str:
        """Return a valid grecaptcha v3 token, minting if needed."""
        # Fast path: cached token still fresh.
        now = time.time()
        if (
            not force
            and self._token
            and (now - self._minted_at) < self.token_ttl
        ):
            return self._token

        async with self._lock:
            now = time.time()
            if (
                not force
                and self._token
                and (now - self._minted_at) < self.token_ttl
            ):
                return self._token

            # Invalidate the old token before minting so a failure doesn't
            # leave a stale value in place.
            if force:
                self._token = ""

            token = await self._mint_once()
            self._token = token
            self._minted_at = time.time()
            log.info(
                "reCAPTCHA: minted new token (len=%d, action=%s)",
                len(token),
                self.action,
            )
            return token

    def invalidate(self) -> None:
        """Drop the cached token so the next call remints."""
        self._token = ""
        self._minted_at = 0.0
