"""
Account management for Arena.ai authentication.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from notionchat.exceptions import NotionChatError

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ArenaAccount:
    """Account information for arena.ai (Chatbot Arena)."""
    token_v2: str = ""  # arena-auth-prod-v1 token
    full_cookie: str = ""
    user_id: str = ""
    user_name: str = ""
    user_email: str = ""
    cookie_domain: str = "arena.ai"
    client_version: str = "1.0.0"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
    )
    extras: dict[str, Any] = field(default_factory=dict)


def parse_browser_cookie(cookie: str) -> dict[str, str]:
    """Parse a cookie string into a dictionary."""
    out: dict[str, str] = {}
    for part in cookie.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, _, value = part.partition("=")
        out[name.strip()] = value.strip()
    return out


def validate_arena_cookie(cookie: str) -> tuple[bool, str]:
    """Validate an arena.ai cookie.

    Returns (is_valid, error_message).
    """
    if not cookie:
        return False, "Cookie is empty."

    parsed = parse_browser_cookie(cookie)

    # Check for arena-auth-prod-v1 cookie
    if not parsed.get("arena-auth-prod-v1"):
        return False, "Cookie is missing 'arena-auth-prod-v1'. Copy the full document.cookie from arena.ai."

    return True, ""


def detect_cookie_domain(cookie: str) -> str:
    """Detect cookie domain from common cookie names.

    Returns the detected domain: "arena.ai", "notion.ai", or "notion.com".
    """
    if not cookie:
        return "arena.ai"

    parsed = parse_browser_cookie(cookie)

    # Check for arena.ai cookies (Chatbot Arena)
    if (
        parsed.get("arena-auth-prod-v1")
        or parsed.get("arena-auth-prod-v1.0")
        or parsed.get("arena-auth-prod-v1.1")
    ):
        return "arena.ai"

    # Check for notion.ai cookies
    if parsed.get("www_notion_so_sid") or parsed.get("_notion_so_session"):
        return "notion.ai"

    # Default to arena.ai (primary target)
    return "arena.ai"


def combine_split_arena_cookies(cookies: list[dict]) -> str | None:
    """Combine split arena-auth-prod-v1.0 and .1 cookies into a single value.

    Google OAuth sometimes creates split cookies due to size limits.
    """
    parts: dict[int, str] = {}
    for cookie in cookies or []:
        name = str(cookie.get("name") or "")
        value = str(cookie.get("value") or "")
        if name == "arena-auth-prod-v1.0":
            parts[0] = value
        elif name == "arena-auth-prod-v1.1":
            parts[1] = value

    if 0 in parts and 1 in parts:
        combined = (parts[0] + parts[1]).strip()
        return combined if combined else None
    elif 0 in parts:
        value = parts[0].strip()
        return value if value else None
    return None


def parse_arena_auth_token(token: str) -> dict | None:
    """Parse arena-auth-prod-v1 token to extract user info.

    The token format is base64-encoded JSON with base64-encoded JWT payload.
    """
    if not token:
        return None
    try:
        # Try direct base64 decode first (some token formats)
        try:
            decoded = base64.b64decode(token).decode("utf-8")
            return json.loads(decoded)
        except Exception:
            pass

        # Try URL-safe base64
        try:
            # Add padding if needed
            padded = token + "=" * (4 - len(token) % 4)
            decoded = base64.urlsafe_b64decode(padded).decode("utf-8")
            return json.loads(decoded)
        except Exception:
            pass

        return None
    except Exception:
        return None


def extract_arena_user_from_token(token: str) -> tuple[str, str]:
    """Extract user ID and email from arena-auth-prod-v1 token.

    Returns (user_id, email) tuple.
    """
    parsed = parse_arena_auth_token(token)
    if not parsed:
        return "", ""

    # Try various possible field names
    user_id = (
        parsed.get("sub")
        or parsed.get("user_id")
        or parsed.get("uid")
        or ""
    )
    email = (
        parsed.get("email")
        or parsed.get("user_email")
        or ""
    )
    return user_id, email


def load_arena_account(path: Path | str) -> ArenaAccount:
    """Load arena account from a JSON file."""
    p = Path(path).expanduser()
    if not p.exists():
        raise NotionChatError(f"Account file not found: {p}", status_code=500)
    try:
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise NotionChatError(f"Invalid account JSON: {e}", status_code=500) from e

    # Validate required fields
    if not data.get("token_v2") and not data.get("arena-auth-prod-v1"):
        raise NotionChatError(
            "Account file missing required field: token_v2 or arena-auth-prod-v1",
            status_code=500,
        )

    # Build ArenaAccount
    token = data.get("token_v2") or data.get("arena-auth-prod-v1") or ""
    return ArenaAccount(
        token_v2=token,
        full_cookie=data.get("full_cookie", ""),
        user_id=data.get("user_id", ""),
        user_name=data.get("user_name", ""),
        user_email=data.get("user_email", ""),
        cookie_domain=data.get("cookie_domain", "arena.ai"),
        client_version=data.get("client_version", "1.0.0"),
        user_agent=data.get("user_agent", ArenaAccount.user_agent),
        extras={k: v for k, v in data.items() if k not in [
            "token_v2", "arena-auth-prod-v1", "full_cookie", "user_id",
            "user_name", "user_email", "cookie_domain", "client_version", "user_agent"
        ]},
    )


def save_arena_account(acc: ArenaAccount, path: Path | str) -> None:
    """Save arena account to a JSON file."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "token_v2": acc.token_v2,
        "full_cookie": acc.full_cookie,
        "user_id": acc.user_id,
        "user_name": acc.user_name,
        "user_email": acc.user_email,
        "cookie_domain": acc.cookie_domain,
        "client_version": acc.client_version,
        "user_agent": acc.user_agent,
    }
    data.update(acc.extras)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_cookie_header(acc: ArenaAccount) -> str:
    """Build cookie header from account."""
    if acc.full_cookie:
        return acc.full_cookie.strip().rstrip(";")
    return f"arena-auth-prod-v1={acc.token_v2}"


def create_account_from_cookie(cookie: str) -> ArenaAccount:
    """Create an ArenaAccount from a cookie string."""
    parsed = parse_browser_cookie(cookie)
    token = parsed.get("arena-auth-prod-v1") or ""

    if not token:
        raise NotionChatError(
            "Cookie is missing 'arena-auth-prod-v1'.",
            status_code=400,
        )

    user_id, user_email = extract_arena_user_from_token(token)

    return ArenaAccount(
        token_v2=token,
        full_cookie=cookie.strip().rstrip(";"),
        user_id=user_id or "arena-user",
        user_email=user_email,
        user_name=user_email or "Arena User",
        cookie_domain="arena.ai",
    )
