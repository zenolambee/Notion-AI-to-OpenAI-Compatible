"""
Configuration management for ArenaChat.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from notionchat.account import (
    ArenaAccount,
    create_account_from_cookie,
    load_arena_account,
    save_arena_account,
)
from notionchat.exceptions import NotionChatError

DEFAULT_BASE_URL = "https://arena.ai/api"


def _resolve_home() -> Path | None:
    """Project home for .env / account files when running from anywhere."""
    raw = os.getenv("ARENACHAT_HOME", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return None


def _load_dotenv_files() -> None:
    home = _resolve_home()
    if home is not None:
        load_dotenv(home / ".env", override=False)
    load_dotenv(override=False)


_load_dotenv_files()


@dataclass(slots=True)
class Settings:
    """Application settings."""
    api_key: str
    host: str
    port: int
    account_path: Path
    base_url: str
    default_model: str
    model_catalog_path: Path = Path("models.json")


def _env_path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default)).expanduser()
    if not p.is_absolute():
        home = _resolve_home()
        if home is not None:
            return (home / p).resolve()
    return p


def load_settings() -> Settings:
    """Load application settings from environment variables."""
    return Settings(
        api_key=os.getenv("ARENACHAT_API_KEY", "sk-arena-chat"),
        host=os.getenv("ARENACHAT_HOST", "127.0.0.1"),
        port=int(os.getenv("ARENACHAT_PORT", "1995")),
        account_path=_env_path("ARENACHAT_ACCOUNT", "arena_account.json"),
        base_url=os.getenv("ARENACHAT_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        # Arena model ids are account-specific UUIDs. A catalog avoids stale,
        # fabricated aliases such as `arena-gpt-4o`.
        default_model=os.getenv("ARENACHAT_DEFAULT_MODEL", ""),
        model_catalog_path=_env_path("ARENACHAT_MODELS_FILE", "models.json"),
    )


def load_account_from_env(settings: Settings) -> ArenaAccount:
    """Load arena account from environment or account file."""
    cookie = os.getenv("ARENA_COOKIE", "").strip()
    account_path = str(settings.account_path)

    # If cookie is provided and valid, create account from it
    if cookie:
        account = create_account_from_cookie(cookie)
        save_arena_account(account, account_path)
        return account

    # Try to load from account file
    if settings.account_path.exists():
        return load_arena_account(settings.account_path)

    raise NotionChatError(
        "No Arena credentials found. Set ARENA_COOKIE in .env or run:\n"
        "  python -m notionchat setup --cookie \"<paste document.cookie>\"",
        status_code=500,
    )
