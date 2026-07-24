"""
ArenaChat CLI entry point.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import uvicorn

from notionchat.config import load_settings
from notionchat.openai_api import create_app
from notionchat.setup_cli import run_interactive_setup


def cmd_serve(_: argparse.Namespace) -> int:
    """Start the API server."""
    home = os.getenv("ARENACHAT_HOME", "").strip()
    if home:
        os.chdir(Path(home).expanduser().resolve())
    settings = load_settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


def cmd_sync_models(_: argparse.Namespace) -> int:
    """Write the known Arena.ai model catalog to models.json."""
    from notionchat.model_catalog import KNOWN_ARENA_MODELS, save_catalog  # noqa: PLC0415
    import time as _time  # noqa: PLC0415

    home = os.getenv("ARENACHAT_HOME", "").strip()
    if home:
        os.chdir(Path(home).expanduser().resolve())

    catalog = {
        "synced_at": _time.time(),
        "source": "hardcoded",
        "models": [dict(m) for m in KNOWN_ARENA_MODELS],
    }
    save_catalog(catalog)

    count = len(catalog["models"])
    print(f"Saved {count} models to models.json")
    if count:
        print("\nAvailable models:")
        for entry in catalog["models"]:
            name = entry.get("name") or entry.get("id")
            mid = entry.get("id")
            provider = entry.get("provider", "")
            suffix = f"  ({provider})" if provider else ""
            print(f"  {name}  →  {mid}{suffix}")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    """Run interactive setup."""
    return asyncio.run(
        run_interactive_setup(
            env_path=Path(args.env) if args.env else None,
            account_path=Path(args.account) if args.account else None,
            cookie=args.cookie,
            api_key=args.api_key,
            host=args.host,
            port=args.port,
            write_cookie_to_env=args.write_cookie if args.write_cookie is not None else None,
            force=args.force,
            yes=args.yes,
        )
    )


def _prog_name() -> str:
    """Get program name from argv."""
    base = Path(sys.argv[0]).name.lower()
    if base in ("arenachat", "arenachat.exe", "arenachat.cmd"):
        return "arenachat"
    if base.startswith("notionchat"):
        return "notionchat"
    return "arenachat"


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    prog = _prog_name()
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Arena.ai OpenAI-compatible API",
    )
    sub = parser.add_subparsers(dest="command")

    # serve command
    serve_p = sub.add_parser("serve", help="Start OpenAI-compatible API server")
    serve_p.set_defaults(func=cmd_serve)

    # setup command
    setup_p = sub.add_parser("setup", help="Interactive wizard: cookie -> account file -> .env")
    setup_p.add_argument("--env", default=".env", help="Path to write environment file (default: .env)")
    setup_p.add_argument("--account", default="arena_account.json", help="Output account file path")
    setup_p.add_argument("--cookie", default=None, help="Skip cookie prompt and use this value")
    setup_p.add_argument("--api-key", default=None, help="API key for ARENACHAT_API_KEY")
    setup_p.add_argument("--host", default=None, help="Bind host for ARENACHAT_HOST")
    setup_p.add_argument("--port", default=None, help="Port for ARENACHAT_PORT")
    setup_p.add_argument(
        "--write-cookie",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store ARENA_COOKIE in .env (default: ask interactively)",
    )
    setup_p.add_argument("--force", action="store_true", help="Overwrite .env without asking")
    setup_p.add_argument("-y", "--yes", action="store_true", help="Accept defaults with minimal prompts")
    setup_p.set_defaults(func=cmd_setup)

    # sync-models command
    sync_p = sub.add_parser("sync-models", help="Fetch and cache Arena.ai model list to models.json")
    sync_p.set_defaults(func=cmd_sync_models)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        print(
            f"\nNew here? Run the interactive setup wizard:\n"
            f"  {prog} setup\n"
            f"Or start the API server:\n"
            f"  {prog} serve\n",
            file=sys.stderr,
        )
        raise SystemExit(1)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
