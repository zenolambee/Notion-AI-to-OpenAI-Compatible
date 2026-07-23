"""
ArenaChat CLI entry point.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
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
    # Make notionchat.* loggers visible under uvicorn.
    log_level = os.getenv("ARENACHAT_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for name in ("notionchat", "notionchat.arena_client", "notionchat.openai_api"):
        logging.getLogger(name).setLevel(log_level)
    settings = load_settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=log_level.lower())
    return 0


def _install_playwright_browsers() -> int:
    """Run `playwright install chromium` in the current interpreter."""
    import subprocess

    cmd = [sys.executable, "-m", "playwright", "install", "chromium"]
    print("Running:", " ".join(cmd))
    try:
        rc = subprocess.call(cmd)
    except FileNotFoundError:
        print(
            "playwright package not found. Install it first:\n"
            "  pip install playwright",
            file=sys.stderr,
        )
        return 1
    return rc


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

    # install-playwright command
    inst_p = sub.add_parser(
        "install-playwright",
        help="Download the headless Chromium used for auto reCAPTCHA minting",
    )
    inst_p.set_defaults(func=lambda _: _install_playwright_browsers())

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
