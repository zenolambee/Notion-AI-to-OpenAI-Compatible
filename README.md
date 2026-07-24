<div align="center">

# ArenaChat - OpenAI-Compatible API for Arena.ai

**Unofficial OpenAI-compatible proxy for Chatbot Arena (arena.ai)**  
Cookie auth · streaming · Cursor / 9router / Postman

[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-OpenAI%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

<br/>

### Support the project

If this saves you time, a coffee (or satoshi) helps keep it maintained.

[![PayPal](https://img.shields.io/badge/PayPal-Donate-00457C?style=for-the-badge&logo=paypal&logoColor=white)](https://paypal.me/captainredz?locale.x=en_US&country.x=ID)
[![Patreon](https://img.shields.io/badge/Patreon-Support-FF424D?style=for-the-badge&logo=patreon&logoColor=white)](https://patreon.com/mughu?utm_medium=unknown&utm_source=join_link&utm_campaign=creatorshare_creator&utm_content=copyLink)
[![Traktir Kopi](https://img.shields.io/badge/Traktir%20Kopi-MuGhu-F59E0B?style=for-the-badge)](https://traktir.mughu.id/)

| | Network | Address |
|:--:|:--|:--|
| **BTC** | Bitcoin | `bc1pq0njd8sqextzrx2xx2qlzuud2m3nkcxqqu2kpyk6gwnufsum2phq6fdvtw` |
| **SOL** | Solana | `HUW8ntbpNGRcdTWc32yat3UCKcsHuiz1Me2QurPcF6iC` |
| **USDT** | Solana | `HUW8ntbpNGRcdTWc32yat3UCKcsHuiz1Me2QurPcF6iC` |
| **USDT** | TRON (TRC20) | `TY5qoR528nTwgkRvQmDMoRpmDRjKXnHMse` |

<p>
  <img src="https://img.shields.io/badge/BTC-Bitcoin-F7931A?style=flat-square&logo=bitcoin&logoColor=white" alt="BTC"/>
  <img src="https://img.shields.io/badge/SOL-Solana-9945FF?style=flat-square&logo=solana&logoColor=white" alt="SOL"/>
  <img src="https://img.shields.io/badge/USDT-Solana-26A17B?style=flat-square&logo=tether&logoColor=white" alt="USDT Solana"/>
  <img src="https://img.shields.io/badge/USDT-TRON%20TRC20-EF0027?style=flat-square&logo=tron&logoColor=white" alt="USDT TRON"/>
</p>

<sub>Send only the matching network asset to each address — wrong-chain transfers are lost forever.</sub>

</div>

---

> **Educational / unofficial** — for learning and research. Not affiliated with or endorsed by Arena.ai. You must comply with Arena.ai's Terms of Service.

OpenAI-compatible HTTP API that routes chat to **Arena.ai (Chatbot Arena)** using your browser session cookie (`arena-auth-prod-v1`).

Works with [Cursor](https://cursor.com), [9router](https://github.com), Postman, or any OpenAI Chat Completions client.

## Features

- **OpenAI-compatible endpoints**
  - `POST /v1/chat/completions` (streaming and non-streaming)
  - `GET /v1/models` (list of available models)
  - `GET /healthz`
- **Browser cookie authentication** — no separate API key required
- **Streaming support** — real-time response streaming
- **Multiple models** — access various AI models available on Arena.ai

## How it works

```
Client (Cursor / 9router / Postman)
        │
        ▼
  ArenaChat (FastAPI)
        │  OpenAI messages → one direct-mode prompt
        ▼
  Arena.ai /nextjs-api/stream/create-evaluation
        │  Arena event records (a0 / ad / a3)
        ▼
  ArenaChat parses response → OpenAI chat.completion
```

## Requirements

- Python 3.11+ recommended
- An Arena.ai (Chatbot Arena) account
- A valid browser session cookie (`arena-auth-prod-v1`)

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/mughu-id/Notion-AI-to-OpenAI-Compatible.git
cd Notion-AI-to-OpenAI-Compatible
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Description |
|----------|-------------|
| `ARENACHAT_API_KEY` | Bearer token clients must send (`Authorization: Bearer ...`) |
| `ARENACHAT_HOST` | Bind host (default `127.0.0.1`) |
| `ARENACHAT_PORT` | Port (default `1995`) |
| `ARENACHAT_ACCOUNT` | Path to account JSON (default `arena_account.json`) |
| `ARENACHAT_BASE_URL` | Arena website origin (default `https://arena.ai`) |
| `ARENACHAT_DEFAULT_MODEL` | Convenience default; use an ID returned by `/v1/models` |
| `ARENACHAT_HOME` | Optional — absolute project folder |
| `ARENA_COOKIE` | Full **Cookie request header** from a logged-in arena.ai tab |
| `ARENA_RECAPTCHA_V3_TOKEN` | Optional, short-lived token from your normal browser request if Arena requires it |

### 3. Get your Arena.ai cookie

1. Log in to [arena.ai](https://arena.ai) and open **Direct mode**.
2. In DevTools (F12), open **Network**, select a request to `arena.ai`, and copy its complete **Cookie** request-header value.
3. It must contain `arena-auth-prod-v1` or its split `arena-auth-prod-v1.0`/`.1` cookies. Keep any `cf_clearance` or `__cf_bm` cookies sent by the browser too.

> Arena can enforce a browser reCAPTCHA check for a request. Complete it in your normal browser. This project deliberately does not solve or bypass CAPTCHA; if the upstream returns a reCAPTCHA rejection, the API now reports that explicitly instead of returning an empty assistant message.

```bash
python -m notionchat setup
```

Or use the interactive wizard:

```bash
python -m notionchat setup --cookie "arena-auth-prod-v1=your_token; cf_clearance=xxx"
```

### 4. Run the server

```bash
python -m notionchat serve
```

Server URL: `http://127.0.0.1:1995` (or your configured host/port).

### 5. Test

```bash
# Health check
curl http://127.0.0.1:1995/healthz

# List available models
curl http://127.0.0.1:1995/v1/models \
  -H "Authorization: Bearer sk-arena-chat"

# Chat completion (non-streaming) — replace the model with an ID from /v1/models
curl http://127.0.0.1:1995/v1/chat/completions \
  -H "Authorization: Bearer sk-arena-chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "arena-gpt-4o",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'

# Chat completion (streaming)
curl http://127.0.0.1:1995/v1/chat/completions \
  -H "Authorization: Bearer sk-arena-chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "arena-gpt-4o",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

## Available Models

Arena's enabled models change frequently. Do **not** rely on the old `arena-*` table: request the current list after configuring your cookie and use an exact returned `id`.

```bash
curl http://127.0.0.1:1995/v1/models \
  -H "Authorization: Bearer sk-arena-chat"
```

For convenience, a request such as `arena-gpt-4o` is normalized to `GPT-4o` when that model is currently present. The current `/v1/models` result remains the source of truth.

## Cursor / 9router setup

1. Run ArenaChat locally (`python -m notionchat serve`).
2. In your router or Cursor custom model settings:
   - **Base URL:** `http://127.0.0.1:1995/v1`
   - **API key:** value of `ARENACHAT_API_KEY` from `.env` (default: `sk-arena-chat`)
   - **Model:** an exact ID returned by `GET /v1/models`

## Project layout

```
notionchat/          # Python package (API, client, account management)
scripts/             # PATH launchers
postman/             # Postman collection
.env.example         # Copy to .env (never commit .env)
pyproject.toml       # package + CLI entry points
requirements.txt
```

## Security

- Never commit `.env`, `arena_account.json`, or any credentials
- Use a strong `ARENACHAT_API_KEY` if the server is reachable on your network
- Default bind is `127.0.0.1` — avoid exposing to the public internet without proper auth

## Troubleshooting

| Problem | Things to try |
|---------|---------------|
| Empty assistant response from an older server | Update/restart the server. The current version returns the upstream error instead of a false successful empty completion. |
| 401 Unauthorized | Verify `ARENACHAT_API_KEY` matches in `.env` and the client. |
| Arena authentication rejected | Re-login to arena.ai and replace `ARENA_COOKIE` with the full Cookie request header. |
| “reCAPTCHA check was not accepted” | Complete the challenge in your normal browser, refresh the session, and retry. This proxy does not bypass CAPTCHA. |
| Model unavailable | Call `GET /v1/models` and use an exact returned model ID. |

Enable server logging:

```bash
python -m notionchat serve
```

## Development

```bash
pip install -r requirements.txt
python -m py_compile notionchat/*.py
python -m notionchat serve
```

## Disclaimer

This software is provided as-is. Using browser session cookies to access Arena.ai's internal APIs may violate Arena.ai's Terms of Service. Use at your own risk.

## License

[MIT](LICENSE)
