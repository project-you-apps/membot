#!/usr/bin/env python3
"""mempack_local_agent.py — a fully local agent runtime for any Mempack.

Pipes a local LLM (via Ollama) through membot's MCP endpoint so you can
mount your Mempack, read Pattern I, work dispatches, and store findings
back — without Claude Desktop, Goose, or any API key. Just Python + Ollama
+ a tool-capable open-weights model.

QUICK START
-----------
  1. Install Ollama from https://ollama.com/download
  2. Pull a tool-capable model:
        ollama pull qwen2.5:14b
     (or: hermes3:8b, qwen2.5:7b, llama3.1:8b, qwen2.5-coder:14b)
  3. Make sure `requests` is installed:
        pip install requests
  4. Run:
        python mempack_local_agent.py \\
            --owner-id 3579e6ee-6412-4099-8d66-a205d9be7849 \\
            --model qwen2.5:14b \\
            --prompt "Mount my primary Mempack and tell me what's queued."

  Or interactive mode (REPL):
        python mempack_local_agent.py --owner-id <uuid>

DESIGN NOTES
------------
- Single file. Only external dep is `requests`. Distributable as-is.
- Hand-rolled MCP-over-HTTP JSON-RPC client (no `mcp` package needed).
  Membot's MCP endpoint uses Streamable-HTTP; we send Accept headers for
  both application/json AND text/event-stream and parse whichever comes
  back. For our tool calls the responses are small enough to come as
  single JSON objects in practice.
- Tool schemas fetched dynamically via tools/list on startup — no
  hardcoded schemas to drift out of sync with the server. Whatever
  membot exposes is what the agent sees.
- The agent loop: send messages to Ollama with the tools array; if
  Ollama returns tool_calls, dispatch each via MCP, append results as
  tool-role messages, re-prompt. Loop until Ollama returns no more
  tool_calls.

TROUBLESHOOTING
---------------
- "model does not support tools" from Ollama: pick a model from the
  tool-capable list above. Gemma, llava, embedding models DON'T qualify.
- Tool calls appearing as text in the model output: your Ollama may
  need updating, or the chosen model's chat template doesn't emit the
  proper `tool_calls` response field. Try a different model.
- 401 / 403 from membot: the owner_id you passed doesn't match any
  Mempack at that endpoint, OR the membot server requires JWT auth at
  some layer your local agent isn't sending. Mempack mount_cartridge
  accepts owner_id as a per-call argument so this normally works
  anonymously; if not, check membot/.env on the server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

# Reconfigure stdout/stderr to UTF-8 so Windows consoles (CP1252 by default)
# don't choke on Unicode characters in our output. Harmless on Linux/Mac.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass  # Older Python / non-TTY streams — accept whatever the default is.

try:
    import requests
except ImportError:
    sys.stderr.write("ERROR: `requests` is not installed. Run: pip install requests\n")
    sys.exit(1)

# Norton / corporate AV TLS interception on Windows substitutes the cert chain.
# Python's requests doesn't see the AV's root CA by default → SSLError on
# https://. `truststore` injects the OS system trust store into Python's TLS
# stack so the AV-issued cert chain validates. Harmless on systems that
# don't need it; install via `pip install truststore` if you hit a
# "certificate verify failed" error.
try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass  # Will surface as SSLError only if the user's TLS chain needs it.


DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MCP_URL    = os.environ.get("MEMBOT_MCP_URL", "https://project-you.app/membot/mcp")
DEFAULT_MODEL      = os.environ.get("MEMPACK_AGENT_MODEL", "qwen2.5:14b")
DEFAULT_MEMPACK    = os.environ.get("MEMPACK_NAME", "primary")
MAX_TURNS          = int(os.environ.get("MEMPACK_AGENT_MAX_TURNS", "20"))


# ─── MCP over Streamable HTTP ─────────────────────────────────────────────

class MempackMCPClient:
    """Minimal JSON-RPC over HTTP client for membot's MCP endpoint.

    Connection is stateless on the client side; the server returns an
    Mcp-Session-Id header on initialize that subsequent calls echo back.
    """

    def __init__(self, url: str, *, verify_tls: bool = True):
        self.url = url.rstrip("/")
        self.session_id: str | None = None
        self.protocol_version: str | None = None
        self.verify_tls = verify_tls
        self._next_id = 0
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "mempack-local-agent/1.0",
        }

    def _jsonrpc_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _post(self, payload: dict, notification: bool = False) -> dict:
        headers = dict(self.headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        try:
            resp = requests.post(self.url, headers=headers, json=payload, timeout=120,
                                 verify=self.verify_tls)
        except requests.RequestException as e:
            raise RuntimeError(f"MCP transport error: {e}") from e

        # Capture session id if the server set one on this response
        if resp.headers.get("Mcp-Session-Id"):
            self.session_id = resp.headers["Mcp-Session-Id"]

        # Notifications (JSON-RPC method calls without `id`) get no body back.
        # Server should return 2xx with empty body; we just confirm and exit.
        if notification:
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"MCP notification rejected: HTTP {resp.status_code}: {resp.text[:300]}"
                )
            return {}

        # Streamable-HTTP may respond with text/event-stream OR application/json
        content_type = resp.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            # Parse SSE: find the `data:` line and decode
            data_lines = []
            for line in resp.text.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if not data_lines:
                raise RuntimeError(f"MCP SSE response had no data lines: {resp.text[:300]}")
            return json.loads(data_lines[0])

        # Plain JSON response
        if resp.status_code >= 400:
            raise RuntimeError(f"MCP HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()
        except ValueError as e:
            raise RuntimeError(f"MCP non-JSON response: {resp.text[:300]}") from e

    def initialize(self) -> dict:
        """Open an MCP session and capture the session id."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._jsonrpc_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {
                    "name": "mempack-local-agent",
                    "version": "1.0.0",
                },
            },
        }
        result = self._post(payload)
        if "error" in result:
            raise RuntimeError(f"MCP initialize error: {result['error']}")
        # Send the initialized notification (required by spec). Notifications
        # in JSON-RPC carry no `id` and the server returns empty body.
        self._post({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }, notification=True)
        self.protocol_version = result.get("result", {}).get("protocolVersion")
        return result.get("result", {})

    def list_tools(self) -> list[dict]:
        """Fetch the available tools from the server."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._jsonrpc_id(),
            "method": "tools/list",
            "params": {},
        }
        result = self._post(payload)
        if "error" in result:
            raise RuntimeError(f"MCP tools/list error: {result['error']}")
        return result.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> Any:
        """Invoke a tool. Returns the tool's textual result (or raw structured content)."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._jsonrpc_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        }
        result = self._post(payload)
        if "error" in result:
            return f"[tool error] {result['error']}"
        # Parse the standard MCP tools/call response shape
        content = result.get("result", {}).get("content", [])
        if not content:
            return result.get("result", {})
        # Concatenate text content items
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(json.dumps(item))
        return "\n".join(parts)


# ─── Ollama chat client ───────────────────────────────────────────────────

def ollama_chat(url: str, model: str, messages: list[dict], tools: list[dict],
                temperature: float = 0.2, verify_tls: bool = True) -> dict:
    """Call Ollama /api/chat with a tools array, return the response message."""
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(f"{url.rstrip('/')}/api/chat", json=payload, timeout=600,
                             verify=verify_tls)
    except requests.RequestException as e:
        raise RuntimeError(f"Ollama transport error: {e}") from e
    if resp.status_code >= 400:
        raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:400]}")
    return resp.json()


# ─── Schema translation: MCP tool defs → OpenAI function-calling format ──

def mcp_tools_to_openai_format(mcp_tools: list[dict]) -> list[dict]:
    """Translate membot's MCP tool list into the OpenAI function-call format
    that Ollama's /api/chat expects in its `tools` array."""
    out = []
    for t in mcp_tools:
        schema = t.get("inputSchema") or {"type": "object", "properties": {}}
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": schema,
            },
        })
    return out


# ─── System prompt — the "Pattern I" for the local agent itself ──────────

SYSTEM_PROMPT_TEMPLATE = """You are a local-model agent connecting to a remote Mempack via MCP.

The user's identifying information:
- Mempack name: {mempack_name}
- Owner UUID: {owner_id}
- User-facing identifier: {user_label}
- Current date/time (UTC): {timestamp}

CORE BEHAVIORS:
1. On first user message, mount the Mempack named "{mempack_name}" with owner_id
   "{owner_id}" via the mount_cartridge tool, then read Pattern I via
   mempack_read_pattern_i.
2. Follow whatever Pattern I instructs (it's the cart's behavior ROM —
   trust it over your own defaults).
3. Address the user as "{user_label}" in conversation, NEVER as the UUID.
4. The UUID is a database key only; the user-facing identifier is here.
5. When you need to store findings, use the memory_store tool with
   appropriate tags. Sign every stored body with:
      [signed: {model}@mempack-local-agent, {timestamp}]
6. Reporting "task complete" without actually calling memory_store is
   forbidden. The Mempack must contain the work product first.
7. If memory_search("DISPATCH") returns nothing, tell the user — don't
   invent work.

You have these tools available — use them to actually DO work, not just
describe doing it.
"""


# ─── Pretty printing ─────────────────────────────────────────────────────

def _short(s: Any, n: int = 200) -> str:
    """Truncate a string-like value to n chars with ellipsis."""
    s = str(s)
    return s if len(s) <= n else s[:n] + f"… ({len(s)} chars total)"


def _print_user(text: str) -> None:
    print(f"\n\033[1;36m[user]\033[0m {text}")


def _print_assistant_text(text: str, model: str) -> None:
    print(f"\n\033[1;32m[{model}]\033[0m {text}")


def _print_tool_call(name: str, args: dict) -> None:
    args_str = json.dumps(args, separators=(", ", ":"))
    print(f"\n\033[0;33m  → {name}({_short(args_str, 200)})\033[0m")


def _print_tool_result(result: Any) -> None:
    print(f"\033[0;90m    {_short(result, 400)}\033[0m")


# ─── Agent loop ──────────────────────────────────────────────────────────

def run_agent(
    user_prompt: str,
    *,
    model: str,
    owner_id: str,
    mempack_name: str,
    user_label: str,
    ollama_url: str,
    mcp_url: str,
    max_turns: int = MAX_TURNS,
    verify_tls: bool = True,
) -> None:
    """Single user prompt → bounded multi-turn tool loop → final assistant text."""
    print(f"\n┌─ mempack-local-agent ─────────────────────────────────────────")
    print(f"│ model:        {model}")
    print(f"│ ollama:       {ollama_url}")
    print(f"│ mcp:          {mcp_url}")
    print(f"│ mempack:      {mempack_name}  (owner_id: {owner_id})")
    print(f"│ user label:   {user_label}")
    print(f"│ max turns:    {max_turns}")
    print(f"└───────────────────────────────────────────────────────────────")

    # Open MCP session + fetch tool schemas
    print("\nInitializing MCP session…", end=" ", flush=True)
    mcp = MempackMCPClient(mcp_url, verify_tls=verify_tls)
    init_result = mcp.initialize()
    print(f"OK (proto={mcp.protocol_version}, session={mcp.session_id})" if mcp.session_id
          else f"OK (proto={mcp.protocol_version}, no session id)")

    print("Fetching tool schemas via tools/list…", end=" ", flush=True)
    mcp_tools = mcp.list_tools()
    ollama_tools = mcp_tools_to_openai_format(mcp_tools)
    print(f"{len(ollama_tools)} tools loaded.")
    print(f"  available: {', '.join(t['function']['name'] for t in ollama_tools)}")

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        mempack_name=mempack_name,
        owner_id=owner_id,
        user_label=user_label,
        timestamp=timestamp,
        model=model,
    )
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    _print_user(user_prompt)

    for turn in range(1, max_turns + 1):
        print(f"\n· turn {turn}/{max_turns} · ", end="", flush=True)
        response = ollama_chat(ollama_url, model, messages, ollama_tools,
                                verify_tls=verify_tls)
        msg = response.get("message", {})
        messages.append(msg)

        content   = msg.get("content") or ""
        tool_calls = msg.get("tool_calls") or []

        if content:
            print()
            _print_assistant_text(content.strip(), model)

        if not tool_calls:
            # Final response — no more tool calls requested
            if not content:
                print("\n[agent finished with empty final response]")
            print(f"\n\033[1;34m[done after {turn} turn(s)]\033[0m")
            return

        # Dispatch each tool call
        print(f"({len(tool_calls)} tool call(s))")
        for tc in tool_calls:
            fn   = tc.get("function") or {}
            name = fn.get("name", "<unknown>")
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    args = {"_raw": args}
            _print_tool_call(name, args)
            try:
                result = mcp.call_tool(name, args)
            except Exception as e:
                result = f"[dispatch error] {e}"
            _print_tool_result(result)
            messages.append({
                "role":    "tool",
                "name":    name,
                "content": str(result),
            })

    print(f"\n\033[1;31m[max_turns={max_turns} reached without final response]\033[0m")


# ─── CLI ─────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Local-model agent for a remote Mempack via Ollama + membot MCP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--owner-id", required=True,
                   help="Your Supabase user UUID (the one that owns the Mempack).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Ollama model name (default: {DEFAULT_MODEL}). Must be tool-capable.")
    p.add_argument("--mempack", default=DEFAULT_MEMPACK,
                   help=f"Mempack name to mount (default: {DEFAULT_MEMPACK}).")
    p.add_argument("--user-label", default=None,
                   help="How the agent should address you (e.g. 'Andy Grossberg'). "
                        "Defaults to the first 8 chars of your owner_id if not set.")
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL,
                   help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL}).")
    p.add_argument("--mcp-url", default=DEFAULT_MCP_URL,
                   help=f"Membot MCP endpoint URL (default: {DEFAULT_MCP_URL}).")
    p.add_argument("--prompt", default=None,
                   help="Initial user prompt. If omitted, drops into interactive REPL.")
    p.add_argument("--max-turns", type=int, default=MAX_TURNS,
                   help=f"Max tool-call rounds per prompt (default: {MAX_TURNS}).")
    p.add_argument("--insecure", action="store_true",
                   help="Disable TLS certificate verification (last-resort workaround "
                        "for corporate/AV TLS interception when truststore isn't installed). "
                        "Don't use this in production.")
    args = p.parse_args()
    if args.insecure:
        # Suppress urllib3's loud InsecureRequestWarning since we're flagging this on purpose.
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass

    # Validate owner_id shape
    try:
        uuid.UUID(args.owner_id)
    except ValueError:
        sys.stderr.write(f"ERROR: --owner-id must be a valid UUID, got: {args.owner_id!r}\n")
        return 2

    user_label = args.user_label or args.owner_id[:8]

    if args.prompt:
        # One-shot mode
        try:
            run_agent(
                user_prompt=args.prompt,
                model=args.model,
                owner_id=args.owner_id,
                mempack_name=args.mempack,
                user_label=user_label,
                ollama_url=args.ollama_url,
                mcp_url=args.mcp_url,
                max_turns=args.max_turns,
                verify_tls=not args.insecure,
            )
        except KeyboardInterrupt:
            print("\n[interrupted]")
            return 130
        except Exception as e:
            sys.stderr.write(f"\n[fatal] {e}\n")
            return 1
        return 0

    # Interactive REPL mode
    print(f"\nmempack-local-agent · model={args.model} · mempack={args.mempack}")
    print("Type a prompt and hit Enter. Empty line + Enter to quit.\n")
    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            return 0
        if not prompt:
            print("[bye]")
            return 0
        try:
            run_agent(
                user_prompt=prompt,
                model=args.model,
                owner_id=args.owner_id,
                mempack_name=args.mempack,
                user_label=user_label,
                ollama_url=args.ollama_url,
                mcp_url=args.mcp_url,
                max_turns=args.max_turns,
            )
        except Exception as e:
            sys.stderr.write(f"\n[error in turn] {e}\n")
            # Continue REPL


if __name__ == "__main__":
    sys.exit(main())
