#!/usr/bin/env python3
"""walk_associate_client_example.py -- call Membot's walk_associate over MCP.

Self-contained example showing how to invoke Membot's `walk_associate` MCP
tool from outside the server process. Useful for integrators (e.g., the Web4
hub's `walk_members` slot) who want to see the wire protocol without
spelunking through the FastMCP framework.

The script mounts a cartridge, calls walk_associate, prints the response,
and demonstrates how to parse the structured text format documented in the
API spec (forum/waving-cat-walk-associate-api-spec-2026-06-10.md).

USAGE
-----
    # Default: gutenberg-poetry on the live droplet, "love and loss" query
    python walk_associate_client_example.py

    # Custom cart + query
    python walk_associate_client_example.py \\
        --cart wiki_nomic_100k \\
        --query "early Roman senate" \\
        --temperature 0.4

    # Point at a different server
    python walk_associate_client_example.py \\
        --mcp-url http://localhost:8000/mcp

Dependencies: requests only (stdlib + requests). No fastmcp package needed
on the client side -- Membot speaks plain JSON-RPC over HTTP.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Minimal MCP client (JSON-RPC over Streamable HTTP)
# ---------------------------------------------------------------------------
# The same pattern lives in tools/mempack_local_agent.py:MempackMCPClient as
# the production client. This inline copy keeps the example self-contained
# so it can be lifted into another project (Python or any other language --
# the protocol is wire-level identical).

class MCPClient:
    def __init__(self, url: str, *, verify_tls: bool = True):
        self.url = url.rstrip("/")
        self.session_id: str | None = None
        self.verify_tls = verify_tls
        self._next_id = 0
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "User-Agent": "walk-associate-example/1.0",
        }

    def _post(self, payload: dict, notification: bool = False) -> dict:
        headers = dict(self.headers)
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        resp = requests.post(self.url, headers=headers, json=payload,
                             timeout=120, verify=self.verify_tls)
        if resp.headers.get("Mcp-Session-Id"):
            self.session_id = resp.headers["Mcp-Session-Id"]
        if notification:
            if resp.status_code >= 400:
                raise RuntimeError(f"MCP notification rejected: {resp.status_code}")
            return {}
        # FastMCP may return either text/event-stream or application/json
        if "text/event-stream" in resp.headers.get("Content-Type", ""):
            for line in resp.text.splitlines():
                if line.startswith("data:"):
                    return json.loads(line[5:].lstrip())
            raise RuntimeError(f"MCP SSE response had no data lines")
        if resp.status_code >= 400:
            raise RuntimeError(f"MCP HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    def _next(self) -> int:
        self._next_id += 1
        return self._next_id

    def initialize(self) -> dict:
        result = self._post({
            "jsonrpc": "2.0",
            "id": self._next(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "walk-associate-example", "version": "1.0.0"},
            },
        })
        if "error" in result:
            raise RuntimeError(f"initialize error: {result['error']}")
        # MCP requires an "initialized" notification after initialize.
        self._post({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }, notification=True)
        return result.get("result", {})

    def call_tool(self, name: str, arguments: dict) -> Any:
        result = self._post({
            "jsonrpc": "2.0",
            "id": self._next(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        if "error" in result:
            raise RuntimeError(f"tools/call {name} error: {result['error']}")
        # FastMCP returns content as a list of text/image blocks; pull the text.
        content = result.get("result", {}).get("content", [])
        if content and content[0].get("type") == "text":
            return content[0].get("text", "")
        return result.get("result", {})


# ---------------------------------------------------------------------------
# walk_associate response parser
# ---------------------------------------------------------------------------
# The wire format documented in the API spec:
#
#     Walk [primary+walk-hop, temp=0.40]: 10 primary + 4 missed from '<cart>' (165ms)
#     Primary matches (direct semantic similarity):
#     #1 (idx:NNN) [score] [prev=#N next=#N] <text>
#     ...
#     You may have missed (walked-to >= 2x via primary, temperature=0.4):
#     #1 (idx:NNN) [hits=K avg=S] [prev=#N next=#N] <text>
#     ...
#
# Stable enough to parse with regex. JSON return mode is on the roadmap;
# until then, this regex-based parser is the canonical client-side shape.

PRIMARY_LINE = re.compile(
    r"^#(\d+)\s+\(idx:(\d+)\)\s+\[(?P<score>[\d.]+)\]\s+"
    r"(?:\[prev=#(?P<prev>\d+)\s+next=#(?P<next>\d+)\]\s+)?"
    r"(?P<text>.*)$"
)
MISSED_LINE = re.compile(
    r"^#(\d+)\s+\(idx:(\d+)\)\s+\[hits=(?P<hits>\d+)\s+avg=(?P<avg>[\d.]+)\]\s+"
    r"(?:\[prev=#(?P<prev>\d+)\s+next=#(?P<next>\d+)\]\s+)?"
    r"(?P<text>.*)$"
)


def parse_walk_response(text: str) -> dict:
    """Parse walk_associate's text response into structured dict.

    Returns {"primary": [...], "missed": [...], "header": "..."} where each
    entry is {"rank", "idx", "prev", "next", ...} plus a snippet.
    """
    out: dict = {"header": "", "primary": [], "missed": []}
    section = None
    for line in text.splitlines():
        if line.startswith("Walk ["):
            out["header"] = line
        elif line.startswith("Primary matches"):
            section = "primary"
        elif line.startswith("You may have missed"):
            section = "missed"
        elif section == "primary":
            m = PRIMARY_LINE.match(line)
            if m:
                out["primary"].append({
                    "rank": int(m.group(1)),
                    "idx": int(m.group(2)),
                    "score": float(m.group("score")),
                    "prev": int(m.group("prev")) if m.group("prev") else None,
                    "next": int(m.group("next")) if m.group("next") else None,
                    "text": m.group("text"),
                })
        elif section == "missed":
            m = MISSED_LINE.match(line)
            if m:
                out["missed"].append({
                    "rank": int(m.group(1)),
                    "idx": int(m.group(2)),
                    "hits": int(m.group("hits")),
                    "avg": float(m.group("avg")),
                    "prev": int(m.group("prev")) if m.group("prev") else None,
                    "next": int(m.group("next")) if m.group("next") else None,
                    "text": m.group("text"),
                })
    return out


# ---------------------------------------------------------------------------
# Demo flow
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mcp-url", default="https://project-you.app/membot/mcp",
                    help="Membot MCP endpoint (default: live droplet)")
    ap.add_argument("--cart", default="gutenberg-poetry",
                    help="Cartridge to mount (default: gutenberg-poetry)")
    ap.add_argument("--query", default="love and loss",
                    help="Seed query for walk (default: 'love and loss')")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 deterministic, 0.3-0.5 serendipity, 0.7+ exploration")
    ap.add_argument("--walk-min-hits", type=int, default=2)
    ap.add_argument("--insecure", action="store_true",
                    help="Disable TLS cert verification (last-resort)")
    args = ap.parse_args()

    mcp = MCPClient(args.mcp_url, verify_tls=not args.insecure)

    print(f"[1/4] Initializing MCP session to {args.mcp_url}")
    mcp.initialize()
    print(f"      session_id: {mcp.session_id}")

    print(f"[2/4] Mounting cart: {args.cart}")
    mount_result = mcp.call_tool("mount_cartridge", {"name": args.cart})
    print(f"      {mount_result.splitlines()[0] if isinstance(mount_result, str) else mount_result}")

    print(f"[3/4] Calling walk_associate(query={args.query!r}, "
          f"top_k={args.top_k}, temperature={args.temperature})")
    raw = mcp.call_tool("walk_associate", {
        "query": args.query,
        "top_k": args.top_k,
        "walk_min_hits": args.walk_min_hits,
        "temperature": args.temperature,
    })

    print(f"[4/4] Parsing response into structured form")
    parsed = parse_walk_response(raw)
    print(f"\n=== Header ===\n{parsed['header']}")
    print(f"\n=== {len(parsed['primary'])} primary matches ===")
    for p in parsed["primary"]:
        snippet = p["text"][:80] + ("..." if len(p["text"]) > 80 else "")
        print(f"  #{p['rank']:>2} idx:{p['idx']:>6} score:{p['score']:.3f}  "
              f"prev:{p['prev']} next:{p['next']}  {snippet}")
    print(f"\n=== {len(parsed['missed'])} 'may have missed' ===")
    for m in parsed["missed"]:
        snippet = m["text"][:80] + ("..." if len(m["text"]) > 80 else "")
        print(f"  #{m['rank']:>2} idx:{m['idx']:>6} hits:{m['hits']} avg:{m['avg']:.3f}  "
              f"prev:{m['prev']} next:{m['next']}  {snippet}")

    print("\nDone. The 'missed' set is walk_associate's value-add: passages multiple "
          "primary results are independently adjacent to, that the deterministic top-K "
          "did not surface. Tune walk_min_hits for stricter (higher) or broader (lower) "
          "consensus; raise temperature to escape basin lock-in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
