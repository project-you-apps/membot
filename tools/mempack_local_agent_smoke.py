#!/usr/bin/env python3
"""mempack_local_agent_smoke.py — write-side smoke test for the local agent.

Verifies the full author-→-store loop by:
  1. Generating a unique marker string
  2. Asking the local agent to store ONE small METHOD pattern containing
     that marker via memory_store
  3. Searching the Mempack via memory_search to confirm the pattern landed

Pass criteria: at least one search hit for the marker. Fail criteria: zero.

USAGE
-----
    python mempack_local_agent_smoke.py \\
        --owner-id 3579e6ee-6412-4099-8d66-a205d9be7849 \\
        --model qwen2.5:14b

  Optional flags (same as mempack_local_agent.py):
    --mempack          (default: primary)
    --ollama-url       (default: http://localhost:11434)
    --mcp-url          (default: https://project-you.app/membot/mcp)
    --max-turns        (default: 10)
    --insecure         (disable TLS verify — last resort)

EXIT CODES
----------
  0   PASS — marker found in Mempack after agent run
  1   FAIL — marker NOT found (agent didn't store, or stored without marker)
  2   bad CLI input
  3   harness subprocess failed before storing anything
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

THIS = Path(__file__).resolve()
HARNESS = THIS.parent / "mempack_local_agent.py"

# Import the MCP client from the harness so we can search before/after
sys.path.insert(0, str(THIS.parent))
try:
    from mempack_local_agent import MempackMCPClient  # noqa: E402
except ImportError as e:
    sys.stderr.write(f"ERROR: can't import MempackMCPClient from {HARNESS}: {e}\n")
    sys.exit(3)


def search_marker(mcp_url: str, marker: str, owner_id: str, mempack: str,
                  verify_tls: bool = True) -> list[str]:
    """Search the user's Mempack for the marker string via MCP memory_search.
    Returns a list of pattern bodies that contain the marker.

    Mounts the Mempack first (the search target must be mounted in this
    session). Returns results that mention the marker explicitly so we
    don't get fooled by lexical-similar hits.
    """
    mcp = MempackMCPClient(mcp_url, verify_tls=verify_tls)
    mcp.initialize()
    # Mount the user's Mempack into this MCP session
    mount_result = mcp.call_tool("mount_cartridge", {
        "name": mempack,
        "owner_id": owner_id,
    })
    # Search for the marker
    search_result = mcp.call_tool("memory_search", {
        "query": marker,
        "top_k": 5,
    })
    text = str(search_result)
    # Crude verification: require the marker substring to appear in results
    hits = []
    if marker in text:
        hits.append(text)
    return hits


def main() -> int:
    p = argparse.ArgumentParser(
        description="Write-side smoke test for mempack_local_agent.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--owner-id", required=True)
    p.add_argument("--model", default="qwen2.5:14b")
    p.add_argument("--mempack", default="primary")
    p.add_argument("--user-label", default=None)
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--mcp-url", default="https://project-you.app/membot/mcp")
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--insecure", action="store_true")
    args = p.parse_args()

    try:
        uuid.UUID(args.owner_id)
    except ValueError:
        sys.stderr.write(f"ERROR: --owner-id must be a valid UUID, got: {args.owner_id!r}\n")
        return 2

    # Generate a marker that's BOTH unique AND something we can read clearly
    # in the agent's stored pattern. Format: SMOKETEST-<8 hex>-<utc-iso>
    marker = f"SMOKETEST-{uuid.uuid4().hex[:8]}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    user_label = args.user_label or args.owner_id[:8]

    print("=" * 70)
    print(" mempack-local-agent · WRITE-SIDE SMOKE TEST")
    print("=" * 70)
    print(f"  model:       {args.model}")
    print(f"  mempack:     {args.mempack}")
    print(f"  owner_id:    {args.owner_id}")
    print(f"  marker:      {marker}")
    print(f"  harness:     {HARNESS}")
    print("=" * 70)
    print()

    # Step 1: Search BEFORE — sanity check the marker doesn't already exist
    print("[1/3] BEFORE: searching for marker (should return zero hits)...")
    try:
        pre_hits = search_marker(args.mcp_url, marker, args.owner_id, args.mempack,
                                  verify_tls=not args.insecure)
    except Exception as e:
        sys.stderr.write(f"  ERROR during BEFORE search: {e}\n")
        return 3
    if pre_hits:
        sys.stderr.write(
            f"  WARNING: BEFORE search already found the marker. Marker collision; "
            f"smoke test inconclusive. Try again.\n"
        )
        return 3
    print("  OK: marker not present (as expected for a fresh marker).")
    print()

    # Step 2: Run the harness with a single-store prompt.
    #
    # IMPORTANT: do NOT include literal `Call function(arg=value)` examples
    # in this prompt. Smaller models (Hermes-3:8b in particular) treat
    # demonstration syntax as "output this as text" rather than "dispatch
    # this as a tool call" — they perfectly format the JSON but emit it
    # in message.content instead of message.tool_calls. The fix is to
    # describe the task in plain language and let the tool schemas
    # (already passed to Ollama via the harness's tools/list fetch)
    # tell the model how to format the actual call.
    iso_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Body has NO leading [METHOD] — the server prepends "[tags] " automatically
    # when memory_store is called with tags="METHOD". Including [METHOD] in the
    # body causes a "[METHOD] [METHOD] ..." double-tag (cosmetic but ugly).
    body_to_store = (
        f"Smoke test marker: {marker}. "
        f"[signed: {args.model}@mempack-local-agent, {iso_now}]"
    )
    prompt = (
        f'You are a smoke-test agent. Two actions, in order:\n\n'
        f'First, mount the Mempack named "{args.mempack}" for the user with '
        f'owner_id {args.owner_id}.\n\n'
        f'Second, store the following text verbatim into that Mempack, tagged '
        f'METHOD. The text MUST be saved exactly as written, including the '
        f'marker and signature:\n\n'
        f'{body_to_store}\n\n'
        f'After the store completes, briefly confirm in one sentence and stop. '
        f'Do not call any other tools, do not search, do not read Pattern I, '
        f'do not invent additional findings. Just mount and store, then stop.\n\n'
        f'(If you need to address me, use "{user_label}". The owner_id above is '
        f'a database key, not my name.)'
    )

    print("[2/3] HARNESS: running the agent with the smoke-test prompt...")
    print(f"  (this can take 30-90 seconds depending on model + machine)")
    print()

    cmd = [
        sys.executable, str(HARNESS),
        "--owner-id", args.owner_id,
        "--model", args.model,
        "--user-label", user_label,
        "--mempack", args.mempack,
        "--ollama-url", args.ollama_url,
        "--mcp-url", args.mcp_url,
        "--max-turns", str(args.max_turns),
        "--prompt", prompt,
    ]
    if args.insecure:
        cmd.append("--insecure")

    try:
        result = subprocess.run(cmd, timeout=600)
    except subprocess.TimeoutExpired:
        sys.stderr.write("\n  ERROR: harness subprocess timed out after 10 min.\n")
        return 3
    except Exception as e:
        sys.stderr.write(f"\n  ERROR: harness subprocess failed to launch: {e}\n")
        return 3

    if result.returncode != 0:
        sys.stderr.write(f"\n  WARNING: harness exited with code {result.returncode}\n")
        # Don't bail here — the harness may have stored anyway. Check below.

    print()
    print("[3/3] AFTER: searching for marker (should return at least one hit)...")
    try:
        post_hits = search_marker(args.mcp_url, marker, args.owner_id, args.mempack,
                                   verify_tls=not args.insecure)
    except Exception as e:
        sys.stderr.write(f"  ERROR during AFTER search: {e}\n")
        return 3

    print()
    print("=" * 70)
    if post_hits:
        print(" RESULT: \033[1;32mPASS\033[0m")
        print(f"   Marker {marker!r} found in Mempack after agent run.")
        print(f"   Hit preview:")
        for hit in post_hits:
            preview = hit if len(hit) <= 400 else hit[:400] + f"… ({len(hit)} chars total)"
            for line in preview.splitlines()[:10]:
                print(f"     {line}")
        print("=" * 70)
        print()
        print("Verify on the dashboard:")
        print(f"  https://project-you.app/membot/app")
        print(f"  → click Mempack → Patterns Browser → search for '{marker}'")
        print()
        return 0
    else:
        print(" RESULT: \033[1;31mFAIL\033[0m")
        print(f"   Marker {marker!r} NOT found in Mempack after agent run.")
        print()
        print("Likely causes:")
        print("  1. The model emitted natural-language 'I stored…' text instead")
        print("     of an actual memory_store tool call (confabulation).")
        print("  2. The model called memory_store but stripped/altered the marker.")
        print("  3. The model called memory_store on the wrong cart (no mount).")
        print()
        print("Diagnostics:")
        print("  - Scroll up to the harness output. Look for a tool-call card")
        print("    labeled 'memory_store' — if absent, the model didn't dispatch.")
        print("  - Check the dashboard activity feed: any new IMPRINT row in the")
        print("    last few minutes? If yes, the store worked but the marker")
        print("    didn't make it through (case 2). If no, the model didn't")
        print("    store (case 1 or 3).")
        print("  - Try a larger or more disciplined model: qwen2.5-coder:14b,")
        print("    hermes3:8b, or hermes3:70b if you have the VRAM.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
