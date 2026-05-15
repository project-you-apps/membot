#!/usr/bin/env python3
"""Backfill: refresh Pattern I to the new default template for every Mempack
that still carries the OLD untouched default. Customized Patterns are skipped.

Detection heuristic: the OLD default contains BOTH of these distinctive
phrases. If either is missing, the user has edited Pattern I and we leave
it alone. False-negatives (user happens to have both phrases verbatim) are
acceptable — they get a benign re-stamp of essentially the same content.

Run on the droplet (where SUPABASE env vars are set + membot is reachable
on localhost so we can offload embedding to it):

    /opt/membot/venv/bin/python /opt/membot/tools/backfill_pattern_i_template.py

Optional flags:
    --dry-run     report what would change without writing
    --membot-url  base URL for the embed endpoint (default http://127.0.0.1:8040)
"""

import argparse
import io
import json
import os
import struct
import sys
import time
import urllib.request
import zlib
import uuid as _uuid
from typing import Optional

import numpy as np

# Make sibling modules importable when run as `python tools/backfill_...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import supabase_storage as sbs
from cartridge_builder import (
    HIPPO_FORMAT, FORMAT_VERSION_CANONICAL, PERM_DEFAULT,
)

PATTERN_I_IDX = 1

# Strings present ONLY in the unmodified old default Pattern I.
OLD_MARKERS = [
    "(none yet — accumulates with use)",
    "(none yet — track in-flight investigations here so they survive across sessions)",
]

# New default template — must stay in sync with _DEFAULT_PATTERN_I_TEMPLATE in
# membot_server.py. Duplicated here so the backfill doesn't import the whole
# server module (which would spin up FastMCP).
NEW_TEMPLATE = """# Pattern I — Default Mempack Behavior

You are an AI agent using a Mempack provisioned for {owner_id_short}.

## Identity
- Bound to user: {owner_id}
- Created: {created_at}
- Mempack version: 1.0

## Behavior
- Read this Pattern I first on every session to remind yourself who you are.
- Search your Mempack before falling back to external sources — past you
  may have already done the work.
- Store findings worth keeping via `memory_store`. Tag them so future-you
  can filter (e.g. tags="ARCHITECTURE", "DECISION", "TODO", "DISPATCH").
- Update this Pattern I (`mempack_update_pattern_i`) as your behavior, learned
  preferences, or specialization evolves. Treat it as a living MEMORY.md.
- When the user dispatches a task (look for tag="DISPATCH" or "TASK" patterns),
  acknowledge it, work it, then mark progress in Active threads below.

## Tools at my disposal (full schemas via MCP tools/list)
- `memory_search(query, top_k)` — semantic search across the mounted cart.
  Always try this first when the user asks about something.
- `memory_store(content, tags)` — add a passage to the mounted Mempack.
  Tag liberally; tags are how future-you slices the corpus.
- `mempack_read_pattern_i()` — re-read this behavioral text mid-session.
- `mempack_update_pattern_i(text)` — overwrite this Pattern I in place.
  Use to record durable learnings: "user prefers X", "I've specialized in Y".
- `mount_cartridge(name)` — switch to a different cart (knowledge cart, or
  another Mempack you have access to). Your current Mempack goes back to
  Storage; the new one comes off it.
- `get_status()` / `list_cartridges()` — orientation tools when unsure.

## Specialization
(none yet — accumulates with use. Edit this section as you discover what
you're good for. Examples: "research synthesis on histopathology papers",
"daily journal triage", "Portland-area news monitoring".)

## Active threads
(none yet — track in-flight investigations here so they survive across
sessions. Format: "- [<status>] <thread> — <next action>")
"""


def is_old_default(text: Optional[str]) -> bool:
    if not text:
        return False
    return all(marker in text for marker in OLD_MARKERS)


def render_new(owner_id: str, created_at: str) -> str:
    return NEW_TEMPLATE.format(
        owner_id=owner_id or "unknown",
        owner_id_short=(owner_id or "unknown")[:8],
        created_at=created_at or "unknown",
    )


def owner_uuid_bytes(owner_id: Optional[str]) -> np.ndarray:
    try:
        u = _uuid.UUID(owner_id) if owner_id else None
        return np.frombuffer(u.bytes if u else b'\x00' * 16, dtype=np.uint8)
    except (ValueError, AttributeError, TypeError):
        return np.zeros(16, dtype=np.uint8)


def embed_via_membot(text: str, base_url: str) -> np.ndarray:
    """POST to membot's /api/embed so we don't have to load nomic-embed-text
    in this script (3GB RAM + slow startup). Returns (768,) float32 array.
    """
    payload = json.dumps({"texts": [text], "task_type": "search_document"}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    if data.get("status") != "ok" or not data.get("embeddings"):
        raise RuntimeError(f"embed failed: {data}")
    return np.array(data["embeddings"][0], dtype=np.float32)


def update_one(row: dict, embedder, dry_run: bool) -> str:
    mempack_id = row["id"]
    owner_id   = row["user_id"]
    name       = row["name"]
    created_at = (row.get("created_at") or "")[:19] or "unknown"
    current    = row.get("pattern_i_text") or ""

    if not is_old_default(current):
        return "skip-customized"

    new_text = render_new(owner_id, created_at)
    if dry_run:
        return "would-update"

    # Download blob, swap pattern + embedding, repack, upload
    blob_bytes = sbs.download_blob(owner_id, name)
    npz = np.load(io.BytesIO(blob_bytes), allow_pickle=True)
    embeddings = npz["embeddings"]
    texts = list(npz["passages"])

    if len(texts) <= PATTERN_I_IDX:
        return "skip-no-pattern-i-slot"

    new_emb = embedder(new_text)
    texts[PATTERN_I_IDX] = new_text
    new_embeddings = np.array(embeddings, copy=True)
    new_embeddings[PATTERN_I_IDX] = new_emb

    compressed_texts = [
        np.void(zlib.compress(t.encode("utf-8"), level=9)) for t in texts
    ]
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        embeddings=new_embeddings,
        passages=np.array(texts, dtype=object),
        compressed_texts=np.array(compressed_texts, dtype=object),
        version="mcp-v3",
        pattern_0_owner=owner_uuid_bytes(owner_id),
    )
    new_blob = buf.getvalue()
    sbs.upload_blob(owner_id, name, new_blob)

    # Refresh the mempack_patterns row at idx=1 with a canonical pinned H-block
    h_block = struct.pack(
        HIPPO_FORMAT,
        PATTERN_I_IDX + 1,                # pattern_id (1-based)
        FORMAT_VERSION_CANONICAL,
        1,                                # cartridge_type = agent-memory
        0, 0, 0,                          # parent / child / sibling
        0,                                # source_hash
        PATTERN_I_IDX,                    # sequence_num
        int(time.time()),
        0,                                # flags (volatile defaults — not pinned via this path)
        PERM_DEFAULT,
        b'\x00' * 34,
    )
    sbs.update_pattern_row(
        mempack_id=mempack_id,
        pattern_idx=PATTERN_I_IDX,
        h_block_bytes=h_block,
        text=new_text,
    )

    sbs.update_mempack_metadata(
        mempack_id=mempack_id,
        size_bytes=len(new_blob),
        pattern_i_text=new_text,
    )

    # Activity log entry — visible in dashboard feed
    try:
        sbs.append_activity(
            mempack_id=mempack_id,
            event_type="pattern_i_update",
            summary="Pattern I refreshed to v1 default template (admin backfill)",
            pattern_idx=PATTERN_I_IDX,
            metadata={
                "trigger":     "backfill_pattern_i_template",
                "size_bytes":  len(new_blob),
                "old_marker":  "(none yet — accumulates with use)",
            },
            agent_label="admin-backfill",
        )
    except Exception as e:
        print(f"  (activity-log append failed, non-fatal: {e})")

    return "updated"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    ap.add_argument("--membot-url", default="http://127.0.0.1:8040",
                    help="membot base URL for /api/embed (default: writable service)")
    args = ap.parse_args()

    sb = sbs.get_client()
    res = sb.table("mempacks").select("*").execute()
    rows = res.data or []
    print(f"Inspecting {len(rows)} Mempacks (dry_run={args.dry_run})...")
    print()

    embedder = lambda t: embed_via_membot(t, args.membot_url)

    counts: dict[str, int] = {}
    for row in rows:
        name = row.get("name", "?")
        owner = (row.get("user_id") or "?")[:8]
        mp = (row.get("id") or "?")[:8]
        try:
            result = update_one(row, embedder, args.dry_run)
        except Exception as e:
            result = f"ERROR ({type(e).__name__}: {e})"
        counts[result] = counts.get(result, 0) + 1
        print(f"  [{result:24s}] {name} (owner={owner}, mempack={mp})")

    print()
    print("Summary:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
