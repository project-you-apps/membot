#!/usr/bin/env python3
"""
migrate_droplet_mempacks_to_supabase.py — one-shot cold-cut migration of
existing droplet-FS Mempacks into Supabase (Postgres rows + Storage blobs).

Walks `/opt/membot/cartridges/users/<uuid>/` for `*.cart.npz` files, reads
each cart's embeddings + texts + manifest sidecar, synthesizes a canonical
12-field H-block array if the NPZ doesn't have one (the smoke-test mempack
from 2026-05-12 doesn't), uploads to Supabase Storage, and inserts rows
into public.mempacks + public.mempack_patterns.

Supports an --map argument to remap synthetic owner_ids (like the smoke-test
ececf504-... UUID that doesn't correspond to a real Supabase user) to real
auth.users IDs. Without --map, the script tries to look up by email.

Usage:
    python migrate_droplet_mempacks_to_supabase.py [--dry-run] [--map OLD=NEW]
                                                   [--owner-email andy.grossberg@gmail.com]
                                                   [--base /opt/membot/cartridges/users]

Idempotent: skips Mempacks whose (user_id, name) row already exists in
public.mempacks. Re-run-safe if a previous run was interrupted mid-flight.

Andy + Claude 2026-05-13.
"""

import argparse
import io
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


# Make sibling modules importable when run from various cwd's
THIS_DIR = Path(__file__).resolve().parent
MEMBOT_DIR = THIS_DIR.parent
sys.path.insert(0, str(MEMBOT_DIR))

import supabase_storage as sbs
from cartridge_builder import (
    HIPPO_FORMAT, HIPPO_SIZE,
    FLAG_PINNED, FLAG_PERISH_ARCHIVAL, FORMAT_VERSION_CANONICAL,
    PERM_R, PERM_DEFAULT,
)


# H-block cartridge_type enum values (packed into uint8)
CART_TYPE_INT_KNOWLEDGE    = 0
CART_TYPE_INT_AGENT_MEMORY = 1


def synthesize_hippocampus(n_patterns: int, cart_type: str = "agent-memory") -> bytes:
    """Synthesize a canonical H-block array for a cart that has no existing one.

    For agent-memory carts: idx=0 is the marker (pinned, read-only), idx=1 is
    Pattern I (pinned, RW), and any idx >= 2 are routine entries (RW).

    Returns: concatenated bytes, N * HIPPO_SIZE long.
    """
    now_ts = int(time.time())
    cart_type_int = (
        CART_TYPE_INT_AGENT_MEMORY if cart_type == "agent-memory"
        else CART_TYPE_INT_KNOWLEDGE
    )

    blocks = []
    for i in range(n_patterns):
        if i == 0:
            # Pattern 0 / marker: pinned + archival + read-only, no links yet
            pattern_id = 0
            flags = FLAG_PINNED | FLAG_PERISH_ARCHIVAL
            perms = PERM_R
            next_ptr = 1 if n_patterns > 1 else 0
        elif i == 1:
            # Pattern I (agent behavioral instructions): pinned + archival + RW
            pattern_id = 1
            flags = FLAG_PINNED | FLAG_PERISH_ARCHIVAL
            perms = PERM_DEFAULT
            next_ptr = 0  # no further patterns chained yet
        else:
            # Pattern N+: routine memory entries — volatile by default (eligible
            # for time-decay GC). Caller can promote to archival explicitly.
            pattern_id = i
            flags = 0
            perms = PERM_DEFAULT
            next_ptr = 0

        block = struct.pack(
            HIPPO_FORMAT,
            pattern_id,                  # pattern_id
            FORMAT_VERSION_CANONICAL,    # format_version
            cart_type_int,               # cartridge_type
            0,                           # parent_ptr
            next_ptr,                    # child_ptr
            0,                           # sibling_ptr
            0,                           # source_hash (synthesized, no source file)
            i,                           # sequence_num
            now_ts,                      # timestamp
            flags,                       # flags
            perms,                       # perms_byte
            b'\x00' * 34,                # reserved
        )
        blocks.append(block)

    return b''.join(blocks)


def load_npz_bytes(path: Path) -> bytes:
    """Read a .cart.npz file as raw bytes (for re-upload to Storage)."""
    return path.read_bytes()


def parse_cart(path: Path) -> dict:
    """Read embeddings + texts + (optional) hippocampus + manifest from a cart.

    Returns dict with keys: embeddings, texts, hippocampus (or None), n_patterns.
    """
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.keys())

    embeddings = npz["embeddings"] if "embeddings" in keys else None
    texts_raw = npz["passages"] if "passages" in keys else (npz["texts"] if "texts" in keys else None)
    texts = [str(t) for t in texts_raw] if texts_raw is not None else []
    hippo = npz["hippocampus"].tobytes() if "hippocampus" in keys else None

    n_patterns = embeddings.shape[0] if embeddings is not None else len(texts)
    return {
        "embeddings": embeddings,
        "texts": texts,
        "hippocampus": hippo,
        "n_patterns": n_patterns,
    }


def parse_manifest(cart_path: Path) -> dict:
    """Read the manifest sidecar (`<name>.cart_manifest.json`) if present."""
    manifest_path = cart_path.parent / cart_path.name.replace(".cart.npz", ".cart_manifest.json")
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [warn] failed to parse manifest at {manifest_path}: {e}")
        return {}


def migrate_one(
    cart_path: Path,
    owner_id: str,
    dry_run: bool = False,
) -> dict:
    """Migrate a single cart to Supabase. Returns a result dict for logging."""
    name = cart_path.stem.replace(".cart", "")  # "primary.cart.npz" -> "primary"
    print(f"\n--- {cart_path} ---")
    print(f"  owner_id: {owner_id}")
    print(f"  name:     {name}")

    # Idempotency check: skip if already migrated
    if not dry_run:
        existing = sbs.get_mempack_by_name(owner_id, name)
        if existing:
            print(f"  [skip] already in Supabase (id={existing['id']}, status={existing['storage_status']})")
            return {"status": "skipped", "reason": "already_migrated", "owner_id": owner_id, "name": name}

    cart = parse_cart(cart_path)
    manifest = parse_manifest(cart_path)
    n = cart["n_patterns"]
    print(f"  patterns: {n}")
    print(f"  has_hippocampus: {cart['hippocampus'] is not None}")

    # Synthesize H-blocks if NPZ didn't have any
    cart_type = manifest.get("cart_type", "agent-memory")
    if cart["hippocampus"] is None:
        print(f"  [synth] generating fresh canonical H-blocks (cart_type={cart_type})")
        hippo_bytes = synthesize_hippocampus(n, cart_type=cart_type)
    else:
        hippo_bytes = cart["hippocampus"]
        if len(hippo_bytes) != n * HIPPO_SIZE:
            print(f"  [warn] hippocampus length {len(hippo_bytes)} != n*HIPPO_SIZE ({n*HIPPO_SIZE}); regenerating")
            hippo_bytes = synthesize_hippocampus(n, cart_type=cart_type)

    blob_bytes = load_npz_bytes(cart_path)
    size_bytes = len(blob_bytes)
    briefing = manifest.get("briefing", "")
    pattern_i_text = cart["texts"][1] if n > 1 else ""

    print(f"  blob size: {size_bytes} bytes")
    print(f"  briefing chars: {len(briefing)}")
    print(f"  pattern_i chars: {len(pattern_i_text)}")

    if dry_run:
        print("  [dry-run] would: insert row, upload blob, insert pattern rows, mark ready, log provision")
        return {"status": "dry_run", "owner_id": owner_id, "name": name, "n_patterns": n, "size_bytes": size_bytes}

    # Real migration
    try:
        row = sbs.insert_mempack(
            owner_id=owner_id,
            name=name,
            pattern_count=n,
            size_bytes=size_bytes,
            briefing=briefing,
            pattern_i_text=pattern_i_text,
            manifest=manifest,
            cart_type=cart_type,
            status="pending",
        )
        mempack_id = row["id"]
        print(f"  [step 1/4] mempack row inserted: id={mempack_id}, status=pending")

        sbs.upload_blob(owner_id, name, blob_bytes)
        print(f"  [step 2/4] blob uploaded to mempacks/{owner_id}/{name}.cart.npz")

        inserted = sbs.insert_pattern_rows(mempack_id, hippo_bytes, cart["texts"])
        print(f"  [step 3/4] {inserted} pattern rows inserted")

        sbs.mark_mempack_ready(mempack_id)
        print(f"  [step 4/4] status -> ready")

        sbs.log_provision(owner_id, mempack_id, "migration", "created")
        print(f"  [done] migrated")
        return {"status": "migrated", "owner_id": owner_id, "name": name, "n_patterns": n,
                "size_bytes": size_bytes, "mempack_id": mempack_id}

    except Exception as e:
        print(f"  [error] migration failed: {e}")
        try:
            sbs.log_provision(owner_id, None, "migration", "failed", error=str(e))
        except Exception:
            pass
        return {"status": "failed", "owner_id": owner_id, "name": name, "error": str(e)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default="/opt/membot/cartridges/users",
                   help="Root dir to walk for existing Mempacks (default: %(default)s)")
    p.add_argument("--map", action="append", default=[],
                   help="Remap owner_id, format OLD_UUID=NEW_UUID. Can be passed multiple times.")
    p.add_argument("--owner-email", default=None,
                   help="If set, look up real user UUID by this email for any owner_id not in --map.")
    p.add_argument("--dry-run", action="store_true",
                   help="Walk and report only; no Supabase writes.")
    args = p.parse_args()

    base = Path(args.base)
    if not base.exists():
        print(f"Base dir {base} does not exist; nothing to migrate.")
        return 0

    # Parse --map entries
    remap = {}
    for entry in args.map:
        if "=" not in entry:
            print(f"[error] --map entry '{entry}' missing '=' separator", file=sys.stderr)
            return 2
        old, new = entry.split("=", 1)
        remap[old.strip()] = new.strip()

    # Find all user dirs
    user_dirs = [d for d in base.iterdir() if d.is_dir()]
    print(f"Found {len(user_dirs)} user dir(s) under {base}")

    # Optional email->UUID lookup for fallback mapping
    email_uuid: Optional[str] = None
    if args.owner_email and not args.dry_run:
        print(f"\nLooking up user UUID for email: {args.owner_email}")
        email_uuid = sbs.find_user_uuid_by_email(args.owner_email)
        if not email_uuid:
            print(f"  [warn] no Supabase user found with email {args.owner_email}")
        else:
            print(f"  resolved to: {email_uuid}")

    results = []
    for user_dir in user_dirs:
        old_owner_id = user_dir.name
        # Resolve owner_id: explicit --map wins, then --owner-email fallback, else use as-is
        if old_owner_id in remap:
            new_owner_id = remap[old_owner_id]
            print(f"\n[map] {old_owner_id} -> {new_owner_id} (via --map)")
        elif email_uuid:
            new_owner_id = email_uuid
            print(f"\n[map] {old_owner_id} -> {new_owner_id} (via --owner-email fallback)")
        else:
            new_owner_id = old_owner_id
            print(f"\n[map] {old_owner_id} -> kept as-is")

        carts = list(user_dir.glob("*.cart.npz"))
        for cart_path in carts:
            result = migrate_one(cart_path, new_owner_id, dry_run=args.dry_run)
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"Migration summary:")
    print(f"  total carts processed: {len(results)}")
    by_status = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")
    print(f"{'='*60}")

    failed = sum(1 for r in results if r["status"] == "failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
