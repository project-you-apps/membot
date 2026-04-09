"""
rebuild_manifests.py — Cart hygiene utility.

Walks every .cart.npz / .cart.pkl file in a directory, computes the actual
fingerprint from the loaded data, and rebuilds the *_manifest.json file if
the stored fingerprint doesn't match. This fixes the "FINGERPRINT MISMATCH"
warnings when mounting carts that were rebuilt without updating their manifest.

Run from the membot directory:
    python scripts/rebuild_manifests.py                  # default: cartridges/
    python scripts/rebuild_manifests.py path/to/dir      # specific dir
    python scripts/rebuild_manifests.py --dry-run        # show what would change
    python scripts/rebuild_manifests.py --force          # rewrite even if matching

Non-destructive by default: only writes when fingerprints differ.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# scripts/ is one level deep — add membot/ to sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_MEMBOT_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _MEMBOT_DIR)

from membot_server import (
    load_cartridge_safe,
    compute_fingerprint,
    verify_manifest,
)


def find_cart_files(directory: str) -> list[str]:
    """Find every cart file (.cart.npz or .cart.pkl) in a directory."""
    out = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if f.endswith(".cart.npz") or f.endswith(".cart.pkl"):
                out.append(os.path.join(root, f))
    return out


def manifest_path_for(cart_path: str) -> str:
    """Compute the manifest path for a cart, matching membot_server's convention.
    cart_path 'foo.cart.npz' → manifest 'foo.cart_manifest.json'
    """
    return cart_path.rsplit(".", 1)[0] + "_manifest.json"


def rebuild_one(cart_path: str, dry_run: bool = False, force: bool = False) -> dict:
    """Inspect one cart and rebuild its manifest if needed.

    Returns a dict with keys: cart_path, status, expected_fp, actual_fp, action.
    status: 'ok' (matches) | 'rebuilt' | 'created' | 'error' | 'skipped'
    """
    result = {
        "cart_path": cart_path,
        "status": "unknown",
        "expected_fp": None,
        "actual_fp": None,
        "action": None,
    }

    try:
        data = load_cartridge_safe(cart_path)
    except Exception as e:
        result["status"] = "error"
        result["action"] = f"load failed: {e}"
        return result

    embeddings = data.get("embeddings")
    texts = data.get("texts", [])
    n_texts = len(texts)
    actual_fp = compute_fingerprint(embeddings, n_texts)
    result["actual_fp"] = actual_fp

    manifest_p = manifest_path_for(cart_path)

    if not os.path.exists(manifest_p):
        # No manifest at all — create one
        result["status"] = "created"
        result["action"] = "no manifest existed; will create"
        if not dry_run:
            _write_manifest(manifest_p, n_texts, actual_fp,
                            has_hippocampus=data.get("hippocampus") is not None)
        return result

    # Manifest exists — load and compare
    try:
        with open(manifest_p, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except Exception as e:
        result["status"] = "error"
        result["action"] = f"manifest read failed: {e}"
        return result

    expected_fp = existing.get("fingerprint", "")
    expected_count = existing.get("count", -1)
    result["expected_fp"] = expected_fp

    fingerprint_matches = (expected_fp == actual_fp)
    count_matches = (expected_count == n_texts)

    if fingerprint_matches and count_matches and not force:
        result["status"] = "ok"
        result["action"] = "fingerprint and count match; no change needed"
        return result

    if force and fingerprint_matches and count_matches:
        result["action"] = "matched but force=True; rewriting timestamp"
    elif not fingerprint_matches and not count_matches:
        result["action"] = (
            f"fingerprint AND count mismatch (expected fp={expected_fp}, "
            f"got {actual_fp}; expected count={expected_count}, got {n_texts})"
        )
    elif not fingerprint_matches:
        result["action"] = (
            f"fingerprint mismatch (expected {expected_fp}, got {actual_fp})"
        )
    else:
        result["action"] = (
            f"count mismatch (expected {expected_count}, got {n_texts})"
        )

    result["status"] = "rebuilt"
    if not dry_run:
        _write_manifest(manifest_p, n_texts, actual_fp,
                        has_hippocampus=data.get("hippocampus") is not None,
                        preserve_extra=existing)
    return result


def _write_manifest(manifest_path: str, n_texts: int, fingerprint: str,
                    has_hippocampus: bool = False, preserve_extra: dict = None):
    """Write a manifest file. Optionally preserves any extra fields from the
    existing manifest (so we don't lose user-added metadata)."""
    base = {
        "version": "mcp-v4",
        "count": n_texts,
        "has_hippocampus": has_hippocampus,
        "fingerprint": fingerprint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rebuilt_by": "scripts/rebuild_manifests.py",
    }
    # If there are non-standard fields in the existing manifest, preserve them
    if preserve_extra:
        for k, v in preserve_extra.items():
            if k not in base and not k.startswith("_"):
                base[k] = v
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild stale cart manifests so verify_integrity works."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=os.path.join(_MEMBOT_DIR, "cartridges"),
        help="Directory to scan for cart files (default: membot/cartridges/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing anything",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rewrite manifests even if fingerprints already match",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"FATAL: not a directory: {args.directory}")
        sys.exit(1)

    print(f"Cart hygiene scan: {args.directory}")
    if args.dry_run:
        print("(DRY RUN — no files will be written)")
    if args.force:
        print("(FORCE — will rewrite even if matching)")
    print()

    cart_files = find_cart_files(args.directory)
    if not cart_files:
        print(f"No cart files found in {args.directory}")
        return

    print(f"Found {len(cart_files)} cart files")
    print()

    counts = {"ok": 0, "rebuilt": 0, "created": 0, "error": 0, "skipped": 0}
    rebuilt_details = []

    for cart_path in cart_files:
        rel = os.path.relpath(cart_path, args.directory)
        result = rebuild_one(cart_path, dry_run=args.dry_run, force=args.force)
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1

        if status == "ok":
            print(f"  ✓ {rel}")
        elif status == "rebuilt":
            print(f"  ✎ {rel}")
            print(f"      {result['action']}")
            rebuilt_details.append(result)
        elif status == "created":
            print(f"  + {rel}  (created new manifest)")
        elif status == "error":
            print(f"  ✗ {rel}  ERROR: {result['action']}")
        else:
            print(f"  ? {rel}  status={status} action={result['action']}")

    print()
    print("=" * 60)
    print(f"Summary: {len(cart_files)} carts scanned")
    print(f"  ok:      {counts['ok']}")
    print(f"  rebuilt: {counts['rebuilt']}")
    print(f"  created: {counts['created']}")
    print(f"  error:   {counts['error']}")
    if args.dry_run:
        print()
        print("DRY RUN — no changes written. Re-run without --dry-run to apply.")
    elif counts["rebuilt"] > 0 or counts["created"] > 0:
        print()
        print("Done. Carts should now mount with verify_integrity=True without warnings.")


if __name__ == "__main__":
    main()
