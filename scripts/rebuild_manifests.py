"""
rebuild_manifests.py — Cart hygiene utility.

Walks every .cart.npz / .cart.pkl file in a directory and brings its
*_manifest.json up to sha256-full-v2 -- but ONLY after checking the existing
manifest under the algorithm that manifest was written with.

WHY THAT ORDER MATTERS, and it is the whole design
---------------------------------------------------
A manifest is an attestation. Rewriting one does not repair a cart; it replaces
the claim about the cart. So a script that rebuilds unconditionally will, on a
cart that has actually changed, compute a fresh hash over the changed bytes and
produce a file that says VERIFIED.
=> Verify first, then upgrade. A cart that fails its own old check is REFUSED,
   and refusing is the point.

Run from the membot directory:
    python scripts/rebuild_manifests.py --dry-run        # read-only; do this first
    python scripts/rebuild_manifests.py                  # default: cartridges/
    python scripts/rebuild_manifests.py path/to/dir      # a specific dir
    python scripts/rebuild_manifests.py --force          # rewrite even if matching
    python scripts/rebuild_manifests.py --accept-content-change   # see --help

Non-destructive by default: writes only where the old manifest verified.
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

# ⭐ Imported from cart_io directly. membot_server re-exports only a subset and NOT
#   the legacy path -- which is the half this script actually needs. The original
#   imported `verify_manifest` from membot_server and then never called it.
from cart_io import (
    load_cartridge_safe,
    compute_fingerprint,
    compute_fingerprint_legacy,
    FINGERPRINT_ALGO,
    LEGACY_ALGO,
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


def rebuild_one(cart_path: str, dry_run: bool = False, force: bool = False,
                accept_content_change: bool = False) -> dict:
    """Inspect one cart and rebuild its manifest if needed.

    THE RULE THIS FUNCTION EXISTS TO ENFORCE -- added 2026-11-13
    ------------------------------------------------------------
    Rewriting a manifest REPLACES the attestation. If the cart on disk has
    changed, a blind rebuild computes a fresh hash over the changed bytes and
    the result LOOKS verified.  => A rebuild can launder tampering.

    The original version did exactly that. It compared a v2 fingerprint against
    whatever string the manifest held, never reading `fingerprint_algo`. For a
    legacy manifest that is 16 chars against 64, so EVERY legacy cart reported
    "fingerprint mismatch" and was rewritten unverified. It imported
    `verify_manifest` and never called it -- a gate present and not wired in.

    => So the OLD manifest is now checked under the OLD algorithm first, and a
       rebuild only proceeds when that check passes.

    THE HONEST LIMIT, which no amount of care removes
    -------------------------------------------------
    trunc64-ends-v1 covered the FIRST embedding, the LAST embedding and a count.
    A cart altered anywhere in the middle still verifies under it.
    => A v1 pass is NOT evidence the content is intact; it is evidence the ENDS
       are. The upgrade is therefore a RE-ATTESTATION by whoever runs this
       script, not a derivation -- and the new manifest records it as such
       rather than silently claiming v2-grade provenance it never had.

    Returns: cart_path, status, expected_fp, actual_fp, prior_algo, action.
    status: 'ok' | 'upgraded' | 'rebuilt' | 'created' | 'refused' | 'error'
    """
    result = {
        "cart_path": cart_path,
        "status": "unknown",
        "expected_fp": None,
        "actual_fp": None,
        "prior_algo": None,
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
    # One unhashable cart must not end the sweep. Before this guard a single
    # zero-row cart raised out of main(), and the 13 carts after it were never
    # examined -- a crash in the instrument reported as silence about the rest.
    try:
        actual_fp = compute_fingerprint(embeddings, n_texts, texts=texts)
    except Exception as e:
        result["status"] = "error"
        result["action"] = f"fingerprint failed: {type(e).__name__}: {e}"
        return result
    result["actual_fp"] = actual_fp
    has_hippo = data.get("hippocampus") is not None

    manifest_p = manifest_path_for(cart_path)

    if not os.path.exists(manifest_p):
        # Nothing to verify against, so this is a FIRST attestation rather than a
        # re-attestation. The distinction is recorded, not glossed.
        result["status"] = "created"
        result["action"] = "no manifest existed; first attestation"
        if not dry_run:
            _write_manifest(manifest_p, n_texts, actual_fp,
                            has_hippocampus=has_hippo, attestation="first")
        return result

    try:
        with open(manifest_p, "r", encoding="utf-8") as f:
            existing = json.load(f)
    except Exception as e:
        result["status"] = "error"
        result["action"] = f"manifest read failed: {e}"
        return result

    expected_fp = existing.get("fingerprint", "")
    expected_count = existing.get("count", -1)
    # A v1 manifest has NO fingerprint_algo field. Its ABSENCE is the version.
    prior_algo = existing.get("fingerprint_algo", LEGACY_ALGO)
    result["expected_fp"] = expected_fp
    result["prior_algo"] = prior_algo
    count_matches = (expected_count == n_texts)

    # ---- already v2 -------------------------------------------------------
    if prior_algo == FINGERPRINT_ALGO:
        if expected_fp == actual_fp and count_matches:
            if not force:
                result["status"] = "ok"
                result["action"] = f"verified under {FINGERPRINT_ALGO}; no change needed"
                return result
            result["status"] = "rebuilt"
            result["action"] = "matched but --force given; rewriting timestamp"
        else:
            detail = []
            if expected_fp != actual_fp:
                detail.append(f"fingerprint {expected_fp[:16]}... != {actual_fp[:16]}...")
            if not count_matches:
                detail.append(f"count {expected_count} != {n_texts}")
            if not accept_content_change:
                result["status"] = "refused"
                result["action"] = (
                    f"INTEGRITY FAILURE under {FINGERPRINT_ALGO}: "
                    + "; ".join(detail)
                    + ". Refusing to rewrite, because that would bless the bytes "
                      "now on disk. If the cart was legitimately rebuilt, re-run "
                      "with --accept-content-change."
                )
                return result
            result["status"] = "rebuilt"
            result["action"] = ("content changed, --accept-content-change given: "
                                + "; ".join(detail))
        if not dry_run:
            _write_manifest(manifest_p, n_texts, actual_fp,
                            has_hippocampus=has_hippo, preserve_extra=existing,
                            attestation="re-attested")
        return result

    # ---- unknown algorithm ------------------------------------------------
    if prior_algo != LEGACY_ALGO:
        result["status"] = "refused"
        result["action"] = (
            f"unknown fingerprint_algo {prior_algo!r}; refusing to replace an "
            f"attestation this script cannot verify"
        )
        return result

    # ---- legacy v1: verify under v1 BEFORE upgrading ----------------------
    # Guarded separately from the v2 call above. The first version of this guard
    # wrapped only `compute_fingerprint`, and the very next run died HERE instead
    # -- a guard placed on one of two call sites, which is not a guard.
    try:
        legacy_fp = compute_fingerprint_legacy(embeddings, n_texts)
    except Exception as e:
        result["status"] = "error"
        result["action"] = f"legacy fingerprint failed: {type(e).__name__}: {e}"
        return result
    if legacy_fp != expected_fp or not count_matches:
        detail = []
        if legacy_fp != expected_fp:
            detail.append(f"legacy fingerprint {expected_fp} != {legacy_fp}")
        if not count_matches:
            detail.append(f"count {expected_count} != {n_texts}")
        if not accept_content_change:
            result["status"] = "refused"
            result["action"] = (
                f"INTEGRITY FAILURE under {LEGACY_ALGO}: " + "; ".join(detail)
                + ". The old manifest does not match the cart even on the ends it "
                  "covered, so upgrading it would attest to content nothing ever "
                  "vouched for. Refusing."
            )
            return result
        result["status"] = "rebuilt"
        result["action"] = ("legacy check FAILED, --accept-content-change given: "
                            + "; ".join(detail))
    else:
        result["status"] = "upgraded"
        result["action"] = (f"verified under {LEGACY_ALGO} (ends + count only); "
                            f"upgraded to {FINGERPRINT_ALGO}")

    if not dry_run:
        _write_manifest(manifest_p, n_texts, actual_fp,
                        has_hippocampus=has_hippo, preserve_extra=existing,
                        attestation="upgraded-from-" + LEGACY_ALGO,
                        prior_algo=prior_algo, prior_fingerprint=expected_fp)
    return result


def _write_manifest(manifest_path: str, n_texts: int, fingerprint: str,
                    has_hippocampus: bool = False, preserve_extra: dict = None,
                    attestation: str = "re-attested",
                    prior_algo: str = None, prior_fingerprint: str = None):
    """Write a manifest file, preserving any extra fields from the existing one.

    `attestation` says what this manifest IS, which a bare hash cannot:
      'first'                  -- nothing existed to check against
      're-attested'            -- the previous manifest verified under its own algo
      'upgraded-from-<algo>'   -- verified under a WEAKER algo, then upgraded
    => A reader a year from now can tell a hash that was checked from one that
       was merely written, which is the whole difference between the two.
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    base = {
        "version": "mcp-v4",
        "count": n_texts,
        "has_hippocampus": has_hippocampus,
        # In `base`, not merely appended, so `preserve_extra` below cannot carry
        # a STALE fingerprint_algo forward from the manifest being replaced.
        "fingerprint_algo": FINGERPRINT_ALGO,
        "fingerprint": fingerprint,
        "attestation": attestation,
        "timestamp": now,
        "rebuilt_by": "scripts/rebuild_manifests.py",
    }
    # Preserve non-standard fields from the existing manifest.
    if preserve_extra:
        for k, v in preserve_extra.items():
            if k not in base and not k.startswith("_"):
                base[k] = v
    # Set AFTER the preserve loop so a stale upgrade record cannot survive one.
    if prior_algo:
        base["upgraded_from_algo"] = prior_algo
        base["upgraded_from_fingerprint"] = prior_fingerprint
        base["upgrade_coverage_note"] = (
            f"{prior_algo} covered the first embedding, the last embedding and a "
            f"count -- nothing in between and no passage text. This manifest "
            f"re-attests the full content as of {now}; it does NOT inherit a "
            f"guarantee from {prior_algo}, because {prior_algo} never made one."
        )
    else:
        base.pop("upgraded_from_algo", None)
        base.pop("upgraded_from_fingerprint", None)
        base.pop("upgrade_coverage_note", None)
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
    parser.add_argument(
        "--accept-content-change", action="store_true",
        help="Rewrite a manifest whose cart FAILS verification under the "
             "manifest's own algorithm. Only use this when you know the cart was "
             "legitimately rebuilt: it replaces an attestation that currently "
             "says the content changed.",
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

    # ASCII only. These lines print to a cp1252 console on this host, where a
    # check mark or an em dash raises UnicodeEncodeError and takes the run with it.
    counts = {"ok": 0, "upgraded": 0, "rebuilt": 0, "created": 0,
              "refused": 0, "error": 0, "skipped": 0}
    refused_details = []

    for cart_path in cart_files:
        rel = os.path.relpath(cart_path, args.directory)
        result = rebuild_one(cart_path, dry_run=args.dry_run, force=args.force,
                             accept_content_change=args.accept_content_change)
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1

        if status == "ok":
            print(f"  [ok]       {rel}")
        elif status == "upgraded":
            print(f"  [upgrade]  {rel}")
            print(f"             {result['action']}")
        elif status == "rebuilt":
            print(f"  [rewrite]  {rel}")
            print(f"             {result['action']}")
        elif status == "created":
            print(f"  [create]   {rel}  (no manifest existed)")
        elif status == "refused":
            print(f"  [REFUSED]  {rel}")
            print(f"             {result['action']}")
            refused_details.append(result)
        elif status == "error":
            print(f"  [error]    {rel}  {result['action']}")
        else:
            print(f"  [?]        {rel}  status={status} action={result['action']}")

    # VERDICT FIRST, computed -- the one line that decides what to do next.
    print()
    print("=" * 66)
    if counts["refused"] or counts["error"]:
        print(f"VERDICT: NOT CLEAN -- {counts['refused']} refused, "
              f"{counts['error']} errored, of {len(cart_files)} carts.")
    else:
        print(f"VERDICT: CLEAN -- all {len(cart_files)} carts accounted for, "
              f"nothing refused.")
    print("=" * 66)
    print(f"  ok (already {FINGERPRINT_ALGO}): {counts['ok']}")
    print(f"  upgraded from {LEGACY_ALGO}:  {counts['upgraded']}")
    print(f"  rewritten:                      {counts['rebuilt']}")
    print(f"  created:                        {counts['created']}")
    print(f"  REFUSED:                        {counts['refused']}")
    print(f"  errors:                         {counts['error']}")

    if refused_details:
        print()
        print("REFUSED -- these carts do not match their own manifests. A rebuild")
        print("would replace the attestation with one computed over the bytes now")
        print("on disk, which is how a rebuild launders a change. Investigate each")
        print("before deciding anything:")
        for r in refused_details:
            print(f"  - {os.path.relpath(r['cart_path'], args.directory)}")

    if counts["upgraded"]:
        print()
        print(f"NOTE on the {counts['upgraded']} upgraded: {LEGACY_ALGO} covered the")
        print("first and last embedding and a count. Passing it means the ENDS")
        print("match, not that the content is intact. Each new manifest records")
        print("this as an upgrade rather than claiming a check it never had.")

    if args.dry_run:
        print()
        print("DRY RUN -- nothing was written. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
