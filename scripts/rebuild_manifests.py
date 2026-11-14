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
    fingerprint_cart_file,
    FINGERPRINT_ALGO,
    V2_ALGO,
    LEGACY_ALGO,
)


_CART_SUFFIXES = (".cart.npz", ".cart.pkl")


def find_cart_files(directory: str) -> list[str]:
    """Find every cart file (.cart.npz or .cart.pkl) in a directory."""
    out = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if f.endswith(_CART_SUFFIXES):
                out.append(os.path.join(root, f))
    return out


def find_unswept(directory: str) -> list[str]:
    """Loadable-looking files this sweep will NOT touch, and must therefore name.

    ⛔ WHY: on 2026-11-13 `test-sysco.cart copy.npz` -- a 9.2 MB Windows duplicate
      of a client-data cart -- sat in cartridges/ and the sweep reported CLEAN.
      `.cart copy.npz` does not end in `.cart.npz`, so it was skipped in silence
      while still being a perfectly loadable npz that a path-taking mount could
      open. An `arxiv_2400k_lite copy.pkl` was there too, on the PICKLE path.

    ⇒ A hygiene tool that says "clean" about the files it happened to match is
      reporting its own glob, not the directory. This makes the gap audible.
      It does not act on them: naming them is the job, deciding is a person's.
    """
    out = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if f.endswith((".npz", ".pkl")) and not f.endswith(_CART_SUFFIXES):
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
    # ⭐ v3 is computed from the FILE, so it covers every array the loader reads
    #   rather than the two this function happens to hold.
    try:
        actual_fp = fingerprint_cart_file(cart_path)
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

    # ---- v2: verify under v2 BEFORE upgrading to v3 -----------------------
    # v2 covered embeddings and passages and nothing else. Passing it means those
    # two match; it says nothing about hippocampus, per_pattern_meta or sign_bits,
    # which the loader also reads. ⇒ Same shape as the v1 upgrade: the check that
    # can be made is made, and the manifest records exactly what it was worth.
    if prior_algo == V2_ALGO:
        try:
            v2_fp = compute_fingerprint(embeddings, n_texts, texts=texts)
        except Exception as e:
            result["status"] = "error"
            result["action"] = f"v2 fingerprint failed: {type(e).__name__}: {e}"
            return result
        if v2_fp != expected_fp or not count_matches:
            detail = []
            if v2_fp != expected_fp:
                detail.append(f"v2 fingerprint {expected_fp[:16]}... != {v2_fp[:16]}...")
            if not count_matches:
                detail.append(f"count {expected_count} != {n_texts}")
            if not accept_content_change:
                result["status"] = "refused"
                result["action"] = (
                    f"INTEGRITY FAILURE under {V2_ALGO}: " + "; ".join(detail)
                    + ". The embeddings or passages have changed since the manifest "
                      "was written, so upgrading it would attest to content nothing "
                      "vouched for. Refusing."
                )
                return result
            result["status"] = "rebuilt"
            result["action"] = ("v2 check FAILED, --accept-content-change given: "
                                + "; ".join(detail))
        else:
            result["status"] = "upgraded"
            result["action"] = (f"verified under {V2_ALGO} (embeddings and passages "
                                f"only); upgraded to {FINGERPRINT_ALGO}")
        if not dry_run:
            _write_manifest(manifest_p, n_texts, actual_fp,
                            has_hippocampus=has_hippo, preserve_extra=existing,
                            attestation="upgraded-from-" + V2_ALGO,
                            prior_algo=prior_algo, prior_fingerprint=expected_fp)
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
        # `rebuilt_by` is in `base`, so the loop below would silently drop it --
        # and it is the only record of WHICH of the six fingerprint
        # implementations wrote this manifest (heartbeat's said
        # `multi_cart.imprint_with_meta`). That is evidence about the codebase,
        # and a sweep is the one moment it can still be captured.
        prior_writer = preserve_extra.get("rebuilt_by")
        if prior_writer and prior_writer != base["rebuilt_by"]:
            base["previously_rebuilt_by"] = prior_writer
        for k, v in preserve_extra.items():
            if k not in base and not k.startswith("_"):
                base[k] = v
    # Set AFTER the preserve loop so a stale upgrade record cannot survive one.
    if prior_algo:
        base["upgraded_from_algo"] = prior_algo
        base["upgraded_from_fingerprint"] = prior_fingerprint
        # ⚠ The note must describe the algorithm actually superseded. It was written
        #   for v1 and would have stated something false about v2, which covered a
        #   great deal more than "the ends and a count".
        if prior_algo == V2_ALGO:
            covered = ("covered every embedding and every passage, and left every "
                       "other array in the cart unsealed -- hippocampus, which "
                       "carries the tombstone flag that hides a passage from search, "
                       "and per_pattern_meta, which carries attribution")
        else:
            covered = ("covered the first embedding, the last embedding and a count "
                       "-- nothing in between and no passage text")
        base["upgrade_coverage_note"] = (
            f"{prior_algo} {covered}. This manifest re-attests the full file as of "
            f"{now}; it does NOT inherit a guarantee from {prior_algo}, because "
            f"{prior_algo} never made one over what is now covered."
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
    # ⛔ The summary used to print "upgraded from <LEGACY_ALGO>" no matter which
    #   algorithm was actually superseded. With v2 manifests on disk it reported 17
    #   upgrades from v1, which was a confident statement about the wrong thing.
    #   ⇒ Count what happened instead of asserting it.
    upgraded_from: dict = {}

    for cart_path in cart_files:
        rel = os.path.relpath(cart_path, args.directory)
        result = rebuild_one(cart_path, dry_run=args.dry_run, force=args.force,
                             accept_content_change=args.accept_content_change)
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1

        if status == "ok":
            print(f"  [ok]       {rel}")
        elif status == "upgraded":
            _from = result.get("prior_algo") or "unknown"
            upgraded_from[_from] = upgraded_from.get(_from, 0) + 1
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
    print(f"  upgraded:                       {counts['upgraded']}")
    for _from, _n in sorted(upgraded_from.items()):
        print(f"      from {_from}: {_n}")
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
        print(f"NOTE on the {counts['upgraded']} upgraded. An upgrade re-attests the "
              f"whole file now;")
        print("it does not inherit a guarantee the old algorithm never made:")
        if LEGACY_ALGO in upgraded_from:
            print(f"  {LEGACY_ALGO}: covered the first and last embedding and a count.")
            print("    Passing it means the ENDS matched, not that content was intact.")
        if V2_ALGO in upgraded_from:
            print(f"  {V2_ALGO}: covered every embedding and every passage, and NO")
            print("    other array -- hippocampus (whose tombstone flag hides a passage")
            print("    from search) and per_pattern_meta (attribution) were unsealed.")

    # Named AFTER the verdict, because a verdict that ignores them is a verdict
    # about the glob rather than about the directory.
    unswept = find_unswept(args.directory)
    if unswept:
        print()
        print(f"NOT SWEPT -- {len(unswept)} loadable file(s) whose name does not end")
        print("in .cart.npz or .cart.pkl. This tool cannot verify them and did not")
        print("touch them. They are still openable by any path-taking mount:")
        for p in unswept:
            rel = os.path.relpath(p, args.directory)
            kind = "PICKLE" if p.endswith(".pkl") else "npz"
            print(f"  - [{kind}] {rel}")

    if args.dry_run:
        print()
        print("DRY RUN -- nothing was written. Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
