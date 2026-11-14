"""Negative control: the rebuild script must REFUSE a cart that fails its manifest.

WHY THIS FILE EXISTS
--------------------
On 2026-11-13 the first real sweep refused exactly one cart: `test-sysco.cart.npz`,
whose v1 manifest from 2026-07-06 did not match the cart even on the two end
embeddings v1 covered.

    legacy fingerprint 45806c9e1a5d0c8a != 7c8a9029f2b96277   (count 47, matching)

That cart was the only real-world integrity failure we had -- and it was 8.8 MB of
a client's actual invoice lines, built to demo cart building in July. Keeping
client data around indefinitely as a test fixture is the thing we warn customers
against, so the cart was deleted and the CONDITION reproduced here instead.

=> The value of that cart was never its content. It was the mismatch. A synthetic
   mismatch is identical for this purpose, weighs nothing, carries no client data,
   and can live in the repo where an 8.8 MB cart never could.

BOTH STATES ARE CHECKED ON PURPOSE. A test that only proves "it refuses a bad
cart" cannot tell a working gate from a gate that refuses everything.

Run:  python tests/test_rebuild_refuses_mismatch.py
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MEMBOT = os.path.dirname(_HERE)
sys.path.insert(0, _MEMBOT)
sys.path.insert(0, os.path.join(_MEMBOT, "scripts"))

from cart_io import FINGERPRINT_ALGO, compute_fingerprint_legacy  # noqa: E402
from rebuild_manifests import rebuild_one  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def build_cart(path: str, rows: int = 6, dim: int = 8):
    """A minimal cart the loader accepts."""
    rng = np.random.default_rng(45806)
    emb = rng.random((rows, dim), dtype=np.float32)
    passages = np.array([f"line item {i}" for i in range(rows)], dtype=object)
    np.savez(path, embeddings=emb, passages=passages)
    return emb, list(passages)


def write_v1_manifest(cart_path: str, fingerprint: str, count: int):
    mp = cart_path.rsplit(".", 1)[0] + "_manifest.json"
    # Shaped like the real test-sysco manifest: version 1, no fingerprint_algo.
    # Its ABSENCE is what marks it as v1.
    with open(mp, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "count": count, "fingerprint": fingerprint,
                   "timestamp": "2026-07-06T10:29:12Z"}, f, indent=2)
    return mp


def main():
    print("Rebuild refusal controls")
    print("=" * 68)
    tmp = tempfile.mkdtemp(prefix="cartgate-")
    try:
        # ---- STATE A: the manifest is WRONG. Must refuse, must not write. ----
        print("\nA. A cart whose manifest does not match it")
        cart_a = os.path.join(tmp, "mismatch.cart.npz")
        emb, passages = build_cart(cart_a)
        mp_a = write_v1_manifest(cart_a, "45806c9e1a5d0c8a", len(passages))
        before = open(mp_a, encoding="utf-8").read()

        r = rebuild_one(cart_a, dry_run=False)
        check("status is 'refused'", r["status"] == "refused", f"got {r['status']!r}")
        check("the reason names the integrity failure",
              "INTEGRITY FAILURE" in (r["action"] or ""))
        check("the manifest on disk is UNCHANGED",
              open(mp_a, encoding="utf-8").read() == before)
        check("it did NOT quietly become sha256-full-v2",
              "fingerprint_algo" not in json.load(open(mp_a, encoding="utf-8")))

        # ---- STATE B: same cart, CORRECT v1 manifest. Must upgrade. ----------
        # Without this the suite cannot distinguish a gate from a brick wall.
        print("\nB. The same cart with a CORRECT v1 manifest")
        cart_b = os.path.join(tmp, "good.cart.npz")
        emb_b, passages_b = build_cart(cart_b)
        true_v1 = compute_fingerprint_legacy(emb_b, len(passages_b))
        mp_b = write_v1_manifest(cart_b, true_v1, len(passages_b))

        r2 = rebuild_one(cart_b, dry_run=False)
        check("status is 'upgraded'", r2["status"] == "upgraded", f"got {r2['status']!r}")
        m2 = json.load(open(mp_b, encoding="utf-8"))
        # ⚠ Tracks the constant, not a version string. This read "sha256-full-v2"
        #   and broke the day v3 shipped, which is a test asserting the version
        #   rather than the property.
        check(f"the new manifest is {FINGERPRINT_ALGO}",
              m2.get("fingerprint_algo") == FINGERPRINT_ALGO)
        check("it records that it was an UPGRADE, not a fresh check",
              m2.get("attestation") == "upgraded-from-trunc64-ends-v1")
        check("the superseded v1 fingerprint is preserved",
              m2.get("upgraded_from_fingerprint") == true_v1)
        check("and the note says v2 does not inherit a v1 guarantee",
              "does NOT inherit" in m2.get("upgrade_coverage_note", ""))

        # ---- STATE C: idempotence. A second pass must say ok, not refuse. ----
        print("\nC. Re-running over the upgraded cart")
        r3 = rebuild_one(cart_b, dry_run=False)
        check("second pass is 'ok', so writer and verifier agree",
              r3["status"] == "ok", f"got {r3['status']!r}")

        # ---- STATE D: a count change must be caught even if the hash isn't --
        print("\nD. A manifest whose count is wrong")
        cart_d = os.path.join(tmp, "badcount.cart.npz")
        emb_d, passages_d = build_cart(cart_d)
        write_v1_manifest(cart_d, compute_fingerprint_legacy(emb_d, 99), 99)
        r4 = rebuild_one(cart_d, dry_run=False)
        check("status is 'refused' on a count mismatch",
              r4["status"] == "refused", f"got {r4['status']!r}")

        # ---- STATE E: the override must actually override -------------------
        print("\nE. --accept-content-change on the mismatched cart")
        r5 = rebuild_one(cart_a, dry_run=False, accept_content_change=True)
        check("the explicit override rewrites it", r5["status"] == "rebuilt",
              f"got {r5['status']!r}")
        check("and only with the flag -- A above proved the default refuses", True)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 68)
    if FAIL:
        print(f"VERDICT: {len(FAIL)} FAILED of {len(PASS) + len(FAIL)}")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    print(f"VERDICT: all {len(PASS)} controls pass")


if __name__ == "__main__":
    main()
