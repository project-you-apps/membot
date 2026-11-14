"""Verification must attest the bytes that were LOADED, not a second read.

THE GAP THIS CLOSES
-------------------
v3 originally verified by calling `fingerprint_cart_file(path)`, which re-reads the
file. But the mount path has already read it. Two reads of one path means the bytes
attested are not provably the bytes being served.

The dangerous direction is specific, and it is the one tested below:

    1. The server loads cart A into memory.
    2. Someone replaces the file with cart B and writes B's manifest.
    3. Verification re-reads the file, sees B, compares against B's manifest,
       and reports VERIFIED.
    4. The server is serving A.

=> A validly sealed file, a clean verdict, and the wrong content in memory.

THE FIX: the loader computes the digest from the read it just did and carries it on
the dict it returns. Verification uses that. The caller cannot supply a different
read, because the parameter has no default and the loader is the only thing that
fills it.

BOTH DIRECTIONS ARE CHECKED, and the file-based function is run alongside as the
differential: showing the new behaviour refuses is only half a result if the old
behaviour would have refused too.

Run:  python tests/test_toctou_load_time_digest.py
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cart_io import (  # noqa: E402
    FINGERPRINT_ALGO,
    LOADED_FP_KEY,
    VERIFIED,
    MISMATCH,
    UNREADABLE,
    fingerprint_cart_file,
    load_npz_cartridge,
    verify_manifest_detail,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def write_cart(path, seed, rows=6):
    rng = np.random.default_rng(seed)
    np.savez(path,
             embeddings=rng.random((rows, 8), dtype=np.float32),
             passages=np.array([f"cart{seed} passage {i}" for i in range(rows)],
                               dtype=object))


def write_manifest(cart_path, fp, count):
    mp = cart_path.rsplit(".", 1)[0] + "_manifest.json"
    with open(mp, "w", encoding="utf-8") as f:
        json.dump({"version": "mcp-v4", "count": count,
                   "fingerprint_algo": FINGERPRINT_ALGO, "fingerprint": fp}, f)
    return mp


def main():
    print("Load-time digest (TOCTOU)")
    print("=" * 70)
    tmp = tempfile.mkdtemp(prefix="toctou-")
    try:
        cart = os.path.join(tmp, "x.cart.npz")

        # ---- the control: nothing swapped --------------------------------
        print("\nCONTROL -- nothing is swapped")
        write_cart(cart, seed=1)
        a = load_npz_cartridge(cart)
        write_manifest(cart, a[LOADED_FP_KEY], len(a["texts"]))
        v, _ = verify_manifest_detail(cart, a["embeddings"], len(a["texts"]),
                                      texts=a["texts"],
                                      loaded_fingerprint=a[LOADED_FP_KEY])
        check("an untouched cart VERIFIES", v == VERIFIED, v)
        check("the loader did produce a digest", bool(a.get(LOADED_FP_KEY)))

        # ---- THE ATTACK: serve A, present a validly sealed B -------------
        print("\nTHE ATTACK -- memory holds A, the file on disk becomes a sealed B")
        loaded_a = load_npz_cartridge(cart)          # memory now holds A
        digest_a = loaded_a[LOADED_FP_KEY]

        write_cart(cart, seed=2)                     # file is now B
        b_on_disk = fingerprint_cart_file(cart)
        write_manifest(cart, b_on_disk, len(loaded_a["texts"]))   # B is validly sealed

        check("the file really did change", digest_a != b_on_disk)

        # New behaviour: verify what was LOADED.
        v_new, msg_new = verify_manifest_detail(
            cart, loaded_a["embeddings"], len(loaded_a["texts"]),
            texts=loaded_a["texts"], loaded_fingerprint=digest_a)
        check("verification REFUSES: the served payload is not the sealed file",
              v_new == MISMATCH, v_new)

        # Old behaviour, run for the differential.
        old_style_fp = fingerprint_cart_file(cart)
        v_old, _ = verify_manifest_detail(
            cart, loaded_a["embeddings"], len(loaded_a["texts"]),
            texts=loaded_a["texts"], loaded_fingerprint=old_style_fp)
        check("and re-reading the file would have said VERIFIED",
              v_old == VERIFIED,
              "this is the gap: a clean verdict while serving other content")

        # ---- a missing digest must refuse, not fall back -----------------
        print("\nA MISSING DIGEST MUST REFUSE")
        v, msg = verify_manifest_detail(cart, loaded_a["embeddings"],
                                        len(loaded_a["texts"]),
                                        texts=loaded_a["texts"],
                                        loaded_fingerprint=None)
        check("no digest -> UNREADABLE, not a silent re-read", v == UNREADABLE, v)
        check("and the message says why", "load time" in msg or "loader" in msg)

        # ---- the parameter cannot be omitted -----------------------------
        print("\nTHE PARAMETER CANNOT BE OMITTED")
        try:
            verify_manifest_detail(cart, loaded_a["embeddings"],
                                   len(loaded_a["texts"]), texts=loaded_a["texts"])
            check("omitting loaded_fingerprint raises TypeError", False,
                  "it was accepted, so a caller can still fall back silently")
        except TypeError:
            check("omitting loaded_fingerprint raises TypeError", True,
                  "no default, same discipline as texts=")

        print("\n" + "=" * 70)
        if FAIL:
            print(f"VERDICT: {len(FAIL)} FAILED of {len(PASS) + len(FAIL)}")
            for f in FAIL:
                print(f"  - {f}")
            return 1
        print(f"VERDICT: all {len(PASS)} controls pass")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
