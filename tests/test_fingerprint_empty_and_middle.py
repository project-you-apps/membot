"""Controls for the 2026-11-13 fingerprint fix: empty carts, and middle coverage.

WHY THIS FILE EXISTS
--------------------
The first real sweep of `scripts/rebuild_manifests.py` died on
`embed-consistency-test.cart.npz`, a cart with zero embedding rows:

    TypeError: memoryview: cannot cast view with zeros in shape or strides

`memoryview().cast()` raises on ANY shape containing a zero, so an empty cart
crashed the hash rather than hashing to a stable value.

=> That is the same class as the missing length guard found in the inline v1
   copies the day before -- removed there, reintroduced here in a new form.
   Hence a permanent control rather than a one-line patch.

Run:  python tests/test_fingerprint_empty_and_middle.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cart_io import (  # noqa: E402
    FINGERPRINT_ALGO,
    LEGACY_ALGO,
    compute_fingerprint,
    compute_fingerprint_legacy,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def main():
    print(f"Fingerprint controls -- {FINGERPRINT_ALGO} vs {LEGACY_ALGO}")
    print("=" * 68)

    # ---- the empty-cart family, which used to raise -----------------------
    print("\nEMPTY AND DEGENERATE SHAPES (all of these raised before the fix)")
    empty768 = np.zeros((0, 768), dtype=np.float32)
    empty384 = np.zeros((0, 384), dtype=np.float32)

    try:
        fp_e768 = compute_fingerprint(empty768, 0, texts=[])
        check("a (0, 768) cart hashes instead of raising", True, fp_e768[:16] + "...")
    except Exception as e:
        fp_e768 = None
        check("a (0, 768) cart hashes instead of raising", False, repr(e))

    try:
        fp_e384 = compute_fingerprint(empty384, 0, texts=[])
        check("a (0, 384) cart hashes instead of raising", True)
    except Exception as e:
        fp_e384 = None
        check("a (0, 384) cart hashes instead of raising", False, repr(e))

    fp_none = compute_fingerprint(None, 0, texts=[])

    # These are the checks that make the empty case MEANINGFUL rather than
    # merely non-crashing. A single shared "empty" digest would be a check
    # identical in both states.
    check("(0, 768) and (0, 384) do NOT collide",
          fp_e768 is not None and fp_e384 is not None and fp_e768 != fp_e384)
    check("an empty array does NOT collide with None embeddings",
          fp_e768 is not None and fp_e768 != fp_none)
    check("the empty digest is stable across calls",
          fp_e768 is not None and fp_e768 == compute_fingerprint(empty768, 0, texts=[]))

    scalar = np.float32(1.5)
    try:
        compute_fingerprint(scalar, 0, texts=[])
        check("a 0-d array returns a row count instead of raising on len()", True)
    except Exception as e:
        check("a 0-d array returns a row count instead of raising on len()",
              False, repr(e))

    # BOTH algorithms, because the first fix covered v2 only and the next sweep
    # died inside the v1 function one cart later. A control on one of two
    # implementations is a control on neither.
    try:
        compute_fingerprint_legacy(scalar, 0)
        check("the LEGACY hash also survives a 0-d array", True)
    except Exception as e:
        check("the LEGACY hash also survives a 0-d array", False, repr(e))
    try:
        compute_fingerprint_legacy(empty768, 0)
        check("the LEGACY hash also survives a (0, 768) array", True)
    except Exception as e:
        check("the LEGACY hash also survives a (0, 768) array", False, repr(e))

    # ---- the regression guard: non-empty carts must be UNTOUCHED ----------
    print("\nNON-EMPTY CARTS ARE UNAFFECTED BY THE FIX")
    rng = np.random.default_rng(20261113)
    base = rng.random((8, 16), dtype=np.float32)
    texts = [f"passage {i}" for i in range(8)]
    fp_base = compute_fingerprint(base, len(texts), texts=texts)
    check("a normal cart still hashes", bool(fp_base))
    check("and is stable across calls",
          fp_base == compute_fingerprint(base.copy(), len(texts), texts=texts))
    check("a non-contiguous view hashes the same as its contiguous copy",
          compute_fingerprint(np.asfortranarray(base).T.T, len(texts),
                              texts=texts) == fp_base)

    # ---- THE control this whole rebuild exists for ------------------------
    # v1 hashed the FIRST embedding, the LAST embedding and a count. So a
    # change in the middle is invisible to it and must be visible to v2.
    # Running BOTH implementations is the point: a suite that passes on the old
    # one as well would be telling us nothing.
    print("\nMIDDLE COVERAGE -- v2 must see what v1 cannot")
    middle = base.copy()
    middle[4, 7] += np.float32(0.5)          # neither first nor last row
    fp_mid_v2 = compute_fingerprint(middle, len(texts), texts=texts)
    v1_base = compute_fingerprint_legacy(base, len(texts))
    v1_mid = compute_fingerprint_legacy(middle, len(texts))

    check("v2 DETECTS a changed middle embedding", fp_mid_v2 != fp_base)
    check("v1 is BLIND to it (this is why v1 was replaced)", v1_mid == v1_base,
          "v1 agreeing here is the expected result, not a bug")

    first = base.copy()
    first[0, 0] += np.float32(0.5)
    last = base.copy()
    last[-1, -1] += np.float32(0.5)
    check("v2 detects a changed FIRST embedding",
          compute_fingerprint(first, len(texts), texts=texts) != fp_base)
    check("v2 detects a changed LAST embedding",
          compute_fingerprint(last, len(texts), texts=texts) != fp_base)
    check("v1 also detects the ends (so v1 was not useless, only partial)",
          compute_fingerprint_legacy(first, len(texts)) != v1_base)

    # ---- passage coverage, which v1 had none of ---------------------------
    print("\nPASSAGE COVERAGE -- v1 hashed no text at all")
    moved = ["passage 0passage 1"] + texts[2:]
    check("v2 detects passage text changing",
          compute_fingerprint(base, len(texts),
                              texts=["CHANGED"] + texts[1:]) != fp_base)
    check("v2 detects a passage BOUNDARY moving without the text changing",
          compute_fingerprint(base, len(moved), texts=moved) != fp_base)

    print("\n" + "=" * 68)
    if FAIL:
        print(f"VERDICT: {len(FAIL)} FAILED of {len(PASS) + len(FAIL)}")
        for f in FAIL:
            print(f"  - {f}")
        sys.exit(1)
    print(f"VERDICT: all {len(PASS)} controls pass")


if __name__ == "__main__":
    main()
