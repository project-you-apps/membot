"""v3 must seal every array the loader reads. v2 sealed two of them.

THE BUG, MEASURED 2026-11-13
----------------------------
On a real cart, with embeddings and passages left identical and only `hippocampus`
tampered with, the v2 fingerprint came back BYTE-IDENTICAL. The seal did not cover it.

That matters more than "an array was unhashed", because of what those arrays carry:

  * `hippocampus` holds the TOMBSTONE flag, and the search path honours it:
        if flags & 0x01: continue
    ⇒ Setting one bit removes a passage from every search result while the cart still
      verifies and still matches the hash a signed package names. Evidence suppression
      that passes the seal.
  * `per_pattern_meta` holds agent_id, written_at, origin and tags.
    ⇒ ATTRIBUTION was unsealed. You could rewrite who wrote a passage and when.

EVERY CHECK HERE IS DIFFERENTIAL, ON PURPOSE
--------------------------------------------
Each tamper is run against BOTH algorithms. "v3 changed" alone would be satisfied by a
hash that changes on everything including nothing; the claim is specifically that v3
catches what v2 missed, so v2 has to be shown missing it in the same breath.

Run:  python tests/test_v3_covers_every_array.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cart_io import (  # noqa: E402
    FINGERPRINT_ALGO,
    V2_ALGO,
    compute_fingerprint,
    fingerprint_members,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def base_members():
    rng = np.random.default_rng(1113)
    return {
        "embeddings": rng.random((6, 8), dtype=np.float32),
        "passages": np.array([f"passage {i}" for i in range(6)], dtype=object),
        "hippocampus": np.zeros((6, 64), dtype=np.uint8),
        "per_pattern_meta": np.array([{"agent_id": "claude", "written_at": 1}] * 6,
                                     dtype=object),
        "sign_bits": np.zeros((6, 2), dtype=np.uint8),
        "pattern0": np.zeros((1, 4), dtype=np.uint8),
    }


def v2_of(m):
    return compute_fingerprint(m["embeddings"], len(m["passages"]),
                               texts=list(m["passages"]))


def main():
    print(f"{FINGERPRINT_ALGO} coverage, measured against {V2_ALGO}")
    print("=" * 72)

    m0 = base_members()
    v3_0, v2_0 = fingerprint_members(m0), v2_of(m0)

    print("\nSTABILITY -- the control for everything below")
    check("v3 is stable across calls", fingerprint_members(base_members()) == v3_0)
    check("v2 is stable across calls", v2_of(base_members()) == v2_0)

    print("\nARRAYS v2 ALREADY COVERED -- both must notice")
    for name, mutate in (
        ("embeddings", lambda m: m.__setitem__(
            "embeddings", m["embeddings"] + np.float32(0.5))),
        ("passages", lambda m: m.__setitem__(
            "passages", np.array(["CHANGED"] + [f"passage {i}" for i in range(1, 6)],
                                 dtype=object))),
    ):
        m = base_members(); mutate(m)
        check(f"v3 detects a changed {name}", fingerprint_members(m) != v3_0)
        check(f"v2 detects a changed {name} too", v2_of(m) != v2_0,
              "so v2 was not useless, only incomplete")

    # ASCII only: this console is cp1252 and a non-ASCII glyph in a print() takes
    # the whole run down. Second time today.
    print("\nARRAYS v2 LEFT UNSEALED -- v3 must notice, v2 must NOT")
    for name in ("hippocampus", "per_pattern_meta", "sign_bits", "pattern0"):
        m = base_members()
        arr = m[name]
        if arr.dtype == object:
            m[name] = np.array([{"agent_id": "someone-else", "written_at": 99}] * 6,
                               dtype=object)
        else:
            m[name] = np.full(arr.shape, 255, dtype=arr.dtype)
        v3_t, v2_t = fingerprint_members(m), v2_of(m)
        check(f"v3 DETECTS a tampered {name}", v3_t != v3_0)
        check(f"v2 is BLIND to it ({name})", v2_t == v2_0,
              "this is the gap v3 closes, not a v2 bug report")

    print("\nTHE TOMBSTONE CASE, specifically")
    m = base_members()
    h = np.array(m["hippocampus"], copy=True)
    h[2, 0] |= 0x01                      # tombstone the third passage
    m["hippocampus"] = h
    check("flipping ONE tombstone bit changes v3", fingerprint_members(m) != v3_0)
    check("and v2 would have called that cart verified", v2_of(m) == v2_0,
          "a passage hidden from every search, seal intact")

    print("\nSTRUCTURE -- things that must and must not matter")
    m = base_members()
    m["bm25_df"] = np.array([3, 1, 4], dtype=np.int32)
    check("ADDING an array changes v3", fingerprint_members(m) != v3_0,
          "so a cart that gains a BM25 index is a different cart, correctly")
    check("and v2 would not have noticed", v2_of(m) == v2_0)

    m = base_members()
    reordered = {k: m[k] for k in reversed(list(m.keys()))}
    check("member ORDER does not change v3", fingerprint_members(reordered) == v3_0,
          "sorted by name, so numpy's ordering cannot move the digest")

    m = base_members()
    m["hippocampus_renamed"] = m.pop("hippocampus")
    check("RENAMING a member changes v3", fingerprint_members(m) != v3_0,
          "the name is hashed with the data")

    m = base_members()
    m["a"] = np.array([1, 2], dtype=np.uint8)
    m["b"] = np.array([3], dtype=np.uint8)
    n = base_members()
    n["a"] = np.array([1], dtype=np.uint8)
    n["b"] = np.array([2, 3], dtype=np.uint8)
    check("moving bytes BETWEEN members changes v3",
          fingerprint_members(m) != fingerprint_members(n),
          "length-prefixed names stop two members cancelling out")

    m = base_members()
    m["embeddings"] = m["embeddings"].astype(np.float64)
    check("dtype alone changes v3", fingerprint_members(m) != v3_0)

    print("\n" + "=" * 72)
    if FAIL:
        print(f"VERDICT: {len(FAIL)} FAILED of {len(PASS) + len(FAIL)}")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print(f"VERDICT: all {len(PASS)} controls pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
