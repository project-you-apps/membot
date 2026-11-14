"""Does a cart carrying an array an older reader does not know about still load?

HUB asked this on 2026-11-13, and it matters more than any rebuild cost:

    "A new array in the .npz costs them nothing as long as older readers ignore keys
     they don't know. If they would refuse, that matters more than our rebuild."

⇒ If an older membot refuses a cart containing an unrecognized array, then putting BM25
  term statistics in the cart format breaks every seat that has not upgraded. That is a
  far larger cost than re-attesting, and it would be the deciding fact.

BOTH DIRECTIONS ARE CHECKED. "It loaded" is not enough on its own -- a loader that
silently returned an empty cart would also "load". So each case asserts the payload came
back INTACT, and the control is the same cart without the extra array.

Run:  python tests/test_unknown_npz_key_compat.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cart_io import (  # noqa: E402
    fingerprint_cart_file,
    load_npz_cartridge,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def main():
    print("Unknown-array compatibility")
    print("=" * 70)
    tmp = tempfile.mkdtemp(prefix="npzcompat-")
    try:
        rng = np.random.default_rng(1113)
        emb = rng.random((6, 12), dtype=np.float32)
        passages = np.array([f"passage {i}" for i in range(6)], dtype=object)

        # ---- the control: an ordinary cart ------------------------------
        plain = os.path.join(tmp, "plain.cart.npz")
        np.savez(plain, embeddings=emb, passages=passages)
        base = load_npz_cartridge(plain)
        check("control: an ordinary cart loads", base is not None)
        check("control: it has the right number of passages",
              len(base.get("texts", [])) == 6, f"got {len(base.get('texts', []))}")

        # ---- the case: the same cart plus an array nothing knows --------
        print("\nTHE SAME CART, PLUS AN ARRAY THE LOADER HAS NEVER HEARD OF")
        extra = os.path.join(tmp, "extra.cart.npz")
        np.savez(extra, embeddings=emb, passages=passages,
                 # what a BM25 index would plausibly look like
                 bm25_df=np.array([3, 1, 4, 1, 5], dtype=np.int32),
                 bm25_doc_len=np.array([11, 7, 9, 13, 8, 10], dtype=np.int32),
                 bm25_vocab=np.array(["alpha", "beta", "gamma"], dtype=object),
                 # and something deliberately meaningless
                 a_key_from_the_future=np.array([0], dtype=np.uint8))

        try:
            got = load_npz_cartridge(extra)
            loaded = True
            err = ""
        except Exception as e:                      # noqa: BLE001
            got, loaded, err = None, False, f"{type(e).__name__}: {e}"

        check("it LOADS rather than refusing", loaded, err)

        # ⛔ "It loaded" proves nothing by itself. A loader that returned an empty
        #   cart would also load. The payload has to come back intact.
        if loaded:
            check("the passages are intact", list(got.get("texts", [])) ==
                  list(base.get("texts", [])))
            check("the embeddings are intact",
                  np.array_equal(got.get("embeddings"), base.get("embeddings")))
            check("the unknown arrays are IGNORED, not surfaced as passages",
                  len(got.get("texts", [])) == 6,
                  f"{len(got.get('texts', []))} passages, expected 6")

        # ---- and the part that is SUPPOSED to change --------------------
        print("\nWHAT THE EXTRA ARRAY MUST STILL CHANGE")
        # ⛔ CORRECTED 2026-11-13, same evening. This check originally asserted the
        #   fingerprint was UNCHANGED by a new array, and said that meant a cart
        #   gaining a BM25 index would keep its attestation. It passed, and the
        #   conclusion was wrong: the digest was unchanged because v2 hashed only
        #   embeddings and passages, which is the gap v3 exists to close.
        # ⇒ Under v3 a cart that gains an array IS a different cart and must say so.
        fp_plain = fingerprint_cart_file(plain)
        fp_extra = fingerprint_cart_file(extra) if loaded else None
        check("v3 DOES change when an array is added",
              fp_extra is not None and fp_extra != fp_plain,
              "a cart that gained a BM25 index is a different cart, and re-attests")
        check("so old readers load it, and the seal still notices it",
              loaded and fp_extra != fp_plain,
              "compatibility and coverage are separate questions, both answered")

        print("\n" + "=" * 70)
        if FAIL:
            print(f"VERDICT: {len(FAIL)} FAILED of {len(PASS) + len(FAIL)}")
            for f in FAIL:
                print(f"  - {f}")
            return 1
        print(f"VERDICT: all {len(PASS)} checks pass")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
