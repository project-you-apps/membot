"""Negative controls for the integrity fingerprint and manifest verdict.

2026-11-12. Gauntlet gate 2, first half.

⛔⛔ WHAT v1 ACTUALLY COVERED:

    h.update(embeddings[0].tobytes())
    h.update(embeddings[-1].tobytes())
    h.update(str(n_texts).encode())
    return h.hexdigest()[:16]

  · the first and last embedding only -- every one in between was free
  · no passage text at all -- a passage could be rewritten undetected
  · 64 bits
  · and verify_manifest returned (True, "no manifest (legacy cartridge)") when the
    manifest was ABSENT, so an unmanifested cart VERIFIED.

⭐⭐ The enforcement was already correct. On 2026-10-18 the boot path was fixed to
  act on its verdict -- Andy: "if it doesn't see if it's ok what's the point of
  checking in the first place?" ⇒ So this is a gate that faithfully enforced a test
  that tested almost nothing, which is worse than a missing gate because it reads
  as cover.

⭐ EVERY CHECK BELOW IS RUN AGAINST v2 *AND* v1. A test that passes on both cannot
  tell them apart, and this file's whole purpose is to tell them apart.

⇒ Run: python tests/test_fingerprint_coverage.py
   No pytest, no GPU, no network.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cart_io as c  # noqa: E402

R: list[tuple[bool, str, str]] = []


def check(name: str, got, want, detail: str = "") -> None:
    R.append((got == want, name, f"got {got!r} want {want!r} {detail}".strip()))


def main() -> int:
    emb = np.arange(24, dtype=np.float32).reshape(8, 3)
    txt = [f"passage {i}" for i in range(8)]
    base = c.compute_fingerprint(emb, len(txt), texts=txt)
    base_v1 = c.compute_fingerprint_legacy(emb, len(txt))

    # ---- POSITIVE CONTROL. Without it, a function returning a constant passes
    #      every "detected" check below by never matching anything.
    check("identical_input_gives_identical_digest",
          c.compute_fingerprint(emb, len(txt), texts=txt), base)

    # ---- D1a · a MIDDLE embedding
    mid = emb.copy(); mid[4, 1] = 999.0
    check("v2_detects_middle_embedding_change",
          c.compute_fingerprint(mid, len(txt), texts=txt) != base, True)
    check("v1_MISSED_middle_embedding_change",
          c.compute_fingerprint_legacy(mid, len(txt)) != base_v1, False,
          "False here is the v1 bug, reproduced on purpose")

    # ---- D1b · passage TEXT
    t2 = list(txt); t2[3] = "REWRITTEN"
    check("v2_detects_passage_text_change",
          c.compute_fingerprint(emb, len(txt), texts=t2) != base, True)
    check("v1_MISSED_passage_text_change",
          c.compute_fingerprint_legacy(emb, len(txt)) != base_v1, False,
          "v1 never hashed text at all")

    # ---- D1c · a moved passage BOUNDARY, which concatenation would hide
    check("boundary_move_is_detected",
          c.compute_fingerprint(emb, 2, texts=["ab", "c"])
          != c.compute_fingerprint(emb, 2, texts=["a", "bc"]), True,
          "length-prefixed, so ['ab','c'] != ['a','bc']")

    # ---- shape and dtype are part of identity
    check("reshape_is_detected",
          c.compute_fingerprint(emb.reshape(3, 8), len(txt), texts=txt) != base, True)
    check("dtype_change_is_detected",
          c.compute_fingerprint(emb.astype(np.float64), len(txt), texts=txt) != base,
          True)

    # ---- D3 · width
    check("v2_is_256_bits", len(base), 64)
    check("v1_was_64_bits", len(base_v1), 16)

    # ---- the API refuses to do half its job
    try:
        c.compute_fingerprint(emb, len(txt))
        R.append((False, "omitting_texts_RAISES", "it returned a weaker hash"))
    except TypeError:
        R.append((True, "omitting_texts_RAISES", "TypeError, as intended"))

    # ---- D2 and the verdicts, against files on disk
    with tempfile.TemporaryDirectory() as tmp:
        cart = os.path.join(tmp, "x.cart.npz")
        np.savez_compressed(cart, embeddings=emb)
        man = cart.rsplit(".", 1)[0] + "_manifest.json"

        # D2 · NO MANIFEST. v1 returned True here.
        v, _ = c.verify_manifest_detail(cart, emb, len(txt), texts=txt)
        check("D2_no_manifest_is_NOT_verified", v, c.NO_MANIFEST)
        ok, _ = c.verify_manifest(cart, emb, len(txt), texts=txt)
        check("D2_no_manifest_ok_is_False", ok, False,
              "v1 returned True with 'no manifest (legacy cartridge)'")

        # ⛔ REWRITTEN 2026-11-13 for v3, and the reason matters more than the edit.
        #   These checks used to alter the ARRAYS PASSED IN while leaving the file
        #   alone. v2 hashed the caller's arrays, so that worked. v3 hashes the FILE,
        #   so altering an argument now proves nothing -- the cart on disk is what is
        #   attested. ⇒ The tamper has to happen to the file, which is also the only
        #   tamper an attacker can actually perform.
        # ⚠ KNOWN LIMIT, written down rather than left implicit: because v3 re-reads
        #   the file, the bytes verified are not provably the bytes a caller already
        #   loaded into RAM. Closing that needs the digest computed during the single
        #   load. See cart_io's note; not done here.
        good = c.fingerprint_cart_file(cart)
        Path(man).write_text(json.dumps({
            "count": len(txt), "fingerprint_algo": c.FINGERPRINT_ALGO,
            "fingerprint": good}), encoding="utf-8")
        v, _ = c.verify_manifest_detail(cart, emb, len(txt), texts=txt)
        check("good_v3_manifest_VERIFIES", v, c.VERIFIED)

        # the same manifest against an altered FILE
        np.savez_compressed(cart, embeddings=mid)
        v, _ = c.verify_manifest_detail(cart, mid, len(txt), texts=txt)
        check("altered_middle_embedding_is_MISMATCH", v, c.MISMATCH)

        # and an array v2 never covered at all
        np.savez_compressed(cart, embeddings=emb,
                            hippocampus=np.full((len(txt), 64), 7, dtype=np.uint8))
        v, _ = c.verify_manifest_detail(cart, emb, len(txt), texts=txt)
        check("added_unsealed_array_is_MISMATCH", v, c.MISMATCH,
              "v2 returned VERIFIED here")
        np.savez_compressed(cart, embeddings=emb)   # restore for the v1 checks below

        # a v1 manifest must be NAMED, not silently mismatched
        Path(man).write_text(json.dumps({
            "count": len(txt), "fingerprint": base_v1}), encoding="utf-8")
        v, msg = c.verify_manifest_detail(cart, emb, len(txt), texts=txt)
        check("v1_manifest_is_LEGACY_not_MISMATCH", v, c.LEGACY_MANIFEST)
        check("legacy_message_says_content_still_matches",
              "still matches" in msg, True,
              "'rebuild me' and 'I was altered' are different emergencies")

        # ⭐⭐ THE CONTRAST THAT IS ITSELF THE FINDING. Against the SAME v1 manifest:
        #   · a MIDDLE embedding altered  -> v1 reports "still matches". It is BLIND.
        #   · the LAST embedding altered  -> v1 reports CONTENT HAS CHANGED.
        #   The first case is D1. This assertion failed on its first run because I
        #   expected v1 to notice a middle change; it cannot, and the test was right.
        v, msg = c.verify_manifest_detail(cart, mid, len(txt), texts=txt)
        check("legacy_is_BLIND_to_a_middle_change", "still matches" in msg, True,
              "a v1 manifest cannot report an altered middle embedding -- this is D1")

        ends = emb.copy(); ends[-1, 0] = -42.0          # the LAST row, which v1 hashes
        v, msg = c.verify_manifest_detail(cart, ends, len(txt), texts=txt)
        check("legacy_DOES_see_a_last_row_change",
              "CONTENT HAS CHANGED" in msg, True,
              "v1 hashed exactly the first and last rows, so it sees only these")

        # an unknown algorithm is refused, not guessed at
        Path(man).write_text(json.dumps({
            "count": len(txt), "fingerprint_algo": "sha3-future-v9",
            "fingerprint": "deadbeef" * 8}), encoding="utf-8")
        v, _ = c.verify_manifest_detail(cart, emb, len(txt), texts=txt)
        check("unknown_algo_is_refused", v, c.UNREADABLE)

        # count disagreement is its own verdict
        Path(man).write_text(json.dumps({
            "count": 999, "fingerprint_algo": c.FINGERPRINT_ALGO,
            "fingerprint": base}), encoding="utf-8")
        v, _ = c.verify_manifest_detail(cart, emb, len(txt), texts=txt)
        check("count_disagreement_is_COUNT_MISMATCH", v, c.COUNT_MISMATCH)

        # unparseable manifest
        Path(man).write_text("{not json", encoding="utf-8")
        v, _ = c.verify_manifest_detail(cart, emb, len(txt), texts=txt)
        check("unparseable_manifest_is_UNREADABLE", v, c.UNREADABLE)

    # ---- ⭐ THE INVARIANT THAT MATTERS MOST: only VERIFIED is a pass.
    for verdict in (c.MISMATCH, c.COUNT_MISMATCH, c.NO_MANIFEST,
                    c.LEGACY_MANIFEST, c.UNREADABLE):
        check(f"only_VERIFIED_passes[{verdict}]", verdict == c.VERIFIED, False)

    width = max(len(n) for _, n, _ in R)
    failed = sum(1 for ok, _, _ in R if not ok)
    for ok, name, detail in R:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print()
    if failed:
        print(f"FAIL: {failed} of {len(R)} checks failed.")
        return 1
    print(f"OK: {len(R)} checks passed, including the positive control and the two")
    print("    v1-reproduction checks that prove v2 and v1 are distinguishable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
