"""Packed sign bits: 8x smaller, and the SAME ANSWERS. 2026-10-18.

S6 from SPEC-the-memory-ceiling-and-the-composite-cart-2026-10-18.md.

None of our carts ship precomputed sign_bits, so every mount derived them with
(embeddings > 0).astype(np.uint8) -- one byte per bit, 768 B/pattern -- while both
scorers have always also read the packed (N, 96) form. Pure waste, invisible until
counted.

⛔ THE SIZE CHECK IS THE EASY HALF. A compression that changes what comes back is a
regression wearing an optimization's clothes, so §2 runs the SAME queries through
both formats and requires identical ranked ids AND identical scores. §4 covers the
append paths, which are where a format change actually breaks things: a packed
corpus growing by an unpacked row is either a ragged vstack or a silently wrong
popcount.

No GPU, no network, no model load. ~25 lines of output.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import membot_server as M          # noqa: E402
import multi_cart as MC            # noqa: E402

RESULTS = []


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


rng = np.random.default_rng(23)
N, DIM = 4000, 768
emb = rng.standard_normal((N, DIM)).astype(np.float32)

print("\n§1 · size")
unpacked = (emb > 0).astype(np.uint8)
packed = M.pack_sign_bits(emb)
check("packed shape is (N, 96)", packed.shape == (N, 96), str(packed.shape))
check("8x smaller", unpacked.nbytes == packed.nbytes * 8,
      f"{unpacked.nbytes/1024:.0f} KB -> {packed.nbytes/1024:.0f} KB")
check("per-pattern 768 B -> 96 B", unpacked.nbytes // N == 768 and packed.nbytes // N == 96)

print("\n§2 · IDENTICAL ANSWERS — the half that matters")
# multi_cart._hamming_scores is the shared scorer and reads both formats.
worst_score_delta = 0.0
rank_mismatches = 0
for t in range(12):
    q = rng.standard_normal(DIM).astype(np.float32)
    s_unpacked = MC._hamming_scores(q, unpacked)
    s_packed = MC._hamming_scores(q, packed)
    worst_score_delta = max(worst_score_delta, float(np.abs(s_unpacked - s_packed).max()))
    top_u = np.argsort(s_unpacked)[-10:][::-1]
    top_p = np.argsort(s_packed)[-10:][::-1]
    if not np.array_equal(top_u, top_p):
        rank_mismatches += 1
check("scores identical across 12 queries", worst_score_delta == 0.0,
      f"max delta {worst_score_delta:.2e}")
check("top-10 ranking identical across 12 queries", rank_mismatches == 0,
      f"{rank_mismatches} mismatches")

print("\n§3 · CONTROL — the comparison must be capable of failing")
# Flip one bit of one pattern. If §2 cannot see this, §2 proves nothing.
corrupt = packed.copy()
corrupt[0, 0] ^= 0xFF
q = emb[0]
s_good = MC._hamming_scores(q, packed)
s_bad = MC._hamming_scores(q, corrupt)
check("an 8-bit corruption DOES move the score",
      float(abs(s_good[0] - s_bad[0])) > 0,
      f"delta {abs(float(s_good[0]) - float(s_bad[0])):.6f}")

print("\n§4 · append paths must match the resident format")
row = M.sign_row_matching(packed, emb[0])
check("packed corpus -> packed row", row.shape == (1, 96), str(row.shape))
check("vstack onto packed corpus works",
      np.vstack([packed, row]).shape == (N + 1, 96))
row_u = M.sign_row_matching(unpacked, emb[0])
check("unpacked corpus -> unpacked row (legacy carts keep working)",
      row_u.shape == (1, 768), str(row_u.shape))
check("empty/None corpus -> unpacked row, not a crash",
      M.sign_row_matching(None, emb[0]).shape == (1, 768))

print("\n§5 · a legacy UNPACKED cart must still mount and score")
# Carts that ship precomputed unpacked sign_bits exist; the scorer branches on width.
check("scorer reads 768-wide corpus", MC._hamming_scores(emb[0], unpacked).shape == (N,))
check("scorer reads 96-wide corpus", MC._hamming_scores(emb[0], packed).shape == (N,))

print("\n§6 · projected saving at scale")
for n, label in ((18_040, "gutenberg-classics"), (34_000_000, "a 100 GB corpus")):
    print(f"    {label:22s} {n*768/1e9:7.2f} GB -> {n*96/1e9:6.3f} GB "
          f"(saves {n*672/1e9:.2f} GB)")

passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print("\n" + "=" * 62)
print(f"VERDICT: {passed}/{total} checks passed" + ("" if passed == total else "  <<< FAILURES"))
for label, ok, detail in RESULTS:
    if not ok:
        print(f"   FAILED: {label}   {detail}")
print("=" * 62)
sys.exit(0 if passed == total else 1)
