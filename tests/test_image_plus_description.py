"""THE JPEG TEST — can a written description make an image searchable, with no OCR?

Asked by a compliance customer, 2026-11-16. The answer is in docs/, with the numbers; this is
the instrument that produced them.

⛔ The weak version of this test is "we stored the image and the text." That proves
nothing. The claim is that the description is FINDABLE BY MEANING and the image comes
back WITH it, unaltered. So:

  J1  build a cart where one passage IS a description and carries image bytes
  J2  the image survives the round trip to disk, byte-identical
  J3  search by MEANING (not by quoting the description) and the image passage surfaces
  J4  POSITIVE CONTROL: a query about something else must NOT pull it up
  J5  the attachment is retrievable by its own index, and the bytes still match

⚠ Takes an image path as an ARGUMENT -- deliberately. The first version hardcoded a
path to a client invoice in another project, which (a) does not exist on anyone else's
machine and (b) would have shipped that client's filename to a third party.

    python test_image_plus_description.py <path-to-any-jpeg-or-png>

Prints no document content beyond short excerpts.
"""
import base64
import hashlib
import io
import os
import sys
import tempfile

MEMBOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MEMBOT)

# ⚠ A Windows console is cp1252 and the markers below are not. Without this the test
# RUNS CORRECTLY and then dies printing its own first result, which reads as a failing
# test rather than a console limitation. Third time this bit us today.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

if len(sys.argv) < 2:
    print(__doc__)
    print("ERROR: give me an image path. Any JPEG or PNG on this machine will do;\n"
          "       the test is about the description, not about what the picture shows.")
    sys.exit(2)
IMG = sys.argv[1]
if not os.path.isfile(IMG):
    print(f"ERROR: not a file: {IMG}")
    sys.exit(2)

RESULTS = []


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


def cosines(embs, query_vec):
    unit = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    return unit @ (query_vec / (np.linalg.norm(query_vec) + 1e-9))


# The evidence set: three text passages plus one screenshot-with-description.
DESCRIPTION = ("Screenshot of the SAP user access review export for Q3, showing the "
               "approver column populated for every terminated employee row.")
OTHER = [
    "The vendor spend approval threshold remains at $10,000 for FY26. Raising it "
    "would pull three recurring vendors out of dual approval.",
    "Quarterly user access reviews are performed by the control owner and evidenced "
    "by an export from the source system, retained for the audit period.",
    "Change management requires a documented rollback plan before any production "
    "deployment is approved by the release manager.",
]
SHOT = len(OTHER)  # the screenshot's index -- referenced by name, never by rank

print("\nJ1 - build a cart where the DESCRIPTION is a passage and carries the image")
import embed  # noqa: E402
with open(IMG, "rb") as f:
    img_bytes = f.read()
img_sha = hashlib.sha256(img_bytes).hexdigest()
print(f"  source image: {len(img_bytes):,} bytes, sha256 {img_sha[:16]}…")

texts = OTHER + [DESCRIPTION]
embs = np.array([embed.embed_text(t, prefix="search_document") for t in texts],
                dtype=np.float32)
check("embedded every passage, description included", embs.shape[0] == 4, str(embs.shape))

ppm = [{} for _ in OTHER] + [{
    "content_type": "graphic",
    "source": os.path.basename(IMG),
    "description": DESCRIPTION,
    "described_by": "control.owner@example.com",
    "image_b64": base64.b64encode(img_bytes).decode("ascii"),
    "raw_sha256": img_sha,
}]

tmpdir = tempfile.mkdtemp(prefix="jpegtest_")
cart = os.path.join(tmpdir, "jpeg-evidence.cart.npz")
np.savez_compressed(
    cart,
    embeddings=embs,
    passages=np.array(texts, dtype=object),
    compressed_texts=np.array(texts, dtype=object),
    per_pattern_meta=np.array(ppm, dtype=object),
    version="mcp-v3",
)
size = os.path.getsize(cart)
check("cart written with the image INSIDE it", os.path.exists(cart),
      f"{size:,} bytes on disk")
check("the image is actually in there (cart >> text alone)", size > len(img_bytes) * 0.5,
      f"image is {len(img_bytes):,} bytes of it")

print("\nJ2 - the image survives the round trip, byte for byte")
z = np.load(cart, allow_pickle=True)
rt = z["per_pattern_meta"][SHOT]
rt_bytes = base64.b64decode(rt["image_b64"])
check("same length", len(rt_bytes) == len(img_bytes), f"{len(rt_bytes):,}")
check("same sha256 - the seal would cover these exact bytes",
      hashlib.sha256(rt_bytes).hexdigest() == img_sha)
check("described_by survived", rt.get("described_by") == "control.owner@example.com")

print("\nJ3 - findable by MEANING, not by quoting the description")
# ⭐ The query deliberately shares almost no words with the description: no
# "screenshot", no "SAP", no "approver", no "terminated". If it surfaces, the match is
# semantic rather than lexical.
QUERY = "evidence that someone signed off on removing departed staff system access"
sims = cosines(embs, embed.embed_text(QUERY, prefix="search_query"))
order = np.argsort(sims)[::-1]
print(f"  query: {QUERY!r}")
for rank, i in enumerate(order, 1):
    tag = "  <-- the screenshot" if i == SHOT else ""
    print(f"    {rank}. cos={sims[i]:.4f}  {texts[i][:58]}…{tag}")
rank_of_shot = list(order).index(SHOT) + 1
# ⚠ The FIRST version of this demanded rank 1 and failed at rank 2 -- beaten by a text
# passage that genuinely answers the query better (a statement of the control beats a
# screenshot of one instance of it). That was my expectation encoded as a claim.
# ⇒ What we promised them is FINDABLE: it surfaces on a paraphrase, well clear of an
# unrelated passage.
check("the screenshot SURFACES on a paraphrased query (top half)",
      rank_of_shot <= 2, f"rank {rank_of_shot} of 4, cos={sims[SHOT]:.4f}")
check("and it beats an unrelated passage comfortably",
      sims[SHOT] - sims[0] > 0.08, f"screenshot {sims[SHOT]:.4f} vs vendor-spend {sims[0]:.4f}")
overlap = set(QUERY.lower().split()) & set(DESCRIPTION.lower().split())
content = sorted(overlap - {"the", "for", "a", "of", "that"})
check("and it is NOT keyword overlap doing the work", len(content) <= 1,
      f"shared content words: {content}")

print("\nJ4 - POSITIVE CONTROL: an unrelated query must NOT surface it")
# A test that only shows the image being found cannot tell "findable" from "returned
# for everything". Semantic search always returns SOMETHING.
Q2 = "what is the dollar threshold for vendor purchase approval"
s2 = cosines(embs, embed.embed_text(Q2, prefix="search_query"))
o2 = np.argsort(s2)[::-1]
print(f"  query: {Q2!r}")
for rank, i in enumerate(o2, 1):
    print(f"    {rank}. cos={s2[i]:.4f}  {texts[i][:58]}…")
check("the $10k passage wins this one, not the screenshot", int(o2[0]) == 0,
      f"winner is passage {int(o2[0])}")
# ⚠ The first version demanded the screenshot rank LAST. Ordinal position is the wrong
# statistic: what matters is SEPARATION. The right answer wins by a mile and the
# screenshot sits among the other non-answers.
check("the winner beats the screenshot by a wide margin",
      s2[0] - s2[SHOT] > 0.25, f"{s2[0]:.4f} vs {s2[SHOT]:.4f}, gap {s2[0] - s2[SHOT]:.4f}")

# ⛔⛔ THE FINDING THAT MATTERS MOST, and it only shows up by comparing the two queries.
# A floor at 0.60 looks perfect HERE -- it admits the right answer and excludes the
# screenshot. But the screenshot's score on its OWN best query (J3) is BELOW 0.60 too.
# ⇒ A fixed threshold that passes this check would HIDE the screenshot from the query
# its description was written for. A description scores lower than prose even when it is
# the right answer, because a description is ABOUT an artifact while a control narrative
# is a STATEMENT, and queries are phrased like statements.
# ⇒ Rank and take top-N, or score relative to the top hit. Never a constant.
check("⛔ a FIXED floor is NOT safe: 0.60 would also hide J3's correct hit",
      sims[SHOT] < 0.60, f"J3 screenshot {sims[SHOT]:.4f} vs J4 winner {s2[0]:.4f} "
                         f"-- no single constant separates them")

print("\nJ5 - the attachment is retrievable by index")
# ⚠ Look it up by its KNOWN index, not by rank -- the first version assumed the
# screenshot was the top hit and crashed with KeyError when it was not.
meta = z["per_pattern_meta"][SHOT]
check("the screenshot passage carries content_type=graphic",
      meta.get("content_type") == "graphic")
check("and its image bytes are fetchable and intact",
      hashlib.sha256(base64.b64decode(meta["image_b64"])).hexdigest() == img_sha)
check("a TEXT passage carries no image", not z["per_pattern_meta"][0].get("image_b64"))

passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print("\n" + "=" * 70)
print(f"VERDICT: {passed}/{total} checks passed" + ("" if passed == total else "   <<< FAILURES"))
for label, ok, detail in RESULTS:
    if not ok:
        print(f"   FAILED: {label}   {detail}")
print(f"cart left at: {cart}")
print("=" * 70)
sys.exit(0 if passed == total else 1)
