"""Enforcement check: no call to a texts-requiring function may omit `texts=`.

WHY THIS IS A TEST AND NOT A GREP I RAN ONCE
---------------------------------------------
On 2026-11-13, upgrading every cart manifest to sha256-full-v2 crashed the server
boot path. `verify_manifest` had `texts=None` as a default, so three call sites
that omitted it compiled fine -- and `verify_manifest_detail` returns
LEGACY_MANIFEST *before* it needs `texts`, so the gap stayed invisible for exactly
as long as every manifest was stale. The fix exposed the bug.

Then, in one morning, the same shape recurred four times:

  1. a guard wrapped `compute_fingerprint` and the sweep died in
     `compute_fingerprint_legacy` one cart later -- a guard on one of two sites
  2. `_hash_embeddings` was fixed for zero-shapes and `compute_fingerprint_legacy`
     had the identical hole, found one cart later
  3. a patch script searched for `verify_manifest(` and not `save_manifest(`, so
     it fixed one function's call sites and left the other's
  4. ⛔ the scanner written to catch #3 used a `(?<![\w.])` lookbehind, which
     excluded `M.save_manifest(...)` -- the exact call that had crashed. The
     detector's own exclusion rule hid the thing it was built to find.

=> A grep is a one-time answer to a recurring question. This file asks it every
   run, and deliberately does NOT exclude method-style calls, because that
   exclusion is what failed.

Run:  python tests/test_no_bare_texts_calls.py
"""

import pathlib
import re
import sys

# Functions that require `texts=`. A call that omits it either raises TypeError or
# silently reproduces the v1 weakness, and both are failures.
FUNCS = ("save_manifest", "compute_fingerprint", "verify_manifest",
         "verify_manifest_detail")

# `compute_fingerprint_legacy` deliberately takes no texts -- it REPRODUCES v1 in
# order to recognize v1 manifests. It is matched by name and excluded by name, not
# by a pattern that might also drop something else.
EXEMPT_NAMES = ("compute_fingerprint_legacy",)

# A frozen snapshot of the June droplet build. Not live, not ours to edit.
SKIP_DIRS = ("droplet-snapshot-2026-06-10", "__pycache__", ".git")

# How many lines after the opening may carry `texts=` for a multi-line call.
# ⛔ Was 6, and `membot_server.py` has a save_manifest call whose `texts=` sits 11
#   lines down past nine permission kwargs. The scanner reported a correct call as
#   bare. ⇒ The first version of this scanner was too NARROW (a lookbehind that
#   dropped method calls) and the second was too BROAD (this window, plus name
#   collisions). Both were reported as a census.
WINDOW = 20

# ⭐ AUDITABLE EXEMPTIONS. Each is (file, snippet, reason). A pattern-based
#   exclusion is what hid `M.save_manifest(...)` in the first place, so exceptions
#   are named one at a time with a reason a reader can check.
# ⛔ AND A STALE EXEMPTION IS A FAILURE, not a free pass: if a snippet no longer
#   appears, this test fails. Otherwise an exemption written for one call quietly
#   licenses the next one that happens to look like it.
EXEMPTIONS = [
    ("subcart_store.py", "_sc.save_manifest(manifest, path)",
     "a DIFFERENT save_manifest: subcart's, which takes (manifest_dict, path). "
     "A dict-first call cannot be membot_server's (cart_path, embeddings, n)."),
    ("subcart_store.py", '_sc.save_manifest(manifest, self.path_for(manifest["id"]))',
     "same subcart function, same reason."),
    ("tests/test_subcart.py", "subcart.save_manifest(m, path)",
     "subcart's save_manifest again, called from its own test."),
    ("tests/test_subcart.py", 'subcart.save_manifest(m, tmp_path / "bad.json")',
     "subcart's save_manifest again, called from its own test."),
    ("tests/test_fingerprint_coverage.py", "c.compute_fingerprint(emb, len(txt))",
     "DELIBERATELY bare: this line asserts that omitting texts raises TypeError. "
     "Passing texts= here would delete the control."),
]

ROOT = pathlib.Path(__file__).resolve().parent.parent


def is_code(line: str) -> bool:
    """Crude but sufficient: skip comment lines and obvious prose.

    The docstring reference `verify_manifest(...)` in test_boot_integrity.py is
    prose, and a scanner that reports it trains the reader to ignore output.
    """
    s = line.strip()
    if s.startswith("#"):
        return True and False  # a comment is not code
    # Prose mentions use an ellipsis argument or sit inside backticks.
    if "(...)" in s or "`" in s:
        return False
    return True


def main() -> int:
    # Name alternation with NO lookbehind: a method call `x.save_manifest(` and a
    # bare `save_manifest(` must both match. Excluding attribute access is the
    # mistake this file exists to prevent.
    pattern = re.compile(r"\b(" + "|".join(FUNCS) + r")\s*\(")
    bare, scanned, calls = [], 0, 0
    used_exemptions = set()

    for path in sorted(ROOT.rglob("*.py")):
        if any(d in path.parts for d in SKIP_DIRS):
            continue
        if path.name == pathlib.Path(__file__).name:
            continue
        scanned += 1
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            m = pattern.search(line)
            if not m:
                continue
            if any(e in line for e in EXEMPT_NAMES):
                continue
            if re.search(r"\bdef\s+" + m.group(1), line):
                continue
            if not is_code(line):
                continue
            calls += 1
            rel = path.relative_to(ROOT).as_posix()
            exempt = next((e for e in EXEMPTIONS
                           if e[0] == rel and e[1] in line), None)
            if exempt:
                used_exemptions.add((exempt[0], exempt[1]))
                continue
            window = " ".join(lines[i:i + WINDOW])
            if "texts=" not in window:
                bare.append((rel, i + 1, m.group(1), line.strip()[:76]))

    print(f"Scanned {scanned} files, {calls} call(s) to {len(FUNCS)} "
          f"texts-requiring functions.")
    print(f"Exemptions: {len(used_exemptions)} of {len(EXEMPTIONS)} matched.")
    print("=" * 70)

    # A stale exemption is a failure. It means the code moved and the licence
    # outlived the thing it was written for.
    stale = [e for e in EXEMPTIONS if (e[0], e[1]) not in used_exemptions]
    if stale:
        print(f"VERDICT: {len(stale)} STALE exemption(s) -- the snippet is gone, so")
        print("         the exemption now licenses nothing and must be removed:")
        for f, snip, why in stale:
            print(f"  {f}: {snip}")
        return 1

    if bare:
        print(f"VERDICT: {len(bare)} BARE call(s) -- each omits texts=")
        for f, n, fn, l in bare:
            print(f"  {f}:{n}  [{fn}]")
            print(f"      {l}")
        return 1

    # POSITIVE CONTROL. A scanner that finds nothing is indistinguishable from a
    # scanner that matches nothing, which is how four of these got through today.
    probe = "    M.save_manifest(CART, emb, 25)"
    m = pattern.search(probe)
    if not m:
        print("VERDICT: FAIL -- the scanner cannot match a METHOD-STYLE call")
        print(f"      probe: {probe.strip()}")
        print("      This is check #4 from the docstring. The scanner is broken,")
        print("      not the tree, and a clean result from it means nothing.")
        return 1
    probe2 = "fingerprint = compute_fingerprint(embeddings, len(texts), texts=texts)"
    if not pattern.search(probe2):
        print("VERDICT: FAIL -- the scanner cannot match a plain call either")
        return 1
    if "texts=" not in probe2:
        print("VERDICT: FAIL -- the texts= detection itself is wrong")
        return 1

    print(f"VERDICT: clean -- 0 bare calls, and the scanner demonstrably matches")
    print(f"         both `M.save_manifest(...)` and a plain call.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
