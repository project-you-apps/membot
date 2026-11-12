"""Negative controls for `is_trusted_directory`.

2026-11-12. Gauntlet step 2, the path-prefix repair.

⛔ THE BUG THIS LOCKS SHUT. The check was:

    if real_path.startswith(real_dir + os.sep) or real_path.startswith(real_dir):

The first clause is correct; the SECOND defeated it. A sibling directory sharing a
prefix was classified as trusted, so `/carts-evil/x.pkl` passed while `/carts` was
the trusted directory. ⇒ And `is_trusted_directory` is the gate the `.pkl` loader
relies on, so this was the gate on the one load path that was actually gated.

⭐⭐ WHY THESE ARE NEGATIVE CONTROLS AND NOT ASSERTIONS. A gate that returns True
for everything passes any test that only checks the happy path. Each case below
names what must be REFUSED, and `test_trusted_child_is_accepted` is the positive
control without which the refusals pass vacuously — a function that returns False
unconditionally would satisfy every other test in this file.

⇒ Run: python tests/test_is_trusted_directory.py
   (no pytest dependency on purpose; this must be runnable anywhere the loader is)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cart_io  # noqa: E402


RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, got, want, detail: str = "") -> None:
    ok = got == want
    RESULTS.append((ok, name, f"got {got!r} want {want!r} {detail}".strip()))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = os.path.realpath(tmp)
        trusted = os.path.join(tmp, "carts")
        sibling = os.path.join(tmp, "carts-evil")      # ⛔ THE BUG: shares the prefix
        nested = os.path.join(trusted, "sub")
        outside = os.path.join(tmp, "elsewhere")
        for d in (trusted, sibling, nested, outside):
            os.makedirs(d, exist_ok=True)

        # Point the module at our temp tree. Restored in `finally`.
        original = cart_io.CARTRIDGE_DIRS
        cart_io.CARTRIDGE_DIRS = [trusted]
        try:
            # ---- POSITIVE CONTROL. Without this the rest pass vacuously. ----
            good = os.path.join(trusted, "ok.pkl")
            Path(good).write_bytes(b"x")
            check("trusted_child_is_accepted",
                  cart_io.is_trusted_directory(good), True)

            nested_file = os.path.join(nested, "ok.pkl")
            Path(nested_file).write_bytes(b"x")
            check("nested_child_is_accepted",
                  cart_io.is_trusted_directory(nested_file), True,
                  "a subdirectory of a trusted dir is still inside it")

            # ---- THE REGRESSION. This is the whole point of the file. ----
            evil = os.path.join(sibling, "x.pkl")
            Path(evil).write_bytes(b"x")
            check("SIBLING_SHARING_PREFIX_IS_REFUSED",
                  cart_io.is_trusted_directory(evil), False,
                  "carts-evil must NOT pass because it starts with carts")

            # ---- Plain outside, the easy case. ----
            out = os.path.join(outside, "x.pkl")
            Path(out).write_bytes(b"x")
            check("unrelated_directory_is_refused",
                  cart_io.is_trusted_directory(out), False)

            # ---- The directory itself is not a file inside itself. ----
            check("trusted_dir_itself_is_refused",
                  cart_io.is_trusted_directory(trusted), False)

            # ---- Traversal that resolves back outside. ----
            trav = os.path.join(trusted, "..", "carts-evil", "x.pkl")
            check("dotdot_traversal_is_refused",
                  cart_io.is_trusted_directory(trav), False,
                  "realpath resolves it to the sibling")

            # ---- Symlink escape. realpath judges the TARGET, deliberately. ----
            link = os.path.join(trusted, "escape.pkl")
            try:
                os.symlink(out, link)
            except (OSError, NotImplementedError, AttributeError) as e:
                RESULTS.append((True, "symlink_escape_is_refused",
                                f"SKIPPED: cannot create symlinks here ({type(e).__name__})"))
            else:
                check("symlink_escape_is_refused",
                      cart_io.is_trusted_directory(link), False,
                      "a link inside a trusted dir pointing out must not launder trust")
        finally:
            cart_io.CARTRIDGE_DIRS = original

    width = max(len(n) for _, n, _ in RESULTS)
    failed = 0
    for ok, name, detail in RESULTS:
        if "SKIPPED" in detail:
            print(f"  SKIP  {name:<{width}}  {detail}")
            continue
        if not ok:
            failed += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")

    print()
    if failed:
        print(f"FAIL: {failed} of {len(RESULTS)} checks failed.")
        print("  [!] If SIBLING_SHARING_PREFIX_IS_REFUSED failed, the second")
        print("      startswith clause is back and the .pkl gate is open.")
        return 1
    print(f"OK: {len(RESULTS)} checks passed, including the positive control.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
