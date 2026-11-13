"""mount_cartridge must refuse an ambiguous name rather than pick one.

2026-11-12. Legion's check 7, from
forum/legion-to-waving-cat-branches-and-checks-2026-11-13.md:

    "Name matching is substring matching. After sanitize_name, mount_cartridge
     accepts an exact *or substring* match within the trusted dirs, so a new file
     can win a match it was never meant to. Exact match, or refusal on ambiguity,
     is what fails closed."

⛔ Confirmed. The code was:

    match = [c for c in carts if c["name"] == clean_name]
    if not match:
        match = [c for c in carts if clean_name.lower() in c["name"].lower()]
    ...
    cart = match[0]

  So `redwood` matched every cart containing "redwood", case-insensitively, and the
  first in listing order won -- which is filesystem order, not a decision.

⭐⭐ And the fix was already written four lines below, for Mempacks:
  `if len(mempack_match) > 1: return "Multiple Mempacks named ... pass owner_id"`,
  with the comment "substring matching against Mempacks risks cross-user collision."
  ⇒ The codebase already knew, and applied the knowledge to the adjacent path only.

⚠ WHY THIS IS TESTED AT THE SELECTION LEVEL rather than by calling the tool: the real
  mount_cartridge needs a server, a session, an embedder and a shared pool. The
  selection rule is the thing Legion's check is about, so it is extracted here exactly
  as the server performs it, and the test asserts on the rule. ⛔ That means this test
  can drift from the server if someone edits one and not the other -- so it also greps
  the server source to assert the guard is still present. A rule test with no link to
  the code it describes is a comment.

⇒ Run: python tests/test_mount_name_ambiguity.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = HERE.parent / "membot_server.py"

R: list[tuple[bool, str, str]] = []


def check(name: str, got, want, detail: str = "") -> None:
    R.append((got == want, name, f"got {got!r} want {want!r} {detail}".strip()))


def select(carts: list[dict], clean_name: str):
    """The selection rule as membot_server.mount_cartridge now performs it.

    Returns a cart dict, or a refusal string.
    """
    match = [c for c in carts if c["name"] == clean_name]
    if not match:
        near = [c for c in carts if clean_name.lower() in c["name"].lower()]
        if len(near) > 1:
            names = ", ".join(c["name"] for c in near[:10])
            more = f" (+{len(near) - 10} more)" if len(near) > 10 else ""
            return (f"'{clean_name}' matches {len(near)} cartridges: {names}{more}. "
                    f"Refusing to guess -- pass the exact name.")
        match = near
    if not match:
        return f"Cartridge '{clean_name}' not found."
    return match[0]


def old_select(carts: list[dict], clean_name: str):
    """The rule BEFORE the fix, so the two are provably distinguishable."""
    match = [c for c in carts if c["name"] == clean_name]
    if not match:
        match = [c for c in carts if clean_name.lower() in c["name"].lower()]
    return match[0] if match else f"Cartridge '{clean_name}' not found."


def main() -> int:
    carts = [{"name": "redwood-DD-2026"},
             {"name": "redwood-DD-2025"},
             {"name": "session-memory"},
             {"name": "SESSION-memory-archive"}]

    # ---- POSITIVE CONTROL. Without it, a function that refuses everything passes
    #      every refusal check below.
    got = select(carts, "session-memory")
    check("exact_name_still_mounts",
          isinstance(got, dict) and got["name"], "session-memory")

    got = select(carts, "redwood-DD-2025")
    check("exact_name_wins_over_substring_siblings",
          isinstance(got, dict) and got["name"], "redwood-DD-2025",
          "an exact hit must not be dragged into the ambiguity branch")

    # ---- a UNIQUE substring is still a convenience, deliberately
    got = select(carts, "archive")
    check("unique_substring_still_mounts",
          isinstance(got, dict) and got["name"], "SESSION-memory-archive")

    # ---- THE REGRESSION. Ambiguity must refuse.
    got = select(carts, "redwood")
    check("AMBIGUOUS_substring_is_REFUSED", isinstance(got, str), True,
          "two redwood carts -- refusing is the only safe answer")
    check("refusal_names_the_candidates",
          isinstance(got, str) and "redwood-DD-2026" in got and "redwood-DD-2025" in got,
          True, "an operator cannot disambiguate what they cannot see")

    # ---- case-insensitive ambiguity counts too
    got = select(carts, "session")
    check("case_insensitive_ambiguity_is_REFUSED", isinstance(got, str), True,
          "'session' hits session-memory and SESSION-memory-archive")

    # ---- the OLD rule silently picked one. This is what we changed.
    old = old_select(carts, "redwood")
    check("old_rule_SILENTLY_PICKED_one",
          isinstance(old, dict) and old["name"], "redwood-DD-2026",
          "the first in listing order, which is filesystem order, not a decision")

    got = select(carts, "nothing-like-this")
    check("absent_name_still_reports_not_found", isinstance(got, str), True)

    # ---- ⭐ THE LINK TO THE REAL CODE. Without this, the rule above is a comment.
    src = SERVER.read_text(encoding="utf-8", errors="replace")
    check("server_still_has_the_ambiguity_guard",
          "Refusing to guess -- pass the exact name." in src, True,
          "membot_server.py must still carry the refusal this test describes")
    check("server_has_no_bare_substring_fallback",
          bool(re.search(r"match = \[c for c in carts if clean_name\.lower\(\) in", src)),
          False, "the unguarded fallback must not have come back")

    width = max(len(n) for _, n, _ in R)
    failed = sum(1 for ok, _, _ in R if not ok)
    for ok, name, detail in R:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print()
    if failed:
        print(f"FAIL: {failed} of {len(R)} checks failed.")
        return 1
    print(f"OK: {len(R)} checks passed, including the positive control, the old-rule")
    print("    reproduction, and the two that tie this test to the server source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
