"""Controls for the per-cart unverified-mount bypass. 2026-11-13.

WHY IT EXISTS
-------------
`--allow-unverified-mount` is GLOBAL. Switching it on to accommodate one stale cart
removes the gate from every cart on the host, and the log reads the same either way.
The droplet's split carts (`arxiv_2400k_index.npz` and friends) have manifests that
predate sha256-full-v2 and are not worth rebuilding yet, so they need an exception --
but an exception, not an amnesty.

⇒ `--allow-unverified-cart NAME` excuses one named cart. Everything else stays
  fail-closed.

BOTH STATES ARE CHECKED THROUGHOUT. A bypass test that only proves "the named cart is
allowed" cannot distinguish a working allowlist from a flag that allows everything --
which is the exact failure the per-cart form exists to prevent.

Run:  python tests/test_percart_bypass.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cart_io import unused_bypass_names, unverified_mount_allowed  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def main():
    print("Per-cart unverified-mount bypass")
    print("=" * 70)

    # ---- the default, which is the control for everything below ----------
    print("\nNO FLAGS -- the gate holds")
    ok, why = unverified_mount_allowed("heartbeat", False, None)
    check("a failing cart is refused when nothing excuses it", ok is False)
    check("and no reason is offered", why == "")

    # ---- the per-cart form -----------------------------------------------
    print("\n--allow-unverified-cart NAME")
    ok, why = unverified_mount_allowed("arxiv_2400k_index", False,
                                       ["arxiv_2400k_index"])
    check("the named cart is excused", ok is True)
    check("and the reason names the flag AND the cart", "allow-unverified-cart" in why
          and "arxiv_2400k_index" in why, why)

    # ⭐⭐ THE CHECK THE WHOLE FEATURE IS FOR. Without this one, an allowlist and a
    #    global amnesty are indistinguishable.
    ok, _ = unverified_mount_allowed("heartbeat", False, ["arxiv_2400k_index"])
    check("A DIFFERENT cart is STILL REFUSED while one is excused", ok is False)

    ok, _ = unverified_mount_allowed("gutenberg-poetry", False,
                                     ["arxiv_2400k_index", "gutenberg-poetry"])
    check("several names may be excused at once", ok is True)

    # ---- matching is exact, deliberately ---------------------------------
    print("\nMATCHING IS EXACT -- a fuzzy rule here would skip integrity checks")
    ok, _ = unverified_mount_allowed("heartbeat", False, ["Heartbeat"])
    check("case does NOT match ('Heartbeat' does not excuse 'heartbeat')", ok is False)
    ok, _ = unverified_mount_allowed("heartbeat", False, ["heart"])
    check("a SUBSTRING does not excuse ('heart' does not excuse 'heartbeat')",
          ok is False, "this is b6eac68's lesson on the path that skips a check")
    ok, _ = unverified_mount_allowed("heart", False, ["heartbeat"])
    check("nor the other way round", ok is False)

    # ---- the global form still works, and says that it is global ---------
    print("\n--allow-unverified-mount -- kept, but it must announce itself")
    ok, why = unverified_mount_allowed("anything-at-all", True, None)
    check("the global flag still excuses any cart", ok is True)
    check("and the reason says GLOBAL, so a log reader can tell the two apart",
          "GLOBAL" in why, why)
    ok, why = unverified_mount_allowed("arxiv_2400k_index", True,
                                       ["arxiv_2400k_index"])
    check("with both, the per-cart reason is reported (more specific wins)",
          ok is True and "allow-unverified-cart" in why, why)

    # ---- a name that matches nothing must be audible ---------------------
    print("\nA BYPASS NAME THAT MATCHES NOTHING")
    have = ["heartbeat", "gutenberg-poetry", "arxiv_2400k_index"]
    check("no names given -> nothing to report", unused_bypass_names(have, None) == [])
    check("every name present -> nothing to report",
          unused_bypass_names(have, ["heartbeat"]) == [])
    check("a typo IS reported, because it fails closed and looks like a broken flag",
          unused_bypass_names(have, ["heartbaet"]) == ["heartbaet"])
    check("reported alongside the ones that did match",
          unused_bypass_names(have, ["heartbeat", "nope"]) == ["nope"])
    check("duplicates are collapsed",
          unused_bypass_names(have, ["nope", "nope"]) == ["nope"])
    check("an empty cart list reports every name rather than none",
          unused_bypass_names([], ["heartbeat"]) == ["heartbeat"])

    print("\n" + "=" * 70)
    if FAIL:
        print(f"VERDICT: {len(FAIL)} FAILED of {len(PASS) + len(FAIL)}")
        for f in FAIL:
            print(f"  - {f}")
        return 1
    print(f"VERDICT: all {len(PASS)} controls pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
