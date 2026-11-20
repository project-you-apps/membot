"""Reading a cart's surface must not run the cart's code. Proven, not asserted.

WHY THIS FILE IS SHAPED THIS WAY
--------------------------------
On 2026-11-19 we found three docstrings in this repository asserting that NPZ was safe when
it was not -- one of them ten days after the same sentence had been corrected elsewhere.
The lesson was not "write better comments." It was that a safety property stated in prose
is worth nothing.

So `read_surface_unopened` does not get a comment claiming it is safe. It gets a cart with
a LIVE PICKLE PAYLOAD and a canary file. If the canary appears, code ran.

The payload is deliberately harmless: it writes a marker file. ⇒ That is enough to prove
execution, and the test also proves the THREAT is real by showing the same cart DOES fire
the canary through the ordinary loader.

Run: python tests/test_surface_does_not_execute.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

import cart_io  # noqa: E402

_passed = 0
_failed = 0


def ok(label, cond, note=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {label}" + (f"   [{note}]" if note else ""))
    else:
        _failed += 1
        print(f"  FAIL  {label}" + (f"   [{note}]" if note else ""))


def section(t):
    print(f"\n{t}")


CANARY = os.path.join(tempfile.gettempdir(), "membot_surface_canary.txt")


def _touch(path):
    """The payload. Harmless on purpose -- it only has to be OBSERVABLE."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("code executed on load")
    return 0


class Canary:
    """An object whose unpickling calls _touch. This is what allow_pickle=True permits."""

    def __reduce__(self):
        return (_touch, (CANARY,))


def build_cart(path):
    """A cart that looks ordinary and carries a live payload in an object array."""
    np.savez_compressed(
        path,
        embeddings=np.zeros((3, 8), dtype=np.float32),
        passages=np.array(["one", "two", "three"], dtype=object),
        per_pattern_meta=np.array([Canary(), {"source": "b.md"}, {"source": "c.md"}],
                                  dtype=object),
        version="mcp-v4",
    )


def clear():
    if os.path.exists(CANARY):
        os.remove(CANARY)


def main():
    d = tempfile.mkdtemp(prefix="surface-exec-")
    cart = os.path.join(d, "trap.cart.npz")
    build_cart(cart)
    ok_built = os.path.exists(cart)

    section("A - the trap cart is well formed")
    ok("cart written", ok_built, f"{os.path.getsize(cart)} bytes")

    section("B - CONTROL: the payload is LIVE, so the threat is real")
    clear()
    fired = False
    try:
        with np.load(cart, allow_pickle=True) as z:
            _ = z["per_pattern_meta"][0]      # unpickles -> executes
        fired = os.path.exists(CANARY)
    except Exception as e:                     # noqa: BLE001
        fired = os.path.exists(CANARY)
        print(f"        (loader raised after the fact: {type(e).__name__})")
    ok("CONTROL: allow_pickle=True FIRES the canary",
       fired,
       "if this fails the payload is inert and section C proves nothing")

    section("C - reading the surface does NOT execute anything")
    clear()
    surf = cart_io.read_surface_unopened(cart)
    ok("surface was read", surf["readable"] is True)
    ok("THE CANARY DID NOT FIRE",
       not os.path.exists(CANARY),
       "no unpickling, no code execution")
    ok("...and it says so: opened is False", surf["opened"] is False)

    section("D - it still learned the useful things")
    ok("passage count read from the header", surf["passages"] == 3, str(surf["passages"]))
    ok("semantic search detected", surf["semantic_search"] is True)
    ok("provenance seen as present and aligned",
       surf["provenance"]["state"] == "aligned", str(surf["provenance"]))
    ok("record count read without reading records",
       surf["provenance"]["records"] == 3)

    section("E - and it is HONEST about what it cannot know")
    ok("populated is None, not an empty dict",
       surf["provenance"]["populated"] is None,
       "{} would read as 'checked, found nothing'")
    ok("the object arrays are NAMED as unread",
       "per_pattern_meta" in surf["object_arrays"] and "passages" in surf["object_arrays"],
       str(surf["object_arrays"]))
    ok("every object array is flagged, not silently skipped",
       all(surf["members"][k]["dtype"] in ("object", "|O") or "O" in surf["members"][k]["dtype"]
           for k in surf["object_arrays"]),
       str({k: surf["members"][k]["dtype"] for k in surf["object_arrays"]}))

    section("F - a corrupt file reports unreadable rather than raising")
    bad = os.path.join(d, "bad.cart.npz")
    with open(bad, "w", encoding="utf-8") as f:
        f.write("not a zip at all")
    s2 = cart_io.read_surface_unopened(bad)
    ok("readable is False", s2["readable"] is False, str(s2.get("error", ""))[:40])
    ok("...and it did not raise", True)

    section("G - the manifest is a CLAIM, crossed against the headers")
    import json as _json
    man = cart[:-4] + "_manifest.json"
    with open(man, "w", encoding="utf-8") as f:
        _json.dump({"version": "mcp-v4", "count": 3, "has_hippocampus": False,
                    "has_per_pattern_meta": True,
                    "timestamp": "2026-11-19T20:00:00Z"}, f)
    s3 = cart_io.read_surface_unopened(cart)
    ok("manifest read with no pickling", s3["manifest"]["present"] is True)
    ok("the timestamp is named written_at, NOT built_at",
       "written_at" in s3["manifest"] and "built_at" not in s3["manifest"],
       "save_cartridge regenerates it; built_at would lie for us")
    ok("an agreeing manifest is consistent", s3["manifest"]["consistent"] is True)

    with open(man, "w", encoding="utf-8") as f:
        _json.dump({"version": "mcp-v4", "count": 116, "has_hippocampus": True,
                    "has_per_pattern_meta": True,
                    "timestamp": "2026-11-19T20:00:00Z"}, f)
    s4 = cart_io.read_surface_unopened(cart)
    ok("a LYING manifest is caught without opening the cart",
       s4["manifest"]["consistent"] is False, str(s4["manifest"].get("disagreements")))
    ok("...and each disagreement is NAMED, not collapsed to a bool",
       len(s4["manifest"]["disagreements"]) == 2,
       "wrong count AND a hippocampus it does not have")
    ok("CONTROL: the canary still did not fire during any of that",
       not os.path.exists(CANARY))

    clear()
    print("\n" + "=" * 70)
    print(f"VERDICT: {_passed}/{_passed + _failed} checks passed")
    print("=" * 70)
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
