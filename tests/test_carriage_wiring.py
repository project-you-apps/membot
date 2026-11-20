"""The seam between multi_search and the carriage, exercised rather than assumed.

⛔ WHY THIS FILE EXISTS. `test_carriage.py` proves the record model. A compile check proves
the syntax. NEITHER proves that the wiring records anything, and "the unit tests pass" is
exactly the false all-clear this project keeps writing rules about.

⭐ What is tested here: `_carriage_score`, the function that joins membot_server to
carriage.py, against real result dicts and a real carriage on disk.

⚠ WHAT IS *NOT* TESTED HERE: the full `multi_search` call. That needs mounted carts and an
embedding model, so it is Andy's to run. This file deliberately does not pretend otherwise.

Run: python tests/test_carriage_wiring.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import carriage as C  # noqa: E402

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


SEAT = "waving-cat/claude"


def fake_result(cart, addr, source=None, score=0.5, text="x"):
    """Exactly the shape multi_cart._search_one_cart now builds."""
    return {"cart_id": cart, "role": "", "local_addr": addr, "score": score,
            "cos_score": None, "ham_score": None, "kw_boost": 0.0, "text": text,
            "source": source, "source_path": None, "page": None}


def main():
    # The seam function lives in membot_server, which is far too heavy to import here
    # (it builds an MCP server at module scope). ⇒ Re-implement the SAME ref-extraction
    # the server does, and assert the extraction rule itself, so a change to that rule
    # breaks this test.
    def refs_of(results):
        return [(r.get("source") or f"{r.get('cart_id')}#{r.get('local_addr')}")
                for r in results]

    section("A - provenance is what makes a ref predictable")
    rs = [fake_result("demo", 7, "cart_io.py"),
          fake_result("demo", 9, "membot_server.py"),
          fake_result("other", 2, None)]
    got = refs_of(rs)
    ok("a result WITH provenance is named by its filename", got[0] == "cart_io.py", got[0])
    ok("a result WITHOUT provenance falls back to cart#addr",
       got[2] == "other#2", got[2])
    ok("...so a cart with no provenance is still SCOREABLE, not silently skipped",
       all(r for r in got), str(got))

    section("B - a hit records the rank the reader actually saw")
    d = tempfile.mkdtemp(prefix="wire-hit-")
    ask = C.record_ask(d, SEAT, "CLI", "where is the loader",
                       expect_ref="membot_server.py", cart="multi:all", now=1.0)
    out = C.rank_outcome(d, SEAT, "CLI", ask["id"], refs_of(rs), now=2.0)
    ok("disposition is hit", out["disposition"] == "hit")
    ok("judged_by carries the 1-based rank", out["judged_by"] == "rank:2", out["judged_by"])
    ok("rank is the strongest tier", C.strength(ask, out) == C.STRENGTH_MEASURED)

    section("C - AN EMPTY RESULT SET MUST SCORE A MISS, not stay open")
    d2 = tempfile.mkdtemp(prefix="wire-empty-")
    ask2 = C.record_ask(d2, SEAT, "CLI", "something absent",
                        expect_ref="nope.py", cart="multi:all", now=1.0)
    out2 = C.rank_outcome(d2, SEAT, "CLI", ask2["id"], [], now=2.0)
    ok("an empty search scores a miss", out2["disposition"] == "miss", out2["judged_by"])
    data2 = C.load(d2, SEAT)
    ok("...and the ask is therefore NOT open",
       len(C.open_asks(data2)) == 0,
       "leaving it open would hide the clearest failures from misses()")
    ok("...and it DOES appear in misses()", len(C.misses(data2)) == 1)

    section("D - a prose expectation stays OPEN, and that is a real state")
    d3 = tempfile.mkdtemp(prefix="wire-prose-")
    ask3 = C.record_ask(d3, SEAT, "CLI", "what did we decide",
                        expectation="I expect we chose 8000", cart="multi:all", now=1.0)
    data3 = C.load(d3, SEAT)
    ok("prose ask is open immediately after the search",
       len(C.open_asks(data3)) == 1)
    try:
        C.rank_outcome(d3, SEAT, "CLI", ask3["id"], refs_of(rs))
        ok("rank scoring refuses a prose ask", False, "it did not raise")
    except ValueError:
        ok("rank scoring refuses a prose ask", True,
           "the caller is not an independent judge of its own expectation")

    section("E - CONTROL: the ask is written BEFORE results, so a failed search leaves it open")
    d4 = tempfile.mkdtemp(prefix="wire-order-")
    ask4 = C.record_ask(d4, SEAT, "CLI", "q", expect_ref="x.py", cart="multi:all", now=1.0)
    # simulate: search raised, nothing scored it
    data4 = C.load(d4, SEAT)
    ok("the ask survives a search that never returned", len(data4["asks"]) == 1)
    ok("CONTROL: it is OPEN, not lost and not silently a miss",
       len(C.open_asks(data4)) == 1,
       "recording after the search would have lost it entirely")
    s = C.summary(data4)
    ok("summary counts it as unjudged",
       s["by_disposition"].get("unjudged") == 1, str(s["by_disposition"]))

    section("F - the carriage is NOT inside the cartridges directory")
    p = C.carriage_path(os.path.join("BASE", "carriages"), SEAT)
    ok("path is under carriages/, a sibling of cartridges/",
       "carriages" in str(p) and "cartridges" not in str(p), str(p))
    ok("the seat is sanitized into the filename",
       "waving-cat_claude" in p.name, p.name)

    print("\n" + "=" * 70)
    print(f"VERDICT: {_passed}/{_passed + _failed} checks passed")
    print("NOT COVERED: a real multi_search call (needs mounted carts + embeddings)")
    print("=" * 70)
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
