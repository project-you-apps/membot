"""A carriage must refuse the shapes that would quietly turn it into a search log.

⭐ THE CONTROL IS SECTION D. `strength()` takes BOTH records because independence is a RELATION
between judge and asker, not a property of the judge. The obvious implementation -- a number per
judge type -- passes every other check in this file. Section D is the only thing that fails
against it, so without D this suite would certify the wrong design.

Run: python tests/test_carriage.py
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


def raises(label, fn, exc=Exception, note=""):
    try:
        fn()
    except exc:
        ok(label, True, note)
        return
    except Exception as e:
        ok(label, False, f"wrong exception: {type(e).__name__}: {e}")
        return
    ok(label, False, "did not raise")


def section(title):
    print(f"\n{title}")


SEAT = "waving-cat/claude"
PEER = "legion/claude"
HUMAN = "waving-cat/andy"


def main():
    d = tempfile.mkdtemp(prefix="carriage-test-")

    # ------------------------------------------------------------------
    section("A - an ask needs a query AND something falsifiable")
    raises("a bare query is refused   (it would be a search log)",
           lambda: C.record_ask(d, SEAT, "CLI", "where did we decide the port"),
           ValueError, "the patent's object, and the thing we promised not to keep")
    raises("an empty query is refused",
           lambda: C.record_ask(d, SEAT, "CLI", "", expectation="something"),
           ValueError)
    a1 = C.record_ask(d, SEAT, "CLI", "where did we decide the port",
                      expectation="I expect we decided 8000", now=1000.0)
    ok("prose expectation is accepted", a1["id"].startswith("ask-"))
    a2 = C.record_ask(d, SEAT, "CLI", "the npz loader finding",
                      expect_ref="docs/FINDING-membot-npz-loader.md", now=1001.0)
    ok("expect_ref is accepted", bool(a2["expect_ref"]))

    # ------------------------------------------------------------------
    section("B - an outcome is a LINK, never a field on the ask")
    o1 = C.record_outcome(d, SEAT, "CLI", a1["id"], "hit",
                          f"person:{HUMAN}", "result 2 said 8000", now=1002.0)
    data = C.load(d, SEAT)
    stored_ask = next(a for a in data["asks"] if a["id"] == a1["id"])
    ok("the ask carries NO disposition",
       "disposition" not in stored_ask,
       "a flag here would be 'confirmed' with nothing confirming it")
    ok("the outcome points at the ask", o1["answers"] == a1["id"])
    ok("a verdict IS stored on the outcome",
       o1["disposition"] == "hit",
       "legitimate: this record IS the confirming artifact")
    raises("an outcome for an unknown ask is refused",
           lambda: C.record_outcome(d, SEAT, "CLI", "ask-nope", "hit", f"person:{HUMAN}"),
           KeyError)
    raises("a judge with no prefix is refused",
           lambda: C.record_outcome(d, SEAT, "CLI", a2["id"], "hit", "andy"),
           ValueError)
    raises("an unknown disposition is refused",
           lambda: C.record_outcome(d, SEAT, "CLI", a2["id"], "probably", f"person:{HUMAN}"),
           ValueError)

    # ------------------------------------------------------------------
    section("C - open asks are DERIVED, and that is the headline number")
    data = C.load(d, SEAT)
    open_ids = {a["id"] for a in C.open_asks(data)}
    ok("the judged ask is not open", a1["id"] not in open_ids)
    ok("the unjudged ask IS open", a2["id"] in open_ids, f"{len(open_ids)} open")

    # ------------------------------------------------------------------
    section("D - CONTROL: strength is a RELATION, not a property of the judge")
    d2 = tempfile.mkdtemp(prefix="carriage-strength-")
    ask = C.record_ask(d2, SEAT, "CLI", "did the fix hold",
                       expectation="I expect the save path keeps all four", now=2000.0)
    same = C.record_outcome(d2, SEAT, "CLI", ask["id"], "hit",
                            f"person:{HUMAN}", "looks right to me", now=2001.0)
    peer = C.record_outcome(d2, SEAT, "CLI", ask["id"], "hit",
                            f"peer:{PEER}", "confirmed on our carts", now=2002.0)
    s_same = C.strength(ask, same)
    s_peer = C.strength(ask, peer)
    ok("OUR human confirming OUR agent scores as SELF",
       s_same == C.STRENGTH_SELF,
       f"{s_same} - same principal, however sincere")
    ok("a PEER confirming the same claim scores INDEPENDENT",
       s_peer == C.STRENGTH_INDEPENDENT, f"{s_peer}")
    ok("CONTROL: identical disposition, DIFFERENT strength",
       s_peer > s_same,
       "a per-judge-type constant would make these equal and pass everything else")

    model = C.record_outcome(d2, SEAT, "CLI", ask["id"], "hit",
                             "model:qwen3-4b@abc123", "judged locally", now=2003.0)
    ok("a model WE run is a different mechanism, not a different principal",
       C.strength(ask, model) == C.STRENGTH_SELF,
       "breaks circularity, confers no independence")
    lex = C.record_outcome(d2, SEAT, "CLI", ask["id"], "hit",
                           "lexical:overlap-v1", "3 terms matched", now=2004.0)
    ok("lexical overlap is the weakest judge", C.strength(ask, lex) == C.STRENGTH_WEAK)
    ok("no outcome at all is strength 0", C.strength(ask, None) == C.STRENGTH_NONE)

    # ------------------------------------------------------------------
    section("E - the rank tier needs no judge, and refuses prose")
    d3 = tempfile.mkdtemp(prefix="carriage-rank-")
    r_prose = C.record_ask(d3, SEAT, "CLI", "the npz thing",
                           expectation="I think it's in cart_io", now=3000.0)
    raises("rank scoring refuses a prose-only ask",
           lambda: C.rank_outcome(d3, SEAT, "CLI", r_prose["id"], ["a", "b"]),
           ValueError, "inferring a target from prose is the circularity this avoids")
    r_ref = C.record_ask(d3, SEAT, "CLI", "the npz loader finding",
                         expect_ref="cart_io.py", now=3001.0)
    hit = C.rank_outcome(d3, SEAT, "CLI", r_ref["id"],
                         ["membot_server.py", "cart_io.py", "notes.py"], now=3002.0)
    ok("a found ref scores a hit at its 1-based rank",
       hit["disposition"] == "hit" and hit["judged_by"] == "rank:2", hit["judged_by"])
    ok("rank is the STRONGEST tier - a measurement cannot be talked into agreeing",
       C.strength(r_ref, hit) == C.STRENGTH_MEASURED,
       "stronger than a frontier model, by design")
    r_ref2 = C.record_ask(d3, SEAT, "CLI", "another", expect_ref="absent.py", now=3003.0)
    miss = C.rank_outcome(d3, SEAT, "CLI", r_ref2["id"], ["a.py", "b.py"], now=3004.0)
    ok("an absent ref scores a miss", miss["disposition"] == "miss", miss["judged_by"])

    # ------------------------------------------------------------------
    section("F - misses are surfaced; there is deliberately NO hit rate")
    data3 = C.load(d3, SEAT)
    m = C.misses(data3)
    ok("the miss appears", len(m) == 1 and m[0]["ask"]["id"] == r_ref2["id"], f"{len(m)}")
    ok("no hit_rate() exists at all",
       not hasattr(C, "hit_rate"),
       "the population is self-selected; an average would measure our mood")
    s = C.summary(data3)
    ok("summary reports COUNTS, not rates",
       all(isinstance(v, (int, bool, dict)) for v in s.values()), str(s["by_disposition"]))

    # ------------------------------------------------------------------
    section("G - corroboration excludes self-judgment by default")
    data2 = C.load(d2, SEAT)
    corr = C.corroborated(data2)
    ok("the ask IS corroborated (a peer looked)", len(corr) == 1)
    ok("...and it is the PEER outcome that qualified",
       corr[0]["outcome"]["judged_by"].startswith("peer:"),
       corr[0]["outcome"]["judged_by"])
    only_self = C.corroborated(data2, min_strength=C.STRENGTH_MEASURED)
    ok("raising the bar above anything present returns nothing",
       len(only_self) == 0, "no rank outcome in this carriage")

    # ------------------------------------------------------------------
    section("H - an unreadable carriage BLOCKS appends rather than starting fresh")
    d4 = tempfile.mkdtemp(prefix="carriage-corrupt-")
    p = C.carriage_path(d4, SEAT)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not json", encoding="utf-8")
    bad = C.load(d4, SEAT)
    ok("a corrupt file reports readable=False", bad["readable"] is False, bad.get("error", "")[:40])
    ok("...and NOT an empty-but-fine carriage", bad["asks"] == [],
       "'asked nothing' and 'cannot read' must stay different facts")
    raises("appending to an unreadable carriage is refused",
           lambda: C.record_ask(d4, SEAT, "CLI", "q", expectation="e"),
           OSError, "silently starting fresh would destroy the record")

    # ------------------------------------------------------------------
    section("I - discard is the easy path, because that is the product")
    ok("the carriage exists before discard", C.carriage_path(d3, SEAT).exists())
    ok("discard removes it", C.discard(d3, SEAT) is True)
    ok("...and it is gone", not C.carriage_path(d3, SEAT).exists())
    ok("discarding nothing is not an error", C.discard(d3, "never-existed") is False)

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"VERDICT: {_passed}/{_passed + _failed} checks passed")
    print("=" * 70)
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
