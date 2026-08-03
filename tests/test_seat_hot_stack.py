"""Tests for per-seat Hot Stack assembly (SPEC-SUBCARTRIDGE-MANIFEST §9 step 4).

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_seat_hot_stack.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import overlay as ov          # noqa: E402
import seat_hot_stack as shs  # noqa: E402
import subcart as sc          # noqa: E402


TEXTS = ["alpha body", "beta body", "gamma body", "delta body"]


def cart_link(count=0, seen=0.0, **canon):
    return {
        "access_count": count,
        "accessed_at": seen,
        "_reserved": {
            "flags": {"agency": "reflective", "salience": "normal", "trust": "trusted"},
            "canonical": {"truth_status": "active", "important": False, "urgent": False,
                          "to_consolidate": False, **canon},
        },
    }


LINKS = [cart_link() for _ in TEXTS]


def fake_score(link, now):
    """Reads BOTH sides on purpose -- attention counters and canonical content fields.

    A scorer that only read access_count would pass even with the merge deleted, which is
    exactly how the real bug survived until it was run against live data.
    """
    canonical = link.get("_reserved", {}).get("canonical", {})
    score = float(link.get("access_count", 0))
    if canonical.get("important"):
        score *= 10
    if canonical.get("truth_status") == "duplicate":
        score *= 0.05
    return score


# ------------------------------------------------------------------ the merge

def test_scoring_link_takes_counters_from_the_seat():
    merged = shs.scoring_link(cart_link(count=99, seen=5.0),
                              ov.as_link({ov._N: 3, ov._T: 7.0}))
    assert merged["access_count"] == 3 and merged["accessed_at"] == 7.0


def test_scoring_link_takes_content_from_the_cart():
    merged = shs.scoring_link(cart_link(truth_status="duplicate", kind="thread"),
                              ov.as_link({}))
    canonical = merged["_reserved"]["canonical"]
    assert canonical["truth_status"] == "duplicate" and canonical["kind"] == "thread"


def test_scoring_link_puts_seat_marks_where_the_scorer_reads_them():
    """The original bug: marks were emitted flat, the scorer reads _reserved.canonical."""
    seat = ov.as_link({ov._K: ["important", "urgent"]})
    canonical = shs.scoring_link(cart_link(), seat)["_reserved"]["canonical"]
    assert canonical["important"] is True and canonical["urgent"] is True


def test_scoring_link_seat_marks_override_the_cart():
    seat = ov.as_link({})                       # this seat has NOT marked it important
    merged = shs.scoring_link(cart_link(important=True), seat)
    assert merged["_reserved"]["canonical"]["important"] is False


def test_scoring_link_does_not_mutate_its_inputs():
    original = cart_link(count=1)
    shs.scoring_link(original, ov.as_link({ov._N: 42, ov._K: ["important"]}))
    assert original["access_count"] == 1
    assert original["_reserved"]["canonical"]["important"] is False


def test_scoring_link_preserves_unknown_cart_fields():
    """Starts from the cart link so fields added later survive without being enumerated."""
    link = cart_link()
    link["origin_ts"] = 12345.0
    link["some_future_field"] = "keep me"
    merged = shs.scoring_link(link, ov.as_link({}))
    assert merged["origin_ts"] == 12345.0 and merged["some_future_field"] == "keep me"


def test_agency_is_only_overridden_when_the_seat_has_one():
    merged = shs.scoring_link(cart_link(), ov.as_link({}))
    assert merged["_reserved"]["flags"]["agency"] == "reflective"
    merged = shs.scoring_link(cart_link(), ov.as_link({ov._A: "blocked"}))
    assert merged["_reserved"]["flags"]["agency"] == "blocked"


# ------------------------------------------------------------------ seat stack

def _seat_with(**counts):
    o = ov.new_overlay("andy", "c.pkl")
    for body, n in counts.items():
        ov.touch(o, sc.content_key(body + " body"), count=n, now=100.0)
    return o


def test_seat_stack_ranks_by_the_seats_own_attention():
    o = _seat_with(alpha=1, beta=9)
    m = sc.new_manifest("all", "c.pkl", mode="all", authored_by="access")
    out = shs.seat_hot_stack(m, TEXTS, LINKS, o, fake_score, top_k=4)
    assert [i["index"] for i in out["items"]] == [1, 0]


def test_seat_stack_never_scores_outside_the_resolved_set():
    """Filter at the INPUT to scoring: an invisible passage must not rank, at any score."""
    o = _seat_with(alpha=1, gamma=1000)
    m = sc.new_manifest("narrow", "c.pkl", keys=[sc.content_key("alpha body")])
    out = shs.seat_hot_stack(m, TEXTS, LINKS, o, fake_score, top_k=4)
    assert [i["index"] for i in out["items"]] == [0]
    assert out["stats"]["resolvable"] == 1


def test_untouched_passages_do_not_rise():
    o = _seat_with(alpha=2)
    m = sc.new_manifest("all", "c.pkl", mode="all", authored_by="access")
    out = shs.seat_hot_stack(m, TEXTS, LINKS, o, fake_score, top_k=8)
    assert out["stats"]["scoreable"] == 1


def test_two_seats_get_different_stacks_from_one_cart():
    a = _seat_with(alpha=5)
    b = _seat_with(gamma=5)
    m = sc.new_manifest("all", "c.pkl", mode="all", authored_by="access")
    ra = shs.seat_hot_stack(m, TEXTS, LINKS, a, fake_score, top_k=4)
    rb = shs.seat_hot_stack(m, TEXTS, LINKS, b, fake_score, top_k=4)
    assert [i["index"] for i in ra["items"]] == [0]
    assert [i["index"] for i in rb["items"]] == [2]


def test_seat_stack_reports_unresolved():
    o = _seat_with(alpha=1)
    m = sc.new_manifest("c", "c.pkl", keys=[sc.content_key("alpha body"), "f" * 40])
    out = shs.seat_hot_stack(m, TEXTS, LINKS, o, fake_score, top_k=4)
    assert out["stats"]["unresolved"] == 1 and out["unresolved"] == ["f" * 40]


def test_content_properties_still_apply_to_a_seat_stack():
    """A duplicate stays suppressed even though truth_status lives on the cart side."""
    links = [cart_link(), cart_link(truth_status="duplicate"), cart_link(), cart_link()]
    o = _seat_with(alpha=5, beta=90)
    m = sc.new_manifest("all", "c.pkl", mode="all", authored_by="access")
    out = shs.seat_hot_stack(m, TEXTS, links, o, fake_score, top_k=4)
    assert [i["index"] for i in out["items"]] == [0, 1]   # 5 beats 90*0.05


# ------------------------------------------------------------------ firm stack

def test_firm_stack_counts_distinct_seats():
    a, b = _seat_with(alpha=1), _seat_with(alpha=1)
    b["seat"] = "dennis"
    out = shs.firm_hot_stack([a, b], TEXTS, LINKS, fake_score, top_k=4)
    assert out["items"][0]["seats"] == 2
    assert out["items"][0]["link"]["access_count"] == 2


def test_firm_stack_masks_to_the_viewer_when_asked():
    a = _seat_with(alpha=1, gamma=50)
    m = sc.new_manifest("narrow", "c.pkl", keys=[sc.content_key("alpha body")])
    unmasked = shs.firm_hot_stack([a], TEXTS, LINKS, fake_score, top_k=4)
    masked = shs.firm_hot_stack([a], TEXTS, LINKS, fake_score, top_k=4, viewer_manifest=m)
    assert unmasked["stats"]["visible"] == 2 and unmasked["stats"]["masked"] is False
    assert masked["stats"]["visible"] == 1 and masked["stats"]["masked"] is True


def test_firm_stack_of_no_seats_is_empty_not_an_error():
    out = shs.firm_hot_stack([], TEXTS, LINKS, fake_score, top_k=4)
    assert out["items"] == [] and out["stats"]["keys"] == 0
