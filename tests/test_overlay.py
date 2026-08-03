"""Tests for per-seat attention overlays (SPEC-SUBCARTRIDGE-MANIFEST §9 step 3).

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_overlay.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import overlay as ov  # noqa: E402
import subcart  # noqa: E402


K1 = "a" * 40
K2 = "b" * 40


def fresh(seat="andy"):
    return ov.new_overlay(seat, "claude_sessions.pkl")


# ------------------------------------------------------------------ basics

def test_new_overlay_validates():
    assert ov.validate_overlay(fresh()) == []


def test_touch_creates_then_increments():
    o = fresh()
    ov.touch(o, K1, now=100.0)
    assert ov.get(o, K1)["access_count"] == 1
    ov.touch(o, K1, now=200.0)
    e = ov.get(o, K1)
    assert e["access_count"] == 2 and e["accessed_at"] == 200.0


def test_birth_is_first_touch_not_last():
    o = fresh()
    ov.touch(o, K1, now=100.0)
    ov.touch(o, K1, now=500.0)
    assert ov.get(o, K1)["born_at"] == 100.0


def test_untouched_key_reads_cold_not_missing():
    """A passage this seat never opened is a real state, not an error."""
    e = ov.get(fresh(), "never-seen")
    assert e["access_count"] == 0 and e["accessed_at"] == 0.0


def test_overlay_is_sparse():
    o = fresh()
    ov.touch(o, K1)
    assert len(o["entries"]) == 1


# ------------------------------------------------------------------ flags / status

def test_marks_round_trip_and_dedupe():
    o = fresh()
    ov.set_mark(o, K1, "important")
    ov.set_mark(o, K1, "important")
    ov.set_mark(o, K1, "urgent")
    e = ov.get(o, K1)
    assert e["important"] and e["urgent"] and not e["to_consolidate"]


def test_mark_off_removes_only_that_mark():
    o = fresh()
    ov.set_mark(o, K1, "important")
    ov.set_mark(o, K1, "urgent")
    ov.set_mark(o, K1, "important", on=False)
    e = ov.get(o, K1)
    assert e["urgent"] and not e["important"]


def test_marking_an_untouched_passage_births_the_row():
    o = fresh()
    ov.set_mark(o, K1, "to_consolidate")
    assert K1 in o["entries"] and ov.get(o, K1)["to_consolidate"]


def test_invalid_mark_and_agency_raise():
    o = fresh()
    with pytest.raises(ValueError):
        ov.set_mark(o, K1, "ACTIVE")
    with pytest.raises(ValueError):
        # `parked` was invented in the first draft; the real vocabulary has no such value.
        ov.set_agency(o, K1, "parked")


def test_agency_accepts_the_real_vocabulary():
    o = fresh()
    for value in ("reflective", "actionable", "done", "blocked"):
        ov.set_agency(o, K1, value)
        assert ov.get(o, K1)["agency"] == value


def test_agency_set_and_cleared():
    o = fresh()
    ov.set_agency(o, K1, "actionable")
    assert ov.get(o, K1)["agency"] == "actionable"
    ov.set_agency(o, K1, None)
    assert ov.get(o, K1)["agency"] is None


# ------------------------------------------------- link shape (scoring contract)

def test_entry_duck_types_as_a_link():
    """compute_hot_score() must consume an overlay entry with no change to scoring code."""
    o = fresh()
    ov.touch(o, K1, now=1234.0)
    link = ov.get(o, K1)
    assert set(("access_count", "accessed_at")).issubset(link)
    assert isinstance(link["access_count"], int)
    assert isinstance(link["accessed_at"], float)


def test_links_for_preserves_order_and_length():
    o = fresh()
    ov.touch(o, K2)
    got = ov.links_for(o, [K1, K2])
    assert len(got) == 2
    assert got[0]["access_count"] == 0 and got[1]["access_count"] == 1


# ------------------------------------------------------------------ isolation

def test_two_seats_do_not_share_attention():
    """The whole point: seat A reading something must not warm it for seat B."""
    a, b = fresh("andy"), fresh("dennis")
    for _ in range(40):
        ov.touch(a, K1)
    assert ov.get(a, K1)["access_count"] == 40
    assert ov.get(b, K1)["access_count"] == 0


# ------------------------------------------------------------------ firm roll-up

def test_rollup_sums_counts_and_counts_distinct_seats():
    a, b = fresh("andy"), fresh("dennis")
    ov.touch(a, K1, now=100.0, count=40)
    ov.touch(b, K1, now=300.0)
    firm = ov.rollup([a, b])
    e = firm["entries"][K1]
    assert e[ov._N] == 41
    assert e[ov._T] == 300.0          # most recent wins
    assert e["seats"] == 2


def test_rollup_distinguishes_one_reader_from_many():
    """40 reads by one person and 1 read by each of 40 are the same count, different facts."""
    solo = fresh("andy")
    ov.touch(solo, K1, count=40)
    crowd = [fresh(f"seat{i}") for i in range(40)]
    for c in crowd:
        ov.touch(c, K1)
    assert ov.rollup([solo])["entries"][K1]["seats"] == 1
    assert ov.rollup(crowd)["entries"][K1]["seats"] == 40


def test_rollup_born_is_the_earliest_across_seats():
    a, b = fresh("a"), fresh("b")
    ov.touch(a, K1, now=900.0)
    ov.touch(b, K1, now=100.0)
    assert ov.rollup([a, b])["entries"][K1][ov._B] == 100.0


def test_rollup_of_nothing_is_empty_not_an_error():
    assert ov.rollup([])["entries"] == {}


# ------------------------------------------------------------------ persistence

def test_save_load_round_trip(tmp_path):
    o = fresh()
    ov.touch(o, K1, now=5.0)
    ov.set_mark(o, K1, "important")
    ov.set_agency(o, K1, "actionable")
    path = tmp_path / "andy.overlay.json"
    ov.save_overlay(o, path)
    assert ov.load_overlay(path) == o


def test_save_leaves_no_temp_files(tmp_path):
    o = fresh()
    ov.touch(o, K1)
    ov.save_overlay(o, tmp_path / "a.json")
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]


def test_overlay_never_carries_the_salt(tmp_path):
    o = fresh()
    o["parent"]["key_salt"] = "pepper"
    assert any("key_salt must NEVER" in p for p in ov.validate_overlay(o))
    with pytest.raises(ValueError):
        ov.save_overlay(o, tmp_path / "bad.json")


def test_salt_id_is_carried_but_salt_is_not():
    o = ov.new_overlay("andy", "c.pkl", key_salt_id=subcart.salt_id("pepper"))
    assert "pepper" not in json.dumps(o)


@pytest.mark.parametrize("mutate,fragment", [
    (lambda o: o.update(format="nope"), "format"),
    (lambda o: o.update(version=99), "version"),
    (lambda o: o.update(seat=""), "seat"),
    (lambda o: o.update(entries=[]), "entries"),
    (lambda o: o["parent"].update(cart=""), "parent.cart"),
])
def test_validation_catches(mutate, fragment):
    o = fresh()
    mutate(o)
    assert any(fragment in p for p in ov.validate_overlay(o))


def test_validate_never_raises_on_garbage():
    for junk in (None, [], "overlay", 7, {}):
        assert ov.validate_overlay(junk)


# ------------------------------------------------------------------ migration

def _link(count=0, seen=0.0, agency="reflective", **canon):
    """A link shaped like the ones actually in the cart, not like the ones I assumed."""
    return {
        "access_count": count,
        "accessed_at": seen,
        "_reserved": {
            "flags": {"agency": agency, "modality": "text", "trust": "trusted",
                      "salience": "normal", "temporal": "indeterminate",
                      "spatial": "indeterminate", "origin": "self",
                      "relational": "linked"},
            "canonical": {"truth_status": "active", "important": False,
                          "urgent": False, "to_consolidate": False, **canon},
        },
    }


def test_seed_skips_passages_with_no_attention():
    texts = ["cold one", "warm one"]
    links = [_link(), _link(count=3, seen=50.0)]
    o = ov.seed_from_links("andy", "c.pkl", texts, links, subcart.content_key)
    assert len(o["entries"]) == 1
    assert ov.get(o, subcart.content_key("warm one"))["access_count"] == 3


def test_seed_treats_reflective_as_no_attention():
    """`reflective` is the resting value on 19,788 links -- counting it makes the overlay
    dense, which is the one thing it must not be."""
    o = ov.seed_from_links("andy", "c.pkl", ["body"], [_link(agency="reflective")],
                           subcart.content_key)
    assert o["entries"] == {}


def test_seed_keeps_a_judged_but_never_opened_passage():
    o = ov.seed_from_links("andy", "c.pkl", ["body"], [_link(agency="blocked")],
                           subcart.content_key)
    assert ov.get(o, subcart.content_key("body"))["agency"] == "blocked"


def test_seed_sums_duplicate_content_instead_of_overwriting():
    """44% of the real cart is multi-sourced; last-writer-wins would discard attention."""
    texts = ["SESSION: a\nsame body", "SESSION: b\nsame body"]
    links = [_link(count=3, seen=10.0), _link(count=4, seen=90.0)]
    o = ov.seed_from_links("andy", "c.pkl", texts, links, subcart.content_key)
    e = ov.get(o, subcart.content_key("same body"))
    assert e["access_count"] == 7
    assert e["accessed_at"] == 90.0


def test_seed_carries_marks_and_agency_from_their_real_locations():
    links = [_link(count=1, seen=5.0, agency="actionable", important=True)]
    o = ov.seed_from_links("andy", "c.pkl", ["body"], links, subcart.content_key)
    e = ov.get(o, subcart.content_key("body"))
    assert e["agency"] == "actionable" and e["important"] and not e["urgent"]


def test_seed_keeps_a_marked_only_passage():
    """Marked important but never opened is attention, and must survive the migration."""
    links = [_link(important=True)]
    o = ov.seed_from_links("andy", "c.pkl", ["body"], links, subcart.content_key)
    assert ov.get(o, subcart.content_key("body"))["important"]


def test_seed_ignores_content_properties():
    """truth_status / kind / duplicate_of_key describe the passage, not the seat."""
    links = [_link(count=1, seen=5.0, truth_status="duplicate", kind="thread",
                   duplicate_of_key="deadbeef")]
    o = ov.seed_from_links("andy", "c.pkl", ["body"], links, subcart.content_key)
    entry = o["entries"][subcart.content_key("body")]
    assert "truth_status" not in json.dumps(entry)
    assert "duplicate_of_key" not in json.dumps(entry)


def test_agency_precedence_is_declared_not_last_writer():
    """Two copies of one body disagreeing about agency must not resolve by write order."""
    assert ov.merge_agency("done", "actionable") == "actionable"
    assert ov.merge_agency("actionable", "done") == "actionable"
    assert ov.merge_agency("actionable", "blocked") == "blocked"
    assert ov.merge_agency("blocked", "done") == "blocked"
    assert ov.merge_agency(None, "done") == "done"
    assert ov.merge_agency("done", None) == "done"
    assert ov.merge_agency(None, None) is None


def test_seed_resolves_conflicting_agency_by_precedence_and_counts_it():
    texts = ["SESSION: a\nsame body", "SESSION: b\nsame body"]
    links = [_link(count=1, seen=1.0, agency="done"),
             _link(count=1, seen=2.0, agency="actionable")]
    o = ov.seed_from_links("andy", "c.pkl", texts, links, subcart.content_key)
    assert ov.get(o, subcart.content_key("same body"))["agency"] == "actionable"
    assert o["merge_conflicts"] == 1


def test_seed_reversed_order_gives_the_same_answer():
    """The point of a precedence rule: order of arrival stops mattering."""
    a = _link(count=1, seen=1.0, agency="done")
    b = _link(count=1, seen=2.0, agency="blocked")
    texts = ["SESSION: x\nbody", "SESSION: y\nbody"]
    forward = ov.seed_from_links("s", "c.pkl", texts, [a, b], subcart.content_key)
    reverse = ov.seed_from_links("s", "c.pkl", texts, [b, a], subcart.content_key)
    key = subcart.content_key("body")
    assert ov.get(forward, key)["agency"] == ov.get(reverse, key)["agency"] == "blocked"


def test_seed_output_is_valid_and_saveable(tmp_path):
    o = ov.seed_from_links("andy", "c.pkl", ["body"], [_link(count=2, seen=7.0)],
                           subcart.content_key)
    ov.save_overlay(o, tmp_path / "seeded.json")
    assert ov.load_overlay(tmp_path / "seeded.json")["seat"] == "andy"


# ------------------------------------------------------------------ integration

def test_overlay_scores_only_over_a_resolved_subcart():
    """Personal Hot Stacks score the seat's resolved set -- never the whole parent."""
    texts = ["alpha body", "beta body", "gamma body"]
    o = fresh()
    for t in texts:
        ov.touch(o, subcart.content_key(t), count=5)

    m = subcart.new_manifest("mine", "c.pkl",
                             keys=[subcart.content_key("alpha body"),
                                   subcart.content_key("gamma body")])
    r = subcart.resolve(m, texts)
    keys = [subcart.content_key(texts[i]) for i in r["indices"]]
    links = ov.links_for(o, keys)

    assert len(links) == 2                       # beta is not scoreable by this seat
    assert all(l["access_count"] == 5 for l in links)


def test_revoked_passage_leaves_attention_intact_but_unscoreable():
    """Revocation loses resolution, not the seat's own history."""
    texts = ["kept", "revoked"]
    o = fresh()
    ov.touch(o, subcart.content_key("revoked"), count=9)
    m = subcart.new_manifest("mine", "c.pkl",
                             keys=[subcart.content_key(t) for t in texts])
    r = subcart.resolve(m, ["kept"])
    assert r["stats"]["unresolved"] == 1
    assert ov.get(o, subcart.content_key("revoked"))["access_count"] == 9
