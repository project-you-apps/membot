"""The changes sidecar — an append-only log of what replaced what.

Exists because the product that actually edits passages (the studio) has no links in its
cart format, so `superseded_by_key` had nowhere to live. A sidecar needs no format change.

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_succession_store.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subcart as sc              # noqa: E402
import succession_store as ss     # noqa: E402


V1 = "the frustration policy is a free parameter"
V2 = "the frustration RESOLUTION policy is a free parameter"
V3 = "frustration resolution policy: one we never varied"


@pytest.fixture
def cart(tmp_path):
    p = tmp_path / "brain.pkl"
    p.write_bytes(b"stub")
    return p


# ------------------------------------------------------------------ empty state

def test_a_cart_with_no_edits_reads_as_empty_and_readable(cart):
    log = ss.load_log(cart)
    assert log["entries"] == [] and log["readable"] is True
    assert ss.build_map(cart) == {}


def test_sidecar_sits_beside_the_cart(tmp_path):
    assert ss.sidecar_path(tmp_path / "brain.pkl").name == "brain.pkl.succession.json"


# ------------------------------------------------------------------ recording

def test_record_maps_old_to_new(cart):
    entry = ss.record(cart, V1, V2, seat="andy", now=100.0)
    assert entry["old_key"] == sc.content_key(V1)
    assert entry["new_key"] == sc.content_key(V2)
    assert ss.build_map(cart) == {sc.content_key(V1): sc.content_key(V2)}


def test_record_keeps_who_and_when_and_why(cart):
    e = ss.record(cart, V1, V2, seat="user_andy", note="typo fix", now=500.0)
    assert e["by"] == "user_andy" and e["at"] == 500.0 and e["note"] == "typo fix"


def test_record_stores_a_preview_not_the_whole_old_text(cart):
    """A full copy would double a heavily-edited cart. The preview is for recognition,
    not recovery — recovery is what a materialised fork is for."""
    long_old = "x" * 5000
    e = ss.record(cart, long_old, V2)
    assert len(e["old_preview"]) == 240


def test_recording_an_identical_edit_is_refused(cart):
    with pytest.raises(ValueError, match="identical"):
        ss.record(cart, V1, V1)


def test_the_log_is_append_only(cart):
    ss.record(cart, V1, V2, now=1.0)
    ss.record(cart, V2, V3, now=2.0)
    assert len(ss.load_log(cart)["entries"]) == 2


def test_writes_are_atomic_leaving_no_temp_files(cart, tmp_path):
    ss.record(cart, V1, V2)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "brain.pkl", "brain.pkl.succession.json"]


# ------------------------------------------------------------------ chains

def test_a_two_edit_chain_maps_each_hop(cart):
    ss.record(cart, V1, V2, now=1.0)
    ss.record(cart, V2, V3, now=2.0)
    mapping = ss.build_map(cart)
    assert mapping[sc.content_key(V1)] == sc.content_key(V2)
    assert mapping[sc.content_key(V2)] == sc.content_key(V3)


def test_the_map_drives_resolve_end_to_end(cart):
    """The whole point: a manifest holding the ORIGINAL key lands on the newest version."""
    ss.record(cart, V1, V2, now=1.0)
    ss.record(cart, V2, V3, now=2.0)
    texts = ["unrelated", V3]
    m = sc.new_manifest("cluster", "brain.pkl", keys=[sc.content_key(V1)], created_at=1.0)
    r = sc.resolve(m, texts, succession=ss.build_map(cart))
    assert r["indices"] == [1] and r["stats"]["followed"] == 1


def test_history_for_walks_the_chain_oldest_first(cart):
    ss.record(cart, V1, V2, seat="a", now=1.0)
    ss.record(cart, V2, V3, seat="b", now=2.0)
    chain = ss.history_for(cart, sc.content_key(V1))
    assert [e["by"] for e in chain] == ["a", "b"]


def test_history_survives_a_revert_cycle(cart):
    """A -> B -> A is real the first time someone undoes an edit."""
    ss.record(cart, V1, V2, now=1.0)
    ss.record(cart, V2, V1, now=2.0)
    chain = ss.history_for(cart, sc.content_key(V1))
    assert len(chain) <= 2                     # terminates rather than looping


def test_history_of_an_unedited_passage_is_empty(cart):
    ss.record(cart, V1, V2)
    assert ss.history_for(cart, sc.content_key("something else")) == []


# ------------------------------------------------------------------ corruption

def test_an_unreadable_log_is_flagged_not_treated_as_empty(cart):
    """'No edits ever happened' and 'we cannot read what happened' are different facts,
    and only one of them is safe to act on."""
    ss.sidecar_path(cart).write_text("{ not json", encoding="utf-8")
    log = ss.load_log(cart)
    assert log["readable"] is False and log["entries"] == []
    assert log["error"]


def test_appending_to_an_unreadable_log_is_refused(cart):
    """Otherwise the append silently discards whatever history was in there."""
    ss.sidecar_path(cart).write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        ss.record(cart, V1, V2)


def test_a_foreign_format_is_not_trusted(cart):
    ss.sidecar_path(cart).write_text(
        json.dumps({"format": "something-else", "entries": [{"old_key": "a", "new_key": "b"}]}),
        encoding="utf-8")
    assert ss.load_log(cart)["readable"] is False
    assert ss.build_map(cart) == {}


# ------------------------------------------------------------------ stats

def test_stats_summarise_the_history(cart):
    ss.record(cart, V1, V2, seat="andy", now=10.0)
    ss.record(cart, V2, V3, seat="claude", now=20.0)
    s = ss.stats(cart)
    assert s["edits"] == 2 and s["editors"] == ["andy", "claude"]
    assert s["first_at"] == 10.0 and s["last_at"] == 20.0


def test_stats_on_an_empty_cart(cart):
    assert ss.stats(cart)["edits"] == 0
