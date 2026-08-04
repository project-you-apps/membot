"""Proposals — what an agent thinks should change, without letting it change anything.

The safety property under test: nothing here mutates a passage, and accepting a proposal
records a DECISION rather than applying it. Conflating those would put passage mutation back
on the agent path by the back door.

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_proposal_store.py -q -p no:cacheprovider
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import proposal_store as ps   # noqa: E402
import subcart as sc          # noqa: E402


BODY = "The on-call rotation changed to fortnightly in June; this says weekly."
KEY = sc.content_key("Redwood's on-call rotation is weekly.")


@pytest.fixture
def cart(tmp_path):
    p = tmp_path / "redwood.pkl"
    p.write_bytes(b"stub")
    return p


# ------------------------------------------------------------------ filing

def test_a_fresh_cart_has_no_proposals(cart):
    assert ps.list_proposals(cart) == []
    assert ps.load(cart)["readable"] is True


def test_propose_stores_who_what_and_why(cart):
    e = ps.propose(cart, KEY, "outdated", BODY, by="agent_qa", now=100.0)
    assert e["target_key"] == KEY and e["kind"] == "outdated"
    assert e["by"] == "agent_qa" and e["at"] == 100.0
    assert e["status"] == ps.STATUS_OPEN


def test_proposals_do_not_touch_the_cart(cart):
    ps.propose(cart, KEY, "correction", BODY, by="agent")
    assert cart.read_bytes() == b"stub"


def test_a_suggested_replacement_is_carried_but_not_applied(cart):
    e = ps.propose(cart, KEY, "correction", BODY, by="agent",
                   suggested_text="Rotation is fortnightly.")
    assert e["suggested_text"] == "Rotation is fortnightly."
    assert cart.read_bytes() == b"stub"          # still nothing changed


def test_bad_kinds_are_refused(cart):
    with pytest.raises(ValueError, match="kind must be"):
        ps.propose(cart, KEY, "delete_it", BODY)


def test_rewrite_is_not_a_valid_kind():
    """A proposal that replaces a whole passage is indistinguishable from an edit, and
    reviewing it is as much work as doing it."""
    assert "rewrite" not in ps.VALID_KINDS


def test_an_empty_body_is_refused(cart):
    with pytest.raises(ValueError, match="what is wrong"):
        ps.propose(cart, KEY, "correction", "   ")


def test_a_non_key_target_is_refused(cart):
    with pytest.raises(ValueError, match="content key"):
        ps.propose(cart, "3", "correction", BODY)


def test_bodies_are_capped(cart):
    e = ps.propose(cart, KEY, "context", "x" * 9000)
    assert len(e["body"]) == ps.MAX_BODY_CHARS


# ------------------------------------------------------------------ deciding

def test_accepting_records_a_decision_and_does_not_edit(cart):
    """THE safety property. Applying is a separate, deliberate act through the studio's
    edit verb, which records succession."""
    e = ps.propose(cart, KEY, "outdated", BODY, by="agent")
    out = ps.resolve(cart, e["id"], ps.STATUS_ACCEPTED, by="andy", note="good catch")
    assert out["status"] == "accepted" and out["resolved_by"] == "andy"
    assert cart.read_bytes() == b"stub"


def test_rejecting_keeps_the_record(cart):
    e = ps.propose(cart, KEY, "question", BODY, by="agent")
    ps.resolve(cart, e["id"], ps.STATUS_REJECTED, by="andy", note="works as intended")
    stored = ps.list_proposals(cart, status="rejected")
    assert len(stored) == 1 and stored[0]["resolution_note"] == "works as intended"


def test_a_decided_proposal_cannot_be_decided_twice(cart):
    """Two owners racing on the same proposal should collide visibly, not silently."""
    e = ps.propose(cart, KEY, "correction", BODY)
    ps.resolve(cart, e["id"], ps.STATUS_ACCEPTED, by="andy")
    with pytest.raises(ValueError, match="already"):
        ps.resolve(cart, e["id"], ps.STATUS_REJECTED, by="someone_else")


def test_only_accept_or_reject_are_valid_decisions(cart):
    e = ps.propose(cart, KEY, "correction", BODY)
    with pytest.raises(ValueError):
        ps.resolve(cart, e["id"], "maybe")


def test_deciding_an_unknown_proposal_raises(cart):
    with pytest.raises(KeyError):
        ps.resolve(cart, "prp_nope", ps.STATUS_ACCEPTED)


# ------------------------------------------------------------------ listing

def test_open_is_the_default_filter(cart):
    a = ps.propose(cart, KEY, "correction", BODY, now=1.0)
    ps.propose(cart, KEY, "context", BODY, now=2.0)
    ps.resolve(cart, a["id"], ps.STATUS_ACCEPTED)
    assert len(ps.list_proposals(cart)) == 1
    assert len(ps.list_proposals(cart, status="")) == 2


def test_listing_is_newest_first(cart):
    ps.propose(cart, KEY, "correction", "older", now=1.0)
    ps.propose(cart, KEY, "correction", "newer", now=9.0)
    assert ps.list_proposals(cart)[0]["body"] == "newer"


def test_filtering_by_target(cart):
    other = sc.content_key("something else")
    ps.propose(cart, KEY, "correction", BODY, now=1.0)
    ps.propose(cart, other, "correction", BODY, now=2.0)
    assert len(ps.list_proposals(cart, target_key=other)) == 1


# ------------------------------------------------------------------ riding along

def test_proposals_ride_back_with_search_results(cart):
    """The whole point of the shape Andy asked for: no separate channel. Whatever renders
    an agent's answer can render the open proposals on the passages it came from."""
    other = sc.content_key("second passage")
    ps.propose(cart, KEY, "outdated", BODY, by="agent_a")
    ps.propose(cart, other, "context", "missing background", by="agent_b")

    grouped = ps.for_results(cart, [KEY, other])
    assert set(grouped) == {KEY, other}
    assert grouped[KEY][0]["kind"] == "outdated"


def test_riding_along_ignores_untouched_passages(cart):
    ps.propose(cart, KEY, "outdated", BODY)
    assert ps.for_results(cart, [sc.content_key("unrelated")]) == {}


def test_riding_along_shows_only_open_ones_by_default(cart):
    e = ps.propose(cart, KEY, "outdated", BODY)
    ps.resolve(cart, e["id"], ps.STATUS_REJECTED)
    assert ps.for_results(cart, [KEY]) == {}


def test_riding_along_with_no_hits_is_empty(cart):
    ps.propose(cart, KEY, "outdated", BODY)
    assert ps.for_results(cart, []) == {}


# ------------------------------------------------------------------ corruption

def test_an_unreadable_file_is_flagged_not_treated_as_empty(cart):
    ps.sidecar_path(cart).write_text("{ not json", encoding="utf-8")
    data = ps.load(cart)
    assert data["readable"] is False and data["proposals"] == []


def test_appending_to_an_unreadable_file_is_refused(cart):
    ps.sidecar_path(cart).write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        ps.propose(cart, KEY, "correction", BODY)


def test_a_foreign_format_is_not_trusted(cart):
    ps.sidecar_path(cart).write_text(
        json.dumps({"format": "other", "proposals": [{"id": "x"}]}), encoding="utf-8")
    assert ps.load(cart)["readable"] is False
    assert ps.list_proposals(cart, status="") == []


def test_writes_are_atomic(cart, tmp_path):
    ps.propose(cart, KEY, "correction", BODY)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "redwood.pkl", "redwood.pkl.proposals.json"]


# ------------------------------------------------------------------ stats

def test_stats_summarise_the_queue(cart):
    a = ps.propose(cart, KEY, "correction", BODY, by="agent_a")
    ps.propose(cart, KEY, "outdated", BODY, by="agent_b")
    ps.resolve(cart, a["id"], ps.STATUS_ACCEPTED, by="andy")
    s = ps.stats(cart)
    assert s["total"] == 2 and s["open"] == 1
    assert s["by_kind"] == {"correction": 1, "outdated": 1}
    assert s["proposers"] == ["agent_a", "agent_b"]
