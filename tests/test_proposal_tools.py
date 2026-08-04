"""The agent-facing proposal tools, exercised against a stubbed mount.

What these pin is the SAFETY SHAPE, not the storage: an agent can file a flag and read the
queue, and cannot accept, reject, edit or delete anything.

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_proposal_tools.py -q -p no:cacheprovider
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MEMBOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEMBOT))

import membot_server as ms     # noqa: E402
import proposal_store as ps    # noqa: E402
import subcart as sc           # noqa: E402

propose_correction = ms.propose_correction.fn
proposals = ms.proposals.fn

TEXTS = ["Redwood's on-call rotation is weekly.", "beta body", "gamma body"]


@pytest.fixture
def mounted(tmp_path, monkeypatch):
    cart = tmp_path / "redwood.pkl"
    cart.write_bytes(b"stub")
    state = ms._new_session()
    state.update({"texts": list(TEXTS), "cartridge_path": str(cart),
                  "cartridge_name": "redwood.pkl"})
    monkeypatch.setattr(ms, "_get_session", lambda sid: state)
    ms._OVERLAY_STORES.clear()
    return state, cart


# ------------------------------------------------------------------ filing

def test_an_agent_can_file_a_proposal(mounted):
    out = propose_correction(0, "outdated", "Changed to fortnightly in June.",
                             session_id="agent_qa")
    assert "Filed proposal" in out and "Nothing was changed" in out


def test_filing_does_not_touch_the_cart(mounted):
    _, cart = mounted
    propose_correction(0, "correction", "This is wrong.", session_id="agent")
    assert cart.read_bytes() == b"stub"


def test_the_proposal_targets_by_content_key_not_index(mounted):
    """An index-keyed proposal points at a stranger after the next rebuild — aimed at the
    one artifact whose entire job is to be reviewed by a human later."""
    _, cart = mounted
    propose_correction(0, "correction", "wrong", session_id="agent")
    stored = ps.list_proposals(cart)[0]
    assert stored["target_key"] == sc.content_key(TEXTS[0])


def test_the_agent_session_is_recorded_as_the_proposer(mounted):
    _, cart = mounted
    propose_correction(1, "context", "missing background", session_id="agent_professor")
    assert ps.list_proposals(cart)[0]["by"] == "agent_professor"


def test_out_of_range_is_refused(mounted):
    _, cart = mounted
    assert "Out of range" in propose_correction(99, "correction", "x", session_id="a")
    assert ps.list_proposals(cart) == []


def test_a_bad_kind_is_refused_with_the_valid_list(mounted):
    out = propose_correction(0, "delete_this", "x", session_id="a")
    assert "Refused" in out and "kind must be" in out


def test_an_empty_body_is_refused(mounted):
    assert "Refused" in propose_correction(0, "correction", "   ", session_id="a")


def test_no_cartridge_mounted(monkeypatch):
    monkeypatch.setattr(ms, "_get_session", lambda sid: {"texts": []})
    assert "No cartridge mounted" in propose_correction(0, "correction", "x")
    assert "No cartridge mounted" in proposals()


# ------------------------------------------------------------------ the safety shape

def test_the_agent_surface_has_no_way_to_decide_a_proposal():
    """Accepting or rejecting is an OWNER action in the studio. If a tool for it ever
    appears here, this test should fail and someone should have to argue for it."""
    tool_names = {name for name in dir(ms)
                  if hasattr(getattr(ms, name, None), "fn")}
    for forbidden in ("accept_proposal", "reject_proposal", "resolve_proposal",
                      "delete_passage", "edit_passage", "remove_passage"):
        assert forbidden not in tool_names, f"{forbidden} must not exist on membot"


def test_reading_the_queue_is_allowed(mounted):
    propose_correction(0, "outdated", "Changed in June.", session_id="agent_qa")
    out = proposals(session_id="anyone")
    assert "1 open proposal" in out and "idx:0" in out and "agent_qa" in out


def test_decided_proposals_leave_the_open_queue(mounted):
    _, cart = mounted
    propose_correction(0, "outdated", "Changed in June.", session_id="agent")
    entry = ps.list_proposals(cart)[0]
    ps.resolve(cart, entry["id"], ps.STATUS_ACCEPTED, by="andy")
    assert "No open proposals" in proposals(session_id="anyone")
    assert "accepted" in proposals(status="accepted", session_id="anyone")


def test_an_unreadable_queue_is_not_reported_as_empty(mounted):
    """'No proposals' and 'we cannot read them' are different facts, and only one means an
    owner has nothing waiting."""
    _, cart = mounted
    ps.sidecar_path(cart).write_text("{ broken", encoding="utf-8")
    out = proposals(session_id="anyone")
    assert "UNREADABLE" in out and "do not assume the queue is empty" in out


def test_a_proposal_on_a_passage_absent_from_this_cart_is_shown_as_such(mounted):
    _, cart = mounted
    ps.propose(cart, "f" * 40, "correction", "from another mount", by="agent")
    assert "passage not in this cart" in proposals(session_id="anyone")
