"""Per-seat hide: the first rung of the delete ladder, and the only free one.

    hidden     this seat's view only   costs nothing, reversible, invisible to others
    tombstone  everyone, bytes kept    needs a durable tombstone in the cart format
    purge      everyone, bytes gone    owner only, irreversible

In a shared brain "this is noise TO ME" is the most common thing a person wants to say about
a passage. Without hide, the only options are live with it or delete it for the whole
company — which is how shared knowledge gets destroyed by people who only wanted a tidier
list.

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_hide.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MEMBOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEMBOT))

import membot_server as ms       # noqa: E402
import overlay as ov             # noqa: E402
import seat_hot_stack as shs     # noqa: E402
import subcart as sc             # noqa: E402

seat_hide = ms.seat_hide.fn
seat_hidden = ms.seat_hidden.fn
seat_hot_stack_tool = ms.seat_hot_stack.fn

TEXTS = ["alpha body", "beta body", "gamma body", "delta body"]


@pytest.fixture
def mounted(tmp_path, monkeypatch):
    cart = tmp_path / "brain.pkl"
    cart.write_bytes(b"stub")
    state = ms._new_session()
    state.update({"texts": list(TEXTS), "cartridge_path": str(cart),
                  "cartridge_name": "brain.pkl"})
    monkeypatch.setattr(ms, "_get_session", lambda sid: state)
    ms._OVERLAY_STORES.clear()          # per-cart-dir cache; tmp_path differs each test
    return state, tmp_path


# ------------------------------------------------------------------ the mark

def test_hidden_is_a_per_seat_mark():
    o = ov.new_overlay("user_a", "c.pkl")
    ov.set_mark(o, "k", ov.HIDDEN_MARK)
    assert ov.is_hidden(o, "k") is True
    assert ov.hidden_keys(o) == {"k"}


def test_unmarked_is_not_hidden():
    o = ov.new_overlay("user_a", "c.pkl")
    ov.touch(o, "k")
    assert ov.is_hidden(o, "k") is False and ov.hidden_keys(o) == set()


def test_hidden_survives_a_save_and_reload(tmp_path):
    o = ov.new_overlay("user_a", "c.pkl")
    ov.set_mark(o, "k", ov.HIDDEN_MARK)
    ov.save_overlay(o, tmp_path / "a.json")
    assert ov.is_hidden(ov.load_overlay(tmp_path / "a.json"), "k") is True


def test_hidden_is_never_seeded_from_a_cart():
    """No cart carries it. It must not become a shared property by accident."""
    links = [{"access_count": 1, "accessed_at": 5.0,
              "_reserved": {"flags": {"agency": "reflective"},
                            "canonical": {"truth_status": "active", "hidden": True}}}]
    o = ov.seed_from_links("user_a", "c.pkl", ["body"], links, sc.content_key)
    assert ov.hidden_keys(o) == set()


def test_hidden_does_not_reach_the_scorers_canonical_block():
    """Hidden is per-seat; leaking it into canonical would make it look cart-wide."""
    seat = ov.as_link({ov._K: ["hidden", "important"]})
    canonical = shs.scoring_link({"_reserved": {"canonical": {}}}, seat)["_reserved"]["canonical"]
    assert canonical.get("important") is True
    assert "hidden" not in canonical


def test_stats_counts_hidden():
    o = ov.new_overlay("user_a", "c.pkl")
    ov.set_mark(o, "k1", ov.HIDDEN_MARK)
    ov.touch(o, "k2")
    assert ov.stats(o)["hidden"] == 1 and ov.stats(o)["rows"] == 2


# ------------------------------------------------------------------ hot stack

def _seat_with_all_touched():
    o = ov.new_overlay("user_a", "c.pkl")
    for t in TEXTS:
        ov.touch(o, sc.content_key(t), count=5)
    return o


def fake_score(link, now):
    return float(link.get("access_count", 0))


def test_hidden_drops_out_of_the_hot_stack():
    o = _seat_with_all_touched()
    ov.set_mark(o, sc.content_key(TEXTS[1]), ov.HIDDEN_MARK)
    m = sc.new_manifest("all", "c.pkl", mode="all", authored_by="access", created_at=1.0)
    out = shs.seat_hot_stack(m, TEXTS, [{}] * len(TEXTS), o, fake_score, top_k=8)
    assert [i["index"] for i in out["items"]] == [0, 2, 3]


def test_hidden_is_absent_not_merely_ranked_low():
    """Scoring it to zero would still show it at the bottom. Hidden means gone."""
    o = _seat_with_all_touched()
    ov.set_mark(o, sc.content_key(TEXTS[1]), ov.HIDDEN_MARK)
    m = sc.new_manifest("all", "c.pkl", mode="all", authored_by="access", created_at=1.0)
    out = shs.seat_hot_stack(m, TEXTS, [{}] * len(TEXTS), o, fake_score, top_k=8)
    assert all(i["index"] != 1 for i in out["items"])


def test_hot_stack_counts_what_it_hid():
    """A list that quietly omits things is as untrustworthy as one that quietly includes."""
    o = _seat_with_all_touched()
    ov.set_mark(o, sc.content_key(TEXTS[1]), ov.HIDDEN_MARK)
    m = sc.new_manifest("all", "c.pkl", mode="all", authored_by="access", created_at=1.0)
    out = shs.seat_hot_stack(m, TEXTS, [{}] * len(TEXTS), o, fake_score, top_k=8)
    assert out["stats"]["hidden"] == 1


def test_one_seats_hiding_does_not_touch_another():
    """The whole point of the rung."""
    a, b = _seat_with_all_touched(), _seat_with_all_touched()
    b["seat"] = "user_b"
    ov.set_mark(a, sc.content_key(TEXTS[1]), ov.HIDDEN_MARK)
    m = sc.new_manifest("all", "c.pkl", mode="all", authored_by="access", created_at=1.0)
    links = [{}] * len(TEXTS)
    ra = shs.seat_hot_stack(m, TEXTS, links, a, fake_score, top_k=8)
    rb = shs.seat_hot_stack(m, TEXTS, links, b, fake_score, top_k=8)
    assert len(ra["items"]) == 3 and len(rb["items"]) == 4


# ------------------------------------------------------------------ the tools

def test_hide_then_list_then_unhide(mounted):
    assert "Hid 2 passage(s)" in seat_hide("1,3", session_id="user_a")
    listed = seat_hidden(session_id="user_a")
    assert "(idx:1)" in listed and "(idx:3)" in listed
    assert "Unhid 2" in seat_hide("1,3", unhide=True, session_id="user_a")
    assert "nothing hidden" in seat_hidden(session_id="user_a")


def test_hide_says_nothing_was_deleted(mounted):
    """The message is the product: people delete because they think hiding is not an option."""
    out = seat_hide("0", session_id="user_a")
    assert "Nothing was deleted" in out and "no other seat is affected" in out


def test_hide_leaves_the_cart_untouched(mounted):
    _, tmp = mounted
    seat_hide("0,1", session_id="user_a")
    assert (tmp / "brain.pkl").read_bytes() == b"stub"


def test_hide_is_per_seat_through_the_tools(mounted):
    seat_hide("1", session_id="user_a")
    assert "nothing hidden" in seat_hidden(session_id="user_b")


def test_hide_rejects_out_of_range(mounted):
    out = seat_hide("0,99", session_id="user_a")
    assert "Out of range" in out and "Nothing changed" in out
    assert "nothing hidden" in seat_hidden(session_id="user_a")


def test_hide_rejects_unparseable(mounted):
    assert "Could not parse" in seat_hide("one,two", session_id="user_a")


def test_tools_report_when_nothing_is_mounted(monkeypatch):
    monkeypatch.setattr(ms, "_get_session", lambda sid: {"texts": []})
    assert "No cartridge mounted" in seat_hide("0")
    assert "No cartridge mounted" in seat_hidden()


def test_hidden_passages_drop_out_of_the_seat_hot_stack_tool(mounted):
    state, _ = mounted
    store = ms._overlay_store_for(state)
    for t in TEXTS:
        store.touch("user_a", sc.content_key(t), count=3)
    seat_hide("2", session_id="user_a")
    out = seat_hot_stack_tool(session_id="user_a")
    assert "idx:2" not in out and "idx:0" in out


def test_listing_reports_hidden_keys_absent_from_this_cart(mounted):
    """A seat can carry hidden marks from another mount, or from a since-removed passage."""
    state, _ = mounted
    store = ms._overlay_store_for(state)
    store.set_mark("user_a", "f" * 40, ov.HIDDEN_MARK)
    assert "not present in this cart" in seat_hidden(session_id="user_a")
