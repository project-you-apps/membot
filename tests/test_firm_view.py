"""The firm view — what the ORGANISATION is warm on.

Andy called this a requirement, not an extra: fifty personal hot stacks do not sum to an
institutional record unless the roll-up is deliberately kept. `firm_hot_stack()` was written
on 2026-08-02 and had ZERO callers until 08-03 — partly because `OverlayStore` had no way to
enumerate seats, so nothing could even assemble its argument.

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_firm_view.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MEMBOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEMBOT))

import membot_server as ms       # noqa: E402
import overlay as ov             # noqa: E402
import subcart as sc             # noqa: E402
from overlay_store import OverlayStore  # noqa: E402

firm_hot_stack = ms.firm_hot_stack.fn

TEXTS = ["alpha body", "beta body", "gamma body", "delta body"]


@pytest.fixture
def mounted(tmp_path, monkeypatch):
    cart = tmp_path / "redwood.pkl"
    cart.write_bytes(b"stub")
    state = ms._new_session()
    state.update({"texts": list(TEXTS), "cartridge_path": str(cart),
                  "cartridge_name": "redwood.pkl"})
    monkeypatch.setattr(ms, "_get_session", lambda sid: state)
    ms._OVERLAY_STORES.clear()
    return state, tmp_path


# ------------------------------------------------------------------ seat enumeration

def test_store_lists_seats_from_disk(tmp_path):
    store = OverlayStore(tmp_path, parent_cart="c.pkl")
    for seat in ("user_a", "user_b"):
        store.touch(seat, "k" * 40)
    store.flush(force=True)
    fresh = OverlayStore(tmp_path, parent_cart="c.pkl")
    assert fresh.seats() == ["user_a", "user_b"]


def test_seats_includes_loaded_but_unflushed(tmp_path):
    """A seat that has touched something but not yet flushed still counts — otherwise the
    firm view lags reality by up to a flush interval."""
    store = OverlayStore(tmp_path, parent_cart="c.pkl", flush_interval=9999)
    store.touch("user_new", "k" * 40)
    assert "user_new" in store.seats()


def test_seats_on_an_empty_directory(tmp_path):
    assert OverlayStore(tmp_path / "nope", parent_cart="c.pkl").seats() == []


# ------------------------------------------------------------------ the signal

def test_distinct_seats_outrank_raw_volume(mounted):
    """One person reading something 40x and 40 people reading it once are the same access
    count and very different facts about a company."""
    state, _ = mounted
    store = ms._overlay_store_for(state)
    store.touch("solo", sc.content_key(TEXTS[0]), count=40)
    for i in range(4):
        store.touch(f"person_{i}", sc.content_key(TEXTS[1]))

    out = firm_hot_stack(session_id="anyone")
    beta_pos = out.index("idx:1")
    alpha_pos = out.index("idx:0")
    assert beta_pos < alpha_pos, "4 people should outrank 1 person reading 40 times"
    assert "4 people" in out and "1 person" in out


def test_it_aggregates_across_seats(mounted):
    state, _ = mounted
    store = ms._overlay_store_for(state)
    store.touch("a", sc.content_key(TEXTS[2]), count=3)
    store.touch("b", sc.content_key(TEXTS[2]), count=2)
    out = firm_hot_stack(session_id="anyone")
    assert "2 people, 5 reads" in out


def test_it_shows_passages_the_viewer_never_touched(mounted):
    """The organisation's view, not yours. That is the whole point of it."""
    state, _ = mounted
    store = ms._overlay_store_for(state)
    store.touch("someone_else", sc.content_key(TEXTS[3]), count=7)
    out = firm_hot_stack(session_id="a_viewer_who_read_nothing")
    assert "idx:3" in out


# ------------------------------------------------------------------ honesty

def test_attention_on_passages_absent_from_this_cart_is_reported(mounted):
    state, _ = mounted
    store = ms._overlay_store_for(state)
    store.touch("a", sc.content_key(TEXTS[0]))
    store.touch("a", "f" * 40)                     # from another mount
    out = firm_hot_stack(session_id="anyone")
    assert "not in this cart" in out


def test_no_seats_yet_says_so(mounted):
    assert "No seats have recorded attention" in firm_hot_stack(session_id="anyone")


def test_seats_present_but_nothing_touched(mounted):
    state, _ = mounted
    store = ms._overlay_store_for(state)
    store.get_overlay("lurker")                    # exists, has touched nothing
    out = firm_hot_stack(session_id="anyone")
    assert "none has touched anything" in out or "No seats have recorded" in out


def test_no_cartridge_mounted(monkeypatch):
    monkeypatch.setattr(ms, "_get_session", lambda sid: {"texts": []})
    assert "No cartridge mounted" in firm_hot_stack()


def test_hidden_does_not_hide_from_the_firm_view(mounted):
    """`hidden` is one seat saying 'noise, to me'. It must not remove a passage from the
    organisation's record — that would let one person edit institutional memory."""
    state, _ = mounted
    store = ms._overlay_store_for(state)
    store.touch("a", sc.content_key(TEXTS[0]), count=5)
    store.set_mark("a", sc.content_key(TEXTS[0]), ov.HIDDEN_MARK)
    assert "idx:0" in firm_hot_stack(session_id="a")
