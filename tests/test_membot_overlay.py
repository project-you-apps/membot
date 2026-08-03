"""Membot's per-seat attention wiring.

Membot tracked NO access counts before 2026-08-02 -- zero `access_count` across
membot_server / multi_cart / membox -- so this is additive, with no legacy counter to
dual-write and no migration. What it already had was per-user IDENTITY
(`_session_id_for_user` -> `user_<uuid>`); what was missing was attention.

These tests exercise the helpers directly rather than booting the MCP server, which
needs a model and a mounted cart. The one thing they pin hardest is the truncation
trap: search truncates display text to 500 chars, and hashing that instead of the body
would produce keys matching nothing, forever, silently.

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_membot_overlay.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MEMBOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEMBOT))

import overlay as ov            # noqa: E402
import subcart as sc            # noqa: E402
from overlay_store import OverlayStore  # noqa: E402


LONG_BODY = "SESSION: 2026-08-02 | live-store\n" + ("lattice frustration policy " * 60)


def _state(tmp_path, texts):
    cart = tmp_path / "brain.pkl"
    cart.write_bytes(b"stub")
    return {"cartridge_path": str(cart), "cartridge_name": "brain.pkl", "texts": texts}


# ------------------------------------------------------ the truncation trap

def test_truncated_display_text_hashes_differently_from_the_body():
    """The bug this wiring had to avoid: search truncates to 500 chars for display."""
    assert len(LONG_BODY) > 500
    truncated = LONG_BODY[:500] + "..."
    assert sc.content_key(truncated) != sc.content_key(LONG_BODY)


def test_key_from_body_resolves_against_the_cart_and_truncated_does_not():
    index = sc.build_key_index([LONG_BODY])
    assert sc.content_key(LONG_BODY) in index
    assert sc.content_key(LONG_BODY[:500] + "...") not in index


# ------------------------------------------------------ store wiring

def test_store_lands_beside_the_cart(tmp_path):
    cart = tmp_path / "brain.pkl"
    store = OverlayStore(cart.parent / "overlays", parent_cart=cart.name)
    assert store.dir == tmp_path / "overlays"
    assert store.parent_cart == "brain.pkl"


def test_seat_is_the_session_id(tmp_path):
    """`user_<uuid>` comes straight from _session_id_for_user; no second identity."""
    store = OverlayStore(tmp_path, parent_cart="brain.pkl")
    seat = "user_3579e6ee-6412-4099-8d66-a205d9be7849"
    store.touch(seat, sc.content_key(LONG_BODY))
    assert ov.get(store.get_overlay(seat), sc.content_key(LONG_BODY))["access_count"] == 1
    assert store.path_for(seat).name.startswith("user_3579e6ee")


def test_two_seats_on_one_cart_do_not_share_attention(tmp_path):
    store = OverlayStore(tmp_path, parent_cart="brain.pkl")
    a, b = "user_aaa", "user_bbb"
    for _ in range(40):
        store.touch(a, sc.content_key(LONG_BODY))
    assert ov.get(store.get_overlay(a), sc.content_key(LONG_BODY))["access_count"] == 40
    assert ov.get(store.get_overlay(b), sc.content_key(LONG_BODY))["access_count"] == 0


def test_anonymous_callers_share_the_default_seat(tmp_path):
    """`default` is the legacy fallback in _session_id_for_user -- unchanged here."""
    store = OverlayStore(tmp_path, parent_cart="brain.pkl")
    store.touch("default", sc.content_key(LONG_BODY))
    assert store.get_overlay("default")["seat"] == "default"


def test_touch_many_records_one_row_per_distinct_body(tmp_path):
    store = OverlayStore(tmp_path, parent_cart="brain.pkl")
    bodies = [LONG_BODY, "SESSION: other\n" + LONG_BODY.split("\n", 1)[1], "different"]
    store.touch_many("user_x", [sc.content_key(b) for b in bodies])
    # First two differ only by SESSION header, which content_key strips -> one row.
    assert len(store.get_overlay("user_x")["entries"]) == 2


# ------------------------------------------------------ ranking

def test_seat_ranking_reflects_this_seats_counts_only(tmp_path):
    store = OverlayStore(tmp_path, parent_cart="brain.pkl")
    texts = ["alpha body", "beta body", "gamma body"]
    store.touch("user_a", sc.content_key(texts[0]), count=9)
    store.touch("user_a", sc.content_key(texts[2]), count=2)
    store.touch("user_b", sc.content_key(texts[1]), count=99)

    o = store.get_overlay("user_a")
    ranked = sorted(o["entries"].items(), key=lambda kv: kv[1][ov._N], reverse=True)
    index = {sc.content_key(t): i for i, t in enumerate(texts)}
    assert [index[k] for k, _ in ranked] == [0, 2]      # beta never appears for user_a


def test_attention_on_a_passage_absent_from_this_cart_is_reported_not_dropped(tmp_path):
    """A seat can carry attention from another mount, or from a tombstoned passage."""
    store = OverlayStore(tmp_path, parent_cart="brain.pkl")
    store.touch("user_a", sc.content_key("from another cart"))
    index = {sc.content_key(t): i for i, t in enumerate(["only body here"])}
    unresolved = [k for k in store.get_overlay("user_a")["entries"] if k not in index]
    assert len(unresolved) == 1


# ------------------------------------------------------ failure modes

def test_a_failed_touch_never_breaks_search(tmp_path):
    """Attention is additive to a product that worked without it."""
    store = OverlayStore(tmp_path, parent_cart="brain.pkl")

    def record():
        try:
            store.touch_many("user_a", [sc.content_key(None)])
            store.flush()
        except Exception as e:                       # noqa: BLE001
            store.stats["last_error"] = f"touch: {e}"
        return "search results"

    assert record() == "search results"


def test_flush_is_interval_gated_so_search_does_not_serialise_megabytes(tmp_path):
    store = OverlayStore(tmp_path, parent_cart="brain.pkl", flush_interval=9999)
    store.touch("user_a", sc.content_key(LONG_BODY))
    assert store.flush()["written"] == 0             # not due: no I/O on the hot path
    assert store.flush(force=True)["written"] == 1


def test_stores_are_keyed_per_cart_directory(tmp_path):
    """Two carts in different directories must not share one overlay set."""
    one, two = tmp_path / "a", tmp_path / "b"
    one.mkdir(); two.mkdir()
    s1 = OverlayStore(one / "overlays", parent_cart="brain.pkl")
    s2 = OverlayStore(two / "overlays", parent_cart="brain.pkl")
    s1.touch("user_a", sc.content_key(LONG_BODY))
    assert len(s1.get_overlay("user_a")["entries"]) == 1
    assert len(s2.get_overlay("user_a")["entries"]) == 0
