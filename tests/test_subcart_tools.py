"""The sub-cart MCP tools, exercised against a stubbed mounted cart.

`membot_server` imports in ~1s and does not load a model at import time, so the tool
bodies can be driven directly with a fake session rather than by booting the server.
That matters because the interesting logic lives IN the tools: index parsing, the "all"
form, out-of-range rejection, and the refusal paths.

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_subcart_tools.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MEMBOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEMBOT))

import membot_server as ms       # noqa: E402
import subcart as sc             # noqa: E402
from subcart_store import SubcartStore, TRASH_DIRNAME  # noqa: E402


TEXTS = ["alpha body", "beta body", "gamma body", "delta body"]

# `@mcp.tool()` wraps each function in a FunctionTool, which is not itself callable.
# `.fn` is the original function — unwrap once here rather than at 21 call sites.
subcart_create = ms.subcart_create.fn
subcart_list = ms.subcart_list.fn
subcart_delete = ms.subcart_delete.fn
seat_hot_stack = ms.seat_hot_stack.fn


@pytest.fixture
def mounted(tmp_path, monkeypatch):
    """A session with a cart mounted at a real path, without touching the real server."""
    cart = tmp_path / "brain.pkl"
    cart.write_bytes(b"stub")
    state = {"texts": list(TEXTS), "cartridge_path": str(cart),
             "cartridge_name": "brain.pkl"}
    monkeypatch.setattr(ms, "_get_session", lambda sid: state)
    return state, tmp_path


# ------------------------------------------------------------------ create

def test_create_from_indices(mounted):
    _, tmp = mounted
    out = subcart_create("alpha-and-gamma", "0,2", session_id="user_a")
    assert "Created sub-cart" in out and "2 passage(s)" in out
    rows = SubcartStore(tmp / "brain.pkl").list()
    assert [r["name"] for r in rows] == ["alpha-and-gamma"]
    assert rows[0]["keys"] == 2


def test_created_manifest_resolves_to_the_chosen_passages(mounted):
    _, tmp = mounted
    subcart_create("pick", "1,3", session_id="user_a")
    manifest = SubcartStore(tmp / "brain.pkl").get("pick")
    assert sc.resolve(manifest, TEXTS)["indices"] == [1, 3]


def test_create_all_stores_no_key_list(mounted):
    _, tmp = mounted
    out = subcart_create("everything", "all", authored_by="access", session_id="user_a")
    assert "mode=all" in out
    manifest = SubcartStore(tmp / "brain.pkl").get("everything")
    assert "keys" not in manifest["membership"]
    assert sc.resolve(manifest, TEXTS)["indices"] == [0, 1, 2, 3]


def test_no_passages_are_copied(mounted):
    """A sub-cart is a reference, not a copy. If passage text lands in the manifest,
    the whole storage argument collapses."""
    _, tmp = mounted
    subcart_create("pick", "0,1", session_id="user_a")
    raw = (tmp / "subcarts").glob("*.subcart.json").__next__().read_text(encoding="utf-8")
    for body in TEXTS:
        assert body not in raw


def test_spaces_in_indices_are_tolerated(mounted):
    assert "Created" in subcart_create("spaced", "0, 2 ,3", session_id="user_a")


# ------------------------------------------------------------------ refusals

def test_out_of_range_index_creates_nothing(mounted):
    _, tmp = mounted
    out = subcart_create("bad", "0,99", session_id="user_a")
    assert "Out of range" in out and "Nothing created" in out
    assert SubcartStore(tmp / "brain.pkl").list() == []


def test_negative_index_is_rejected(mounted):
    assert "Out of range" in subcart_create("bad", "-1", session_id="user_a")


def test_unparseable_indices_are_rejected(mounted):
    out = subcart_create("bad", "three,four", session_id="user_a")
    assert "Could not parse" in out


def test_duplicate_name_is_refused_with_a_reason(mounted):
    subcart_create("dupe", "0", session_id="user_a")
    out = subcart_create("dupe", "1", session_id="user_a")
    assert "Refused" in out and "already exists" in out


def test_blank_name_is_refused(mounted):
    assert "name is required" in subcart_create("   ", "0", session_id="user_a")


def test_tools_report_when_nothing_is_mounted(monkeypatch):
    monkeypatch.setattr(ms, "_get_session", lambda sid: {"texts": []})
    assert "No cartridge mounted" in subcart_create("x", "0")
    assert "No cartridge mounted" in subcart_list()
    assert "No cartridge mounted" in subcart_delete("x")
    assert "No cartridge mounted" in seat_hot_stack()


# ------------------------------------------------------------------ list

def test_list_shows_resolution_counts(mounted):
    subcart_create("pick", "0,1", session_id="user_a")
    out = subcart_list(session_id="user_a")
    assert "pick" in out and "2 resolved" in out


def test_list_reports_unresolved_rather_than_hiding_it(mounted):
    """A key with no passage behind it is tombstoned, revoked, or from another cart."""
    state, tmp = mounted
    store = SubcartStore(tmp / "brain.pkl")
    store.create(sc.new_manifest("ghosts", "brain.pkl",
                                 keys=[sc.content_key(TEXTS[0]), "f" * 40],
                                 created_at=1.0))
    out = subcart_list(session_id="user_a")
    assert "1 resolved" in out and "1 unresolved" in out


def test_list_flags_a_corrupt_manifest(mounted):
    state, tmp = mounted
    subcart_create("good", "0", session_id="user_a")
    (tmp / "subcarts" / "broken.subcart.json").write_text("{ nope", encoding="utf-8")
    out = subcart_list(session_id="user_a")
    assert "INVALID" in out and "good" in out


def test_list_when_empty(mounted):
    assert "No sub-carts defined" in subcart_list(session_id="user_a")


# ------------------------------------------------------------------ delete

def test_delete_moves_to_trash_and_says_so(mounted):
    _, tmp = mounted
    subcart_create("temp", "0", session_id="user_a")
    out = subcart_delete("temp", session_id="user_a")
    assert "moved to" in out and "No passages were removed" in out
    assert list((tmp / "subcarts" / TRASH_DIRNAME).glob("*.subcart.json"))


def test_delete_leaves_the_cart_untouched(mounted):
    _, tmp = mounted
    subcart_create("temp", "0", session_id="user_a")
    subcart_delete("temp", session_id="user_a")
    assert (tmp / "brain.pkl").read_bytes() == b"stub"


def test_delete_unknown_name_is_a_clear_message(mounted):
    assert "No sub-cart named" in subcart_delete("ghost", session_id="user_a")


# ------------------------------------------------------------------ seat stack

def test_seat_hot_stack_before_any_attention(mounted):
    assert "no recorded attention" in seat_hot_stack(session_id="user_a")


def test_seat_hot_stack_ranks_this_seats_own_reads(mounted):
    state, tmp = mounted
    store = ms._overlay_store_for(state)
    store.touch("user_a", sc.content_key(TEXTS[2]), count=9)
    store.touch("user_a", sc.content_key(TEXTS[0]), count=2)
    out = seat_hot_stack(session_id="user_a")
    assert "idx:2" in out and out.index("idx:2") < out.index("idx:0")


def test_two_seats_see_different_stacks(mounted):
    state, tmp = mounted
    store = ms._overlay_store_for(state)
    store.touch("user_a", sc.content_key(TEXTS[0]), count=5)
    store.touch("user_b", sc.content_key(TEXTS[3]), count=5)
    assert "idx:0" in seat_hot_stack(session_id="user_a")
    assert "idx:3" in seat_hot_stack(session_id="user_b")
    assert "idx:3" not in seat_hot_stack(session_id="user_a")
