"""The two-session data-loss scenario, driven through membot's own tools.

    Andy mounts, Claude mounts, Andy deletes a passage and saves,
    Claude saves  ->  Andy's deletion is silently reverted.

That was whole-cart last-writer-wins. This checks the guard actually fires where it needs
to, rather than only in the unit tests for the module it lives in.

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_generation_wiring.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

MEMBOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MEMBOT))

import cart_generation as cg   # noqa: E402
import membot_server as ms     # noqa: E402

get_status = ms.get_status.fn


@pytest.fixture
def shared_cart(tmp_path):
    cart = tmp_path / "company-brain.cart.npz"
    cart.write_bytes(b"cart bytes")
    return cart


def session_for(cart, generation):
    """A session state as `mount_cartridge` would leave it.

    Built from `_new_session()` rather than hand-rolled, so it cannot drift out of sync
    with whatever fields the server actually reads -- the first version omitted
    `binary_corpus` and failed with a KeyError that said nothing about generations.
    """
    state = ms._new_session()
    state.update({
        "cartridge_name": "company-brain",
        "cartridge_path": str(cart),
        "cart_generation": generation,
        "texts": ["alpha", "beta"],
    })
    return state


# ------------------------------------------------------------------ the guard

def test_second_saver_is_refused_not_silently_applied(shared_cart):
    cg.bump_generation(shared_cart, seat="setup")
    andy = session_for(shared_cart, 1)
    claude = session_for(shared_cart, 1)          # both mounted at generation 1

    cg.bump_generation(shared_cart, seat="user_andy")     # Andy saves
    andy["cart_generation"] = 2

    cg.assert_writable(shared_cart, andy["cart_generation"])   # Andy stays current
    with pytest.raises(cg.StaleCartError):
        cg.assert_writable(shared_cart, claude["cart_generation"])


def test_the_refusal_says_who_and_how_far(shared_cart):
    cg.bump_generation(shared_cart, seat="user_andy", now=1000.0)
    try:
        cg.assert_writable(shared_cart, 0)
    except cg.StaleCartError as e:
        assert e.status["updated_by"] == "user_andy" and e.status["behind"] == 1
    else:
        pytest.fail("expected refusal")


# ------------------------------------------------------------------ status surface

def test_status_reports_a_current_view(shared_cart, monkeypatch):
    cg.bump_generation(shared_cart, seat="setup")
    monkeypatch.setattr(ms, "_get_session", lambda sid: session_for(shared_cart, 1))
    assert "View: current (gen 1)" in get_status(session_id="user_a")


def test_status_reports_a_stale_view_with_the_reason(shared_cart, monkeypatch):
    cg.bump_generation(shared_cart, seat="setup")
    cg.bump_generation(shared_cart, seat="user_andy")
    monkeypatch.setattr(ms, "_get_session", lambda sid: session_for(shared_cart, 1))
    out = get_status(session_id="user_claude")
    assert "View: STALE" in out and "user_andy" in out and "1 save behind" in out


def test_status_reports_an_out_of_band_change(shared_cart, monkeypatch):
    """droplet_sync, an scp, a manual copy -- none of them go through us."""
    cg.bump_generation(shared_cart, seat="setup")
    shared_cart.write_bytes(b"replaced by something else entirely")
    monkeypatch.setattr(ms, "_get_session", lambda sid: session_for(shared_cart, 1))
    assert "outside the app" in get_status(session_id="user_a")


def test_status_without_a_mounted_cart_does_not_mention_generation(monkeypatch):
    state = session_for(Path("nowhere"), 0)
    state["cartridge_path"] = None
    monkeypatch.setattr(ms, "_get_session", lambda sid: state)
    assert "View:" not in get_status(session_id="user_a")


def test_a_generation_check_failure_never_breaks_status(monkeypatch, shared_cart):
    """Reporting freshness is a nicety; get_status answering at all is not."""
    monkeypatch.setattr(ms, "_get_session", lambda sid: session_for(shared_cart, 1))
    monkeypatch.setattr(ms._cart_generation, "check_stale",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk gone")))
    assert "Cartridge:" in get_status(session_id="user_a")


# ------------------------------------------------------------------ scope

def test_saving_under_a_new_name_is_a_copy_not_a_conflict(shared_cart, tmp_path):
    """'Save as' must not be blocked by staleness -- nothing is being overwritten."""
    cg.bump_generation(shared_cart, seat="andy")
    fresh = tmp_path / "my-fork.cart.npz"
    assert cg.assert_writable(fresh, 0)["stale"] is False


def test_a_brand_new_cart_saves_without_ceremony(tmp_path):
    assert cg.assert_writable(tmp_path / "new.cart.npz", 0)["stale"] is False
