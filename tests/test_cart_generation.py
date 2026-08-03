"""Cart generation: detect a stale view, refuse a stale write.

The scenario under test is the one that motivated it: two sessions hold independent copies
of one cart, A saves, B saves, and B silently writes back what A deleted.

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_cart_generation.py -q
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cart_generation as cg  # noqa: E402


def cart(tmp_path, body=b"cart bytes"):
    p = tmp_path / "brain.pkl"
    p.write_bytes(body)
    return p


# ------------------------------------------------------------------ reading

def test_a_cart_with_no_sidecar_is_generation_zero(tmp_path):
    """Every cart in existence predates this module. Not an error."""
    status = cg.read_generation(cart(tmp_path))
    assert status["generation"] == 0 and status["has_sidecar"] is False
    assert status["out_of_band"] is False


def test_reading_a_missing_cart_does_not_raise(tmp_path):
    assert cg.read_generation(tmp_path / "nope.pkl")["generation"] == 0


def test_a_corrupt_sidecar_reports_out_of_band_rather_than_breaking(tmp_path):
    """We genuinely do not know the generation -- practically the same as a stranger
    having written the cart. Must not make the cart unusable."""
    c = cart(tmp_path)
    cg.sidecar_path(c).write_text("{ not json", encoding="utf-8")
    status = cg.read_generation(c)
    assert status["out_of_band"] is True and status["generation"] == 0


def test_sidecar_sits_beside_the_cart(tmp_path):
    assert cg.sidecar_path(tmp_path / "brain.pkl").name == "brain.pkl.generation.json"


# ------------------------------------------------------------------ bumping

def test_bump_increments_and_records_who(tmp_path):
    c = cart(tmp_path)
    out = cg.bump_generation(c, seat="user_a", now=100.0)
    assert out["generation"] == 1 and out["updated_by"] == "user_a"
    assert cg.read_generation(c)["generation"] == 1


def test_bumps_accumulate(tmp_path):
    c = cart(tmp_path)
    for _ in range(3):
        cg.bump_generation(c, seat="user_a")
    assert cg.read_generation(c)["generation"] == 3


def test_bump_writes_atomically_leaving_no_temp_files(tmp_path):
    c = cart(tmp_path)
    cg.bump_generation(c, seat="user_a")
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "brain.pkl", "brain.pkl.generation.json"]


def test_bump_stamps_the_carts_size(tmp_path):
    c = cart(tmp_path, b"x" * 500)
    assert cg.bump_generation(c)["cart_size"] == 500


# ------------------------------------------------------------------ out-of-band

def test_a_cart_rewritten_behind_our_back_is_flagged(tmp_path):
    """droplet_sync, an scp, a manual copy -- none of them go through us."""
    c = cart(tmp_path, b"original")
    cg.bump_generation(c, seat="user_a")
    assert cg.read_generation(c)["out_of_band"] is False

    c.write_bytes(b"replaced by something else entirely")   # out of band
    assert cg.read_generation(c)["out_of_band"] is True


def test_out_of_band_makes_a_view_stale_even_at_the_same_generation(tmp_path):
    c = cart(tmp_path, b"original")
    cg.bump_generation(c, seat="user_a")
    c.write_bytes(b"different bytes now")
    status = cg.check_stale(c, loaded_generation=1)
    assert status["behind"] == 0 and status["stale"] is True


def test_small_mtime_drift_is_tolerated(tmp_path):
    """Copies and filesystems round mtime differently. A false alarm on every file move
    would train people to ignore the warning."""
    c = cart(tmp_path)
    cg.bump_generation(c)
    st = c.stat()
    os.utime(c, (st.st_atime, st.st_mtime + 1.0))
    assert cg.read_generation(c)["out_of_band"] is False


# ------------------------------------------------------------------ staleness

def test_a_current_view_is_not_stale(tmp_path):
    c = cart(tmp_path)
    cg.bump_generation(c, seat="user_a")
    assert cg.check_stale(c, 1)["stale"] is False


def test_a_view_behind_by_two_says_so(tmp_path):
    c = cart(tmp_path)
    cg.bump_generation(c, seat="user_a")
    cg.bump_generation(c, seat="user_a")
    cg.bump_generation(c, seat="user_a")
    status = cg.check_stale(c, 1)
    assert status["behind"] == 2 and status["current"] == 3 and status["stale"] is True


def test_a_view_ahead_of_disk_is_not_reported_as_behind(tmp_path):
    """Can happen after a restore or a rollback. Negative 'behind' would be nonsense."""
    assert cg.check_stale(cart(tmp_path), 5)["behind"] == 0


# ------------------------------------------------------------------ the guard

def test_the_data_loss_scenario_is_now_refused(tmp_path):
    """A mounts, B mounts, A saves, B saves -- B's save used to silently write back
    whatever A deleted."""
    c = cart(tmp_path)
    cg.bump_generation(c, seat="setup")
    a_view = b_view = cg.read_generation(c)["generation"]      # both mounted at gen 1

    cg.bump_generation(c, seat="andy")                          # A saves
    cg.assert_writable(c, a_view + 1, seat="andy")              # A is current

    with pytest.raises(cg.StaleCartError) as excinfo:
        cg.assert_writable(c, b_view, seat="claude")            # B is not
    assert "behind" in str(excinfo.value)
    assert excinfo.value.status["behind"] == 1


def test_the_error_carries_the_status_for_the_ui(tmp_path):
    c = cart(tmp_path)
    cg.bump_generation(c, seat="andy", now=500.0)
    try:
        cg.assert_writable(c, 0)
    except cg.StaleCartError as e:
        assert e.status["updated_by"] == "andy" and e.status["updated_at"] == 500.0
    else:
        pytest.fail("expected StaleCartError")


def test_a_first_save_to_a_fresh_cart_is_allowed(tmp_path):
    """Generation 0 view against a generation 0 cart. Nothing to conflict with."""
    assert cg.assert_writable(cart(tmp_path), 0)["stale"] is False


def test_out_of_band_blocks_a_write_too(tmp_path):
    c = cart(tmp_path, b"original")
    cg.bump_generation(c, seat="a")
    c.write_bytes(b"changed underneath us")
    with pytest.raises(cg.StaleCartError):
        cg.assert_writable(c, 1)


# ------------------------------------------------------------------ messages

def test_message_for_a_normal_stale_view_reads_as_news_not_a_decision():
    msg = cg.describe_stale({"behind": 1, "updated_by": "Andy", "out_of_band": False})
    assert "Andy" in msg and "1 save behind" in msg and "Reload" in msg


def test_message_pluralises():
    assert "3 saves behind" in cg.describe_stale(
        {"behind": 3, "updated_by": "Andy", "out_of_band": False})


def test_message_for_out_of_band_says_something_different():
    """'Someone changed this behind our back' and 'you are two saves behind' want
    different responses from a human."""
    msg = cg.describe_stale({"behind": 0, "out_of_band": True})
    assert "outside the app" in msg


def test_message_when_current():
    assert cg.describe_stale({"behind": 0, "out_of_band": False}) == "Your view is current."


def test_unknown_updater_does_not_produce_an_empty_name():
    assert "someone else" in cg.describe_stale(
        {"behind": 1, "updated_by": "", "out_of_band": False})
