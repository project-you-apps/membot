"""Tests for the live per-seat overlay store (SPEC-SUBCARTRIDGE-MANIFEST §10 step 5).

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_overlay_store.py -q
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import overlay as ov              # noqa: E402
from overlay_store import OverlayStore  # noqa: E402


K1, K2 = "a" * 40, "b" * 40


def store(tmp_path, **kw):
    return OverlayStore(tmp_path, parent_cart="c.pkl", **kw)


# ------------------------------------------------------------------ basics

def test_touch_is_visible_immediately_without_a_flush(tmp_path):
    """The hot path must not need I/O to be correct."""
    s = store(tmp_path)
    s.touch("andy", K1)
    assert ov.get(s.get_overlay("andy"), K1)["access_count"] == 1
    assert list(tmp_path.iterdir()) == []


def test_new_seat_is_created_not_an_error(tmp_path):
    s = store(tmp_path)
    assert s.get_overlay("someone-new")["seat"] == "someone-new"


def test_seats_are_isolated(tmp_path):
    s = store(tmp_path)
    s.touch("andy", K1, count=5)
    assert ov.get(s.get_overlay("dennis"), K1)["access_count"] == 0


def test_touch_many_bumps_all_keys(tmp_path):
    s = store(tmp_path)
    s.touch_many("andy", [K1, K2], now=100.0)
    o = s.get_overlay("andy")
    assert ov.get(o, K1)["access_count"] == 1 and ov.get(o, K2)["access_count"] == 1
    assert s.stats["touches"] == 2


def test_touch_many_with_no_hits_does_not_dirty(tmp_path):
    s = store(tmp_path)
    s.touch_many("andy", [])
    assert s.report()["seats_dirty"] == 0


# ------------------------------------------------------------------ flushing

def test_flush_is_interval_gated(tmp_path):
    s = store(tmp_path, flush_interval=9999)
    s.touch("andy", K1)
    assert s.flush()["written"] == 0            # not due
    assert s.flush(force=True)["written"] == 1


def test_flush_persists_and_reloads(tmp_path):
    s = store(tmp_path, flush_interval=0)
    s.touch("andy", K1, count=3, now=50.0)
    s.flush(force=True)
    fresh = store(tmp_path)
    assert ov.get(fresh.get_overlay("andy"), K1)["access_count"] == 3


def test_flush_clears_dirty(tmp_path):
    s = store(tmp_path)
    s.touch("andy", K1)
    s.flush(force=True)
    assert s.report()["seats_dirty"] == 0
    assert s.flush(force=True)["written"] == 0


def test_only_dirty_seats_are_written(tmp_path):
    s = store(tmp_path)
    s.touch("andy", K1)
    s.get_overlay("dennis")                      # loaded but untouched
    assert s.flush(force=True)["seats"] == ["andy"]


def test_close_flushes(tmp_path):
    s = store(tmp_path, flush_interval=9999)
    s.touch("andy", K1)
    s.close()
    assert (tmp_path / "andy.overlay.json").exists()


def test_a_failing_flush_stays_dirty_and_retries(tmp_path):
    """A dropped write must not be forgotten -- that is the check_reload lesson."""
    s = store(tmp_path)
    s.touch("andy", K1)
    s._overlays["andy"]["format"] = "corrupt"     # save_overlay will refuse this
    out = s.flush(force=True)
    assert out["failed"] == 1 and s.report()["seats_dirty"] == 1
    assert s.stats["flush_errors"] == 1 and "flush andy" in s.stats["last_error"]

    s._overlays["andy"]["format"] = ov.FORMAT     # repaired
    assert s.flush(force=True)["written"] == 1


def test_flush_never_raises_into_the_caller(tmp_path):
    s = store(tmp_path)
    s.touch("andy", K1)
    s._overlays["andy"]["entries"] = "not a dict"
    s.flush(force=True)                           # must not raise


# ------------------------------------------------------------------ resilience

def test_a_corrupt_overlay_on_disk_does_not_break_search(tmp_path):
    (tmp_path / "andy.overlay.json").write_text("{ this is not json", encoding="utf-8")
    s = store(tmp_path)
    o = s.get_overlay("andy")
    assert o["entries"] == {}
    # and it must be VISIBLE that we started over, not silently indistinguishable from new
    assert "load andy" in s.stats["last_error"]


def test_seat_names_are_sanitised_for_paths(tmp_path):
    s = store(tmp_path)
    assert s.path_for("../../etc/passwd").name == "....etcpasswd.overlay.json"
    assert s.path_for("a b/c").parent == tmp_path


def test_unusable_seat_name_raises(tmp_path):
    with pytest.raises(ValueError):
        store(tmp_path).path_for("///")


def test_flush_writes_atomically_leaving_no_temp_files(tmp_path):
    s = store(tmp_path)
    s.touch("andy", K1)
    s.flush(force=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["andy.overlay.json"]


# ------------------------------------------------------------------ concurrency

def test_concurrent_touches_lose_nothing(tmp_path):
    """The search path is threaded; a dropped increment would be invisible forever."""
    s = store(tmp_path)
    def worker():
        for _ in range(200):
            s.touch("andy", K1)
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ov.get(s.get_overlay("andy"), K1)["access_count"] == 1600


def test_concurrent_touches_across_seats(tmp_path):
    s = store(tmp_path)
    def worker(seat):
        for _ in range(100):
            s.touch(seat, K1)
    threads = [threading.Thread(target=worker, args=(f"seat{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i in range(6):
        assert ov.get(s.get_overlay(f"seat{i}"), K1)["access_count"] == 100


def test_flush_during_touches_is_safe(tmp_path):
    """Serialising a live overlay while it is mutated is a torn write or an outright
    'dictionary changed size during iteration'. The search path is threaded, so this is a
    when, not an if."""
    s = store(tmp_path)
    s.touch("andy", K1)                    # guarantee there is something to flush
    started = threading.Event()
    stop = threading.Event()
    def toucher():
        started.set()
        i = 0
        while not stop.is_set():
            s.touch("andy", f"{i % 500:040d}")   # keeps ADDING keys, not just incrementing
            i += 1
    t = threading.Thread(target=toucher)
    t.start()
    started.wait(2)
    try:
        for _ in range(20):
            s.flush(force=True)            # must not raise
    finally:
        stop.set()
        t.join()
    assert ov.load_overlay(tmp_path / "andy.overlay.json")["seat"] == "andy"


def test_a_touch_racing_a_flush_is_not_marked_clean(tmp_path):
    """The lost-update window: bump arrives after serialisation, before dirty is cleared.
    Marking the seat clean there would leave that increment in RAM only."""
    s = store(tmp_path)
    s.touch("andy", K1)

    real_save = ov.save_overlay
    def save_then_race(o, path):
        real_save(o, path)
        s.touch("andy", K2)               # the racing write
    ov.save_overlay = save_then_race
    try:
        s.flush(force=True)
    finally:
        ov.save_overlay = real_save

    assert s.report()["seats_dirty"] == 1, "racing touch was marked clean"
    s.flush(force=True)
    assert ov.get(ov.load_overlay(tmp_path / "andy.overlay.json"), K2)["access_count"] == 1


def test_snapshot_is_decoupled_from_the_live_overlay(tmp_path):
    """What gets written must be the state at snapshot time, not whatever it became."""
    s = store(tmp_path)
    s.touch("andy", K1, count=1)

    real_save = ov.save_overlay
    captured = {}
    def capture(o, path):
        s.touch("andy", K1, count=99)      # mutate the LIVE overlay mid-write
        captured["written"] = ov.get(o, K1)["access_count"]
        real_save(o, path)
    ov.save_overlay = capture
    try:
        s.flush(force=True)
    finally:
        ov.save_overlay = real_save

    assert captured["written"] == 1                       # snapshot, not live state
    assert ov.get(s.get_overlay("andy"), K1)["access_count"] == 100


# ------------------------------------------------------------------ reporting

def test_report_surfaces_dirt_and_cadence(tmp_path):
    s = store(tmp_path, flush_interval=123)
    s.touch("andy", K1)
    r = s.report()
    assert r["seats_loaded"] == 1 and r["seats_dirty"] == 1
    assert r["rows"]["andy"] == 1 and r["flush_interval_s"] == 123
