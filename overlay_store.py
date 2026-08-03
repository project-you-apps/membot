"""Live per-seat attention: hold overlays in memory, bump on access, flush atomically.

Step 5 of `docs/SPEC-SUBCARTRIDGE-MANIFEST.md` §10, and the one that makes the rest true
rather than true-at-migration-time.

WHY THIS IS NOT OPTIONAL
------------------------
A seeded overlay is correct at the instant it is written and stale immediately after. We
watched it happen: twenty minutes after seeding seat #1, two passages had `cart_n` exactly
one higher than `seat_n` because the live cart kept bumping counters while we worked. Left
alone, every seat diverges from the moment of migration and the per-seat Hot Stack slowly
becomes fiction.

WHY IT IS NOT JUST "WRITE ON EVERY SEARCH"
-------------------------------------------
Seat #1's overlay is 1.94 MB. Writing it per search would put a multi-megabyte serialise on
the hot path of the most frequent operation in the system. So: bump in memory, flush on an
interval, and flush atomically -- the same shape `persist_access_stats` already uses, for the
same reason.

DUAL-WRITE, DELIBERATELY
------------------------
The cart's flat counters keep being bumped by their existing code. This module writes
ALONGSIDE them, never instead of them. Every unmigrated caller keeps working, the migration
stays reversible by deleting a directory, and the two can be compared for drift -- which is
how the missing scoring join got caught in the first place. Dropping the flat write is a
separate decision for a day when nothing reads it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import overlay as _ov

DEFAULT_FLUSH_INTERVAL = 300.0        # seconds; matches persist_access_stats' cadence
DEFAULT_SEAT = "andy"                 # single-seat compatibility until profiles land


class OverlayStore:
    """Thread-safe, lazily-loaded, periodically-flushed collection of seat overlays.

    One process may serve many seats, so everything here is keyed by seat and guarded. The
    lock is held only around dict mutation -- never across a file write -- because the whole
    point of this class is to keep I/O off the search path.
    """

    def __init__(self, directory, parent_cart: str = "",
                 flush_interval: float = DEFAULT_FLUSH_INTERVAL,
                 key_salt_id: str = ""):
        self.dir = Path(directory)
        self.parent_cart = parent_cart
        self.flush_interval = flush_interval
        self.key_salt_id = key_salt_id
        self._overlays: dict = {}
        self._dirty: set = set()
        # Per-seat mutation counter. A flush clears `dirty` ONLY if the version is unchanged
        # since the snapshot was taken -- otherwise a touch arriving mid-write would be
        # marked clean while never reaching disk, and would then survive only in RAM.
        self._version: dict = {}
        self._last_flush = time.time()
        self._lock = threading.RLock()
        self.stats = {"touches": 0, "flushes": 0, "flush_errors": 0,
                      "last_flush_at": 0.0, "last_error": ""}

    # ------------------------------------------------------------------ paths

    def path_for(self, seat: str) -> Path:
        safe = "".join(c for c in seat if c.isalnum() or c in "-_.@")
        if not safe:
            raise ValueError(f"seat name {seat!r} has no usable characters")
        return self.dir / f"{safe}.overlay.json"

    # ------------------------------------------------------------------ access

    def seats(self) -> list:
        """Every seat with an overlay on disk, plus any loaded but not yet flushed.

        Needed by the FIRM view: rolling up "what is this organisation warm on" requires
        enumerating seats, and until now the store only ever loaded one on demand. That gap
        is why `firm_hot_stack` shipped with no possible caller — the function existed but
        nothing could assemble its argument.
        """
        found = set()
        try:
            for path in self.dir.glob(f"*{'.overlay.json'}"):
                found.add(path.name[: -len(".overlay.json")])
        except OSError as e:
            self.stats["last_error"] = f"seats: {e}"
        with self._lock:
            found.update(self._overlays)
        return sorted(found)

    def get_overlay(self, seat: str) -> dict:
        """Load from disk on first use, create if absent. Never raises on a missing file."""
        with self._lock:
            o = self._overlays.get(seat)
            if o is not None:
                return o
        path = self.path_for(seat)
        try:
            loaded = _ov.load_overlay(path) if path.exists() else None
        except (ValueError, OSError) as e:
            # A corrupt overlay must not take down search. Start a fresh one and SAY SO --
            # silently continuing with an empty overlay would look identical to a new seat.
            self.stats["last_error"] = f"load {seat}: {e}"
            loaded = None
        if loaded is None:
            loaded = _ov.new_overlay(seat, self.parent_cart, self.key_salt_id)
        with self._lock:
            return self._overlays.setdefault(seat, loaded)

    # ------------------------------------------------------------------ writes

    def _mark_dirty(self, seat: str) -> None:
        """Caller must hold the lock."""
        self._dirty.add(seat)
        self._version[seat] = self._version.get(seat, 0) + 1

    def touch(self, seat: str, key: str, now: float = 0.0, count: int = 1) -> None:
        o = self.get_overlay(seat)
        with self._lock:
            _ov.touch(o, key, now=now, count=count)
            self._mark_dirty(seat)
            self.stats["touches"] += 1

    def touch_many(self, seat: str, keys, now: float = 0.0) -> None:
        """One search returns many hits; bump them under a single lock acquisition."""
        o = self.get_overlay(seat)
        now = now or time.time()
        with self._lock:
            for key in keys:
                _ov.touch(o, key, now=now)
            if keys:
                self._mark_dirty(seat)
                self.stats["touches"] += len(keys)

    def set_mark(self, seat: str, key: str, mark: str, on: bool = True) -> None:
        o = self.get_overlay(seat)
        with self._lock:
            _ov.set_mark(o, key, mark, on)
            self._mark_dirty(seat)

    def set_agency(self, seat: str, key: str, agency) -> None:
        o = self.get_overlay(seat)
        with self._lock:
            _ov.set_agency(o, key, agency)
            self._mark_dirty(seat)

    # ------------------------------------------------------------------ flush

    def due(self, now: float = 0.0) -> bool:
        return (now or time.time()) - self._last_flush >= self.flush_interval

    def flush(self, force: bool = False, now: float = 0.0) -> dict:
        """Write dirty overlays. Returns a report; NEVER raises into the caller.

        A failed flush leaves the seat dirty so the next attempt retries it, and increments
        a counter rather than logging once and moving on. `check_reload()` earned that
        lesson the hard way on 2026-07-31: an `except: pass` around a load meant the server
        served stale data indefinitely with nothing to show for it.
        """
        now = now or time.time()
        if not force and not self.due(now):
            return {"written": 0, "skipped": "not due"}

        # Snapshot UNDER the lock. Serialising the live overlay while another thread touches
        # it is a torn write at best and a "dictionary changed size during iteration" crash
        # at worst -- and the search path is threaded, so that is a when, not an if. Copying
        # ~20k small dicts costs single-digit milliseconds; holding the lock across the file
        # write would instead stall every search for the duration of a 2 MB serialise.
        failed = []
        with self._lock:
            pending = sorted(self._dirty)
            snapshots = {}
            for seat in pending:
                o = self._overlays.get(seat)
                if o is None:
                    continue
                try:
                    snapshots[seat] = (
                        self._version.get(seat, 0),
                        {**o, "entries": {k: dict(v) for k, v in o["entries"].items()}},
                    )
                except Exception as e:
                    # One malformed overlay must not stop every other seat from flushing.
                    failed.append(seat)
                    self.stats["flush_errors"] += 1
                    self.stats["last_error"] = f"snapshot {seat}: {e}"

        written = []
        for seat, (version, snapshot) in snapshots.items():
            try:
                _ov.save_overlay(snapshot, self.path_for(seat))
                written.append((seat, version))
            except (ValueError, OSError) as e:
                failed.append(seat)
                self.stats["flush_errors"] += 1
                self.stats["last_error"] = f"flush {seat}: {e}"

        with self._lock:
            for seat, version in written:
                # Cleared only if nothing changed while we were writing. A touch that raced
                # the write leaves the seat dirty so the next flush picks it up, instead of
                # living in RAM until the process exits.
                if self._version.get(seat, 0) == version:
                    self._dirty.discard(seat)
            self._last_flush = now
            self.stats["flushes"] += 1
            self.stats["last_flush_at"] = now

        return {"written": len(written), "failed": len(failed),
                "seats": [s for s, _ in written]}

    def close(self) -> dict:
        return self.flush(force=True)

    # ------------------------------------------------------------------ status

    def report(self) -> dict:
        with self._lock:
            return {
                "seats_loaded": len(self._overlays),
                "seats_dirty": len(self._dirty),
                "rows": {s: len(o["entries"]) for s, o in self._overlays.items()},
                "flush_interval_s": self.flush_interval,
                "seconds_since_flush": round(time.time() - self._last_flush, 1),
                **self.stats,
            }
