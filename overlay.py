"""Per-seat attention overlay: what ONE seat has touched in a shared cart.

Implements step 3 of `docs/SPEC-SUBCARTRIDGE-MANIFEST.md` §9.

THE SPLIT THIS ENFORCES
-----------------------
    content properties  -> the cart, shared by everyone
        text, embedding, duplicate_of_key, provenance, kind
    attention properties -> here, one file per seat
        access_count, accessed_at, ACTIVE/THREAD/IMP, actionable status

Before this module, everything in the second list lived in the first: `access_count` and
`accessed_at` are single integers stored flat on each link in memory_server.py. The Hot Stack
was not shared-by-design, it was single-tenant by construction -- invisible because there had
only ever been one seat.

WHY THIS MAKES WRITES CHEAPER, NOT MORE EXPENSIVE
-------------------------------------------------
Counter-intuitive and load-bearing. Today every search bumps a counter inside the one shared
cart, which is the single-writer contention that has bitten us repeatedly. Per-seat overlays
are separate files: 200 seats write 200 different places and never contend. Sharding the
hottest write path in the system falls out of the privacy requirement for free.

ENTRIES DUCK-TYPE AS LINKS ON PURPOSE
-------------------------------------
`as_link()` emits `{"access_count", "accessed_at", ...}` so `compute_hot_score()` scores an
overlay entry with NO change to the scoring code. If the shapes had diverged, we would be
maintaining two decay models that must agree -- the same setup that made two copies of
content_key worth a dedicated parity test.

Stdlib only. Both the VPS API and the Membot server import it.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

FORMAT = "vps-overlay"
VERSION = 1

# READ OFF THE LIVE 38,655-LINK CART, NOT INVENTED. The first draft of this module guessed
# ("actionable", "blocked", "done", "parked") -- inventing `parked` and omitting `reflective`,
# which is the single most common value in the cart (19,788). The vocabulary was available to
# read the whole time.
VALID_AGENCY = ("reflective", "actionable", "done", "blocked")

# When several links share one content_key (44% of the real cart is multi-sourced) they can
# disagree about agency -- the same body marked `actionable` on one copy and `done` on
# another. Silent last-writer-wins was the first implementation and it is exactly the class
# of bug this project keeps paying for, so the merge is declared instead:
#
#   blocked > actionable > done > reflective
#
# Rationale: surfacing something already finished costs a glance; hiding something still
# actionable costs the work. `blocked` outranks everything because it is the value that
# carries information you cannot reconstruct by looking at the passage.
AGENCY_PRECEDENCE = {"blocked": 3, "actionable": 2, "done": 1, "reflective": 0}


def merge_agency(a, b):
    """Highest-precedence value wins. Either argument may be None."""
    if not a:
        return b
    if not b:
        return a
    return a if AGENCY_PRECEDENCE.get(a, 0) >= AGENCY_PRECEDENCE.get(b, 0) else b

# WHICH OF THE EXISTING PER-LINK FIELDS ARE ATTENTION, AND WHICH ARE CONTENT
# -------------------------------------------------------------------------
# Of `_reserved.flags` (the 8 hippocampus dimensions) SEVEN describe the passage itself --
# modality, temporal, spatial, trust, origin, relational, salience -- and stay in the cart,
# shared. Only `agency` is "what should be done about this", which is per-seat: what is
# actionable for Andy is not actionable for Dennis.
#
# Of `_reserved.canonical`: truth_status / kind / duplicate_of_key are properties of the
# CONTENT and stay global. important / urgent / to_consolidate are one person's judgement
# and belong here.
ATTENTION_BOOLS = ("important", "urgent", "to_consolidate")

# Compact on-disk keys. An overlay is one row per touched passage and is rewritten often;
# "access_count" costs 6x "n" across tens of thousands of rows for no added clarity on disk.
_N = "n"        # access_count
_T = "t"        # accessed_at  (epoch)
_A = "a"        # agency
_K = "k"        # attention booleans, as a sorted list of the ones that are True
_B = "b"        # born -- first time THIS SEAT touched it


def new_overlay(seat: str, parent_cart: str, key_salt_id: str = "") -> dict:
    return {
        "format": FORMAT,
        "version": VERSION,
        "seat": seat,
        "parent": {"cart": parent_cart, "key_salt_id": key_salt_id},
        "entries": {},
        "created_at": time.time(),
    }


def validate_overlay(overlay: dict) -> list:
    """Return a list of problems; empty means valid. Never raises on bad input."""
    problems = []
    if not isinstance(overlay, dict):
        return ["overlay is not an object"]
    if overlay.get("format") != FORMAT:
        problems.append(f"format must be {FORMAT!r}, got {overlay.get('format')!r}")
    if overlay.get("version") != VERSION:
        problems.append(f"unsupported version {overlay.get('version')!r}")
    if not overlay.get("seat"):
        problems.append("missing seat")
    if not isinstance(overlay.get("entries"), dict):
        problems.append("entries must be an object keyed by content_key")
    parent = overlay.get("parent")
    if not isinstance(parent, dict) or not parent.get("cart"):
        problems.append("parent.cart is required")
    if isinstance(parent, dict) and "key_salt" in parent:
        problems.append(
            "parent.key_salt must NEVER appear in an overlay -- an overlay outlives access, "
            "so shipping the salt in it hands a revoked seat the confirmation oracle."
        )
    return problems


# ------------------------------------------------------------------ mutation

def touch(overlay: dict, key: str, now: float = 0.0, count: int = 1) -> dict:
    """Record that this seat accessed `key`. Creates the entry on first touch."""
    now = now or time.time()
    entry = overlay["entries"].get(key)
    if entry is None:
        entry = {_N: 0, _T: 0.0, _B: now}
        overlay["entries"][key] = entry
    entry[_N] = entry.get(_N, 0) + count
    entry[_T] = now
    return entry


def _ensure(overlay: dict, key: str, now: float = 0.0) -> dict:
    """Marking something is itself an act of attention -- birth the row, never drop it."""
    entry = overlay["entries"].get(key)
    if entry is None:
        entry = {_N: 0, _T: 0.0, _B: now or time.time()}
        overlay["entries"][key] = entry
    return entry


def set_mark(overlay: dict, key: str, mark: str, on: bool = True,
             now: float = 0.0) -> dict:
    """Set one of the per-seat attention booleans (important / urgent / to_consolidate)."""
    if mark not in ATTENTION_BOOLS:
        raise ValueError(f"mark must be one of {ATTENTION_BOOLS}, got {mark!r}")
    entry = _ensure(overlay, key, now)
    marks = set(entry.get(_K, []))
    marks.add(mark) if on else marks.discard(mark)
    if marks:
        entry[_K] = sorted(marks)
    else:
        entry.pop(_K, None)
    return entry


def set_agency(overlay: dict, key: str, agency, now: float = 0.0) -> dict:
    """Set this seat's disposition toward a passage. `None` clears it."""
    if agency is not None and agency not in VALID_AGENCY:
        raise ValueError(f"agency must be one of {VALID_AGENCY} or None, got {agency!r}")
    entry = _ensure(overlay, key, now)
    if agency is None:
        entry.pop(_A, None)
    else:
        entry[_A] = agency
    return entry


# ------------------------------------------------------------------ reading

def as_link(entry: dict) -> dict:
    """Expand a compact entry into the link shape `compute_hot_score()` already consumes."""
    marks = set(entry.get(_K, []))
    link = {
        "access_count": entry.get(_N, 0),
        "accessed_at": entry.get(_T, 0.0),
        "agency": entry.get(_A),
        "born_at": entry.get(_B, 0.0),
    }
    link.update({m: (m in marks) for m in ATTENTION_BOOLS})
    return link


def get(overlay: dict, key: str) -> dict:
    entry = overlay["entries"].get(key)
    return as_link(entry) if entry else as_link({})


def links_for(overlay: dict, keys) -> list:
    """Link-shaped rows for `keys`, in the order given. Untouched keys score as cold, not
    as missing -- a passage this seat has never opened is a real state, not an error."""
    return [get(overlay, k) for k in keys]


def stats(overlay: dict) -> dict:
    entries = overlay["entries"]
    agency = {}
    for e in entries.values():
        a = e.get(_A)
        if a:
            agency[a] = agency.get(a, 0) + 1
    return {
        "seat": overlay.get("seat"),
        "rows": len(entries),
        "touched": sum(1 for e in entries.values() if e.get(_N, 0) > 0),
        "marked": sum(1 for e in entries.values() if e.get(_K)),
        "total_accesses": sum(e.get(_N, 0) for e in entries.values()),
        "agency": agency,
    }


# ------------------------------------------------------------------ firm roll-up

def rollup(overlays, now: float = 0.0) -> dict:
    """Aggregate N seat overlays into the FIRM-LEVEL attention signal.

    Andy, 2026-08-02: the company brain "keeps an episodic record of the firm's evolution
    over time as an emergent property of the Hot Stack system." Fifty personal overlays do
    NOT sum to an institutional record unless the roll-up is deliberately kept -- per-seat
    attention alone silently deletes the firm's memory of itself.

    `seats` (how many distinct people touched a passage) is the genuinely new signal here.
    One person reading something 40 times and 40 people reading it once are the same
    `access_count` and very different facts about a company.

    LEAK NOTE: this aggregate is computed firm-wide and masked at PRESENTATION, so a viewer's
    company-view ranking is shaped by items they cannot see. Accepted for the company view
    specifically, because being firm-wide is that view's entire purpose. Personal Hot Stacks
    must be scored over the seat's resolved set only -- never through this function.
    """
    now = now or time.time()
    merged: dict = {}
    for overlay in overlays:
        seat = overlay.get("seat")
        for key, entry in overlay.get("entries", {}).items():
            agg = merged.get(key)
            if agg is None:
                agg = {_N: 0, _T: 0.0, _B: entry.get(_B, now), "seats": set()}
                merged[key] = agg
            agg[_N] += entry.get(_N, 0)
            agg[_T] = max(agg[_T], entry.get(_T, 0.0))
            agg[_B] = min(agg[_B], entry.get(_B, now) or now)
            if entry.get(_N, 0) > 0 or entry.get(_K) or entry.get(_A):
                agg["seats"].add(seat)

    return {
        "format": FORMAT,
        "version": VERSION,
        "seat": "__firm__",
        "entries": {k: {_N: v[_N], _T: v[_T], _B: v[_B], "seats": len(v["seats"])}
                    for k, v in merged.items()},
        "computed_at": now,
    }


# ------------------------------------------------------------------ persistence

def save_overlay(overlay: dict, path) -> None:
    """Atomic write. A torn overlay is a seat's whole history, and we have already lost a
    cart to a crash mid-write (droplet OOM, 2026-07)."""
    problems = validate_overlay(overlay)
    if problems:
        raise ValueError("refusing to write invalid overlay: " + "; ".join(problems))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(overlay, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_overlay(path) -> dict:
    overlay = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate_overlay(overlay)
    if problems:
        raise ValueError(f"invalid overlay {path}: " + "; ".join(problems))
    return overlay


# ------------------------------------------------------------------ migration

def seed_from_links(seat: str, parent_cart: str, texts, links,
                    key_fn, key_salt: str = "", key_salt_id: str = "") -> dict:
    """Build seat #1's overlay from the attention already sitting flat in a cart.

    PURELY ADDITIVE -- reads the cart, writes a new file, removes nothing. The existing flat
    counters keep working for every caller that has not migrated, so this can land without a
    coordinated switchover.

    Passages with no attention at all are SKIPPED rather than written as zero rows: an
    overlay is sparse by design, and 38,650 zero rows would make the common case expensive to
    prove nothing about.

    Duplicate content keys are SUMMED, not overwritten. 44% of our cart is multi-sourced --
    the same body arriving from several syncs -- and taking the last writer would silently
    discard real attention against the earlier copies.
    """
    overlay = new_overlay(seat, parent_cart, key_salt_id)
    entries = overlay["entries"]

    for text, link in zip(texts, links):
        if not isinstance(link, dict):
            continue
        reserved = link.get("_reserved") or {}
        canonical = reserved.get("canonical") or {}

        count = link.get("access_count", 0) or 0
        seen = link.get("accessed_at", 0.0) or 0.0
        agency = (reserved.get("flags") or {}).get("agency")
        marks = sorted(m for m in ATTENTION_BOOLS if canonical.get(m))

        # `reflective` is the resting value on 19,788 links and carries no judgement, so it
        # is not on its own evidence that this seat ever engaged with the passage. Treating
        # it as attention would make the overlay dense -- the one thing it must not be.
        meaningful_agency = agency if agency in VALID_AGENCY and agency != "reflective" else None
        if not count and not seen and not marks and not meaningful_agency:
            continue

        key = key_fn(text, key_salt) if key_salt else key_fn(text)
        entry = entries.get(key)
        if entry is None:
            entry = {_N: 0, _T: 0.0, _B: seen}
            entries[key] = entry
        entry[_N] += count
        entry[_T] = max(entry.get(_T, 0.0), seen)
        if not entry.get(_B) or (seen and seen < entry[_B]):
            entry[_B] = seen
        if marks:
            entry[_K] = sorted(set(entry.get(_K, [])) | set(marks))
        if meaningful_agency:
            existing = entry.get(_A)
            if existing and existing != meaningful_agency:
                overlay["merge_conflicts"] = overlay.get("merge_conflicts", 0) + 1
            entry[_A] = merge_agency(existing, meaningful_agency)
    return overlay
