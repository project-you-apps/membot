"""The changes sidecar: an append-only log of what replaced what.

WHY A SIDECAR AND NOT A FIELD ON THE PASSAGE
--------------------------------------------
`subcart.record_succession()` writes `superseded_by_key` into a link's `_reserved.canonical`.
That works for carts that HAVE links — the session cart does. Studio carts do not: their pkl
holds `version`, `embeddings`, `passages`, `compressed_lens` and nothing else.

So the product that actually edits passages had nowhere to record an edit. Adding links to
the studio cart format would be a format change, a canon update and a fleet notice; a sidecar
is none of those, and Andy had already pointed at it: *"carts are several files in one so they
can carry a 'changes' sidecar."*

A LOG, NOT A MAP
----------------
The resolver only needs `old_key -> new_key`. Storing a log instead costs a few bytes per
edit and buys the change history for free — who changed what, when, and what the text was
before. That is the difference between "your cluster followed the edit" and "your cluster
followed the edit, here is what changed and who did it", which is the whole reason anyone
wants an audit trail.

Entry N of the log also lines up with generation N from `cart_generation`, so the two
sidecars read together as one history.

CONTENT-KEYED, LIKE EVERYTHING ELSE
-----------------------------------
Keys, never indices. A rebuild renumbers passages and would silently repoint every entry —
which on 2026-07-29 put five thread flags on five unrelated passages.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import subcart as _sc

SIDECAR_SUFFIX = ".succession.json"
FORMAT = "vps-succession"
VERSION = 1


def sidecar_path(cart_path) -> Path:
    cart = Path(cart_path)
    return cart.with_name(cart.name + SIDECAR_SUFFIX)


def load_log(cart_path) -> dict:
    """Full sidecar. NEVER raises — a cart with no edit history is the normal case."""
    path = sidecar_path(cart_path)
    empty = {"format": FORMAT, "version": VERSION, "entries": [], "readable": True}
    if not path.exists():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        # An unreadable history must not make the cart unusable, but it must NOT look like
        # an empty one either — "no edits ever happened" and "we cannot read what happened"
        # are different facts, and only one of them is safe to act on.
        return {**empty, "readable": False, "error": str(e)}
    if data.get("format") != FORMAT:
        return {**empty, "readable": False, "error": f"unexpected format {data.get('format')!r}"}
    data.setdefault("entries", [])
    data["readable"] = True
    return data


def build_map(cart_path, salt: str = "") -> dict:
    """`old_key -> new_key`, ready to hand to `subcart.resolve(succession=…)`.

    Later entries win: editing a passage twice should land a reader on the newest version,
    and the chain-walk in `follow_succession` handles the multi-hop case anyway.
    """
    log = load_log(cart_path)
    mapping = {}
    for entry in log.get("entries", []):
        old_key, new_key = entry.get("old_key"), entry.get("new_key")
        if isinstance(old_key, str) and isinstance(new_key, str) and old_key != new_key:
            mapping[old_key] = new_key
    return mapping


def record(cart_path, old_text: str, new_text: str, seat: str = "",
           note: str = "", salt: str = "", now: float = 0.0,
           old_preview_chars: int = 240) -> dict:
    """Append one edit to the log. Returns the entry written.

    Stores a PREVIEW of the old text, not the whole thing. A full copy would double the cart
    on a heavily-edited corpus, and the point of the preview is to let a human recognise what
    changed — not to be a backup. Recovery of the full prior text is what a materialised fork
    is for.
    """
    now = now or time.time()
    old_key = _sc.content_key(old_text, salt)
    new_key = _sc.content_key(new_text, salt)
    if old_key == new_key:
        raise ValueError("old and new text are identical — nothing was superseded")

    log = load_log(cart_path)
    if not log.get("readable", True):
        # Appending to a history we cannot read would silently discard whatever is in it.
        raise ValueError(f"refusing to append to an unreadable succession log: "
                         f"{log.get('error')}")

    entry = {
        "old_key": old_key,
        "new_key": new_key,
        "at": now,
        "by": seat or "",
        "note": note or "",
        "old_preview": (old_text or "")[:old_preview_chars],
    }
    log["entries"].append(entry)

    path = sidecar_path(cart_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT, "version": VERSION, "entries": log["entries"]}
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return entry


def history_for(cart_path, key: str, salt: str = "") -> list:
    """Every edit in this passage's chain, oldest first — for an audit or a UI timeline."""
    log = load_log(cart_path)
    by_old = {e["old_key"]: e for e in log.get("entries", []) if e.get("old_key")}
    chain, seen, current = [], {key}, key
    while current in by_old:
        entry = by_old[current]
        chain.append(entry)
        current = entry.get("new_key")
        if not current or current in seen:
            break                      # cycle guard: a revert makes A -> B -> A real
        seen.add(current)
    return chain


def stats(cart_path) -> dict:
    log = load_log(cart_path)
    entries = log.get("entries", [])
    editors = {e.get("by") for e in entries if e.get("by")}
    return {
        "readable": log.get("readable", True),
        "edits": len(entries),
        "editors": sorted(editors),
        "first_at": entries[0]["at"] if entries else 0.0,
        "last_at": entries[-1]["at"] if entries else 0.0,
    }
