"""Proposals: what an agent thinks should change, without letting it change anything.

THE DECISION THIS IMPLEMENTS (2026-08-03)
-----------------------------------------
No destructive CRUD in membot. Three reasons, in ascending order of importance:

1. Appending is recoverable; deleting is not. The surface with the least human oversight
   should get only the recoverable verb.
2. Agents are steerable by their inputs. A poisoned document in a shared company brain can
   instruct an agent reading it to delete other documents -- which is why Membraine has a
   poison guard. If membot literally cannot delete, that attack has no teeth.
3. Andy: tight security is outside his wheelhouse. A capability whose safety only I can
   verify is a single point of failure, and that point is me.

So an agent's edit becomes a PROPOSAL. It says "this passage is out of date, here is the
correction"; a human accepts it in the studio, where there is a UI, a confirm, and an owner.
Worst case is "an agent filed a bad suggestion" rather than "an agent deleted the runbook".

A COMMENT IS STILL A WRITE -- and that is fine
----------------------------------------------
The line is not read-vs-write, it is DESTRUCTIVE vs ADDITIVE. A proposal only ever appends
to a sidecar, is fully reversible, and changes no passage. That is precisely what the
`commenter` access level already means: read + annotate, no write. The level existed in the
schema, the CHECK constraint and the UI badge since 2026-07-26 and did nothing until now.

CONTENT-KEYED
-------------
A proposal targets a passage by content key, so it survives a rebuild and follows an edit
chain. A proposal keyed by index would point at a stranger after the next reindex -- the
2026-07-29 failure, aimed at the one artifact whose entire job is to be reviewed by a human
later.

Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

SIDECAR_SUFFIX = ".proposals.json"
FORMAT = "vps-proposals"
VERSION = 1

# What an agent may propose. Deliberately small: each maps to something a human can act on
# in one click. "rewrite" is NOT here -- a proposal that replaces a whole passage wholesale
# is indistinguishable from an edit, and reviewing it is as much work as doing it.
VALID_KINDS = ("correction", "outdated", "duplicate", "context", "question")

STATUS_OPEN = "open"
STATUS_ACCEPTED = "accepted"
STATUS_REJECTED = "rejected"
VALID_STATUSES = (STATUS_OPEN, STATUS_ACCEPTED, STATUS_REJECTED)

MAX_BODY_CHARS = 2000


def sidecar_path(cart_path) -> Path:
    cart = Path(cart_path)
    return cart.with_name(cart.name + SIDECAR_SUFFIX)


def load(cart_path) -> dict:
    """All proposals. NEVER raises — a cart with none is the normal case."""
    path = sidecar_path(cart_path)
    empty = {"format": FORMAT, "version": VERSION, "proposals": [], "readable": True}
    if not path.exists():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        # "No proposals" and "we cannot read the proposals" are different facts. Only one
        # of them means an owner has nothing waiting.
        return {**empty, "readable": False, "error": str(e)}
    if data.get("format") != FORMAT:
        return {**empty, "readable": False,
                "error": f"unexpected format {data.get('format')!r}"}
    data.setdefault("proposals", [])
    data["readable"] = True
    return data


def _write(cart_path, proposals) -> None:
    path = sidecar_path(cart_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT, "version": VERSION, "proposals": proposals}
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


def propose(cart_path, target_key: str, kind: str, body: str, by: str = "",
            suggested_text: str = "", now: float = 0.0) -> dict:
    """File a proposal against a passage. Additive only; changes nothing.

    Returns the stored entry. Raises ValueError on bad input rather than storing junk an
    owner would have to wade through.
    """
    now = now or time.time()
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {VALID_KINDS}, got {kind!r}")
    if not isinstance(target_key, str) or len(target_key) < 8:
        raise ValueError("target_key must be a content key")
    body = (body or "").strip()
    if not body:
        raise ValueError("a proposal needs a body saying what is wrong and why")

    data = load(cart_path)
    if not data.get("readable", True):
        raise ValueError(f"refusing to append to an unreadable proposal file: "
                         f"{data.get('error')}")

    entry = {
        "id": "prp_" + hashlib.sha1(
            f"{target_key}\x00{by}\x00{now}\x00{body[:80]}".encode("utf-8")
        ).hexdigest()[:16],
        "target_key": target_key,
        "kind": kind,
        "body": body[:MAX_BODY_CHARS],
        "suggested_text": (suggested_text or "")[:MAX_BODY_CHARS],
        "by": by or "",
        "at": now,
        "status": STATUS_OPEN,
        "resolved_by": "",
        "resolved_at": 0.0,
        "resolution_note": "",
    }
    data["proposals"].append(entry)
    _write(cart_path, data["proposals"])
    return entry


def resolve(cart_path, proposal_id: str, status: str, by: str = "",
            note: str = "", now: float = 0.0) -> dict:
    """Accept or reject. RECORDS THE DECISION; never applies it.

    Accepting a proposal does NOT edit the passage. Applying is a separate, deliberate act
    through the studio's edit verb, which records succession. Conflating the two would put
    passage mutation back on the agent path by the back door -- the exact thing this module
    exists to prevent.
    """
    now = now or time.time()
    if status not in (STATUS_ACCEPTED, STATUS_REJECTED):
        raise ValueError(f"status must be accepted or rejected, got {status!r}")

    data = load(cart_path)
    if not data.get("readable", True):
        raise ValueError(f"unreadable proposal file: {data.get('error')}")

    for entry in data["proposals"]:
        if entry.get("id") == proposal_id:
            if entry.get("status") != STATUS_OPEN:
                raise ValueError(f"proposal {proposal_id} is already "
                                 f"{entry.get('status')} — decided by "
                                 f"{entry.get('resolved_by') or 'someone'}")
            entry["status"] = status
            entry["resolved_by"] = by or ""
            entry["resolved_at"] = now
            entry["resolution_note"] = (note or "")[:MAX_BODY_CHARS]
            _write(cart_path, data["proposals"])
            return entry
    raise KeyError(f"no proposal with id {proposal_id!r}")


def list_proposals(cart_path, status: str = STATUS_OPEN, target_key: str = "") -> list:
    """Proposals, filtered. `status=""` returns every one regardless of state."""
    data = load(cart_path)
    out = data.get("proposals", [])
    if status:
        out = [p for p in out if p.get("status") == status]
    if target_key:
        out = [p for p in out if p.get("target_key") == target_key]
    return sorted(out, key=lambda p: p.get("at", 0.0), reverse=True)


def for_results(cart_path, target_keys, status: str = STATUS_OPEN) -> dict:
    """`content_key -> [proposals]` for a set of search hits.

    This is what lets a proposal RIDE BACK WITH THE RESULTS instead of needing its own
    channel: whatever surface renders an agent's answer can render the open proposals on the
    passages that answer came from, at no extra round trip.
    """
    wanted = set(target_keys or [])
    if not wanted:
        return {}
    grouped: dict = {}
    for p in list_proposals(cart_path, status=status):
        key = p.get("target_key")
        if key in wanted:
            grouped.setdefault(key, []).append(p)
    return grouped


def stats(cart_path) -> dict:
    data = load(cart_path)
    proposals = data.get("proposals", [])
    counts: dict = {}
    kinds: dict = {}
    for p in proposals:
        counts[p.get("status", "?")] = counts.get(p.get("status", "?"), 0) + 1
        kinds[p.get("kind", "?")] = kinds.get(p.get("kind", "?"), 0) + 1
    return {
        "readable": data.get("readable", True),
        "total": len(proposals),
        "by_status": counts,
        "by_kind": kinds,
        "open": counts.get(STATUS_OPEN, 0),
        "proposers": sorted({p.get("by") for p in proposals if p.get("by")}),
    }
