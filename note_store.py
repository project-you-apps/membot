"""Sticky notes: the WHY layer. Every compliance tool records who/what/when; none record why.

THE GAP THIS FILLS (SPEC-sticky-notes-the-why-layer-2026-10-26)
---------------------------------------------------------------
`by` + `surface` + `at` is the write-attribution triple: WHO, WHERE, WHEN. It is also what
every audit log in the world already has. The question an auditor actually asks is "why did
you make that judgment", and that one stops being answerable the day the person leaves --
because WHY cannot be reconstructed afterward. It has to be captured at the moment of the
decision. A sticky note is that capture.

A sticky note is NOT a compliance feature with a general-purpose fallback. It is a general
cart annotation of which the SOX/FDICIA version is one configuration (Andy, 2026-10-25:
"I want it for us to use also... on entries in the session cart or anywhere else we have a
cart"). That reframe is what made it small.

FOUR DECISIONS THIS IMPLEMENTS, NOT UP FOR REVISITING
-----------------------------------------------------
1. Both per-seat AND shared -- but DEFAULT PRIVATE, and that is a product constraint rather
   than a default someone may flip. Andy: sharing is "a little like having your boss look
   over your shoulder". A shared "why" that is on by default is surveillance. A note is BORN
   private; `share()` is a second, explicit verb, and only the author may take it.
2. Follow the proposals precedent. Same sidecar shape, same content-keying, same
   `for_results` delivery, same atomic write, same validate-and-raise. One shape to learn.
3. INERT in v1. Nothing in the retrieval path reads a note, and nothing in this module
   touches scoring or ranking. Note-affects-ranking is a 1.5 feature that needs its own test
   block, because the moment a note scores it is a new failure surface to measure. There is
   deliberately no score, weight, rank or boost field anywhere in the record.
4. Nothing goes in the passage text. The sidecar IS the metadata layer, so legibility and
   filterability are both free and the cart format does not change.

CONTENT-KEYED, LIKE A PROPOSAL
------------------------------
`target_key` is the content key of the passage, never an index. A note keyed by index points
at a stranger after the next reindex -- the 2026-07-29 failure, and it would aim it at the
one record whose whole job is to be read by a human much later.

APPEND-ONLY. THERE IS NO DELETE.
--------------------------------
Appending is recoverable; deleting is not. A note is corrected by `supersede()`, which writes
a NEW note pointing at the old one; the original stays exactly as written. The only in-place
mutation in this module is the private->shared scope flip, which is the author's own act and
is itself attributed (`shared_by` / `shared_at`), mirroring `proposal_store.resolve()`.

THE SURVEILLANCE BOUNDARY, WHICH IS A READ-TIME RULE
----------------------------------------------------
A seat sees its own private notes plus every shared note. Never another seat's private note.
This is enforced in ONE place, `_visible()`, and every read path goes through it -- a filter
on `list_notes` with an unfiltered `for_results` would be the obvious bug and would leak one
seat's reasoning into another seat's search results.

An empty `seat` means an UNIDENTIFIED caller, and it sees shared notes only. It does not mean
"no filter". Fail closed: on a single-user desktop install there is no authentication to
stamp a seat from (spec §7.2), and the safe reading of "I do not know who you are" is never
"show me everyone's private reasoning".

Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

SIDECAR_SUFFIX = ".notes.json"
FORMAT = "vps-notes"
VERSION = 1

# What a note can be. Small on purpose, like proposal kinds: each is a distinct thing a
# reader does something different with.
#   why        -- the reasoning behind a judgment. The reason this module exists.
#   context    -- what was true at the time that the record does not say.
#   caution    -- a trap for the next reader.
#   decision   -- what was settled, so it is not re-litigated.
#   prediction -- an expectation recorded BEFORE the outcome, which makes it falsifiable and
#                 makes the wrong ones the informative ones. v1 stores it and does nothing
#                 with it; scoring a prediction against its outcome is 1.5 work.
# ⚠ A KIND IS PERMANENT ON EVERY NOTE THAT USED IT. Notes cannot be deleted or edited, so
# adding a kind later is cheap and removing one is not. Add deliberately.
#
# `recommendation` added 2026-11-07 (Andy). It pairs with `decision` without overlapping:
# decision records what WAS chosen, recommendation records what someone thinks SHOULD be.
# ⛔ It is deliberately NOT called `proposal`. That word is taken by proposal_store, which
# is a different and stronger thing -- an agent's suggested REPLACEMENT TEXT for a passage,
# with a resolution lifecycle (accept/reject, resolved_by). A note kind called `proposal`
# would be that mechanism with the workflow removed: the name would promise an accept/
# reject the note cannot deliver, and two things called "proposal" where one quietly does
# not work is worse than either alone.
# ⭐ A recommendation promises no workflow at all, which is why it is safe on an
# append-only record. Acting on one is a later `decision` note, optionally superseding it.
VALID_KINDS = ("why", "context", "caution", "decision", "prediction", "recommendation")

SCOPE_PRIVATE = "private"
SCOPE_SHARED = "shared"
VALID_SCOPES = (SCOPE_PRIVATE, SCOPE_SHARED)

# Where the write physically came from. Part of the attribution triple, and like `by` it is
# stamped from what the server verified, not from what the caller claimed.
VALID_SURFACES = ("CLI", "VSC", "WEB")

MAX_BODY_CHARS = 2000


def sidecar_path(cart_path) -> Path:
    cart = Path(cart_path)
    return cart.with_name(cart.name + SIDECAR_SUFFIX)


def load(cart_path) -> dict:
    """Every note on the cart, unfiltered. NEVER raises -- a cart with none is normal.

    `readable` is the load-bearing field. "This cart has no notes" and "we cannot read this
    cart's notes" are DIFFERENT FACTS, and only the first one means the reader has nothing
    waiting for them. Collapsing them would turn a corrupt sidecar into a silent, confident
    "no reasoning was recorded here" -- the worst possible lie for a provenance record.
    """
    path = sidecar_path(cart_path)
    empty = {"format": FORMAT, "version": VERSION, "notes": [], "readable": True}
    if not path.exists():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {**empty, "readable": False, "error": str(e)}
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        fmt = data.get("format") if isinstance(data, dict) else type(data).__name__
        return {**empty, "readable": False, "error": f"unexpected format {fmt!r}"}
    notes = data.get("notes")
    if not isinstance(notes, list):
        return {**empty, "readable": False, "error": "notes is not a list"}
    data["readable"] = True
    return data


def _write(cart_path, notes) -> None:
    """Atomic replace. A half-written sidecar reads as unreadable, which blocks appends, so
    the write must be all-or-nothing rather than merely usually fine."""
    path = sidecar_path(cart_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT, "version": VERSION, "notes": notes}
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        _replace_with_retry(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ⚠ WINDOWS, MEASURED 2026-11-01: `os.replace` fails intermittently with WinError 32
# ("the process cannot access the file because it is being used by another process") at
# roughly 1 run in 6. Nothing of ours holds the handle -- the file was just closed -- so
# the usual cause is a scanner or the search indexer opening the newly created temp file
# in the moment between close and replace. POSIX has no equivalent: a rename over an open
# file is legal there.
#
# ⛔ WHY IT IS WORTH HANDLING RATHER THAN TOLERATING. The caller sees a refused write, and
#   for this module a refused write means a note was NOT recorded. A note is the one copy
#   of reasoning nobody can reconstruct later, so losing one to a transient lock loses the
#   asset -- and it would look like the author never wrote it.
#
# ⓘ `proposal_store._write` and `membox` use this same mkstemp/replace shape and have the
#   same latent flake. Not changed here: they are pre-existing and out of today's scope,
#   but a reader hitting WinError 32 there should know why.
def _replace_with_retry(tmp, path, attempts: int = 5, delay: float = 0.02) -> None:
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Deliberately only PermissionError. A genuine problem -- a missing directory,
            # a read-only volume, a bad path -- must fail immediately rather than be
            # retried five times and then reported late.
            if i == attempts - 1:
                raise
            time.sleep(delay * (2 ** i))


def _load_writable(cart_path) -> dict:
    data = load(cart_path)
    if not data.get("readable", True):
        # Appending to a file we cannot parse would silently discard every note already in
        # it. Refuse instead; a human can look at the file.
        raise ValueError(f"refusing to append to an unreadable note file: {data.get('error')}")
    return data


def _check_identity(by: str, surface: str) -> tuple:
    """Validate the attribution stamp.

    ⛔ `by` and `surface` ARE ARGUMENTS, BUT THEY ARE NOT THE CALLER'S CHOICE. The server
    passes the seat it authenticated and the surface it served the request on; a note tool
    exposed to an agent must never let that agent name itself. "A tag the writer supplies is
    a claim" (CBP, 2026-10-25) -- and provenance filters downstream will read this field as
    fact, so a claim stored here becomes a fact later. All this function can do is refuse the
    empty and the unknown, so that a caller which forgot to stamp cannot produce an
    anonymous note that looks attributed.
    """
    by = (by or "").strip()
    if not by:
        raise ValueError("`by` must be the authenticated seat, stamped by the caller's "
                         "server -- an unattributed note is not a provenance record")
    surface = (surface or "").strip()
    if surface not in VALID_SURFACES:
        raise ValueError(f"surface must be one of {VALID_SURFACES}, got {surface!r}")
    return by, surface


def _check_target_key(target_key: str) -> str:
    # Length is the only cheap structural signal that this is a content key and not an
    # ordinal. "7" passes no test we could write; it fails this one.
    if not isinstance(target_key, str) or len(target_key.strip()) < 8:
        raise ValueError("target_key must be a CONTENT key, never an index -- an "
                         "index-keyed note points at a stranger after the next reindex")
    return target_key.strip()


def _check_body(body: str) -> str:
    """Refuse an over-long body; do not truncate it.

    DIVERGENCE FROM `proposal_store`, DELIBERATE: a proposal truncates at MAX_BODY_CHARS
    because its body is advisory prose a human reads beside `suggested_text`, which carries
    the payload. A note's body IS the payload -- it is the only copy of reasoning that cannot
    be reconstructed later. Silently cutting it at 2000 chars would store a mutilated record
    that still looks complete, and the part we dropped is the part nobody can rewrite.
    Refusing hands the problem back to a caller that can split the note or shorten it.
    """
    body = (body or "").strip()
    if not body:
        raise ValueError("a note needs a body -- an empty why is not a why")
    if len(body) > MAX_BODY_CHARS:
        raise ValueError(f"body is {len(body)} chars, over the {MAX_BODY_CHARS} cap; "
                         f"refusing rather than truncating the one copy of the reasoning")
    return body


def _mint_id(existing_ids, target_key: str, by: str, surface: str, now: float,
             body: str) -> str:
    base = f"{target_key}\x00{by}\x00{surface}\x00{now}\x00{body[:80]}"
    nid = "nte_" + hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    # Two notes with the same seat, target, body prefix and timestamp are possible when a
    # caller passes an explicit `now` (tests, imports, replays). A duplicate id would make
    # supersede() and share() address the wrong record, so salt until it is unique.
    salt = 0
    while nid in existing_ids:
        salt += 1
        nid = "nte_" + hashlib.sha1(
            f"{base}\x00{salt}".encode("utf-8")).hexdigest()[:16]
    return nid


def _visible(note: dict, seat: str) -> bool:
    """THE SURVEILLANCE BOUNDARY, in one place so there is one thing to get right.

    A seat sees its own private notes plus every shared note. An empty seat is an
    unidentified caller and sees shared notes only -- "I do not know who you are" must never
    widen to "here is everyone's private reasoning".
    """
    if note.get("scope") == SCOPE_SHARED:
        return True
    seat = (seat or "").strip()
    return bool(seat) and note.get("by") == seat


def add_note(cart_path, target_key: str, kind: str, body: str, by: str, surface: str,
             scope: str = SCOPE_PRIVATE, supersedes: str = "", now: float = 0.0) -> dict:
    """Attach a note to a passage by content key. Additive; changes no passage and no score.

    `scope` defaults to private, and that default is the product constraint -- a note is born
    private and becomes shared only when its author says so.

    Raises ValueError on bad input rather than storing junk a reader would have to wade
    through.
    """
    now = now or time.time()
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {VALID_KINDS}, got {kind!r}")
    if scope not in VALID_SCOPES:
        raise ValueError(f"scope must be one of {VALID_SCOPES}, got {scope!r}")
    target_key = _check_target_key(target_key)
    body = _check_body(body)
    by, surface = _check_identity(by, surface)

    data = _load_writable(cart_path)
    notes = data["notes"]

    supersedes = (supersedes or "").strip()
    if supersedes:
        prior = _find(notes, supersedes)
        if not _visible(prior, by):
            # Superseding a note you cannot see is not a correction, it is a write into
            # someone else's private reasoning.
            raise PermissionError(f"seat {by!r} cannot supersede note {supersedes!r}")

    entry = {
        "id": _mint_id({n.get("id") for n in notes}, target_key, by, surface, now, body),
        "target_key": target_key,
        "kind": kind,
        "body": body,
        "scope": scope,
        "by": by,
        "surface": surface,
        "at": now,
        "supersedes": supersedes,
        # Attribution for the sharing ACT itself, empty until share() is called. Not in the
        # spec's §3 record; added for the same reason proposal_store carries resolved_by --
        # a state change a human can take is a state change someone will later ask about.
        "shared_by": "",
        "shared_at": 0.0,
    }
    notes.append(entry)
    _write(cart_path, notes)
    return entry


def _find(notes, note_id: str) -> dict:
    for n in notes:
        if n.get("id") == note_id:
            return n
    raise KeyError(f"no note with id {note_id!r}")


def supersede(cart_path, note_id: str, body: str, by: str, surface: str,
              now: float = 0.0) -> dict:
    """Correct a note by writing a NEW one that points at it. Nothing is deleted or edited.

    The original stays byte-identical, because what someone believed at the time is itself
    the record -- a why layer that quietly rewrites its own history is worth nothing to an
    auditor. The correction inherits the original's kind and target, and its SCOPE: a
    correction to a shared statement is shared, and a correction to a private note stays
    private, so superseding can never widen visibility by accident.
    """
    now = now or time.time()
    by, surface = _check_identity(by, surface)
    body = _check_body(body)

    data = _load_writable(cart_path)
    prior = _find(data["notes"], note_id)
    if not _visible(prior, by):
        raise PermissionError(f"seat {by!r} cannot supersede note {note_id!r}")

    return add_note(cart_path, target_key=prior["target_key"], kind=prior.get("kind", "why"),
                    body=body, by=by, surface=surface,
                    scope=prior.get("scope", SCOPE_PRIVATE), supersedes=note_id, now=now)


def share(cart_path, note_id: str, by: str, now: float = 0.0) -> dict:
    """Promote a private note to shared. ONLY the author may do this.

    This is the second explicit verb from decision 1. Sharing is an act the author takes, and
    it is specifically NOT a setting another seat -- or an admin, or a background job -- can
    flip on their behalf. If anyone else could make a private note shared, "default private"
    would be decoration, and the boss-over-the-shoulder failure would be one API call away.
    """
    now = now or time.time()
    by = (by or "").strip()
    if not by:
        raise ValueError("`by` must be the authenticated seat")

    data = _load_writable(cart_path)
    entry = _find(data["notes"], note_id)

    if entry.get("by") != by:
        raise PermissionError(
            f"only the author may share a note: {note_id!r} belongs to "
            f"{entry.get('by')!r}, not {by!r}")
    if entry.get("scope") == SCOPE_SHARED:
        raise ValueError(f"note {note_id!r} is already shared")

    entry["scope"] = SCOPE_SHARED
    entry["shared_by"] = by
    entry["shared_at"] = now
    _write(cart_path, data["notes"])
    return entry


def superseded_ids(notes) -> set:
    """Ids that some later note corrects. Derived, never stored -- a stored flag on the old
    note would mean editing it, and the old note is never edited."""
    return {n.get("supersedes") for n in notes if n.get("supersedes")}


def list_notes(cart_path, seat: str = "", scope: str = "", target_key: str = "") -> list:
    """Notes this seat is allowed to see, newest first.

    Visibility is applied FIRST and always; `scope` then narrows what is already visible, so
    `scope="private"` means "my private notes", never "everyone's". Ordering is strictly by
    `at` -- chronological, never ranked. Nothing here scores anything.

    This is the audit view, so it includes superseded notes: the wrong version and its
    correction are both part of the record.
    """
    data = load(cart_path)
    out = [n for n in data.get("notes", []) if _visible(n, seat)]
    if scope:
        if scope not in VALID_SCOPES:
            raise ValueError(f"scope must be one of {VALID_SCOPES} or '', got {scope!r}")
        out = [n for n in out if n.get("scope") == scope]
    if target_key:
        out = [n for n in out if n.get("target_key") == target_key]
    return sorted(out, key=lambda n: n.get("at", 0.0), reverse=True)


def for_results(cart_path, target_keys, seat: str = "",
                include_superseded: bool = False) -> dict:
    """`content_key -> [notes]` for a set of search hits.

    This is what lets a note RIDE BACK WITH THE RESULTS instead of needing its own channel:
    whatever surface renders an answer can render the reasoning recorded on the passages that
    answer came from, at no extra round trip.

    ⛔ It filters by the SAME `_visible()` rule as `list_notes`. A delivery path that skipped
    the check would hand one seat's private reasoning to another inside their search results
    -- quieter than a leak through a list call, and worse.

    Superseded notes are omitted by default: on delivery the reader wants the current why,
    not the one that was corrected. `list_notes` still shows both.

    ⛔ Returns metadata only. Nothing in this dict is read by scoring or ranking in v1.
    """
    wanted = set(target_keys or [])
    if not wanted:
        return {}
    data = load(cart_path)
    notes = data.get("notes", [])
    stale = set() if include_superseded else superseded_ids(notes)
    grouped: dict = {}
    for n in sorted(notes, key=lambda n: n.get("at", 0.0), reverse=True):
        if n.get("target_key") not in wanted:
            continue
        if not _visible(n, seat):
            continue
        if n.get("id") in stale:
            continue
        grouped.setdefault(n["target_key"], []).append(n)
    return grouped


def stats(cart_path) -> dict:
    """Counts over EVERY note, for an owner looking at a cart. Not a read path for a seat --
    it reports no bodies and no targets, so it cannot be used to peek at private reasoning.
    """
    data = load(cart_path)
    notes = data.get("notes", [])
    kinds: dict = {}
    scopes: dict = {}
    surfaces: dict = {}
    for n in notes:
        kinds[n.get("kind", "?")] = kinds.get(n.get("kind", "?"), 0) + 1
        scopes[n.get("scope", "?")] = scopes.get(n.get("scope", "?"), 0) + 1
        surfaces[n.get("surface", "?")] = surfaces.get(n.get("surface", "?"), 0) + 1
    return {
        "readable": data.get("readable", True),
        "error": data.get("error", ""),
        "total": len(notes),
        "by_kind": kinds,
        "by_scope": scopes,
        "by_surface": surfaces,
        "private": scopes.get(SCOPE_PRIVATE, 0),
        "shared": scopes.get(SCOPE_SHARED, 0),
        "superseded": len(superseded_ids(notes)),
        "targets": len({n.get("target_key") for n in notes if n.get("target_key")}),
        "authors": sorted({n.get("by") for n in notes if n.get("by")}),
    }
