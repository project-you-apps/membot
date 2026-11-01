"""Subscriptions: "what changed on the things I watch", with the WHY attached.

THE SCOPE DECISION THIS IMPLEMENTS (2026-11-01)
-----------------------------------------------
The customer requirement is "notify subscribers when and where a document changed." That
sentence contains three mechanisms and only two of them are cheap:

    a subscription registry   -> this file
    change detection          -> already exists; it is `journal.py`
    a delivery channel        -> email/webhook/daemon. BLOCKED, and deliberately absent.

⛔ THERE IS NO DELIVERY CHANNEL IN THIS MODULE, ON PURPOSE. No mailer, no webhook, no
   poller, no scheduler, no background thread. Everything here is PULL: a seat asks, and
   gets an answer computed at the moment it asked. Adding "just a small email send" here
   would reopen the mail-sender question this scope decision was taken to avoid, and it
   would do it inside the one module whose correctness is a privacy boundary.

⭐⭐ AND THE DIFFERENTIATOR WAS NEVER THE NOTIFICATION -- IT IS THE PAYLOAD.
   "File changed, line 40" is commodity. Every document system diffs, and a diff says
   THAT something changed, never WHY. The payload nobody else can assemble is:

       "This form was superseded, here is what it descends from, who vouched for the
        replacement -- and whether that identity was VERIFIED or merely claimed -- and
        why the old one stopped being current."

   The first three clauses come out of the hash-chained journal. The last one comes out of
   the sticky notes, which is the only place a "why" is recorded at the moment of the
   decision rather than reconstructed after the author has left. `what_changed()` is the
   function that joins them, and it is the only reason this file is worth writing.

⇒ So this is the PULL half of the requirement: it carries all of the differentiating value
  and none of the blocked parts. A push loop on top of it is later, somebody else's call,
  and is a transport -- not a redesign of anything in here.

WHY NO DIFF ENGINE
------------------
`journal.py` landed as an append-only, hash-chained ledger of every mutation that routes
through the one door. That is the change feed. Diffing two cart snapshots to discover a
change we already recorded would be slower, lossier (a diff cannot tell a tombstone from a
deletion, or name who did it), and would reintroduce the thing the journal exists to
remove: a second, unchained account of history.

CONTENT-KEYED, NEVER INDEX-KEYED
--------------------------------
A subscription targets a passage by CONTENT KEY. An index-keyed subscription points at a
stranger after the next reindex -- the 2026-07-29 failure -- and here it would do it to a
record whose entire job is to tell someone their document changed. The worst version of
that bug is not "no notification": it is a confident notification about the wrong document.

⚠ THE SEAM THIS LEAVES, STATED RATHER THAN HIDDEN. The journal's `target` today is
  `{"cart": <name>, "idx": <n>}` -- an INDEX, because the mutation sites know the index and
  not the key. A subscription is a KEY. Bridging the two requires the cart's texts, which
  this module deliberately does not load (stdlib only, no pickle, no embeddings). So
  `what_changed` accepts an optional `key_resolver` from the caller that does have them,
  and when it is absent it SAYS SO in the result (`unresolved_index_events`,
  `notices`) instead of silently returning zero rows for a passage subscription. "A verdict
  on an empty result set" is the failure mode that would otherwise live here, and it would
  read as "nothing changed on your document."

APPEND-ONLY WHERE IT CAN BE
---------------------------
`unsubscribe` flips `active` to False with attribution; it does not remove the row. Who
stopped watching what, and when, is exactly the sort of fact an auditor asks about after an
incident, and it costs a few bytes to keep.

`mark_seen` MOVES FORWARD ONLY. A rewound marker would silently re-hide changes somebody
already looked at and answered for -- it is the one mutation here that could destroy
information, so it is refused rather than applied, and the refusal is visible in the return
value.

PER SEAT, NEVER GLOBAL
----------------------
The high-water mark is keyed by seat because two people have different last-looks. One
global mark means the second person to look is told nothing changed.

THE PRIVACY BOUNDARY, WHICH IS THE MOST LIKELY BUG IN THIS FILE
---------------------------------------------------------------
A subscription must never become a way to read another seat's private notes. `what_changed`
reaches the notes ONLY through `note_store.for_results(..., seat=...)` and
`note_store.list_notes(seat=...)`, both of which apply `note_store._visible()`. This module
never opens the notes sidecar itself, so there is no code path here that COULD widen
visibility. That is a structural choice, not a careful one: careful decays.

⛔ A consequence, and it is the right one: when an event's only handle on a passage is a
   note id for a note this seat cannot see, the event stays UNRESOLVED rather than being
   keyed through an unfiltered read. Failing to match is a missing notification; matching by
   reading somebody's private note is a leak.

INTEGRITY IS NOT A FOOTNOTE
---------------------------
If `journal.verify()` reports a broken chain, `what_changed` does not hand back rows. A
change read out of a journal that may have been edited is not evidence, and a caller that
cannot tell the difference will render it as though it were. The rows are WITHHELD with a
count, the verdict line leads with the break, and a caller that genuinely wants forensics
passes `allow_broken_chain=True` and gets rows that each carry `trustworthy: False`.

A torn tail is the exception, because it is the one break a power failure causes and
`journal.replay` already tolerates it. That case is reported as degraded, not as tampering.

Stdlib only. Sidecar at `<cart>.subscriptions.json`, beside the other sidecars.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import journal as _journal
import note_store as _notes

SIDECAR_SUFFIX = ".subscriptions.json"
FORMAT = "vps-subscriptions"
VERSION = 1

SCOPE_PASSAGE = "passage"
SCOPE_CART = "cart"
VALID_SCOPES = (SCOPE_PASSAGE, SCOPE_CART)

# A seat becomes a dict key, a sidecar field and a log line. Same bound membot_server and
# seat_identity put on it, for the same reasons.
MAX_SEAT_CHARS = 128

# How many changes one call will render. A rendered change carries notes and provenance, so
# an unbounded answer is both a context bomb for an agent caller and a latency surprise for
# a human one. Truncation is REPORTED, never silent.
DEFAULT_LIMIT = 500
MAX_LIMIT = 5000

# `since_seq=FROM_BEGINNING` means "ignore my high-water mark and replay everything". A
# plain 0 means "use my mark", which is the thing a client should not have to remember.
FROM_BEGINNING = -1

# Payload/target fields that, when present, already carry a content key. Checked in this
# order. The journal's vocabulary is closed but its payloads are not, so this is written to
# accept the key under whichever name a mutation site used rather than to mandate one.
_KEY_FIELDS = ("target_key", "content_key", "key", "passage_key")

# Payload/target fields naming what an entry supersedes. Same reasoning.
_SUPERSEDES_KEY_FIELDS = ("superseded_key", "supersedes_key", "old_key", "from_key",
                          "replaces_key", "previous_key")
_SUPERSEDES_ID_FIELDS = ("supersedes", "superseded_id", "replaces", "supersedes_note_id")

# Ops that change what a document IS or who may see it. Everything in the journal is a
# mutation already (`_journal_event` is mutations-only by design), so this is not a filter --
# it is a labeling aid for the headline, which is why there is no "other ops are dropped"
# branch anywhere below.
_OP_PHRASE = {
    "store": "stored",
    "tombstone": "tombstoned",
    "corroborate": "corroborated",
    "supersede": "superseded",
    "perm_change": "permissions changed",
    "flag": "flagged",
    "note": "annotated",
}


# ---------------------------------------------------------------------------
# paths and sidecar io
# ---------------------------------------------------------------------------

def sidecar_path(cart_path) -> Path:
    cart = Path(cart_path)
    return cart.with_name(cart.name + SIDECAR_SUFFIX)


def load(cart_path) -> dict:
    """Every subscription and mark on the cart, unfiltered. NEVER raises.

    `readable` is the load-bearing field, for the same reason it is in `note_store.load`:
    "this cart has nobody watching it" and "we cannot read who is watching it" are
    DIFFERENT FACTS. Collapsing them turns a corrupt sidecar into a confident "nobody is
    subscribed, nothing to tell anyone" -- which is silence that looks like an answer.
    """
    path = sidecar_path(cart_path)
    empty = {"format": FORMAT, "version": VERSION, "subscriptions": [], "marks": {},
             "readable": True}
    if not path.exists():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {**empty, "readable": False, "error": str(e)}
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        fmt = data.get("format") if isinstance(data, dict) else type(data).__name__
        return {**empty, "readable": False, "error": f"unexpected format {fmt!r}"}
    subs = data.get("subscriptions")
    if not isinstance(subs, list):
        return {**empty, "readable": False, "error": "subscriptions is not a list"}
    marks = data.get("marks")
    if marks is None:
        data["marks"] = marks = {}
    if not isinstance(marks, dict):
        return {**empty, "readable": False, "error": "marks is not a dict"}
    data["readable"] = True
    return data


def _write(cart_path, subscriptions, marks) -> None:
    """Atomic replace. A half-written sidecar reads as unreadable, which blocks further
    writes, so this must be all-or-nothing rather than usually fine."""
    path = sidecar_path(cart_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": FORMAT, "version": VERSION,
               "subscriptions": subscriptions, "marks": marks}
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


# ⚠ Same measured Windows flake `note_store._write` documents: `os.replace` fails
# intermittently with WinError 32 at roughly 1 run in 6, because a scanner or the search
# indexer opens the freshly created temp file between close and replace. Handled rather than
# tolerated because a refused write here means a subscription was not recorded, and a seat
# that believes it is watching a document and is not would get silence it reads as "nothing
# changed."
def _replace_with_retry(tmp, path, attempts: int = 5, delay: float = 0.02) -> None:
    for i in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Only PermissionError. A missing directory or a read-only volume must fail at
            # once rather than be retried five times and reported late.
            if i == attempts - 1:
                raise
            time.sleep(delay * (2 ** i))


def _load_writable(cart_path) -> dict:
    data = load(cart_path)
    if not data.get("readable", True):
        # Writing over a file we cannot parse would discard every subscription in it.
        raise ValueError(f"refusing to write to an unreadable subscriptions file: "
                         f"{data.get('error')}")
    return data


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _check_seat(seat: str) -> str:
    """Validate the seat stamp.

    ⛔ `seat` IS AN ARGUMENT, BUT IT IS NOT THE CALLER'S CHOICE. The server passes the seat
    it authenticated (`seat_identity.resolve_seat`); a subscription tool exposed to an agent
    must never let that agent name itself, because every read gate in this module is keyed
    on this one string. A caller that could pick its own seat could pick somebody else's and
    read their change feed -- and, through it, their private notes.

    All this function can do is refuse the empty and the oversized, so that a caller which
    forgot to stamp cannot produce an unattributed subscription that looks attributed.
    """
    seat = (seat or "").strip()
    if not seat:
        raise ValueError("`seat` must be the authenticated seat, stamped by the caller's "
                         "server -- the caller does not pick its own identity, and an "
                         "unattributed subscription has no one to notify")
    if len(seat) > MAX_SEAT_CHARS:
        raise ValueError(f"seat is {len(seat)} chars, over the {MAX_SEAT_CHARS} cap")
    return seat


def _check_target_key(target_key: str) -> str:
    # Length is the only cheap structural signal that this is a content key and not an
    # ordinal. "7" passes no test we could write; it fails this one. Same gate as
    # note_store._check_target_key, same reason.
    if not isinstance(target_key, str) or len(target_key.strip()) < 8:
        raise ValueError("target_key must be a CONTENT key, never an index -- an "
                         "index-keyed subscription points at a stranger after the next "
                         "reindex, and would then report the wrong document as changed")
    return target_key.strip()


def _mint_id(existing_ids, seat: str, scope: str, target_key: str, now: float) -> str:
    base = f"{seat}\x00{scope}\x00{target_key}\x00{now}"
    sid = "sub_" + hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    # Two subscriptions can share seat, scope, target and timestamp when a caller passes an
    # explicit `now` (tests, imports, replays). A duplicate id would make `unsubscribe`
    # address the wrong row, so salt until it is unique.
    salt = 0
    while sid in existing_ids:
        salt += 1
        sid = "sub_" + hashlib.sha1(f"{base}\x00{salt}".encode("utf-8")).hexdigest()[:16]
    return sid


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def subscribe(cart_path, seat: str, target_key: str = "", scope: str = SCOPE_PASSAGE,
              now: float = 0.0) -> dict:
    """Register that `seat` watches something on this cart. Additive; changes no passage.

    `scope="passage"` needs a `target_key` (a content key). `scope="cart"` must NOT have
    one: the journal is a per-cart sidecar, so "the whole cart" is every entry in the file
    and a key would only be a lie about narrowing.

    IDEMPOTENT. Subscribing twice to the same target returns the existing row with
    `created: False` rather than minting a duplicate, because a duplicate would make the
    same change report twice and would make `unsubscribe` look broken -- one row goes
    inactive and the feed keeps arriving.

    Raises ValueError on bad input rather than storing junk in a registry whose job is to
    decide who gets told what.
    """
    now = now or time.time()
    seat = _check_seat(seat)
    if scope not in VALID_SCOPES:
        raise ValueError(f"scope must be one of {VALID_SCOPES}, got {scope!r}")
    if scope == SCOPE_PASSAGE:
        target_key = _check_target_key(target_key)
    else:
        if (target_key or "").strip():
            raise ValueError("a cart-scope subscription takes no target_key: the journal is "
                             "a per-cart sidecar, so cart scope already means every entry "
                             "in it")
        target_key = ""

    data = _load_writable(cart_path)
    subs = data["subscriptions"]

    for row in subs:
        if (row.get("seat") == seat and row.get("scope") == scope
                and row.get("target_key", "") == target_key and row.get("active", True)):
            out = dict(row)
            out["created"] = False
            return out

    entry = {
        "id": _mint_id({s.get("id") for s in subs}, seat, scope, target_key, now),
        "seat": seat,
        "scope": scope,
        "target_key": target_key,
        "at": float(now),
        "active": True,
        # Attribution for the STOP, empty until unsubscribe is called. Who stopped watching
        # a document and when is a question an auditor asks after an incident.
        "unsubscribed_at": 0.0,
        "unsubscribed_by": "",
    }
    subs.append(entry)
    _write(cart_path, subs, data["marks"])
    out = dict(entry)
    out["created"] = True
    return out


def unsubscribe(cart_path, seat: str, subscription_id: str, now: float = 0.0) -> dict:
    """Stop watching. Flips `active` to False WITH attribution; removes nothing.

    ⛔ ONLY THE OWNING SEAT MAY DO THIS. If any seat could cancel another's subscription,
    the quietest possible attack on a compliance notification system would be one API call:
    the target keeps asking "what changed on my documents", keeps being told nothing, and
    nothing in their answer looks wrong.
    """
    now = now or time.time()
    seat = _check_seat(seat)
    sid = (subscription_id or "").strip()
    if not sid:
        raise ValueError("subscription_id is required")

    data = _load_writable(cart_path)
    for row in data["subscriptions"]:
        if row.get("id") != sid:
            continue
        if row.get("seat") != seat:
            # Deliberately the same message whether or not the row exists for someone else,
            # so this cannot be used to enumerate other seats' subscriptions.
            raise PermissionError(f"seat {seat!r} does not own subscription {sid!r}")
        if not row.get("active", True):
            raise ValueError(f"subscription {sid!r} is already inactive (stopped "
                             f"{row.get('unsubscribed_at')})")
        row["active"] = False
        row["unsubscribed_at"] = float(now)
        row["unsubscribed_by"] = seat
        _write(cart_path, data["subscriptions"], data["marks"])
        return dict(row)
    raise KeyError(f"no subscription with id {sid!r} for seat {seat!r}")


def list_subscriptions(cart_path, seat: str = "", include_inactive: bool = False) -> list:
    """What THIS seat watches, newest first.

    ⛔ FAILS CLOSED, same rule as `note_store.list_notes`: an empty `seat` is an
    UNIDENTIFIED caller, not "no filter", and it gets an empty list. On a single-user desktop
    install there is nothing to authenticate against, and the safe reading of "I do not know
    who you are" is never "here is everyone's watch list". A watch list is a map of what a
    person considers important, which in a compliance account is itself sensitive.

    The owner-level view is `stats()`, which reports counts and never targets.
    """
    seat = (seat or "").strip()
    if not seat:
        return []
    data = load(cart_path)
    out = [s for s in data.get("subscriptions", []) if s.get("seat") == seat]
    if not include_inactive:
        out = [s for s in out if s.get("active", True)]
    return sorted(out, key=lambda s: s.get("at", 0.0), reverse=True)


# ---------------------------------------------------------------------------
# high-water mark, per seat
# ---------------------------------------------------------------------------

def last_seen(cart_path, seat: str) -> int:
    """The seat's high-water mark, or 0. Returns 0 for an unreadable sidecar too -- callers
    that need to tell those apart read `load()["readable"]`, and `what_changed` does."""
    seat = _check_seat(seat)
    data = load(cart_path)
    row = (data.get("marks") or {}).get(seat)
    if isinstance(row, dict) and isinstance(row.get("seq"), int) \
            and not isinstance(row.get("seq"), bool):
        return max(0, row["seq"])
    return 0


def mark_seen(cart_path, seat: str, seq: int, now: float = 0.0) -> dict:
    """Record that this seat has looked as far as `seq`. FORWARD ONLY.

    ⛔ A REWIND IS REFUSED, NOT APPLIED, and that is the whole point of the function. A mark
    that could move backward would silently re-hide changes somebody had already read and
    answered for -- and in a compliance account "I was never told" and "I was told and
    forgot" are the two things an audit has to be able to distinguish. Moving the mark back
    makes the record say the first when the second happened.

    It is refused in the RETURN VALUE rather than by raising, because a client replaying its
    own stale cursor is a normal race (two tabs, two devices) and not an error. The return
    says exactly what happened, so a caller can tell a no-op from a move:

        {"seat", "seq": the mark NOW, "previous_seq", "advanced": bool,
         "refused_rewind": bool, "requested_seq", "at"}

    ⚠ `what_changed` does NOT call this. Reading is not acknowledging: if a look advanced the
      mark, a client that crashed while rendering would lose those changes for ever, and
      nothing afterwards could tell that it had. Acknowledging is a separate, explicit act.
    """
    now = now or time.time()
    seat = _check_seat(seat)
    if not isinstance(seq, int) or isinstance(seq, bool):
        raise ValueError(f"seq must be an int, got {seq!r}")
    if seq < 0:
        raise ValueError(f"seq must not be negative, got {seq!r}")

    data = _load_writable(cart_path)
    marks = data["marks"]
    prior_row = marks.get(seat) if isinstance(marks.get(seat), dict) else {}
    prior = prior_row.get("seq", 0)
    if not isinstance(prior, int) or isinstance(prior, bool) or prior < 0:
        prior = 0

    if seq <= prior:
        return {"seat": seat, "seq": prior, "previous_seq": prior, "advanced": False,
                "refused_rewind": seq < prior, "requested_seq": seq,
                "at": prior_row.get("at", 0.0)}

    marks[seat] = {"seq": seq, "at": float(now), "by": seat}
    _write(cart_path, data["subscriptions"], marks)
    return {"seat": seat, "seq": seq, "previous_seq": prior, "advanced": True,
            "refused_rewind": False, "requested_seq": seq, "at": float(now)}


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------

def _iso(ts) -> str:
    if not isinstance(ts, (int, float)) or isinstance(ts, bool):
        return ""
    try:
        return datetime.datetime.fromtimestamp(float(ts),
                                               datetime.timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return ""


def _first_field(*dicts, fields=()) -> str:
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for f in fields:
            v = d.get(f)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def _who(entry: dict) -> dict:
    """WHO, and -- the part an auditor actually cares about -- whether that was VERIFIED.

    `_journal_event` stamps `payload["identity_source"]` from `seat_identity`, which is the
    only field in the system that distinguishes a seat derived from a verified JWT from a
    string a caller typed. A change feed that reported the writer without that distinction
    would present a claim as a fact, which is precisely the thing provenance is sold on.

    ⚠ ABSENT IS NOT "UNVERIFIED". An entry written before the identity stamp existed, or by
      a path that does not stamp, has no opinion recorded -- so `identity_recorded` is False
      and `identity_source` is "unrecorded". Folding that into "claimed" would invent a
      finding; folding it into "verified" would launder one.
    """
    payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
    src = payload.get("identity_source")
    recorded = isinstance(src, str) and bool(src.strip())
    return {
        "writer": entry.get("writer", ""),
        "surface": entry.get("surface", ""),
        "identity_recorded": recorded,
        "identity_source": src.strip() if recorded else "unrecorded",
        "identity_verified": recorded and src.strip() == "verified",
        "identity_overridden": bool(payload.get("identity_overridden")),
        "identity_refused": bool(payload.get("identity_refused")),
    }


def _supersedes(entry: dict) -> dict | None:
    """What this entry says it descends from, or None.

    ⛔ READ OUT OF THE ENTRY, NEVER INFERRED. If the journal does not say what something
    supersedes, this returns None rather than guessing from timestamps or adjacency -- a
    fabricated lineage in a compliance payload is worse than an absent one, because the
    reader cannot tell it was invented.
    """
    target = entry.get("target") if isinstance(entry.get("target"), dict) else {}
    payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
    key = _first_field(payload, target, fields=_SUPERSEDES_KEY_FIELDS)
    ident = _first_field(payload, target, fields=_SUPERSEDES_ID_FIELDS)
    # A `perm_change` from private to shared is recorded with from/to, which is a state
    # transition rather than a lineage. Surfaced separately so neither is mistaken for the
    # other.
    transition = None
    if isinstance(payload.get("from"), str) or isinstance(payload.get("to"), str):
        transition = {"from": payload.get("from", ""), "to": payload.get("to", "")}
    if not key and not ident and transition is None:
        return None
    return {"key": key, "id": ident, "transition": transition}


def _candidate_keys(entry: dict, key_resolver, errors: list) -> list:
    """Every content key this entry could be about, best first.

    Order matters: an explicit key on the entry beats a resolver lookup, because the entry
    is the record and the resolver is an opinion about the cart as it stands right now.
    """
    target = entry.get("target") if isinstance(entry.get("target"), dict) else {}
    payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
    out = []
    explicit = _first_field(target, payload, fields=_KEY_FIELDS)
    if explicit:
        out.append(explicit)
    if key_resolver is not None:
        resolved = ""
        try:
            if isinstance(key_resolver, dict):
                idx = target.get("idx")
                if isinstance(idx, int) and not isinstance(idx, bool):
                    resolved = key_resolver.get(idx) or ""
                    if not resolved:
                        # Also allow a resolver keyed by the string form, which is what a
                        # JSON round trip of an int-keyed dict produces.
                        resolved = key_resolver.get(str(idx)) or ""
            elif callable(key_resolver):
                resolved = key_resolver(target) or ""
        except Exception as e:                            # noqa: BLE001
            # A caller-supplied resolver must not be able to take down the change feed. The
            # failure is counted and reported rather than swallowed, because a resolver that
            # throws on every row looks exactly like "nothing changed".
            errors.append(f"{type(e).__name__}: {e}")
            resolved = ""
        if isinstance(resolved, str) and resolved.strip() and resolved.strip() not in out:
            out.append(resolved.strip())
    return out


def _headline(seq: int, op: str, who: dict, target_key: str, sup, why_count: int) -> str:
    """One computed line, so a surface can render a feed without walking the dict.

    ⚠ COMPUTED, never asserted. Every clause below is read off the entry; a clause with no
    data behind it is omitted rather than filled with a plausible default.
    """
    bits = [f"seq {seq}", _OP_PHRASE.get(op, op or "?")]
    writer = who.get("writer") or "?"
    if who.get("identity_verified"):
        bits.append(f"by {writer} (verified)")
    elif who.get("identity_recorded"):
        bits.append(f"by {writer} (unverified {who.get('identity_source')})")
    else:
        bits.append(f"by {writer} (identity not recorded)")
    if target_key:
        bits.append(f"passage {target_key[:12]}")
    if isinstance(sup, dict):
        if sup.get("key"):
            bits.append(f"supersedes {sup['key'][:12]}")
        elif sup.get("id"):
            bits.append(f"supersedes {sup['id']}")
        elif sup.get("transition"):
            t = sup["transition"]
            bits.append(f"{t.get('from') or '?'} -> {t.get('to') or '?'}")
    bits.append(f"{why_count} note explains why" if why_count == 1
                else f"{why_count} notes explain why" if why_count
                else "no recorded why")
    # ASCII separator deliberately: a headline is printed to a Windows console, where the
    # default cp1252 codec raises on a middot and takes the whole feed down with it.
    return " | ".join(bits)


# ---------------------------------------------------------------------------
# what_changed -- the one that matters
# ---------------------------------------------------------------------------

def what_changed(cart_path, seat: str, since_seq: int = 0, now: float = 0.0,
                 key_resolver=None, limit: int = DEFAULT_LIMIT,
                 allow_broken_chain: bool = False) -> dict:
    """What happened, since when, to the things this seat watches -- with the WHY attached.

    This is the function the whole module exists for. Everything else is bookkeeping that
    makes this call answerable.

    Args:
        cart_path:  the cart. Its journal and note sidecars sit beside it.
        seat:       the AUTHENTICATED seat. Not the caller's choice (see `_check_seat`).
                    Every visibility decision in the answer keys on this one string.
        since_seq:  0 -> use this seat's high-water mark ("since I last looked", with no
                    cursor for the client to remember). A positive value is used verbatim,
                    so an explicit replay is possible. `FROM_BEGINNING` (-1) ignores the
                    mark and returns the full history this seat is entitled to.
        now:        when the look happened. Recorded in the answer; advances nothing.
        key_resolver: optional `{idx: content_key}` dict, or `callable(target) -> key`.
                    The journal's target is an index and a subscription is a content key;
                    this is the bridge, and it comes from the caller because resolving it
                    needs the cart's texts, which this module deliberately does not load.
                    Absent, index-only events can still match a CART-scope subscription but
                    not a passage one, and the count of those is reported.
        limit:      cap on rendered changes. Truncation is reported, never silent.
        allow_broken_chain: forensics escape hatch. See below.

    ⚠ INTEGRITY FIRST. If `journal.verify()` reports a broken chain the rows are WITHHELD:
      `ok` is False, `changes` is empty, `changes_withheld` is True, `withheld_count` says
      how many rows would have been returned, and `verdict_line` leads with the break.
      Changes read out of a journal that may have been edited are not evidence, and a caller
      that cannot tell the difference will present them as if they were. `allow_broken_chain`
      returns them anyway, each stamped `trustworthy: False`.

      A TORN TAIL is the one exception, because it is the only break a power failure can
      cause and `journal.replay` already tolerates it. That is reported as
      `integrity["degraded"]`, with the rows intact.

    ⚠ MISSING vs EMPTY JOURNAL are different facts, told apart the way `note_store.load`
      tells unreadable from empty. No journal at all means nothing has ever routed through
      the one door for this cart, which is a louder statement than "nothing has happened
      since you last looked", and it is the state a wiring bug produces.

    Never advances the mark -- call `mark_seen` for that, deliberately and separately.
    """
    now = now or time.time()
    seat = _check_seat(seat)
    if not isinstance(since_seq, int) or isinstance(since_seq, bool):
        raise ValueError(f"since_seq must be an int, got {since_seq!r}")
    if since_seq < FROM_BEGINNING:
        raise ValueError(f"since_seq must be >= {FROM_BEGINNING}, got {since_seq}")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError(f"limit must be a positive int, got {limit!r}")
    limit = min(limit, MAX_LIMIT)

    cart = Path(cart_path)
    jpath = _journal.journal_path_for(cart)

    out = {
        "format": FORMAT,
        "version": VERSION,
        "verdict_line": "",          # filled last, computed from everything below
        "ok": False,
        "cart": str(cart),
        "seat": seat,
        "at": float(now),
        "journal_path": str(jpath),
        "journal_exists": jpath.exists(),
        "journal_empty": False,
        "subscriptions_readable": True,
        "notes_readable": True,
        "error": "",
        "integrity": {"checked": False, "ok": False, "degraded": False,
                      "verdict_line": "", "first_break": None, "entries": 0, "breaks": 0},
        "since_seq": 0,
        "since_seq_source": "none",
        "last_seen_seq": 0,
        "head_seq": 0,
        "watching": [],
        "watched_keys": [],
        "cart_scope": False,
        "changes": [],
        "change_count": 0,
        "order": "seq_desc",
        "truncated": False,
        "withheld_older": 0,
        "changes_withheld": False,
        "withheld_count": 0,
        "unresolved_index_events": 0,
        "resolver_errors": [],
        "notices": [],
    }

    # ---- 1. the registry. Unreadable is a REFUSAL to answer, not an empty answer.
    sub_data = load(cart)
    if not sub_data.get("readable", True):
        out["subscriptions_readable"] = False
        out["error"] = str(sub_data.get("error", "unreadable subscriptions sidecar"))
        out["notices"].append(
            "The subscriptions sidecar cannot be read, so we do not know what you watch. "
            "This is NOT the same as watching nothing -- no conclusion about your documents "
            "can be drawn from this answer.")
        out["verdict_line"] = (f"CANNOT ANSWER: subscriptions sidecar for {cart.name} is "
                               f"unreadable ({out['error']}) -- this is not 'nothing "
                               f"changed'")
        return out

    rows = [s for s in sub_data.get("subscriptions", [])
            if s.get("seat") == seat and s.get("active", True)]
    rows.sort(key=lambda s: s.get("at", 0.0), reverse=True)
    out["watching"] = rows
    key_to_subs: dict = {}
    cart_sub_ids = []
    for row in rows:
        if row.get("scope") == SCOPE_CART:
            cart_sub_ids.append(row.get("id", ""))
        else:
            k = row.get("target_key", "")
            if k:
                key_to_subs.setdefault(k, []).append(row.get("id", ""))
    out["watched_keys"] = sorted(key_to_subs)
    out["cart_scope"] = bool(cart_sub_ids)

    # ---- 2. the cursor. A stored mark beats a client's memory of one.
    mark = last_seen(cart, seat)
    out["last_seen_seq"] = mark
    if since_seq == FROM_BEGINNING:
        effective_since = 0
        out["since_seq_source"] = "from_beginning"
    elif since_seq > 0:
        effective_since = since_seq
        out["since_seq_source"] = "argument"
    else:
        effective_since = mark
        out["since_seq_source"] = "mark" if mark else "none"
    out["since_seq"] = effective_since

    # ---- 3. integrity, BEFORE reading a single row out of the file.
    v = _journal.verify(jpath)
    no_journal = (not v.get("exists")) or (
        isinstance(v.get("first_break"), dict)
        and v["first_break"].get("reason") == "no_journal")
    if no_journal:
        # Nothing has happened yet -- but say WHICH nothing. "No journal exists" is the
        # state a wiring bug produces, and it must not read as "your documents are quiet".
        out["ok"] = True
        out["integrity"] = {"checked": True, "ok": True, "degraded": False,
                            "verdict_line": v.get("verdict_line", ""),
                            "first_break": None, "entries": 0, "breaks": 0}
        out["notices"].append(
            f"No journal exists at {jpath.name}. Nothing has been recorded for this cart "
            f"yet -- which is a different fact from 'no changes since you last looked', and "
            f"is what an unwired mutation path looks like.")
        out["verdict_line"] = (f"NO JOURNAL for {cart.name}: nothing has been recorded yet "
                              f"({len(rows)} active subscription"
                              f"{'' if len(rows) == 1 else 's'} for {seat})")
        return out

    first_break = v.get("first_break") if isinstance(v.get("first_break"), dict) else None
    only_torn_tail = (v.get("breaks") == 1 and first_break is not None
                      and first_break.get("reasons") == ["truncated_tail"]) \
        or bool(v.get("torn_tail_confirmed_by_hwm"))
    chain_ok = bool(v.get("ok"))
    degraded = (not chain_ok) and only_torn_tail
    out["integrity"] = {
        "checked": True,
        "ok": chain_ok,
        "degraded": degraded,
        "verdict_line": v.get("verdict_line", ""),
        "first_break": first_break,
        "entries": v.get("entries", 0),
        "breaks": v.get("breaks", 0),
    }
    tampered = (not chain_ok) and not degraded
    if degraded:
        out["notices"].append(
            "The journal's last line is torn, which is what an interrupted write looks like "
            "rather than tampering. Everything before it verifies; the torn entry is not "
            "reported below.")
    if v.get("empty"):
        out["journal_empty"] = True

    # ---- 4. the rows. strict=False because the verdict is already computed above and
    #         reported; raising here would replace a structured answer with a traceback.
    try:
        entries = _journal.replay(jpath, strict=False)
    except Exception as e:                                # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out["notices"].append("The journal could not be read at all, which is not the same "
                              "as nothing having happened.")
        out["verdict_line"] = f"CANNOT ANSWER: journal for {cart.name} unreadable ({e})"
        return out

    seqs = [e.get("seq") for e in entries
            if isinstance(e.get("seq"), int) and not isinstance(e.get("seq"), bool)]
    out["head_seq"] = max(seqs) if seqs else 0

    # ---- 5. match against the watch list, and resolve keys.
    resolver_errors: list = []
    matched = []
    unresolved = 0
    for entry in entries:
        seq = entry.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq <= effective_since:
            continue
        keys = _candidate_keys(entry, key_resolver, resolver_errors)
        hit_key = next((k for k in keys if k in key_to_subs), "")
        if hit_key:
            matched.append((entry, hit_key, SCOPE_PASSAGE, list(key_to_subs[hit_key])))
            continue
        if cart_sub_ids:
            # The journal is this cart's own sidecar, so every entry in it is about this
            # cart by construction. Cart scope therefore matches everything -- there is no
            # cart-name comparison to get wrong.
            matched.append((entry, keys[0] if keys else "", SCOPE_CART, list(cart_sub_ids)))
            continue
        if key_to_subs and not keys:
            # A passage subscription exists and this event carries no key we can resolve. It
            # is NOT a miss we may be quiet about: without a resolver this is the normal
            # case, and a silent zero here reads as "your document did not change".
            unresolved += 1
    out["unresolved_index_events"] = unresolved
    out["resolver_errors"] = resolver_errors[:16]
    if unresolved and key_resolver is None:
        out["notices"].append(
            f"{unresolved} event(s) since seq {effective_since} name a passage only by "
            f"index, and no key_resolver was supplied, so they could not be matched to a "
            f"passage subscription. Pass key_resolver={{idx: content_key}} from the caller "
            f"that holds the cart's texts. These events are NOT evidence that nothing "
            f"changed on the passages you watch.")
    elif unresolved:
        out["notices"].append(
            f"{unresolved} event(s) could not be resolved to a content key even with a "
            f"resolver -- the passage they name may no longer be in the cart.")
    if resolver_errors:
        out["notices"].append(f"key_resolver raised on {len(resolver_errors)} event(s); "
                              f"those events were left unresolved.")

    matched.sort(key=lambda m: m[0].get("seq") or 0, reverse=True)
    if len(matched) > limit:
        out["truncated"] = True
        out["withheld_older"] = len(matched) - limit
        out["notices"].append(
            f"Showing the {limit} newest of {len(matched)} changes; "
            f"{len(matched) - limit} OLDER change(s) were dropped. Raise `limit` or lower "
            f"`since_seq` -- the dropped ones are the oldest, not the least important.")
        matched = matched[:limit]

    # ---- 6. the WHY, fetched ONLY through note_store's visibility rule.
    #
    # ⛔ This module never opens the notes sidecar. Both calls below filter on `seat` inside
    #    note_store._visible(), which is the single place that boundary is implemented. A
    #    "quick" direct read here is the leak this file is most likely to grow.
    note_keys = {k for _, k, _, _ in matched if k}
    for entry, _k, _s, _ids in matched:
        sup = _supersedes(entry)
        if sup and sup.get("key"):
            # The notes on the OLD key are the ones that say why it stopped being current,
            # which is the clause of the payload nobody else can send.
            note_keys.add(sup["key"])
    notes_by_key: dict = {}
    notes_readable = True
    notes_error = ""
    try:
        nd = _notes.load(cart)
        notes_readable = bool(nd.get("readable", True))
        notes_error = str(nd.get("error", ""))
        if note_keys:
            notes_by_key = _notes.for_results(cart, sorted(note_keys), seat=seat)
    except Exception as e:                                # noqa: BLE001
        notes_readable = False
        notes_error = f"{type(e).__name__}: {e}"
    out["notes_readable"] = notes_readable
    if not notes_readable:
        out["notices"].append(
            f"The notes sidecar is unreadable ({notes_error}), so no WHY could be attached "
            f"to these changes. That is not the same as no reasoning having been recorded.")

    # Notes named by id in a payload (`note`, `perm_change`). Pulled from the seat-filtered
    # list, so a note this seat may not see is simply absent -- never fetched and then
    # hidden, which is how a filter gets forgotten.
    visible_by_id: dict = {}
    if notes_readable:
        try:
            for n in _notes.list_notes(cart, seat=seat):
                if n.get("id"):
                    visible_by_id[n["id"]] = n
        except Exception:                                  # noqa: BLE001
            visible_by_id = {}

    # ---- 7. render.
    changes = []
    for entry, hit_key, how, sub_ids in matched:
        who = _who(entry)
        sup = _supersedes(entry)
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}

        why = list(notes_by_key.get(hit_key, [])) if hit_key else []
        seen_ids = {n.get("id") for n in why}
        nid = payload.get("note_id")
        if isinstance(nid, str) and nid in visible_by_id and nid not in seen_ids:
            why.append(visible_by_id[nid])
            seen_ids.add(nid)

        why_prior = []
        if sup and sup.get("key"):
            why_prior = [n for n in notes_by_key.get(sup["key"], [])
                         if n.get("id") not in seen_ids]

        row = {
            "seq": entry.get("seq"),
            "op": entry.get("op", ""),
            "headline": "",
            "when": {
                "event_ts": entry.get("event_ts", 0.0),
                "ingest_ts": entry.get("ingest_ts", 0.0),
                "event_iso": _iso(entry.get("event_ts")),
                "ingest_iso": _iso(entry.get("ingest_ts")),
            },
            "who": who,
            "where": {
                "target": entry.get("target", {}),
                "target_key": hit_key,
                "matched_by": how,
                "subscription_ids": sub_ids,
            },
            "supersedes": sup,
            "why": why,
            "why_count": len(why),
            "why_prior": why_prior,
            "payload": payload,
            "entry_hash": entry.get("hash", ""),
            "trustworthy": not tampered,
        }
        row["headline"] = _headline(row["seq"], row["op"], who, hit_key, sup, len(why))
        changes.append(row)

    # ---- 8. the integrity gate, applied to the rendered rows.
    if tampered and not allow_broken_chain:
        out["changes"] = []
        out["change_count"] = 0
        out["changes_withheld"] = True
        out["withheld_count"] = len(changes)
        out["ok"] = False
        out["notices"].append(
            f"{len(changes)} change(s) matched your subscriptions but were WITHHELD: the "
            f"journal's hash chain is broken, so these rows are not evidence. Investigate "
            f"with `python journal.py verify {jpath}`; pass allow_broken_chain=True only "
            f"for forensics, and never render the result as a notification.")
        out["verdict_line"] = (f"INTEGRITY FAILURE for {cart.name}: "
                               f"{v.get('verdict_line', 'chain broken')} -- "
                               f"{len(changes)} matching change(s) withheld")
        return out

    out["changes"] = changes
    out["change_count"] = len(changes)
    out["ok"] = (not tampered)
    if tampered:
        out["notices"].insert(0,
            "RETURNED UNDER allow_broken_chain: the journal's hash chain is broken, so "
            "every row below is marked trustworthy=False and must not be presented as a "
            "notification.")
        out["verdict_line"] = (f"UNTRUSTED: {len(changes)} change(s) for {seat} on "
                               f"{cart.name}, read from a BROKEN chain "
                               f"({v.get('verdict_line', '')})")
        return out

    # The verdict line, computed. It leads with the count because that is the one number a
    # human reads, and it names the watch list because zero changes with zero subscriptions
    # is a different answer from zero changes with ten.
    if not rows:
        out["verdict_line"] = (f"{seat} watches nothing on {cart.name} -- 0 changes "
                               f"reported because there is no subscription, not because the "
                               f"cart is quiet (head seq {out['head_seq']})")
    else:
        scope_bit = (f"{len(out['watched_keys'])} passage"
                     f"{'' if len(out['watched_keys']) == 1 else 's'}"
                     + (" + whole cart" if out["cart_scope"] else ""))
        why_total = sum(c["why_count"] for c in changes)
        tail = []
        if degraded:
            tail.append("chain degraded (torn tail)")
        if unresolved:
            tail.append(f"{unresolved} unresolved")
        if out["truncated"]:
            tail.append(f"{out['withheld_older']} older dropped")
        out["verdict_line"] = (
            f"{len(changes)} change(s) since seq {effective_since} on {scope_bit} for "
            f"{seat}; {why_total} explaining note(s); head seq {out['head_seq']}"
            + (" -- " + ", ".join(tail) if tail else ""))
    return out


# ---------------------------------------------------------------------------
# stats -- the owner view
# ---------------------------------------------------------------------------

def stats(cart_path) -> dict:
    """Counts over every subscription, for an owner looking at a cart.

    Deliberately reports NO target keys and no per-seat watch lists. A watch list is a map
    of what somebody considers important, so an owner-level summary that leaked targets
    would be a quieter version of the leak `list_subscriptions` fails closed to prevent.
    """
    data = load(cart_path)
    subs = data.get("subscriptions", [])
    by_scope: dict = {}
    active = 0
    for s in subs:
        by_scope[s.get("scope", "?")] = by_scope.get(s.get("scope", "?"), 0) + 1
        if s.get("active", True):
            active += 1
    marks = data.get("marks", {}) if isinstance(data.get("marks"), dict) else {}
    mark_seqs = [m.get("seq") for m in marks.values()
                 if isinstance(m, dict) and isinstance(m.get("seq"), int)]
    return {
        "readable": data.get("readable", True),
        "error": data.get("error", ""),
        "total": len(subs),
        "active": active,
        "inactive": len(subs) - active,
        "by_scope": by_scope,
        "seats": sorted({s.get("seat") for s in subs if s.get("seat")}),
        "marked_seats": len(mark_seqs),
        "lowest_mark": min(mark_seqs) if mark_seqs else 0,
        "highest_mark": max(mark_seqs) if mark_seqs else 0,
    }


# ---------------------------------------------------------------------------
# CLI -- so an operator can look at a feed without writing code. The verdict line comes
# FIRST and is computed, not asserted. Read-only: there is no subscribe verb here, because
# a registry edit should carry a server-stamped seat and a shell argument is a claim.
# ---------------------------------------------------------------------------

def _main(argv) -> int:
    if len(argv) < 3 or argv[1] not in ("changed", "list", "stats"):
        print("usage: python subscriptions.py {changed|list|stats} <cart> [seat] [since]")
        return 2
    cmd, cart = argv[1], argv[2]
    if cmd == "stats":
        print(json.dumps(stats(cart), indent=2, sort_keys=True))
        return 0
    seat = argv[3] if len(argv) > 3 else ""
    if cmd == "list":
        for row in list_subscriptions(cart, seat):
            print(f"{row['id']}  {row['scope']:<8} {row.get('target_key', '') or '(cart)'}")
        return 0
    since = int(argv[4]) if len(argv) > 4 else 0
    res = what_changed(cart, seat, since_seq=since)
    print(res["verdict_line"])
    for notice in res["notices"]:
        print(f"  ! {notice}")
    for c in res["changes"]:
        print(f"  {c['headline']}")
        for n in c["why"]:
            print(f"      why ({n.get('kind')}, {n.get('scope')}): {n.get('body', '')[:120]}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
