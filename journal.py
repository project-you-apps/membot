"""The provenance journal: an append-only, hash-chained record of every mutation.

THE DECISION THIS IMPLEMENTS (docs/DESIGN-WAL-2026-09-02.md)
-----------------------------------------------------------
The purpose of this log changed between its scoping (2026-07-21) and its design
(2026-09-02), and the change is the whole reason this file looks the way it does:

    2026-07-21 purpose   crash safety     -> a WAL is a buffer. Merge it, drop it.
    2026-09-02 purpose   PROVENANCE       -> the history IS the asset.

Under the second purpose, "who touched what, when, and what it descends from" is the
thing being sold. So:

1. NO BYPASS (§2.1). Every mutation produces a record because there is no other way to
   mutate. That is an architecture property, not a feature, and it is NOT satisfied by
   this module alone -- it is satisfied by there being exactly one door into mutation and
   that door calling `append()`. This module is the door's latch, not the door.
   See docs/AUDIT-membot-mutation-paths-2026-11-01.md for the doors that still exist.
2. IMMUTABLE (§2.2). Append-only, never rewritten in place.
3. VERIFIABLE (§2.3). Each record carries the previous record's hash. Without chaining
   you can *claim* completeness; with it you can *prove* it. Deletion and reordering both
   break the chain. This is the same seal discipline we ran with Grant all week --
   hash-lock, verify on receipt, never retro-edit, correct by erratum -- pointed inward.

WHY `log_activity()` WAS NOT ENOUGH
-----------------------------------
It is a Python list. Bounded, in-memory, gone on restart. The only event record the
system kept did not survive a reboot. Every corroboration, every tombstone, every
permission change was written to a list and forgotten.

And the deeper lesson from the corroboration endpoint -- built 2026-08-23, complete,
reachable, used zero times in nine days:

    Write-attribution works because it is automatic and unavoidable.
    Corroboration does not work because it is merely available.

    => If provenance is the moat, no part of it may depend on anyone remembering.

That is why `append()` is cheap, why `for_passage()` exists (§8: provenance travels WITH
the result, not on a dashboard nobody reads), and why nothing in this module is optional.

⛔ THERE IS NO `compact()` AND NO `delete()`. ON PURPOSE.
---------------------------------------------------------
§3 reverses the 2026-07-21 answer ("I'm not worried about WAL size"), which was correct
for a crash-safety WAL and is wrong for a ledger:

    A WAL you can compact is a buffer. A WAL you cannot compact is a ledger.

The cart is a materialized view of this journal, rebuildable by replay. If storage ever
becomes a problem the answer is cold-tiering old segments -- moving bytes, keeping the
chain -- never deletion. A compaction function here would be the single change that turns
the asset back into a buffer, so it does not exist and should not be added. The same goes
for a delete or an "amend": history is corrected by APPENDING an erratum (`op="supersede"`
pointing at the wrong record), which is itself a journal entry and is itself chained.

There is deliberately also no repair path for a broken chain. `verify()` names the break
and a human decides. Silent repair is indistinguishable from tampering, which is the one
thing this file exists to make impossible.

Stdlib only. JSONL, one event per line, greppable by design (7/21: "must be inspectable
on disk").
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

FORMAT = "vps-journal"
VERSION = 1

JOURNAL_SUFFIX = ".journal.jsonl"

# The fields that go into the hash, in the order §5 lists them. "hash" itself is excluded:
# it is the output. Everything PRECEDING it is covered, which is what makes a payload edit,
# a writer swap, a timestamp nudge and a re-chain all detectable.
HASHED_FIELDS = (
    "seq",
    "event_ts",
    "ingest_ts",
    "writer",
    "surface",
    "op",
    "target",
    "payload",
    "prev_hash",
)

# Every field of a stored entry, in §5 order.
ENTRY_FIELDS = HASHED_FIELDS + ("hash",)

# seq 1 chains to this. A real hash would be indistinguishable from a record we lost, so
# the genesis link is a value no sha256 can produce by accident.
GENESIS_PREV_HASH = "sha256:" + ("0" * 64)

# §5. Deliberately closed sets. An unrecognized surface or op is refused rather than
# stored, because `project_universal_write_attribution_invariant_2026-07-22` is only worth
# anything if the vocabulary is auditable -- a free-text `op` column means nobody can ever
# ask "show me every permission change" and trust the answer. Adding a verb here is a
# deliberate act, and that is the point.
VALID_SURFACES = ("CLI", "VSC", "WEB", "MCP")
VALID_OPS = ("store", "tombstone", "corroborate", "supersede", "perm_change", "flag")

# A payload with an inlined embedding is the expected large case (§4: serialize the
# embedding, do not re-embed on replay -- re-embedding makes replay model-version
# dependent, so it is not deterministic, so it is not a reconstruction). 4 MB is roughly a
# 100k-dim float list; well past anything legitimate, and short of a line nothing can read.
MAX_LINE_BYTES = 4 * 1024 * 1024


class JournalIntegrityError(Exception):
    """The chain is broken. Loud, not a warning (§6.3)."""


# ---------------------------------------------------------------------------
# paths and hashing
# ---------------------------------------------------------------------------

def journal_path_for(cart_path) -> Path:
    """`<cart>.journal.jsonl` -- a sidecar, so a shard carries its own journal (§4)."""
    cart = Path(cart_path)
    return cart.with_name(cart.name + JOURNAL_SUFFIX)


def _canonical(core: dict) -> str:
    """The exact bytes that get hashed.

    sort_keys so key order on disk cannot change the hash; separators so whitespace
    cannot; allow_nan=False because NaN is not JSON and would round-trip differently in
    every other reader. Note we hash the PARSED fields, never the raw line -- so a
    pretty-printed or reordered line still verifies, and the on-disk format stays free to
    be human-readable.
    """
    return json.dumps(core, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def compute_hash(entry: dict) -> str:
    core = {k: entry[k] for k in HASHED_FIELDS}
    blob = _canonical(core).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# locking
# ---------------------------------------------------------------------------
#
# Two processes appending must not interleave a partial line or hand out the same seq.
# A lock is the only thing standing between "append-only" and "append-mostly".
#
# Windows-primary (`msvcrt.locking` on a byte range of a sidecar .lock file), with a POSIX
# path (`fcntl.flock`). We lock a SIDECAR rather than the journal itself so that a reader
# holding the journal open -- grep, a tail, replay -- never blocks a writer, and so the
# lock state is never confused with file length.
#
# ⛔ The fallback is honest. If neither locking primitive is available or the lock cannot
# be taken, we DO proceed (losing the journal entry is worse than racing for it) but the
# return value of `append()` says `_locked: False` and carries `_lock_error`. A caller that
# needs the stronger guarantee can refuse. Silently proceeding while claiming a lock would
# be a check that is identical in both states, which is not a check.

_LOCK_TIMEOUT_S = 10.0
_LOCK_POLL_S = 0.01

try:  # Windows
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

try:  # POSIX
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows
    fcntl = None


class _FileLock:
    """Exclusive advisory lock on `<journal>.lock`. Never raises on failure to lock."""

    def __init__(self, journal_path: Path):
        self.path = journal_path.with_name(journal_path.name + ".lock")
        self.locked = False
        self.error = ""
        self._fh = None

    def __enter__(self) -> "_FileLock":
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a+b")
        except OSError as e:
            self.error = f"cannot open lock file {self.path}: {e}"
            return self
        if msvcrt is None and fcntl is None:  # pragma: no cover
            self.error = "no locking primitive on this platform (no msvcrt, no fcntl)"
            return self
        deadline = time.time() + _LOCK_TIMEOUT_S
        last = ""
        while True:
            try:
                if msvcrt is not None:
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.locked = True
                return self
            except OSError as e:
                last = str(e)
                if time.time() >= deadline:
                    self.error = (f"lock on {self.path} still held after "
                                  f"{_LOCK_TIMEOUT_S:g}s: {last}")
                    return self
                time.sleep(_LOCK_POLL_S)

    def __exit__(self, *exc) -> None:
        if self._fh is None:
            return
        try:
            if self.locked:
                if msvcrt is not None:
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self.locked = False
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


# ---------------------------------------------------------------------------
# tail read
# ---------------------------------------------------------------------------

def _last_line(path: Path) -> str:
    """The last complete-looking line, read by seeking from the end.

    Appending must not cost a full scan: a provenance journal is never compacted, so it
    only ever gets longer, and an O(n) append would make the ledger quadratic over its
    lifetime -- a cost that shows up exactly when the asset is most valuable.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return ""
    if size == 0:
        return ""
    chunk = 8192
    with open(path, "rb") as fh:
        buf = b""
        pos = size
        while pos > 0:
            step = min(chunk, pos)
            pos -= step
            fh.seek(pos)
            buf = fh.read(step) + buf
            stripped = buf.rstrip(b"\r\n")
            if b"\n" in stripped:
                buf = stripped
                break
            if len(buf) > MAX_LINE_BYTES * 2:
                break
    line = buf.rstrip(b"\r\n").split(b"\n")[-1]
    return line.decode("utf-8", errors="replace")


def _tail_state(path: Path) -> tuple:
    """`(last_seq, last_hash)` for the next append. Raises if the tail is unreadable.

    Refusing here is the point. If the final line is torn or tampered, appending on top of
    it would bury the break under a valid-looking chain -- the next `verify()` would still
    find it, but only after we had written over the evidence of where it started.
    """
    line = _last_line(path)
    if not line:
        return 0, GENESIS_PREV_HASH
    try:
        entry = json.loads(line)
    except ValueError as e:
        raise JournalIntegrityError(
            f"refusing to append: last line of {path} is not valid JSON ({e}). "
            f"Run journal.verify() -- a torn tail and a tampered tail look the same "
            f"from here, and only one of them is innocent."
        )
    if not isinstance(entry, dict) or "seq" not in entry or "hash" not in entry:
        raise JournalIntegrityError(
            f"refusing to append: last line of {path} is not a journal entry "
            f"(no seq/hash). Run journal.verify()."
        )
    seq = entry.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
        raise JournalIntegrityError(
            f"refusing to append: last line of {path} has a non-integer seq {seq!r}."
        )
    return seq, entry["hash"]


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------

def append(journal_path, writer: str, surface: str, op: str, target: dict,
           payload: dict, event_ts: float = 0.0) -> dict:
    """Append one entry. Returns the stored entry plus the lock verdict.

    `hash` covers every preceding field; `prev_hash` chains to seq-1. `seq` is monotonic
    and GAPLESS -- a gap is a detected deletion, which is the only reason a counter is
    worth carrying when we already have hashes.

    Returns the entry as written, with two extra keys that are NOT stored and NOT hashed:

        _locked      bool -- whether an OS-level exclusive lock was actually held
        _lock_error  str  -- why not, if not

    Raises ValueError on bad input rather than storing junk in a ledger nobody can prune,
    and JournalIntegrityError if the existing tail cannot be chained onto.
    """
    path = Path(journal_path)

    # ---- validate before touching anything. A ledger cannot be cleaned up later.
    writer = (writer or "").strip()
    if not writer:
        raise ValueError("writer is required: an unattributed entry is not provenance")
    if surface not in VALID_SURFACES:
        raise ValueError(f"surface must be one of {VALID_SURFACES}, got {surface!r}")
    if op not in VALID_OPS:
        raise ValueError(f"op must be one of {VALID_OPS}, got {op!r} -- add new verbs to "
                         f"journal.VALID_OPS deliberately, so the vocabulary stays "
                         f"auditable")
    if not isinstance(target, dict) or not target:
        raise ValueError("target must be a non-empty dict, e.g. "
                         "{'cart': 'andy.cart', 'idx': 41234}")
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")
    if not isinstance(event_ts, (int, float)) or isinstance(event_ts, bool):
        raise ValueError(f"event_ts must be a number, got {event_ts!r}")
    if event_ts < 0:
        raise ValueError(f"event_ts must not be negative, got {event_ts!r}")

    # ---- two clocks, sampled independently, NEVER derived from each other.
    #
    # `project_event_time_vs_ingest_time_2026-07-30` -- one bug in three systems -- is
    # what happens when one of these is computed from the other. So:
    #
    #   event_ts   when it HAPPENED. The caller states it when it knows (a backdated
    #              import, a replayed federation batch). When it does not, the event is
    #              this call, and we take our OWN reading here, before any lock wait.
    #   ingest_ts  when we RECORDED it. Taken below, under the lock, immediately before
    #              the write.
    #
    # They are two observations of two different events, not one value copied twice. The
    # gap between them is real information -- it is the lock wait and the queue depth --
    # and a system that derives one from the other destroys exactly that.
    if not event_ts:
        event_ts = time.time()
    event_ts = float(event_ts)

    with _FileLock(path) as lock:
        last_seq, last_hash = _tail_state(path)
        entry = {
            "seq": last_seq + 1,
            "event_ts": event_ts,
            "ingest_ts": float(time.time()),
            "writer": writer,
            "surface": surface,
            "op": op,
            "target": target,
            "payload": payload,
            "prev_hash": last_hash,
        }
        try:
            entry["hash"] = compute_hash(entry)
            line = json.dumps({k: entry[k] for k in ENTRY_FIELDS},
                              sort_keys=True, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"target/payload must be JSON-serializable: {e}")
        blob = (line + "\n").encode("utf-8")
        if len(blob) > MAX_LINE_BYTES:
            raise ValueError(f"entry is {len(blob)} bytes, over the "
                             f"{MAX_LINE_BYTES} limit; store the body by reference")
        if b"\n" in blob[:-1]:  # defensive: json escapes newlines, but one event per line
            raise ValueError("serialized entry contains a newline; refusing to write")

        path.parent.mkdir(parents=True, exist_ok=True)
        # Append + flush + fsync rather than mkstemp/os.replace. The sidecar stores in this
        # tree write whole files atomically by replacement; a journal cannot -- rewriting
        # the file to add a line is a rewrite of history by definition, and would also
        # make every append O(n). A single write() of one line under an exclusive lock,
        # fsynced, is the append-only equivalent: the line either lands whole or the
        # crash-torn remainder is detectable as a truncated tail (see verify()).
        with open(path, "ab") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())

        out = dict(entry)
        out["_locked"] = lock.locked
        out["_lock_error"] = lock.error
        return out


def append_to_cart(cart_path, *args, **kwargs) -> dict:
    """`append()` against the journal sidecar of a cart. Convenience, same semantics."""
    return append(journal_path_for(cart_path), *args, **kwargs)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def _walk(path: Path):
    """Yield `(line_no, raw_line, parsed_or_None, parse_error)` for every non-blank line."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                yield line_no, raw, json.loads(stripped), ""
            except ValueError as e:
                yield line_no, raw, None, str(e)


def verify(journal_path) -> dict:
    """Walk the whole chain. Returns a structured verdict naming the FIRST broken link.

    A broken chain is LOUD (§6.3): `ok` is False and `first_break` says which record,
    which line, and which checks failed -- never a generic "integrity error".

    Reasons, and what each one actually means:

        malformed_json     a line in the middle is not JSON. Someone edited the file.
        truncated_tail     the LAST line is incomplete. Distinguished from tampering
                           because it is the one break a power failure can cause: the
                           entries before it are intact and the chain up to them verifies.
        missing_field      an entry is missing one of §5's fields.
        seq_gap            found_seq > expected. A RECORD WAS DELETED.
        seq_regression     found_seq <= expected. Records were reordered or duplicated.
        prev_hash_mismatch this record does not chain to the one before it.
        hash_mismatch      this record's own hash does not cover its own contents. The
                           record was edited in place.

    Deletion and reordering are told apart by `missing_seqs`: a deletion leaves a hole in
    the seq set, a reorder does not. Both break the chain; only one loses data.
    """
    path = Path(journal_path)
    verdict = {
        "path": str(path),
        "format": FORMAT,
        "version": VERSION,
        "exists": path.exists(),
        "ok": False,
        "entries": 0,
        "first_seq": 0,
        "last_seq": 0,
        "missing_seqs": [],
        "duplicate_seqs": [],
        "truncated_tail": False,
        "first_break": None,
        "breaks": 0,
    }
    # Every exit from this function sets `verdict_line`. A caller that prints the verdict
    # must never hit a KeyError on the path where there is nothing to verify -- "no
    # journal" is the single most alarming answer this function can give, and it is the
    # one an early return is most likely to swallow.
    if not path.exists():
        verdict["first_break"] = {"reason": "no_journal",
                                  "detail": f"{path} does not exist"}
        verdict["verdict_line"] = f"NO JOURNAL at {path} -- nothing to verify"
        return verdict
    if path.stat().st_size == 0:
        # An empty journal is a valid, verified, zero-length chain. It is NOT a pass for a
        # system that should have written something -- "a verdict on an empty result set"
        # is the caller's trap, so we say so explicitly rather than just returning ok.
        verdict["ok"] = True
        verdict["empty"] = True
        verdict["verdict_line"] = (f"EMPTY: {path} exists and holds 0 entries -- "
                                   f"a verified chain of nothing, not a record of nothing")
        return verdict

    lines = list(_walk(path))
    seqs = []
    expected_seq = 1
    prev_hash = GENESIS_PREV_HASH
    breaks = []

    for idx, (line_no, raw, entry, parse_error) in enumerate(lines):
        is_last = (idx == len(lines) - 1)
        if entry is None or not isinstance(entry, dict):
            reason = "truncated_tail" if is_last else "malformed_json"
            if reason == "truncated_tail":
                verdict["truncated_tail"] = True
            breaks.append({
                "line_no": line_no,
                "seq": None,
                "reasons": [reason],
                "detail": parse_error or "line is not a JSON object",
                "raw_preview": raw.strip()[:160],
            })
            break  # nothing after an unparseable line can be chained to it

        missing = [f for f in ENTRY_FIELDS if f not in entry]
        if missing:
            reason = "truncated_tail" if (is_last and "hash" in missing) else "missing_field"
            if reason == "truncated_tail":
                verdict["truncated_tail"] = True
            breaks.append({
                "line_no": line_no,
                "seq": entry.get("seq"),
                "reasons": [reason],
                "detail": f"missing field(s): {', '.join(missing)}",
            })
            break

        reasons = []
        detail = []
        found_seq = entry["seq"]
        if found_seq != expected_seq:
            if isinstance(found_seq, int) and not isinstance(found_seq, bool) \
                    and found_seq > expected_seq:
                reasons.append("seq_gap")
            else:
                reasons.append("seq_regression")
            detail.append(f"expected seq {expected_seq}, found {found_seq!r}")
        if entry["prev_hash"] != prev_hash:
            reasons.append("prev_hash_mismatch")
            detail.append(f"prev_hash {entry['prev_hash']!r} does not chain to "
                          f"{prev_hash!r}")
        try:
            recomputed = compute_hash(entry)
        except (TypeError, ValueError) as e:
            recomputed = ""
            reasons.append("hash_mismatch")
            detail.append(f"cannot hash entry: {e}")
        if recomputed and recomputed != entry["hash"]:
            reasons.append("hash_mismatch")
            detail.append(f"stored hash {entry['hash']!r}, recomputed {recomputed!r}")

        if isinstance(found_seq, int) and not isinstance(found_seq, bool):
            seqs.append(found_seq)
        if reasons:
            breaks.append({
                "line_no": line_no,
                "seq": found_seq,
                "reasons": reasons,
                "detail": "; ".join(detail),
                "expected_seq": expected_seq,
                "expected_prev_hash": prev_hash,
                "found_prev_hash": entry["prev_hash"],
                "stored_hash": entry["hash"],
                "recomputed_hash": recomputed,
                "writer": entry.get("writer"),
                "op": entry.get("op"),
            })
            # Keep walking: the chain is already broken, but the SHAPE of the damage (one
            # edited record vs. a wholesale rewrite from here on) is what tells a human
            # whether this was a mistake or an attack.
        # Chain onward from what is ON DISK, not from what we recomputed. Chaining from
        # the recomputed hash would silently "heal" a tampered record for every record
        # after it, and report one break where there is a whole rewritten suffix.
        prev_hash = entry["hash"]
        expected_seq = (found_seq + 1) if isinstance(found_seq, int) \
            and not isinstance(found_seq, bool) else expected_seq + 1

    verdict["entries"] = len([e for _, _, e, _ in lines if isinstance(e, dict)])
    if seqs:
        verdict["first_seq"] = min(seqs)
        verdict["last_seq"] = max(seqs)
        seen = set()
        dupes = sorted({s for s in seqs if s in seen or seen.add(s)})
        verdict["duplicate_seqs"] = dupes
        verdict["missing_seqs"] = [s for s in range(1, max(seqs) + 1) if s not in seen][:64]
    verdict["breaks"] = len(breaks)
    verdict["first_break"] = breaks[0] if breaks else None
    verdict["ok"] = not breaks
    if breaks:
        b = breaks[0]
        verdict["verdict_line"] = (
            f"BROKEN at line {b['line_no']} (seq {b['seq']}): "
            f"{', '.join(b['reasons'])} -- {b.get('detail', '')}"
        )
    else:
        verdict["verdict_line"] = (
            f"OK: {verdict['entries']} entries, seq {verdict['first_seq']}.."
            f"{verdict['last_seq']}, chain intact"
        )
    return verdict


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def replay(journal_path, strict: bool = True) -> list:
    """Entries in seq order, for state reconstruction.

    §4 crash recovery: load the cart, THEN replay the journal ("boot time is coffee
    sipping time"). Replay is deterministic because embeddings are inlined in the payload
    and never re-embedded -- re-embedding would make reconstruction depend on which model
    version happened to be installed, which is not reconstruction.

    `strict=True` (default) raises JournalIntegrityError on a broken chain, because §6.3
    says a broken chain is loud and booting on top of one would launder it. A truncated
    tail -- the one break a crash can cause -- is dropped with the rest of the chain
    intact, which is the honest reading of an interrupted write.

    `strict=False` returns whatever parses, for forensics on a journal you already know is
    damaged. Never use it on a boot path.
    """
    path = Path(journal_path)
    if not path.exists():
        return []
    v = verify(path)
    if not v["ok"] and strict:
        b = v["first_break"] or {}
        only_truncated_tail = (v["breaks"] == 1 and b.get("reasons") == ["truncated_tail"])
        if not only_truncated_tail:
            raise JournalIntegrityError(v.get("verdict_line", "journal chain is broken"))

    out = [e for _, _, e, _ in _walk(path)
           if isinstance(e, dict) and isinstance(e.get("seq"), int)
           and not isinstance(e.get("seq"), bool)]
    # Sort by seq, not by file order. On a healthy journal they are identical; where they
    # differ the file lied and `verify()` has already said so.
    out.sort(key=lambda e: e["seq"])
    return out


# ---------------------------------------------------------------------------
# surfacing (§8): provenance travels WITH the result
# ---------------------------------------------------------------------------

def _target_matches(entry_target, want) -> bool:
    if not isinstance(entry_target, dict):
        return False
    if isinstance(want, dict):
        # Every key the caller named must match. A caller naming only {"cart": x} gets
        # every entry for that cart; naming {"cart": x, "idx": n} gets one passage.
        return all(entry_target.get(k) == v for k, v in want.items())
    # A bare string matches any target VALUE -- a content key, a cart name, an index
    # rendered as text. Deliberately permissive on the read side: a missed provenance
    # line is a silent loss of the one signal the reader cannot skip.
    return any(v == want or str(v) == str(want) for v in entry_target.values())


def for_passage(journal_path, target, n: int = 3) -> list:
    """The N most recent entries referencing one passage, MOST RECENT FIRST.

    §8 is the whole point of this function. Three surfacing options were considered and
    the two dashboard-shaped ones were rejected:

        a changelog window on the Hot Stack      -> a log nobody reads
        a rolling 10-item corpus-wide changelog  -> ambient, mostly irrelevant, trains
                                                    the reader to skip it
        THE N MOST RECENT ENTRIES FOR THIS       -> contextual. The reader cannot skip it,
        PASSAGE, RETURNED WITH IT                   because it is attached to the thing
                                                    they asked for.

    And it answers the question that matters at that moment -- should I trust this passage
    right now -- rather than the question a dashboard answers, which is what has been
    happening lately.

    Ordered by `seq` descending, NOT by timestamp. seq is the order things were recorded
    and it cannot be forged without breaking the chain; `event_ts` is caller-supplied and a
    backdated import would otherwise bury a recent tombstone.
    """
    path = Path(journal_path)
    if not path.exists() or n <= 0:
        return []
    hits = []
    for _, _, entry, _ in _walk(path):
        if not isinstance(entry, dict):
            continue
        if _target_matches(entry.get("target"), target):
            hits.append(entry)
    hits.sort(key=lambda e: e.get("seq") or 0, reverse=True)
    return hits[:n]


def for_results(journal_path, targets, n: int = 3) -> dict:
    """`{target_key: [entries]}` for a batch of search hits, in one pass over the file.

    Same motive as `proposal_store.for_results`: provenance rides back with the results
    instead of needing its own channel and its own round trip per hit.
    """
    wanted = list(targets or [])
    if not wanted:
        return {}
    path = Path(journal_path)
    if not path.exists():
        return {t if isinstance(t, str) else json.dumps(t, sort_keys=True): [] for t in wanted}
    keys = [t if isinstance(t, str) else json.dumps(t, sort_keys=True) for t in wanted]
    grouped = {k: [] for k in keys}
    for _, _, entry, _ in _walk(path):
        if not isinstance(entry, dict):
            continue
        et = entry.get("target")
        for want, key in zip(wanted, keys):
            if _target_matches(et, want):
                grouped[key].append(entry)
    for key in grouped:
        grouped[key].sort(key=lambda e: e.get("seq") or 0, reverse=True)
        grouped[key] = grouped[key][:n]
    return grouped


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def stats(journal_path) -> dict:
    """Shape of the journal. Cheap; says nothing about integrity.

    `gapless` here is an arithmetic check on the seq range, not a chain verification --
    `verify()` is the only thing that can say the journal is intact, and this function
    deliberately does not return a key that could be mistaken for its answer.
    """
    path = Path(journal_path)
    out = {
        "path": str(path),
        "format": FORMAT,
        "version": VERSION,
        "exists": path.exists(),
        "bytes": 0,
        "entries": 0,
        "malformed_lines": 0,
        "first_seq": 0,
        "last_seq": 0,
        "gapless": True,
        "by_op": {},
        "by_writer": {},
        "by_surface": {},
        "writers": [],
        "first_event_ts": 0.0,
        "last_event_ts": 0.0,
        "first_ingest_ts": 0.0,
        "last_ingest_ts": 0.0,
    }
    if not path.exists():
        return out
    out["bytes"] = path.stat().st_size
    seqs = []
    event_ts = []
    ingest_ts = []
    for _, _, entry, _ in _walk(path):
        if not isinstance(entry, dict):
            out["malformed_lines"] += 1
            continue
        out["entries"] += 1
        for field, bucket in (("op", "by_op"), ("writer", "by_writer"),
                              ("surface", "by_surface")):
            key = entry.get(field) or "?"
            out[bucket][key] = out[bucket].get(key, 0) + 1
        s = entry.get("seq")
        if isinstance(s, int) and not isinstance(s, bool):
            seqs.append(s)
        if isinstance(entry.get("event_ts"), (int, float)):
            event_ts.append(float(entry["event_ts"]))
        if isinstance(entry.get("ingest_ts"), (int, float)):
            ingest_ts.append(float(entry["ingest_ts"]))
    if seqs:
        out["first_seq"] = min(seqs)
        out["last_seq"] = max(seqs)
        out["gapless"] = (out["last_seq"] - out["first_seq"] + 1) == len(set(seqs)) \
            and len(set(seqs)) == len(seqs)
    if event_ts:
        out["first_event_ts"] = min(event_ts)
        out["last_event_ts"] = max(event_ts)
    if ingest_ts:
        out["first_ingest_ts"] = min(ingest_ts)
        out["last_ingest_ts"] = max(ingest_ts)
    out["writers"] = sorted(out["by_writer"])
    return out


# ---------------------------------------------------------------------------
# CLI -- `python -m journal verify <path>` so an operator can check a chain without
# writing code. The verdict line comes FIRST and is computed, not asserted.
# ---------------------------------------------------------------------------

def _main(argv) -> int:
    if len(argv) < 3 or argv[1] not in ("verify", "stats", "tail"):
        print("usage: python journal.py {verify|stats|tail} <journal.jsonl> [n]")
        return 2
    cmd, path = argv[1], argv[2]
    if cmd == "verify":
        v = verify(path)
        print(v["verdict_line"])
        if not v["ok"]:
            print(json.dumps(v["first_break"], indent=2, sort_keys=True))
        return 0 if v["ok"] else 1
    if cmd == "stats":
        print(json.dumps(stats(path), indent=2, sort_keys=True))
        return 0
    n = int(argv[3]) if len(argv) > 3 else 10
    for entry in replay(path, strict=False)[-n:]:
        print(f"{entry.get('seq'):>8}  {entry.get('op'):<12} "
              f"{entry.get('writer')}/{entry.get('surface'):<4} "
              f"{json.dumps(entry.get('target'), sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
