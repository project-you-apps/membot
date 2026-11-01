"""Tests for the provenance journal.

⛔ THE RULE THESE TESTS ARE WRITTEN AGAINST
-------------------------------------------
"EXISTENCE IS NOT OPERATION -- a check identical in both states is not a check."

A test that a hash chain *has* a hash field passes whether or not the chain works. So
every test below is paired with the question "would this fail if the mechanism were
switched off?":

    test_clean_chain_verifies          fails if verify() always says broken
                                       (the positive control -- it is what makes every
                                        detection test below mean something)
    test_tampered_payload_*            fails if the hash does not cover the payload
    test_hash_covers_every_field       fails if ANY §5 field is left out of the hash
    test_deleted_middle_record_*       fails if seq is not gapless OR prev_hash unchained
    test_reordered_records_*           fails if order is not pinned by the chain
    test_truncated_tail_*              fails if a torn write looks like tampering
    test_replay_*                      fails if replay trusts file order
    test_for_passage_*                 fails if the target filter leaks other passages
    test_two_processes_*               fails if the OS-level lock is not taken
    test_event_ts_not_derived          fails if one timestamp is computed from the other
    test_no_compaction_api             fails the day somebody adds compact() or delete()

Run per CLAUDE_DO_NOT_RUN_TESTS.md:
    python -m pytest test_journal.py > /tmp/journal.log 2>&1   # then grep the log
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import journal  # noqa: E402

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _seed(path, count=5, writer="andy", surface="CLI", op="store", cart="andy.cart"):
    """A clean chain of `count` entries, one per passage index."""
    out = []
    for i in range(count):
        out.append(journal.append(
            path, writer=writer, surface=surface, op=op,
            target={"cart": cart, "idx": 1000 + i},
            payload={"text": f"passage {i}", "embedding": [0.1 * i, 0.2 * i]},
        ))
    return out


def _lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def _write_lines(path, lines):
    Path(path).write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# positive control
# ---------------------------------------------------------------------------

def test_clean_chain_verifies(tmp_path):
    """The control. If this fails, no detection test below proves anything."""
    p = tmp_path / "j.jsonl"
    _seed(p, 6)
    v = journal.verify(p)
    assert v["ok"] is True, v
    assert v["first_break"] is None
    assert v["entries"] == 6
    assert (v["first_seq"], v["last_seq"]) == (1, 6)
    assert v["missing_seqs"] == []
    assert v["duplicate_seqs"] == []
    assert v["truncated_tail"] is False
    assert "chain intact" in v["verdict_line"]


def test_first_entry_chains_to_genesis(tmp_path):
    p = tmp_path / "j.jsonl"
    e = journal.append(p, "andy", "CLI", "store", {"cart": "c", "idx": 1}, {})
    assert e["seq"] == 1
    assert e["prev_hash"] == journal.GENESIS_PREV_HASH
    assert e["hash"].startswith("sha256:")
    assert e["hash"] != journal.GENESIS_PREV_HASH


def test_every_entry_chains_to_its_predecessor(tmp_path):
    p = tmp_path / "j.jsonl"
    entries = _seed(p, 5)
    for prev, cur in zip(entries, entries[1:]):
        assert cur["prev_hash"] == prev["hash"]
        assert cur["seq"] == prev["seq"] + 1


# ---------------------------------------------------------------------------
# tampering
# ---------------------------------------------------------------------------

def test_tampered_payload_in_middle_record_is_detected(tmp_path):
    """Edit the content of record 3 in place, keep its hash. Must be caught AT record 3."""
    p = tmp_path / "j.jsonl"
    _seed(p, 5)
    lines = _lines(p)
    entry = json.loads(lines[2])
    assert entry["seq"] == 3
    entry["payload"]["text"] = "quietly rewritten"      # the forgery
    lines[2] = json.dumps(entry, sort_keys=True)        # hash deliberately left alone
    _write_lines(p, lines)

    v = journal.verify(p)
    assert v["ok"] is False
    b = v["first_break"]
    assert b["seq"] == 3 and b["line_no"] == 3, b
    assert "hash_mismatch" in b["reasons"], b
    # seq and prev_hash are still fine -- the forger only changed the body, and the hash
    # is the only thing that notices. That is the point of hashing the whole record.
    assert "seq_gap" not in b["reasons"]
    assert "prev_hash_mismatch" not in b["reasons"]
    assert b["stored_hash"] != b["recomputed_hash"]


def test_tampered_writer_is_detected(tmp_path):
    """Attribution is the asset. Changing who wrote it must break the seal."""
    p = tmp_path / "j.jsonl"
    _seed(p, 3, writer="andy")
    lines = _lines(p)
    entry = json.loads(lines[1])
    entry["writer"] = "someone-else"
    lines[1] = json.dumps(entry, sort_keys=True)
    _write_lines(p, lines)
    v = journal.verify(p)
    assert v["ok"] is False
    assert "hash_mismatch" in v["first_break"]["reasons"]
    assert v["first_break"]["seq"] == 2


@pytest.mark.parametrize("field,bad", [
    ("seq", 99),
    ("event_ts", 1.0),
    ("ingest_ts", 1.0),
    ("writer", "mallory"),
    ("surface", "WEB"),
    ("op", "tombstone"),
    ("target", {"cart": "other.cart", "idx": 7}),
    ("payload", {"text": "different"}),
    ("prev_hash", journal.GENESIS_PREV_HASH),
])
def test_hash_covers_every_field(tmp_path, field, bad):
    """§5: the hash is over EVERY preceding field.

    One case per field. If any field were left out of HASHED_FIELDS, exactly one of these
    nine would pass while the record verified -- which is how a field quietly stops being
    covered without anyone noticing.
    """
    p = tmp_path / f"j_{field}.jsonl"
    _seed(p, 3)
    lines = _lines(p)
    entry = json.loads(lines[1])
    assert entry[field] != bad, "test fixture must actually change the value"
    entry[field] = bad
    lines[1] = json.dumps(entry, sort_keys=True)
    _write_lines(p, lines)
    v = journal.verify(p)
    assert v["ok"] is False, f"editing {field} went undetected"
    assert "hash_mismatch" in v["first_break"]["reasons"], \
        f"{field} is not covered by the hash: {v['first_break']}"


def test_rehashing_a_tampered_record_still_breaks_the_chain(tmp_path):
    """The thorough forger recomputes the hash. The NEXT record's prev_hash catches it.

    This is the test that makes the chain worth having at all: a per-record hash alone is
    defeated by anyone who can run sha256.
    """
    p = tmp_path / "j.jsonl"
    _seed(p, 5)
    lines = _lines(p)
    entry = json.loads(lines[2])
    entry["payload"]["text"] = "rewritten and resealed"
    entry["hash"] = journal.compute_hash(entry)          # a valid hash of a false record
    lines[2] = json.dumps(entry, sort_keys=True)
    _write_lines(p, lines)

    v = journal.verify(p)
    assert v["ok"] is False
    b = v["first_break"]
    assert b["seq"] == 4, b          # record 3 self-verifies; record 4 no longer chains
    assert "prev_hash_mismatch" in b["reasons"], b
    assert "hash_mismatch" not in b["reasons"], b


# ---------------------------------------------------------------------------
# deletion
# ---------------------------------------------------------------------------

def test_deleted_middle_record_is_detected_as_a_seq_gap(tmp_path):
    p = tmp_path / "j.jsonl"
    _seed(p, 5)
    lines = _lines(p)
    del lines[2]                                   # seq 3 is gone
    _write_lines(p, lines)

    v = journal.verify(p)
    assert v["ok"] is False
    b = v["first_break"]
    assert b["seq"] == 4 and b["expected_seq"] == 3, b
    assert "seq_gap" in b["reasons"], b
    assert v["missing_seqs"] == [3], v
    assert v["entries"] == 4


def test_deleted_middle_record_is_also_detected_as_a_chain_break(tmp_path):
    """Belt and braces, separately asserted: the hash chain catches it too.

    Deleting a record and renumbering every later seq would defeat the counter. It cannot
    defeat the chain, which is why both checks exist and why they are tested apart.
    """
    p = tmp_path / "j.jsonl"
    _seed(p, 5)
    lines = _lines(p)
    del lines[2]
    _write_lines(p, lines)
    b = journal.verify(p)["first_break"]
    assert "prev_hash_mismatch" in b["reasons"], b
    assert b["found_prev_hash"] != b["expected_prev_hash"]


def test_deleted_record_with_renumbered_seqs_is_still_detected(tmp_path):
    """The seq counter alone is not the defense. Renumber it away; the chain holds."""
    p = tmp_path / "j.jsonl"
    _seed(p, 5)
    lines = _lines(p)
    del lines[2]
    kept = [json.loads(ln) for ln in lines]
    for i, entry in enumerate(kept, start=1):
        entry["seq"] = i                           # cover the tracks
    _write_lines(p, [json.dumps(e, sort_keys=True) for e in kept])

    v = journal.verify(p)
    assert v["ok"] is False
    assert v["missing_seqs"] == [], "renumbering hid the gap, as expected"
    b = v["first_break"]
    assert "hash_mismatch" in b["reasons"] or "prev_hash_mismatch" in b["reasons"], b


def test_deleting_the_tail_is_detected_by_the_high_water_mark(tmp_path):
    """⚠ THE ORIGINAL VERSION OF THIS TEST PINNED THE HOLE; this one pins the fix.

    The insight it recorded is still exactly right and is preserved as the control below:
    chopping records off the END leaves a perfectly valid shorter chain, and NOTHING
    INSIDE THE FILE can detect it. Detection requires an external witness.

    ⇒ 2026-11-01 added that witness -- a `{seq, hash}` mark in a sidecar, written after
      each append. So the hole is closed to the extent that an external witness closes
      it: a trailing deletion must now edit a SECOND file to stay invisible.

    ⛔ Tamper-EVIDENCE, not tamper-proofing. Someone with write access to the directory
      can update both files; only anchoring the head hash somewhere we do not control
      makes this detectable by a party who does not trust us. That is still unbuilt.
    """
    p = tmp_path / "j.jsonl"
    _seed(p, 6)
    lines = _lines(p)
    _write_lines(p, lines[:3])

    v = journal.verify(p)
    assert v["ok"] is False, "a cleanly truncated tail must not verify"
    assert "entries_deleted_from_tail" in v["first_break"]["reasons"]
    assert v["last_seq"] == 3
    assert v["hwm_seq"] == 6, "the mark is the witness that says three entries are gone"


def test_a_crash_torn_tail_is_distinguished_from_a_deletion(tmp_path):
    """⛔⛔ THE BRANCH THAT BREAKS BOOT IF IT IS WRONG, and it had no test.

    Both cases leave the mark AHEAD of the newest readable entry, so without this
    distinction a hard kill mid-write would be reported as tampering and replay-on-boot
    would refuse to start. The discriminator is that a crash leaves the last line
    physically TORN, because the mark is written only after that line is fsynced.

    ⚠ Asserting only "the torn case still verifies" would be a check identical in both
      states: it would pass if the whole tail-deletion check were deleted. So this test
      asserts BOTH SIDES in one place -- same mark offset, opposite verdicts.
    """
    torn = tmp_path / "torn.jsonl"
    _seed(torn, 5)
    raw = Path(torn).read_bytes()
    Path(torn).write_bytes(raw[: len(raw) - 40])       # tear the newest line
    v_torn = journal.verify(torn)

    clean_cut = tmp_path / "cut.jsonl"
    _seed(clean_cut, 5)
    _write_lines(clean_cut, _lines(clean_cut)[:4])     # remove it cleanly instead
    v_cut = journal.verify(clean_cut)

    # Same shape of evidence: in both, the mark says 5 and only 4 entries are readable.
    assert v_torn["hwm_seq"] == 5 and v_cut["hwm_seq"] == 5
    assert v_torn["last_seq"] == 4 and v_cut["last_seq"] == 4

    # ⓘ BOTH are `ok: False` -- a torn tail was already a break before the mark existed,
    #   and `replay()` carries the tolerance rather than `verify()`. So "ok" is NOT the
    #   discriminator; the REASON is, and the consequence that matters is whether the
    #   server can still boot.
    torn_reasons = (v_torn["first_break"] or {}).get("reasons", [])
    cut_reasons = (v_cut["first_break"] or {}).get("reasons", [])

    assert "entries_deleted_from_tail" not in torn_reasons, (
        f"A CRASH MID-WRITE IS BEING REPORTED AS DELETION, which accuses an operator of "
        f"tampering every time the process is killed. reasons: {torn_reasons}"
    )
    assert v_torn.get("torn_tail_confirmed_by_hwm") is True
    assert "entries_deleted_from_tail" in cut_reasons, (
        f"a clean trailing deletion was tolerated as if it were a crash, which would let "
        f"anyone hide their last action by deleting one line. reasons: {cut_reasons}"
    )

    # ⭐ The consequence, asserted directly rather than inferred: a crash must still boot,
    #   and a deletion must not be quietly replayed as though nothing were missing.
    assert [e["seq"] for e in journal.replay(torn)] == [1, 2, 3, 4], (
        "replay-on-boot cannot recover from a crash mid-write"
    )
    with pytest.raises(journal.JournalIntegrityError):
        journal.replay(clean_cut)


def test_resealing_the_final_entry_is_detected(tmp_path):
    """The in-file chain cannot see this: rewrite the last entry, recompute its own hash,
    and every link still agrees. The mark's recorded hash is the only witness."""
    p = tmp_path / "j.jsonl"
    _seed(p, 4)
    lines = _lines(p)
    last = json.loads(lines[-1])
    last["payload"] = {"text": "rewritten after the fact", "embedding": [0.0]}
    last["hash"] = journal.compute_hash(last)          # reseal it properly
    _write_lines(p, lines[:-1] + [json.dumps(last, sort_keys=True)])

    v = journal.verify(p)
    assert v["ok"] is False, f"A RESEALED FINAL ENTRY VERIFIED CLEAN: {v['verdict_line']}"
    assert "tail_entry_rewritten" in v["first_break"]["reasons"], (
        f"detected, but reported as something else, so the operator is told the wrong "
        f"thing: {v['first_break']}"
    )


def test_a_mark_behind_the_file_is_reported_but_not_a_break(tmp_path):
    """A failed mark write leaves a STALE mark while the chain itself is intact. That is
    an anomaly worth surfacing and NOT a reason to call the journal broken -- treating it
    as tampering would turn a disk hiccup into a refusal to boot."""
    p = tmp_path / "j.jsonl"
    _seed(p, 4)
    journal._write_hwm(p, 2, journal.read_hwm(p)["hash"])   # rewind the mark

    v = journal.verify(p)
    assert v["ok"] is True, "a stale mark must not invalidate an intact chain"
    assert v.get("hwm_stale") is True, "but it must be visible, not swallowed"


def test_without_the_mark_a_truncated_tail_is_still_undetectable(tmp_path):
    """⛔ THE CONTROL, and the original lesson kept intact: the CHAIN cannot do this.

    If this ever starts failing, the chain is somehow catching trailing deletion on its
    own, which would mean the mark is unnecessary -- and that is worth knowing rather than
    assuming.
    """
    p = tmp_path / "j.jsonl"
    _seed(p, 6)
    lines = _lines(p)
    _write_lines(p, lines[:3])
    journal.hwm_path_for(p).unlink()

    v = journal.verify(p)
    assert v["ok"] is True, "a truncated-at-the-end chain IS internally consistent"
    assert v["hwm_present"] is False


# ---------------------------------------------------------------------------
# reordering
# ---------------------------------------------------------------------------

def test_reordered_records_are_detected(tmp_path):
    p = tmp_path / "j.jsonl"
    _seed(p, 5)
    lines = _lines(p)
    lines[2], lines[3] = lines[3], lines[2]        # swap seq 3 and seq 4
    _write_lines(p, lines)

    v = journal.verify(p)
    assert v["ok"] is False
    b = v["first_break"]
    assert b["line_no"] == 3 and b["seq"] == 4, b
    assert "prev_hash_mismatch" in b["reasons"], b


def test_reordering_is_distinguished_from_deletion(tmp_path):
    """Same break, different damage. `missing_seqs` tells them apart.

    A reorder keeps every record (nothing lost, order lied about); a deletion leaves a
    hole. An operator needs to know which, and a single "integrity error" cannot say.
    """
    reordered = tmp_path / "reordered.jsonl"
    deleted = tmp_path / "deleted.jsonl"
    _seed(reordered, 5)
    _seed(deleted, 5)

    lines = _lines(reordered)
    lines[2], lines[3] = lines[3], lines[2]
    _write_lines(reordered, lines)

    lines = _lines(deleted)
    del lines[2]
    _write_lines(deleted, lines)

    vr = journal.verify(reordered)
    vd = journal.verify(deleted)
    assert vr["ok"] is False and vd["ok"] is False
    assert vr["entries"] == 5 and vr["missing_seqs"] == []      # nothing lost
    assert vd["entries"] == 4 and vd["missing_seqs"] == [3]     # seq 3 is gone
    assert vr["truncated_tail"] is False and vd["truncated_tail"] is False


def test_duplicated_record_is_detected(tmp_path):
    p = tmp_path / "j.jsonl"
    _seed(p, 4)
    lines = _lines(p)
    lines.insert(2, lines[2])                      # seq 3 twice
    _write_lines(p, lines)
    v = journal.verify(p)
    assert v["ok"] is False
    assert v["duplicate_seqs"] == [3], v
    assert "seq_regression" in v["first_break"]["reasons"], v["first_break"]


# ---------------------------------------------------------------------------
# truncation (a crash) vs. tampering (a person)
# ---------------------------------------------------------------------------

def test_truncated_last_line_is_detected(tmp_path):
    p = tmp_path / "j.jsonl"
    _seed(p, 4)
    raw = Path(p).read_bytes()
    Path(p).write_bytes(raw[: len(raw) - 40])      # tear the final write in half
    v = journal.verify(p)
    assert v["ok"] is False
    assert v["truncated_tail"] is True
    assert v["first_break"]["line_no"] == 4


def test_truncated_tail_is_distinguished_from_tampering(tmp_path):
    """Two broken journals, two different verdicts. Not one generic failure.

    A torn tail is what a power cut looks like and the records before it are intact.
    An edited middle record is what a person looks like. Reporting both as "chain broken"
    would make the loud failure useless -- every crash would read as an attack.
    """
    torn = tmp_path / "torn.jsonl"
    edited = tmp_path / "edited.jsonl"
    _seed(torn, 4)
    _seed(edited, 4)

    raw = Path(torn).read_bytes()
    Path(torn).write_bytes(raw[: len(raw) - 40])

    lines = _lines(edited)
    e = json.loads(lines[1])
    e["payload"]["text"] = "edited"
    lines[1] = json.dumps(e, sort_keys=True)
    _write_lines(edited, lines)

    vt = journal.verify(torn)
    ve = journal.verify(edited)
    assert vt["first_break"]["reasons"] == ["truncated_tail"], vt["first_break"]
    assert vt["truncated_tail"] is True
    assert ve["truncated_tail"] is False
    assert "hash_mismatch" in ve["first_break"]["reasons"]
    assert vt["first_break"]["reasons"] != ve["first_break"]["reasons"]


def test_malformed_middle_line_is_not_called_a_truncated_tail(tmp_path):
    p = tmp_path / "j.jsonl"
    _seed(p, 4)
    lines = _lines(p)
    lines[1] = lines[1][: len(lines[1]) // 2]      # garbage in the MIDDLE
    _write_lines(p, lines)
    v = journal.verify(p)
    assert v["ok"] is False
    assert v["first_break"]["reasons"] == ["malformed_json"], v["first_break"]
    assert v["truncated_tail"] is False


def test_append_refuses_to_chain_onto_a_torn_tail(tmp_path):
    """Appending on top of a break buries the evidence of where it started."""
    p = tmp_path / "j.jsonl"
    _seed(p, 3)
    raw = Path(p).read_bytes()
    Path(p).write_bytes(raw[: len(raw) - 30])
    with pytest.raises(journal.JournalIntegrityError):
        journal.append(p, "andy", "CLI", "store", {"cart": "c", "idx": 9}, {})


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def test_replay_returns_entries_in_seq_order(tmp_path):
    p = tmp_path / "j.jsonl"
    _seed(p, 7)
    entries = journal.replay(p)
    assert [e["seq"] for e in entries] == [1, 2, 3, 4, 5, 6, 7]
    assert [e["target"]["idx"] for e in entries] == list(range(1000, 1007))


def test_replay_orders_by_seq_not_by_file_order(tmp_path):
    """⚠ Fails if replay just reads lines top to bottom.

    §4 crash recovery replays the journal onto the cart. If replay honored file order, a
    reordered file would reconstruct the WRONG state -- a tombstone applied before the
    store it tombstones leaves the passage alive.
    """
    p = tmp_path / "j.jsonl"
    _seed(p, 5)
    lines = _lines(p)
    lines[2], lines[3] = lines[3], lines[2]
    _write_lines(p, lines)
    entries = journal.replay(p, strict=False)
    assert [e["seq"] for e in entries] == [1, 2, 3, 4, 5]


def test_replay_is_loud_on_a_broken_chain(tmp_path):
    """§6.3: a broken chain is loud, not a warning. Booting on one would launder it."""
    p = tmp_path / "j.jsonl"
    _seed(p, 5)
    lines = _lines(p)
    del lines[2]
    _write_lines(p, lines)
    with pytest.raises(journal.JournalIntegrityError) as ei:
        journal.replay(p)
    assert "seq" in str(ei.value).lower()
    assert len(journal.replay(p, strict=False)) == 4       # forensics still possible


def test_replay_survives_a_crash_torn_tail(tmp_path):
    """The one break a boot path should tolerate: the record that was mid-write."""
    p = tmp_path / "j.jsonl"
    _seed(p, 5)
    raw = Path(p).read_bytes()
    Path(p).write_bytes(raw[: len(raw) - 40])
    entries = journal.replay(p)
    assert [e["seq"] for e in entries] == [1, 2, 3, 4]


def test_replay_preserves_inlined_embeddings(tmp_path):
    """§4: serialize the embedding; do not re-embed on replay."""
    p = tmp_path / "j.jsonl"
    vec = [0.015625, -0.5, 1.0, 3.14159]
    journal.append(p, "andy", "CLI", "store", {"cart": "c", "idx": 1},
                   {"text": "hello", "embedding": vec})
    assert journal.replay(p)[0]["payload"]["embedding"] == vec


def test_replay_of_a_missing_journal_is_empty_not_an_error(tmp_path):
    assert journal.replay(tmp_path / "nope.jsonl") == []
    v = journal.verify(tmp_path / "nope.jsonl")
    assert v["ok"] is False and v["first_break"]["reason"] == "no_journal"


@pytest.mark.parametrize("state", ["missing", "empty", "clean", "broken"])
def test_every_verify_path_returns_a_verdict_line(tmp_path, state):
    """⚠ Found by running the CLI against a missing file: two early returns had no
    verdict line, so the operator check crashed on the single most alarming answer.

    "Andy sees only output" -- a verdict the caller cannot print is not a verdict.
    """
    p = tmp_path / f"{state}.jsonl"
    if state == "empty":
        p.write_bytes(b"")
    elif state in ("clean", "broken"):
        _seed(p, 3)
        if state == "broken":
            lines = _lines(p)
            del lines[1]
            _write_lines(p, lines)
    v = journal.verify(p)
    assert v.get("verdict_line"), f"no verdict_line for the {state} journal: {v}"
    assert v["ok"] is (state in ("empty", "clean"))


# ---------------------------------------------------------------------------
# §8 -- provenance travels WITH the result
# ---------------------------------------------------------------------------

def test_for_passage_returns_only_that_target_most_recent_first(tmp_path):
    p = tmp_path / "j.jsonl"
    want = {"cart": "andy.cart", "idx": 23285}
    other = {"cart": "andy.cart", "idx": 21100}
    journal.append(p, "andy", "WEB", "store", want, {"n": 1})
    journal.append(p, "claude", "MCP", "store", other, {"n": 2})
    journal.append(p, "andy", "WEB", "corroborate", want, {"n": 3})
    journal.append(p, "agent:sage-3", "MCP", "store", other, {"n": 4})
    journal.append(p, "andy", "CLI", "tombstone", want, {"n": 5})

    hits = journal.for_passage(p, want, n=3)
    assert [h["payload"]["n"] for h in hits] == [5, 3, 1], hits
    assert all(h["target"] == want for h in hits), "another passage leaked in"
    assert {h["op"] for h in hits} == {"store", "corroborate", "tombstone"}

    assert [h["payload"]["n"] for h in journal.for_passage(p, other, n=3)] == [4, 2]
    assert journal.for_passage(p, {"cart": "andy.cart", "idx": 99999}) == []


def test_for_passage_respects_n(tmp_path):
    p = tmp_path / "j.jsonl"
    t = {"cart": "c", "idx": 1}
    for i in range(6):
        journal.append(p, "andy", "CLI", "store", t, {"n": i})
    assert [h["payload"]["n"] for h in journal.for_passage(p, t, n=2)] == [5, 4]
    assert len(journal.for_passage(p, t, n=3)) == 3
    assert journal.for_passage(p, t, n=0) == []


def test_for_passage_orders_by_seq_not_event_ts(tmp_path):
    """⚠ A backdated import must not bury a recent tombstone.

    `event_ts` is caller-supplied. If ordering trusted it, anyone who can set a timestamp
    could push the most important provenance line out of the N that get shown.
    """
    p = tmp_path / "j.jsonl"
    t = {"cart": "c", "idx": 1}
    journal.append(p, "andy", "CLI", "store", t, {"n": 1}, event_ts=1_000_000.0)
    journal.append(p, "andy", "CLI", "tombstone", t, {"n": 2}, event_ts=1.0)
    hits = journal.for_passage(p, t, n=1)
    assert hits[0]["payload"]["n"] == 2, "ordered by event_ts, not seq"
    assert hits[0]["op"] == "tombstone"


def test_for_passage_matches_a_cart_wide_target_subset(tmp_path):
    p = tmp_path / "j.jsonl"
    journal.append(p, "andy", "CLI", "store", {"cart": "a.cart", "idx": 1}, {})
    journal.append(p, "andy", "CLI", "store", {"cart": "b.cart", "idx": 2}, {})
    journal.append(p, "andy", "CLI", "store", {"cart": "a.cart", "idx": 3}, {})
    hits = journal.for_passage(p, {"cart": "a.cart"}, n=10)
    assert [h["target"]["idx"] for h in hits] == [3, 1]


def test_for_results_groups_a_batch_of_hits(tmp_path):
    p = tmp_path / "j.jsonl"
    a = {"cart": "c", "idx": 1}
    b = {"cart": "c", "idx": 2}
    journal.append(p, "andy", "CLI", "store", a, {"n": 1})
    journal.append(p, "andy", "CLI", "store", b, {"n": 2})
    journal.append(p, "andy", "CLI", "corroborate", a, {"n": 3})
    grouped = journal.for_results(p, [a, b], n=5)
    assert len(grouped) == 2
    by_n = {k: [e["payload"]["n"] for e in v] for k, v in grouped.items()}
    assert sorted(by_n.values()) == [[2], [3, 1]]


# ---------------------------------------------------------------------------
# concurrency -- the lock
# ---------------------------------------------------------------------------

_WORKER = '''\
import sys
sys.path.insert(0, sys.argv[1])
import journal
path, writer, count = sys.argv[2], sys.argv[3], int(sys.argv[4])
unlocked = 0
for i in range(count):
    e = journal.append(path, writer, "MCP", "store",
                       {"cart": "shared.cart", "idx": i},
                       {"writer": writer, "i": i})
    if not e["_locked"]:
        unlocked += 1
print("unlocked=%d" % unlocked)
'''


def test_two_processes_appending_both_land_with_gapless_seqs(tmp_path):
    """⚠ Fails if the OS-level lock is not actually taken.

    Two processes, 30 appends each, no sleeps -- so the read-tail / hash / write window is
    genuinely contested. Without the lock this produces duplicate seqs (both processes
    read the same tail) or an interleaved partial line, and `verify()` says so. With it,
    60 entries numbered 1..60 and an intact chain.

    This is the test that makes "append-only" mean something: a journal that loses an
    entry to a race has the same evidentiary value as no journal.
    """
    p = tmp_path / "shared.jsonl"
    worker = tmp_path / "worker_append.py"
    worker.write_text(_WORKER, encoding="utf-8")

    n = 30
    procs = [
        subprocess.Popen([sys.executable, str(worker), str(HERE), str(p), w, str(n)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for w in ("proc-a", "proc-b")
    ]
    outs = [proc.communicate(timeout=120) for proc in procs]
    for proc, (out, err) in zip(procs, outs):
        assert proc.returncode == 0, f"worker failed: {err[-2000:]}"
        assert "unlocked=0" in out, f"a worker proceeded without a lock: {out!r}"

    v = journal.verify(p)
    assert v["ok"] is True, v.get("verdict_line")
    assert v["entries"] == 2 * n
    assert v["duplicate_seqs"] == []
    assert v["missing_seqs"] == []
    assert [e["seq"] for e in journal.replay(p)] == list(range(1, 2 * n + 1))

    s = journal.stats(p)
    assert s["by_writer"] == {"proc-a": n, "proc-b": n}, s["by_writer"]
    assert s["gapless"] is True


def test_append_reports_whether_it_held_a_lock(tmp_path):
    """The fallback is honest: `_locked` is a real observation, not a constant.

    If this platform could not lock, the return value says so and a caller can refuse --
    which is the whole difference between a fallback and a silent downgrade.
    """
    p = tmp_path / "j.jsonl"
    e = journal.append(p, "andy", "CLI", "store", {"cart": "c", "idx": 1}, {})
    assert e["_locked"] is True, e.get("_lock_error")
    assert e["_lock_error"] == ""
    # and the bookkeeping keys are NOT part of the record on disk or of the hash
    stored = json.loads(_lines(p)[0])
    assert "_locked" not in stored and "_lock_error" not in stored
    assert set(stored) == set(journal.ENTRY_FIELDS)
    assert journal.verify(p)["ok"] is True


def test_lock_file_is_a_sidecar_not_the_journal(tmp_path):
    p = tmp_path / "j.jsonl"
    journal.append(p, "andy", "CLI", "store", {"cart": "c", "idx": 1}, {})
    assert (tmp_path / "j.jsonl.lock").exists()
    assert journal.verify(p)["ok"] is True, "locking must not write into the journal"


# ---------------------------------------------------------------------------
# the two clocks
# ---------------------------------------------------------------------------

def test_event_ts_not_derived_from_ingest_ts(tmp_path):
    """§5: both stored, NEVER derived. One bug in three systems.

    A backdated event -- an import, a replayed federation batch -- must keep the time it
    HAPPENED and separately record the time we RECORDED it. If either were computed from
    the other, this test fails.
    """
    p = tmp_path / "j.jsonl"
    happened = 1_756_000_000.0
    e = journal.append(p, "andy", "WEB", "store", {"cart": "c", "idx": 1},
                       {"text": "an old document"}, event_ts=happened)
    assert e["event_ts"] == happened
    assert e["ingest_ts"] > happened + 1_000_000, "ingest_ts was copied from event_ts"
    stored = json.loads(_lines(p)[0])
    assert stored["event_ts"] == happened
    assert stored["ingest_ts"] == e["ingest_ts"]
    assert stored["event_ts"] != stored["ingest_ts"]


def test_both_timestamps_are_present_when_event_ts_is_omitted(tmp_path):
    """Omitting event_ts means "the event is this call" -- not "leave the field blank"."""
    p = tmp_path / "j.jsonl"
    e = journal.append(p, "andy", "CLI", "store", {"cart": "c", "idx": 1}, {})
    assert e["event_ts"] > 1_700_000_000.0
    assert e["ingest_ts"] > 1_700_000_000.0
    assert e["event_ts"] <= e["ingest_ts"]


def test_timestamps_are_covered_by_the_hash(tmp_path):
    """Nudging a timestamp is the quietest possible forgery, so it must be sealed."""
    p = tmp_path / "j.jsonl"
    _seed(p, 3)
    lines = _lines(p)
    e = json.loads(lines[1])
    e["ingest_ts"] = e["ingest_ts"] - 86400.0
    lines[1] = json.dumps(e, sort_keys=True)
    _write_lines(p, lines)
    assert "hash_mismatch" in journal.verify(p)["first_break"]["reasons"]


# ---------------------------------------------------------------------------
# validation -- refuse junk rather than store it in something unprunable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,frag", [
    (dict(writer="", surface="CLI", op="store"), "writer"),
    (dict(writer="  ", surface="CLI", op="store"), "writer"),
    (dict(writer="andy", surface="SMS", op="store"), "surface"),
    (dict(writer="andy", surface="cli", op="store"), "surface"),
    (dict(writer="andy", surface="CLI", op="delete"), "op"),
    (dict(writer="andy", surface="CLI", op="compact"), "op"),
])
def test_append_refuses_bad_attribution(tmp_path, kwargs, frag):
    p = tmp_path / "j.jsonl"
    with pytest.raises(ValueError) as ei:
        journal.append(p, target={"cart": "c", "idx": 1}, payload={}, **kwargs)
    assert frag in str(ei.value)
    assert not p.exists(), "a refused entry must not create the journal"


@pytest.mark.parametrize("target", [None, {}, "andy.cart", 41234, []])
def test_append_refuses_a_bad_target(tmp_path, target):
    p = tmp_path / "j.jsonl"
    with pytest.raises(ValueError) as ei:
        journal.append(p, "andy", "CLI", "store", target, {})
    assert "target" in str(ei.value)


def test_append_refuses_an_unserializable_payload(tmp_path):
    p = tmp_path / "j.jsonl"
    with pytest.raises(ValueError):
        journal.append(p, "andy", "CLI", "store", {"cart": "c", "idx": 1},
                       {"blob": object()})


def test_append_refuses_a_nan_payload(tmp_path):
    """NaN is not JSON. Stored, it would hash differently in every other reader."""
    p = tmp_path / "j.jsonl"
    with pytest.raises(ValueError):
        journal.append(p, "andy", "CLI", "store", {"cart": "c", "idx": 1},
                       {"embedding": [float("nan")]})


def test_append_refuses_a_negative_event_ts(tmp_path):
    p = tmp_path / "j.jsonl"
    with pytest.raises(ValueError):
        journal.append(p, "andy", "CLI", "store", {"cart": "c", "idx": 1}, {},
                       event_ts=-1.0)


def test_a_refused_entry_does_not_break_a_live_chain(tmp_path):
    p = tmp_path / "j.jsonl"
    _seed(p, 3)
    with pytest.raises(ValueError):
        journal.append(p, "andy", "FAX", "store", {"cart": "c", "idx": 1}, {})
    v = journal.verify(p)
    assert v["ok"] is True and v["entries"] == 3


# ---------------------------------------------------------------------------
# the file format itself
# ---------------------------------------------------------------------------

def test_one_event_per_line_and_greppable(tmp_path):
    """7/21: must be inspectable on disk. A multi-line record is not greppable."""
    p = tmp_path / "j.jsonl"
    journal.append(p, "andy", "CLI", "store", {"cart": "c", "idx": 1},
                   {"text": "line one\nline two\nline three"})
    journal.append(p, "andy", "CLI", "tombstone", {"cart": "c", "idx": 1}, {})
    raw = Path(p).read_text(encoding="utf-8")
    assert raw.count("\n") == 2, "an embedded newline split a record across lines"
    assert len(_lines(p)) == 2
    assert [ln for ln in _lines(p) if '"op":"tombstone"' in ln.replace(" ", "")]
    assert journal.replay(p)[0]["payload"]["text"].count("\n") == 2


def test_non_ascii_survives_round_trip(tmp_path):
    p = tmp_path / "j.jsonl"
    journal.append(p, "andy", "CLI", "store", {"cart": "café.cart", "idx": 1},
                   {"text": "松乃木 — Matsunoki"})
    v = journal.verify(p)
    assert v["ok"] is True
    assert journal.replay(p)[0]["payload"]["text"] == "松乃木 — Matsunoki"


def test_hash_is_independent_of_on_disk_key_order(tmp_path):
    """We hash the parsed fields, not the raw bytes, so the file stays free to be readable.

    Reformatting a line must NOT read as tampering -- otherwise `jq .` over the journal
    looks like an attack and nobody ever inspects it.
    """
    p = tmp_path / "j.jsonl"
    _seed(p, 3)
    lines = _lines(p)
    e = json.loads(lines[1])
    reversed_order = {k: e[k] for k in reversed(list(e))}
    rewritten = json.dumps(reversed_order, sort_keys=False, separators=(" , ", " : "))
    assert list(reversed_order) != list(e), "fixture must actually reorder the keys"
    assert rewritten != lines[1]
    lines[1] = rewritten
    _write_lines(p, lines)
    assert journal.verify(p)["ok"] is True


def test_journal_path_for_is_a_cart_sidecar(tmp_path):
    cart = tmp_path / "andy.cart"
    assert journal.journal_path_for(cart).name == "andy.cart.journal.jsonl"
    journal.append_to_cart(cart, "andy", "CLI", "store", {"cart": "andy.cart", "idx": 1}, {})
    assert journal.verify(journal.journal_path_for(cart))["ok"] is True


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

def test_stats_counts_by_op_writer_and_surface(tmp_path):
    p = tmp_path / "j.jsonl"
    journal.append(p, "andy", "WEB", "store", {"cart": "c", "idx": 1}, {})
    journal.append(p, "andy", "CLI", "tombstone", {"cart": "c", "idx": 1}, {})
    journal.append(p, "agent:sage-3", "MCP", "store", {"cart": "c", "idx": 2}, {})
    s = journal.stats(p)
    assert s["entries"] == 3
    assert s["by_op"] == {"store": 2, "tombstone": 1}
    assert s["by_writer"] == {"andy": 2, "agent:sage-3": 1}
    assert s["by_surface"] == {"WEB": 1, "CLI": 1, "MCP": 1}
    assert s["writers"] == ["agent:sage-3", "andy"]
    assert s["first_seq"] == 1 and s["last_seq"] == 3 and s["gapless"] is True
    assert s["bytes"] > 0


def test_stats_reports_a_gap(tmp_path):
    p = tmp_path / "j.jsonl"
    _seed(p, 5)
    lines = _lines(p)
    del lines[2]
    _write_lines(p, lines)
    s = journal.stats(p)
    assert s["entries"] == 4 and s["gapless"] is False


def test_stats_on_a_missing_journal(tmp_path):
    s = journal.stats(tmp_path / "nope.jsonl")
    assert s["exists"] is False and s["entries"] == 0 and s["bytes"] == 0


def test_empty_journal_verifies_but_says_it_is_empty(tmp_path):
    """⛔ "a verdict on an EMPTY result set" -- ok=True here must not read as "we logged"."""
    p = tmp_path / "j.jsonl"
    p.write_bytes(b"")
    v = journal.verify(p)
    assert v["ok"] is True
    assert v.get("empty") is True
    assert v["entries"] == 0


# ---------------------------------------------------------------------------
# §3 -- the thing that must never exist
# ---------------------------------------------------------------------------

def test_no_compaction_api():
    """§3: a WAL you can compact is a buffer; a WAL you cannot compact is a ledger.

    This test exists to fail on the day somebody adds the convenience. It is the only
    place the decision is enforced rather than merely written down.
    """
    forbidden = ("compact", "compaction", "prune", "truncate", "delete", "remove",
                 "rewrite", "purge", "vacuum", "amend", "repair", "fix")
    found = [name for name in dir(journal)
             if not name.startswith("_") and callable(getattr(journal, name))
             and any(f in name.lower() for f in forbidden)]
    assert found == [], f"the journal must have no destructive verb; found {found}"


def test_append_only_on_disk(tmp_path):
    """Every existing byte is still there after an append, at the same offset."""
    p = tmp_path / "j.jsonl"
    _seed(p, 3)
    before = Path(p).read_bytes()
    journal.append(p, "andy", "CLI", "corroborate", {"cart": "c", "idx": 1000}, {})
    after = Path(p).read_bytes()
    assert after.startswith(before), "an append rewrote earlier bytes"
    assert len(after) > len(before)


def test_history_is_corrected_by_appending_an_erratum(tmp_path):
    """The sanctioned way to fix a wrong record: supersede it. The wrong one stays."""
    p = tmp_path / "j.jsonl"
    wrong = journal.append(p, "andy", "WEB", "store", {"cart": "c", "idx": 7},
                           {"text": "the vendor is Sysco"})
    journal.append(p, "andy", "WEB", "supersede", {"cart": "c", "idx": 7},
                   {"supersedes_seq": wrong["seq"], "text": "the vendor is US Foods"})
    entries = journal.replay(p)
    assert len(entries) == 2
    assert entries[0]["payload"]["text"] == "the vendor is Sysco", "history was edited"
    assert entries[1]["payload"]["supersedes_seq"] == 1
    assert journal.verify(p)["ok"] is True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_verify_exit_code_tracks_the_verdict(tmp_path):
    """An operator check is worthless if it exits 0 on a broken chain."""
    good = tmp_path / "good.jsonl"
    bad = tmp_path / "bad.jsonl"
    _seed(good, 3)
    _seed(bad, 3)
    lines = _lines(bad)
    del lines[1]
    _write_lines(bad, lines)

    env = dict(os.environ)
    mod = str(HERE / "journal.py")
    r_good = subprocess.run([sys.executable, mod, "verify", str(good)],
                            capture_output=True, text=True, env=env, timeout=60)
    r_bad = subprocess.run([sys.executable, mod, "verify", str(bad)],
                           capture_output=True, text=True, env=env, timeout=60)
    assert r_good.returncode == 0, r_good.stderr[-2000:]
    assert "OK:" in r_good.stdout
    assert r_bad.returncode == 1, r_bad.stdout
    assert "BROKEN" in r_bad.stdout
