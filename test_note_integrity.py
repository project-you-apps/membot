"""Can we actually tell that a note was removed?

⛔ THE CONTROL PAIR IS test_a_removed_note_is_DETECTED and test_an_intact_cart_is_clean.
Neither means anything alone: a reconciler that always reported tampering would pass the
first, and one that never did would pass the second.

Run:  python -m pytest test_note_integrity.py -q      (from the membot directory)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import journal
import note_integrity
import note_store

KEY = "a" * 40
SEAT = "seat-alice"


@pytest.fixture()
def cart():
    d = Path(tempfile.mkdtemp())
    p = d / "integrity-probe.cart.npz"
    p.write_bytes(b"")
    return p


def _add(cart, body: str) -> dict:
    """Create a note the way both real surfaces do: store it, then journal it."""
    n = note_store.add_note(str(cart), KEY, "why", body, by=SEAT, surface="WEB",
                            scope="shared")
    journal.append_to_cart(str(cart), writer=SEAT, surface="WEB", op="note",
                           target={"target_key": KEY},
                           payload={"note_id": n["id"], "kind": "why"})
    return n


def _splice_out(cart, note_id: str) -> None:
    """Remove a note from notes.json the way an editor would: cleanly, leaving valid JSON.

    ⭐ This is the whole threat model. The file still parses, every remaining note still
    reads correctly, and nothing in note_store objects."""
    p = note_store.sidecar_path(str(cart))
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    notes = data["notes"] if isinstance(data, dict) else data
    kept = [n for n in notes if n.get("id") != note_id]
    assert len(kept) == len(notes) - 1, "the splice did not remove anything"
    if isinstance(data, dict):
        data["notes"] = kept
    else:
        data = kept
    Path(p).write_text(json.dumps(data), encoding="utf-8")


def test_an_intact_cart_is_clean(cart):
    _add(cart, "retention is 7 years because FDICIA")
    _add(cart, "confirmed with the examiner")
    r = note_integrity.reconcile(str(cart))
    assert r["ok"] is True, r
    assert r["checked"] == 2
    assert r["orphaned_in_journal"] == []
    assert r["unwitnessed_in_notes"] == []


def test_a_removed_note_is_DETECTED(cart):
    """⭐⭐ THE POINT. The notes file still parses and still looks right. The chain
    disagrees, and that disagreement is the detection."""
    keep = _add(cart, "retention is 7 years because FDICIA")
    doomed = _add(cart, "the inconvenient one")
    _splice_out(cart, doomed["id"])

    # The sidecar is perfectly valid afterwards -- that is why this check is needed.
    assert note_store.load(str(cart))["readable"] is True
    assert len(note_store.load(str(cart))["notes"]) == 1

    r = note_integrity.reconcile(str(cart))
    assert r["ok"] is False, r
    assert doomed["id"] in r["orphaned_in_journal"], r
    assert keep["id"] not in r["orphaned_in_journal"]
    assert "MISSING" in r["verdict_line"]


def test_the_journal_itself_still_verifies_after_the_splice(cart):
    """⛔ Removing a note does NOT break the hash chain -- which is exactly why
    verify() alone could never have caught this, and why reconcile() exists."""
    n = _add(cart, "the inconvenient one")
    _splice_out(cart, n["id"])
    v = journal.verify(journal.journal_path_for(str(cart)))
    assert v["ok"] is True, "the splice broke the chain; then reconcile would be redundant"


def test_a_note_with_no_journal_entry_is_unwitnessed_NOT_tampering(cart):
    """⚠ The false-positive guard. A note written before journalling existed must not be
    reported as a removal -- a check that cries wolf on legacy data gets switched off,
    and then it is not a check. ⭐ But it is also NOT attested."""
    # A journal exists (something else was journalled); this note simply is not in it.
    journal.append_to_cart(str(cart), writer=SEAT, surface="WEB", op="store",
                           target={"target_key": KEY}, payload={})
    note_store.add_note(str(cart), KEY, "why", "legacy, pre-journal", by=SEAT,
                        surface="WEB", scope="shared")
    r = note_integrity.reconcile(str(cart))
    assert r["ok"] is True, r
    assert r["attested"] is False, "an unwitnessed note must not read as verified"
    assert len(r["unwitnessed_in_notes"]) == 1
    assert r["orphaned_in_journal"] == []
    assert "predate" in r["verdict_line"]


def test_no_journal_at_all_is_ok_but_NOT_attested(cart):
    """⭐⭐ THE DISTINCTION THE FIRST DRAFT GOT WRONG, and its own tests caught: this state
    and the one above were asserted to have OPPOSITE verdicts while being, at the time,
    indistinguishable.

    A cart written before journalling existed is not evidence of tampering, so `ok`. And
    nothing here can attest anything, so NOT `attested`. One boolean could not carry both,
    and collapsing them forces a choice between crying wolf on every legacy cart and
    showing green for a cart with no witness at all."""
    note_store.add_note(str(cart), KEY, "why", "unwitnessed", by=SEAT, surface="WEB")
    r = note_integrity.reconcile(str(cart))
    assert r["ok"] is True, r
    assert r["attested"] is False, r
    assert r["journal_readable"] is False
    assert "cannot be detected" in r["verdict_line"]


def test_a_missing_journal_FILE_is_not_an_empty_one(cart):
    """⛔ replay() returns [] for a path with no file, exactly as it does for a journal
    that exists and saw nothing. Inferring absence from an exception collapsed 'there is
    no witness' into 'the witness saw nothing' -- opposite reports."""
    note_store.add_note(str(cart), KEY, "why", "n", by=SEAT, surface="WEB")
    assert note_integrity.reconcile(str(cart))["journal_readable"] is False

    journal.append_to_cart(str(cart), writer=SEAT, surface="WEB", op="store",
                           target={"target_key": KEY}, payload={})
    assert note_integrity.reconcile(str(cart))["journal_readable"] is True


def test_an_intact_cart_is_ATTESTED(cart):
    """The green signal, and the control for every 'ok but not attested' case above."""
    _add(cart, "witnessed properly")
    r = note_integrity.reconcile(str(cart))
    assert r["ok"] is True and r["attested"] is True, r


def test_an_empty_cart_reports_checked_zero(cart):
    """⛔ EXISTENCE IS NOT OPERATION. `ok` on an empty comparison must be readable as
    'nothing to check', not as 'verified'. `checked` is what makes that legible."""
    journal.append_to_cart(str(cart), writer=SEAT, surface="WEB", op="store",
                           target={"target_key": KEY}, payload={})
    r = note_integrity.reconcile(str(cart))
    assert r["checked"] == 0
    assert "nothing to reconcile" in r["verdict_line"]


def test_an_unreadable_sidecar_is_not_clean(cart, monkeypatch):
    monkeypatch.setattr(note_store, "load",
                        lambda _p: {"readable": False, "notes": [], "error": "boom"})
    r = note_integrity.reconcile(str(cart))
    assert r["ok"] is False
    assert r["checked"] == 0
    assert "could not be read" in r["verdict_line"]


def test_only_note_ops_are_reconciled(cart):
    """A `store` entry carrying a stray note_id must not become a phantom orphan."""
    journal.append_to_cart(str(cart), writer=SEAT, surface="WEB", op="store",
                           target={"target_key": KEY},
                           payload={"note_id": "not-really-a-note"})
    _add(cart, "a real one")
    r = note_integrity.reconcile(str(cart))
    assert r["ok"] is True, r
    assert "not-really-a-note" not in r["orphaned_in_journal"]
