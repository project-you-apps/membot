"""Did anyone remove a note from this discussion?

THE GAP THIS CLOSES (named 2026-11-04, from Andy's "every note append is part of the note
chain")
-------------------------------------------------------------------------------------
`journal.jsonl` is hash-chained, so removing an entry from its middle is detectable and
removing one from its tail is caught by the `.hwm` sidecar. `notes.json` is a **plain JSON
array**. Delete an object from it and nothing notices: the file still parses, the remaining
notes still read correctly, and every existing check passes. For a feature whose entire
claim is "the reasoning behind this record is preserved", a silently removable note is the
one failure that matters.

⭐⭐ WHY THERE IS NO SECOND HASH CHAIN HERE, AND THAT IS THE POINT
The obvious fix was to chain `notes.json` the way `journal.jsonl` is chained. That would
have been a second chaining implementation to keep correct, a format change, and a
migration for every existing sidecar.

It is also unnecessary, because **the witness already exists**. Every path that creates a
note already writes a journal entry carrying its `note_id`:

    membot_server.note_add      -> _journal_event(..., "note", {"note_id": ...})
    membot_server.note_share    -> _journal_event(..., {"note_id": ...})
    api/provenance_routes       -> journal.append_to_cart(..., op="note", {"note_id": ...})

So the hash chain already attests that note X existed. What was missing was nobody ever
ASKING the two files whether they agree. ⇒ This module is that question, and it adds zero
new chaining: one implementation, as ruled.

WHAT A DISAGREEMENT MEANS
-------------------------
    orphaned_in_journal   the chain says note X was written; notes.json has no X.
                          ⛔ THE ATTACK, and the reason this module exists. To forge it you
                          must also edit the journal, which breaks the chain, or truncate
                          it, which the high-water mark catches.
    unwitnessed_in_notes  notes.json has note Y; no journal entry mentions it.
                          ⚠ NOT an attack and usually not even a problem -- it is what a
                          note written before journalling existed looks like, and what a
                          note written while the journal was unwritable looks like. Named
                          separately and never counted as tampering, because a check that
                          cries wolf on legacy data gets turned off, and then it is not a
                          check at all.

⛔ AND IT REPORTS, IT DOES NOT REPAIR. Re-adding a missing note would be inventing a record;
deleting an unwitnessed one would be destroying one. Both are decisions for a human who can
see the verdict.
"""

from __future__ import annotations

import journal
import note_store

# Every op whose payload names a note. Kept explicit rather than "any op with a note_id",
# so a future op that happens to carry one does not quietly join the reconciliation.
NOTE_OPS = ("note", "note_share", "supersede")


def _journalled_note_ids(journal_path) -> tuple[dict, bool]:
    """{note_id: first seq that mentioned it}, and whether the journal was readable.

    ⚠ Uses `replay(strict=False)`. A journal with a torn tail still tells the truth about
    every entry BEFORE the tear, and refusing to read it would turn an unclean shutdown
    into "all your notes look forged". `verify()` remains the only thing that speaks to
    integrity, and the caller is handed its verdict separately.

    ⛔ EXISTENCE IS TESTED, NOT INFERRED FROM AN EXCEPTION. `replay()` on a path with no
    file returns `[]`, exactly like a journal that exists and is empty -- so catching an
    error would have collapsed "there is no witness" into "the witness saw nothing". Those
    deserve opposite reporting, and the first draft of this function got it wrong.
    """
    from pathlib import Path as _Path
    if not _Path(journal_path).exists():
        return {}, False
    try:
        entries = journal.replay(journal_path, strict=False)
    except Exception:                                        # noqa: BLE001
        return {}, False

    seen: dict = {}
    for e in entries:
        if e.get("op") not in NOTE_OPS:
            continue
        nid = (e.get("payload") or {}).get("note_id")
        if nid and nid not in seen:
            seen[nid] = e.get("seq")
    return seen, True


def reconcile(cart_path) -> dict:
    """Compare `notes.json` against the hash-chained journal. Reports; never repairs.

    Returns a dict a UI can render directly:

    ⭐⭐ TWO BOOLEANS, NOT ONE, and the first draft's single `ok` was the bug. It was
    being asked to mean both "nothing was removed" and "and I was able to check properly",
    which are different claims — the tests for this module asserted opposite verdicts for
    the SAME state before the split existed. A legacy cart with notes and no journal is
    not evidence of tampering (so `ok`), and is also not verified (so not `attested`).
    Collapsing them forces a choice between crying wolf on every old cart and showing green
    for a cart with no witness at all. Both are how a check stops being a check.

        ok                     No note is known to be MISSING. ⛔ Not a clean bill of
                               health: an unwitnessed note leaves it True.
        attested               ⭐ THE GREEN SIGNAL. True only when the journal was readable
                               AND every note in the sidecar is witnessed by it. This is
                               what a UI may render as "verified"; `ok` is not.
        checked                how many notes were compared, so neither boolean can be
                               read off an empty comparison.
        journal_readable       False when there is no journal file at all, in which case
                               removal CANNOT be detected here and the verdict says so.
        orphaned_in_journal    note ids the chain attests and the sidecar lacks.
        unwitnessed_in_notes   note ids in the sidecar that no entry mentions.
        verdict_line           one sentence, computed, for a human.
    """
    jpath = journal.journal_path_for(str(cart_path))
    journalled, readable = _journalled_note_ids(jpath)

    data = note_store.load(cart_path)
    notes_readable = bool(data.get("readable", True))
    notes = data.get("notes", []) or []
    note_ids = {n.get("id") for n in notes if n.get("id")}

    # ⛔ Two different "cannot answer" cases, both of which must refuse to return a clean
    # verdict. Saying ok=True because a file would not load is how a check becomes
    # decoration -- it would read green in exactly the state an attacker would arrange.
    if not notes_readable:
        # ⛔ Not ok AND not attested: a file that would not load is the one state where
        # saying anything reassuring would be a guess.
        return {
            "ok": False, "attested": False, "checked": 0,
            "journal_readable": readable, "notes_readable": False,
            "orphaned_in_journal": [], "unwitnessed_in_notes": [],
            "verdict_line": "The notes sidecar could not be read, so nothing can be said "
                            "about whether a note is missing.",
        }
    if not readable:
        # ⭐ ok, because nothing indicates removal -- but NOT attested, because nothing
        # could. This is every cart written before journalling existed, and reporting it
        # as tampering is how a check gets switched off.
        return {
            "ok": True, "attested": False, "checked": len(note_ids),
            "journal_readable": False, "notes_readable": True,
            "orphaned_in_journal": [], "unwitnessed_in_notes": sorted(note_ids),
            "verdict_line": ("No journal beside this cart, so removal of a note cannot be "
                             "detected here. Nothing suggests a note is missing, and "
                             "nothing attests that one is not."),
        }

    orphaned = sorted(nid for nid in journalled if nid not in note_ids)
    unwitnessed = sorted(nid for nid in note_ids if nid not in journalled)

    # Only removal makes this False. An unwitnessed note is a provenance gap worth naming
    # and is not evidence that anyone took anything away.
    ok = not orphaned
    # ⭐ The green signal is strictly stronger: readable witness AND nothing unaccounted
    # for in either direction.
    attested = ok and not unwitnessed

    if orphaned:
        line = (f"{len(orphaned)} note(s) recorded in the hash-chained journal are MISSING "
                f"from the notes file. Someone removed them without touching the chain.")
    elif unwitnessed:
        line = (f"No notes are missing. {len(unwitnessed)} note(s) predate journalling or "
                f"were written while the journal was unwritable, so their creation is not "
                f"independently attested.")
    elif not note_ids:
        line = "No notes on this cart yet, so there is nothing to reconcile."
    else:
        line = (f"All {len(note_ids)} note(s) are accounted for in the hash-chained "
                f"journal. None has been removed.")

    return {
        "ok": ok,
        "attested": attested,
        "checked": len(note_ids),
        "journal_readable": True,
        "notes_readable": True,
        "orphaned_in_journal": orphaned,
        "unwitnessed_in_notes": unwitnessed,
        "verdict_line": line,
    }
