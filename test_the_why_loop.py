"""The whole loop, end to end, through the real MCP tools — and the demo script.

Four modules landed on 2026-11-01: verified seat identity, a hash-chained provenance
journal, sticky notes, and subscriptions. Each has its own suite with its own
disabled-mechanism control. ⛔ None of those answers the only question a buyer asks:

    "Someone changed a document I depend on. Tell me what changed, who changed it, and
     WHY the old version stopped being current."

So this file runs that sentence. It is deliberately readable top to bottom as the demo,
because a test that doubles as the script cannot drift from what the product does.

⚠ WHAT THIS DOES NOT COVER, said here rather than discovered in the room: nothing in
  membot currently emits `op="supersede"`. The payload carries `supersedes` and
  `why_prior` and `what_changed` renders both, but they stay empty until some code path
  records a supersession -- membot has no destructive verbs, so a passage is never
  replaced in place. ⇒ The "what it descends from" clause is BUILT AND LATENT. Demo the
  change-plus-why; do not promise the descent chain.
"""

import os
import sys
import tempfile

import numpy as np

import journal
import membot_server as ms
import subscriptions

AUDITOR = "seat-auditor"
PREPARER = "seat-preparer"

PASSAGES = [
    "AP control 4.2: invoices over $10,000 require dual approval.",
    "Retention: AP documents are held for 7 years from period close.",
]


def _mount(seat, cart_path):
    ms._server_config["read_only"] = False
    state = ms._get_session(seat)
    state.update({
        "cartridge_name": "firm-controls",
        "cartridge_path": cart_path,
        "mount_read_only": False,
        "mempack_id": None,
        "texts": list(PASSAGES),
        "embeddings": None,
        "binary_corpus": None,
        "metadata": [],
    })
    state.pop("_dedup_set", None)
    state.pop("_dedup_n", None)
    ms.embed_text = lambda text, prefix="search_document": np.ones(768, dtype=np.float32)
    return state


def _call(tool, **kw):
    fn = getattr(getattr(ms, tool), "fn", getattr(ms, tool))
    return fn(**kw)


def test_the_whole_loop():
    cart = os.path.join(tempfile.mkdtemp(), "firm-controls.cart.npz")
    _mount(AUDITOR, cart)
    _mount(PREPARER, cart)

    # 1 · The auditor watches the retention passage.
    watched = _call("watch", idx=1, session_id=AUDITOR)
    assert "Watching passage 1" in watched, watched

    # 2 · The preparer changes something and records WHY, shared so others can see it.
    # New text, not a restatement: memory_store dedups byte-identical re-stores, so
    # reusing PASSAGES[1] here would be silently skipped and the demo would be resting
    # entirely on the note.
    _call("memory_store",
          content="Retention: AP documents are held for 7 years from period close, "
                  "superseding the 5-year schedule cited in the 2023 manual.",
          session_id=PREPARER)
    _call("note_add", idx=1, kind="decision",
          body="Moved from 5 to 7 years because the examiner cited FDICIA in the 2025 "
               "cycle. The 5-year schedule is not merely old, it is non-compliant.",
          session_id=PREPARER)
    note_id = [n for n in __import__("note_store").list_notes(cart, seat=PREPARER)][0]["id"]
    _call("note_share", note_id=note_id, session_id=PREPARER)

    # 3 · The auditor asks what changed. This is the sentence the product exists for.
    out = _call("what_changed", session_id=AUDITOR)

    assert "CANNOT ANSWER" not in out and "NO JOURNAL" not in out, out
    assert "preparer" in out, f"the change does not name who made it:\n{out}"
    assert "examiner cited FDICIA" in out, (
        f"⛔ THE WHY DID NOT TRAVEL. The alert is commodity without it:\n{out}"
    )
    return out


def test_the_auditor_never_sees_a_private_note():
    """⭐⭐ The boundary that makes a note worth writing honestly. A subscription must not
    become a back door into another seat's private reasoning."""
    cart = os.path.join(tempfile.mkdtemp(), "firm-controls.cart.npz")
    _mount(AUDITOR, cart)
    _mount(PREPARER, cart)

    _call("watch", session_id=AUDITOR)                 # watch the whole cart
    _call("note_add", idx=1, kind="caution",
          body="PRIVATE-CANARY I am not sure this reading is right yet",
          session_id=PREPARER)

    out = _call("what_changed", session_id=AUDITOR)
    assert "PRIVATE-CANARY" not in out, (
        f"A PRIVATE NOTE LEAKED THROUGH what_changed:\n{out}"
    )


def test_unwatched_changes_do_not_appear():
    """The control for the feed. A tool that returned everything would pass the loop test
    above and be useless -- and would leak across kitchens, firms and clients."""
    cart = os.path.join(tempfile.mkdtemp(), "firm-controls.cart.npz")
    _mount(AUDITOR, cart)
    _mount(PREPARER, cart)

    _call("watch", idx=0, session_id=AUDITOR)          # watches passage 0 ONLY
    _call("note_add", idx=1, kind="why", body="OFF-TARGET-CANARY", session_id=PREPARER)

    out = _call("what_changed", session_id=AUDITOR)
    assert "OFF-TARGET-CANARY" not in out, (
        f"a change to an UNWATCHED passage was reported:\n{out}"
    )


def test_a_broken_journal_withholds_the_rows():
    """⛔ Changes read out of a journal that may have been edited are not evidence, and a
    caller who cannot tell the difference will present them as if they were."""
    cart = os.path.join(tempfile.mkdtemp(), "firm-controls.cart.npz")
    _mount(AUDITOR, cart)
    _mount(PREPARER, cart)
    _call("watch", session_id=AUDITOR)
    for i in range(3):
        _call("note_add", idx=i % 2, kind="why", body=f"note {i}", session_id=PREPARER)

    jpath = journal.journal_path_for(cart)
    lines = open(jpath, encoding="utf-8").read().splitlines()
    with open(jpath, "w", encoding="utf-8") as fh:     # remove a MIDDLE entry
        fh.write("\n".join([lines[0], lines[2]]) + "\n")

    out = _call("what_changed", session_id=AUDITOR)
    assert "note 2" not in out, f"rows served from a broken journal:\n{out}"
    assert ("CANNOT ANSWER" in out or "BROKEN" in out or "integrity" in out.lower()), (
        f"the integrity problem was not reported to the caller:\n{out}"
    )


def test_acknowledging_is_separate_from_reading():
    """Reading is not acknowledging. A client that crashed mid-render would otherwise lose
    those changes with nothing able to tell afterwards."""
    cart = os.path.join(tempfile.mkdtemp(), "firm-controls.cart.npz")
    _mount(AUDITOR, cart)
    _mount(PREPARER, cart)
    _call("watch", session_id=AUDITOR)
    _call("note_add", idx=1, kind="why", body="first change", session_id=PREPARER)

    _call("what_changed", session_id=AUDITOR)          # read, do NOT acknowledge
    assert subscriptions.last_seen(cart, AUDITOR) == 0, (
        "reading advanced the mark; an unrendered change would be lost silently"
    )

    _call("what_changed", mark_seen=True, session_id=AUDITOR)
    assert subscriptions.last_seen(cart, AUDITOR) > 0, (
        "mark_seen=True did not acknowledge anything"
    )

    out = _call("what_changed", session_id=AUDITOR)
    assert "first change" not in out, f"an acknowledged change came back:\n{out}"


if __name__ == "__main__":
    failures = 0
    demo = None
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                r = fn()
                if name == "test_the_whole_loop":
                    demo = r
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}\n      {e}")
            except Exception as e:                     # noqa: BLE001
                failures += 1
                print(f"ERROR {name}\n      {type(e).__name__}: {e}")
    if demo:
        print("\n--- what the auditor actually sees (the demo) ---")
        print(demo)
    print(f"\n{'FAIL' if failures else 'PASS'} -- {failures} failure(s)")
    sys.exit(1 if failures else 0)
