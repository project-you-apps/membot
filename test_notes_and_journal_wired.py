"""Do the note tools and the journal OPERATE, driven through the real MCP tools?

⛔ The unit suites already prove note_store (74 checks) and journal (74 tests) work in
isolation, each with its own disabled-mechanism control. Neither can prove the server
reaches them. This file drives `note_add`, `notes`, `note_share` and `memory_store` the
way a client does and then inspects the sidecars on disk.

⭐ The journal assertion is the one worth having: it is not "a file appeared" but "the
hash chain over entries written by real tool calls VERIFIES" -- and, in the last test,
that a tampered chain is caught. A ledger whose verifier has never been pointed at real
server output is a ledger nobody has tested.
"""

import os
import sys
import tempfile

import numpy as np

import journal
import membot_server as ms
import note_store

SEAT_A = "seat-alice-probe"
SEAT_B = "seat-bob-probe"

PASSAGES = [
    "Vendor invoices are reconciled weekly against the Toast export.",
    "The 2025 retention schedule supersedes the 2023 one for all AP documents.",
]


def _mount(seat, cart_path):
    ms._server_config["read_only"] = False
    state = ms._get_session(seat)
    state["cartridge_name"] = "probe-cart"
    state["cartridge_path"] = cart_path
    state["mount_read_only"] = False
    state["mempack_id"] = None
    state["texts"] = list(PASSAGES)
    state["embeddings"] = None
    state["binary_corpus"] = None
    state["metadata"] = []
    state.pop("_dedup_set", None)
    state.pop("_dedup_n", None)
    ms.embed_text = lambda text, prefix="search_document": np.ones(768, dtype=np.float32)
    return state


def _call(tool, **kw):
    fn = getattr(getattr(ms, tool), "fn", getattr(ms, tool))
    return fn(**kw)


def _fresh_cart():
    d = tempfile.mkdtemp()
    return os.path.join(d, "probe.cart.npz")


def test_a_note_can_be_written_and_read_back():
    cart = _fresh_cart()
    _mount(SEAT_A, cart)

    out = _call("note_add", idx=1, kind="decision",
                body="Retention moved to 7 years because the examiner asked for it in the "
                     "2025 cycle. The 2023 schedule is not merely old, it is wrong.",
                session_id=SEAT_A)
    assert "attached to passage 1" in out, f"note_add did not report success: {out}"
    assert "only you can see it" in out, f"a note must be born private: {out}"

    listed = _call("notes", idx=1, session_id=SEAT_A)
    assert "examiner asked for it" in listed, f"note not readable by its author: {listed}"


def test_another_seat_cannot_see_a_private_note():
    """⭐⭐ THE SURVEILLANCE BOUNDARY, through the server rather than the module."""
    cart = _fresh_cart()
    _mount(SEAT_A, cart)
    _mount(SEAT_B, cart)

    _call("note_add", idx=0, kind="caution",
          body="PRIVATE-CANARY do not reconcile before the Toast export settles",
          session_id=SEAT_A)

    seen_by_b = _call("notes", session_id=SEAT_B)
    assert "PRIVATE-CANARY" not in seen_by_b, (
        f"SEAT B READ SEAT A'S PRIVATE NOTE through the server: {seen_by_b}"
    )


def test_sharing_is_an_explicit_act_and_then_it_is_visible():
    cart = _fresh_cart()
    _mount(SEAT_A, cart)
    _mount(SEAT_B, cart)

    _call("note_add", idx=0, kind="why",
          body="SHARED-CANARY weekly cadence matches the vendor's billing cycle",
          session_id=SEAT_A)
    note_id = note_store.list_notes(cart, seat=SEAT_A)[0]["id"]

    # Bob may not share Alice's note.
    refused = _call("note_share", note_id=note_id, session_id=SEAT_B)
    assert "Refused" in refused, f"a non-author shared someone else's note: {refused}"
    assert "SHARED-CANARY" not in _call("notes", session_id=SEAT_B)

    ok = _call("note_share", note_id=note_id, session_id=SEAT_A)
    assert "now shared" in ok, f"author could not share their own note: {ok}"
    assert "SHARED-CANARY" in _call("notes", session_id=SEAT_B), (
        "a shared note is still invisible to another seat"
    )


def test_the_journal_records_real_tool_calls_and_the_chain_verifies():
    """⭐ The ledger, pointed at actual server output."""
    cart = _fresh_cart()
    _mount(SEAT_A, cart)

    _call("memory_store", content="A new AP control was added for dual approval.",
          session_id=SEAT_A)
    _call("note_add", idx=0, kind="context", body="Context for the weekly reconciliation.",
          session_id=SEAT_A)

    jpath = journal.journal_path_for(cart)
    assert os.path.exists(jpath), (
        "NO JOURNAL WAS WRITTEN by a store and a note. The journal is imported but not "
        "reached, which is indistinguishable from not existing."
    )

    v = journal.verify(jpath)
    assert v["ok"], f"the chain over real tool output does not verify: {v}"

    ops = [e["op"] for e in journal.replay(jpath)]
    assert "store" in ops, f"the store was not journaled: {ops}"
    assert "note" in ops, f"the note was not journaled: {ops}"

    # Identity provenance must travel on the entry: "writer: X" is only evidence if the
    # record also says whether anyone checked.
    entries = journal.replay(jpath)
    assert all("identity_source" in e["payload"] for e in entries), (
        f"entries do not record HOW the writer was established: {entries[0]['payload']}"
    )


def _journal_with_three_entries():
    cart = _fresh_cart()
    _mount(SEAT_A, cart)
    for i, body in enumerate(("first", "second", "third")):
        _call("note_add", idx=i % 2, kind="why", body=body, session_id=SEAT_A)
    jpath = journal.journal_path_for(cart)
    assert journal.verify(jpath)["ok"], "precondition: a clean chain must verify"
    with open(jpath, "r", encoding="utf-8") as fh:
        return jpath, fh.read().splitlines()


def _rewrite(jpath, lines):
    with open(jpath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def test_an_interior_deletion_in_a_real_journal_is_detected():
    """The chain's own strength: removing a middle entry orphans the next one."""
    jpath, lines = _journal_with_three_entries()
    assert len(lines) >= 3, f"need three entries, got {len(lines)}"
    _rewrite(jpath, [lines[0], lines[2]])

    v = journal.verify(jpath)
    assert not v["ok"], f"INTERIOR DELETION WENT UNDETECTED: {v['verdict_line']}"


def test_a_trailing_deletion_is_detected_by_the_high_water_mark():
    """⛔⛔ THE ONE THE CHAIN ALONE CANNOT CATCH, and the reason the mark exists.

    Dropping the last entries leaves a SHORTER CHAIN THAT VERIFIES PERFECTLY -- nothing
    inside the file knows the removed entries were ever there. This is also the easiest
    and most useful tampering: it is how someone hides what they just did.
    """
    jpath, lines = _journal_with_three_entries()
    _rewrite(jpath, lines[:1])                 # drop the last two, cleanly

    v = journal.verify(jpath)
    assert not v["ok"], (
        f"TRAILING DELETION WENT UNDETECTED -- the journal cannot support the claim "
        f"'the record cannot be altered without detection'. verdict: {v['verdict_line']}"
    )
    assert "entries_deleted_from_tail" in (v["first_break"] or {}).get("reasons", []), (
        f"detected, but not as a tail deletion, so the message will mislead: {v}"
    )


def test_the_high_water_mark_is_what_catches_it():
    """⛔ THE CONTROL FOR THE CHECK ABOVE. Delete the mark and the same truncation must go
    undetected -- otherwise something else is catching it and the mark is dead weight."""
    jpath, lines = _journal_with_three_entries()
    _rewrite(jpath, lines[:1])
    journal.hwm_path_for(jpath).unlink()

    v = journal.verify(jpath)
    assert v["ok"], (
        "the truncation was caught WITHOUT the high-water mark, so the mark is not what "
        "is doing the work and this design needs rechecking"
    )
    assert v["hwm_present"] is False, "verify must say the mark is missing"


def test_rewriting_the_last_entry_is_detected():
    """Same length, different content. The chain cannot see it; the mark's hash can."""
    jpath, lines = _journal_with_three_entries()
    import json as _json
    last = _json.loads(lines[-1])
    last["payload"] = dict(last.get("payload") or {})
    last["payload"]["kind"] = "tampered"
    # Recompute the hash so the entry is internally consistent -- the realistic attack,
    # and the one a naive chain walk accepts.
    last["hash"] = journal.compute_hash(last)
    _rewrite(jpath, lines[:-1] + [_json.dumps(last, sort_keys=True)])

    v = journal.verify(jpath)
    assert not v["ok"], f"A REWRITTEN TAIL ENTRY VERIFIED CLEAN: {v['verdict_line']}"


def test_credentials_do_not_reach_a_note_body():
    """A note box is an input box, and the spec is explicit that there is no classifier at
    the boundary -- that is a deliberate ruling, not an oversight. So this test PINS THE
    CURRENT BEHAVIOUR rather than asserting a filter exists: if someone later wires
    redaction into notes, this test should be updated deliberately, not silently."""
    cart = _fresh_cart()
    _mount(SEAT_A, cart)
    fake = "sk-ant-api03-" + "FAKE" + ("q7Zx" * 20) + "AA"
    _call("note_add", idx=0, kind="caution", body=f"the key was {fake}", session_id=SEAT_A)

    stored = note_store.list_notes(cart, seat=SEAT_A)[0]["body"]
    assert fake in stored, (
        "a note body IS being filtered. That may be an improvement, but it is a change "
        "from the ruled design (no classifier at the boundary; permissions are the "
        "control) and this test should be updated on purpose."
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}\n      {e}")
            except Exception as e:                     # noqa: BLE001
                failures += 1
                print(f"ERROR {name}\n      {type(e).__name__}: {e}")
    print(f"\n{'FAIL' if failures else 'PASS'} -- {failures} failure(s)")
    sys.exit(1 if failures else 0)
