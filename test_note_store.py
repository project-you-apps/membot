#!/usr/bin/env python3
"""note_store: the four decisions, and the surveillance boundary in particular.

HOUSE RULE THIS FILE IS WRITTEN AGAINST: "existence is not operation -- a check identical in
both states is not a check." Every case below is built so that it FAILS if the mechanism it
names were removed, which for a visibility rule means each leak test is paired with a
positive control. "Seat B sees nothing" passes just as happily against a store that returns
nothing to anybody, so every such case also asserts that the right seat DOES see the note.

Covers: round-trip across a reload, default-private, the private/shared read boundary on BOTH
read paths (list_notes and for_results -- one filtered and the other not is the obvious bug),
author-only sharing, supersede leaving the original untouched, over-long and malformed input
being refused rather than stored, and an unreadable sidecar reporting as unreadable rather
than as "no notes".

No network, no GPU. Writes only inside a temp dir it removes.

Run from membot/:  python test_note_store.py
"""
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_MEMBOT_DIR = _HERE if os.path.exists(os.path.join(_HERE, "note_store.py")) \
    else os.path.dirname(_HERE)
sys.path.insert(0, _MEMBOT_DIR)

import note_store as ns                                 # noqa: E402

failures = []
checks = 0

KEY_A = "c0ffee1234567890abcdef"        # a content key: sha1-shaped, not an ordinal
KEY_B = "deadbeef9876543210fedcba"
SEAT_A = "seat-a@example.com"
SEAT_B = "seat-b@example.com"


def check(label, ok, detail=""):
    global checks
    checks += 1
    if not ok:
        failures.append(label)
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if detail and not ok else ""))


def raises(exc, fn, *a, **kw):
    """Returns (did_raise, message). Catches only `exc`, so an unexpected error still blows
    up loudly instead of being scored as a correct refusal."""
    try:
        fn(*a, **kw)
    except exc as e:
        return True, str(e)
    return False, ""


def fresh_cart():
    d = tempfile.mkdtemp(prefix="note-store-test-")
    return os.path.join(d, "demo.cart"), d


# ---------------------------------------------------------------- 1 · round trip + reload
def t_round_trip():
    cart, d = fresh_cart()
    try:
        entry = ns.add_note(cart, KEY_A, "why", "Vendor code was wrong on the invoice.",
                            by=SEAT_A, surface="CLI", now=1000.0)
        check("1.1 sidecar lands beside the cart as <cart>.notes.json",
              os.path.exists(cart + ".notes.json"),
              f"looked for {cart + '.notes.json'}")
        check("1.2 id is nte_ + 16 hex", entry["id"].startswith("nte_")
              and len(entry["id"]) == 20, entry["id"])

        # A RELOAD, not the in-memory dict: read the file back through a fresh load.
        got = ns.list_notes(cart, seat=SEAT_A)
        check("1.3 exactly one note survives the reload", len(got) == 1, repr(got))
        rec = got[0] if got else {}
        spec_fields = ("id", "target_key", "kind", "body", "scope", "by", "surface", "at",
                       "supersedes")
        missing = [f for f in spec_fields if f not in rec]
        check("1.4 every spec §3 field is on the reloaded record", not missing,
              f"missing {missing}")
        check("1.5 reloaded values match what was written",
              rec.get("target_key") == KEY_A and rec.get("kind") == "why"
              and rec.get("by") == SEAT_A and rec.get("surface") == "CLI"
              and rec.get("at") == 1000.0
              and rec.get("body") == "Vendor code was wrong on the invoice.",
              repr(rec))

        # Inert in v1: nothing in the record can be read by scoring or ranking, because
        # there is nothing scoring-shaped on it. This fails the day someone adds one.
        scoreish = [k for k in rec
                    if any(w in k.lower() for w in ("score", "rank", "weight", "boost"))]
        check("1.6 the record carries no scoring field (v1 is inert)", not scoreish,
              f"found {scoreish}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- 2 · born private
def t_default_private():
    cart, d = fresh_cart()
    try:
        entry = ns.add_note(cart, KEY_A, "decision", "Chose per-seat over shared.",
                            by=SEAT_A, surface="VSC", now=1000.0)
        check("2.1 scope defaults to private when not passed",
              entry["scope"] == ns.SCOPE_PRIVATE, repr(entry.get("scope")))
        check("2.2 and it is private on disk too, not just in the return value",
              ns.list_notes(cart, seat=SEAT_A)[0]["scope"] == ns.SCOPE_PRIVATE)
        check("2.3 the module's own default constant is private",
              ns.SCOPE_PRIVATE == "private")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- 3 · the boundary
def t_private_is_not_visible_to_another_seat():
    cart, d = fresh_cart()
    try:
        a = ns.add_note(cart, KEY_A, "why", "A's private reasoning.",
                        by=SEAT_A, surface="CLI", now=1000.0)

        # POSITIVE CONTROL first: the author must see it, or "B sees nothing" proves nothing.
        mine = ns.list_notes(cart, seat=SEAT_A)
        check("3.1 CONTROL: the author sees their own private note via list_notes",
              [n["id"] for n in mine] == [a["id"]], repr(mine))
        mine_r = ns.for_results(cart, [KEY_A], seat=SEAT_A)
        check("3.2 CONTROL: the author sees it via for_results too",
              [n["id"] for n in mine_r.get(KEY_A, [])] == [a["id"]], repr(mine_r))

        theirs = ns.list_notes(cart, seat=SEAT_B)
        check("3.3 seat B cannot see A's private note via list_notes",
              theirs == [], repr(theirs))
        theirs_r = ns.for_results(cart, [KEY_A], seat=SEAT_B)
        check("3.4 seat B cannot see A's private note via for_results",
              theirs_r == {}, repr(theirs_r))

        # An unidentified caller is not an omniscient one.
        check("3.5 an empty seat sees no private note via list_notes",
              ns.list_notes(cart, seat="") == [])
        check("3.6 an empty seat sees no private note via for_results",
              ns.for_results(cart, [KEY_A], seat="") == {})

        # scope="private" narrows what is already visible; it never widens.
        check("3.7 scope='private' for seat B is still empty, not 'all private notes'",
              ns.list_notes(cart, seat=SEAT_B, scope="private") == [])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_shared_is_visible_to_another_seat():
    cart, d = fresh_cart()
    try:
        s = ns.add_note(cart, KEY_A, "caution", "This vendor bills in two currencies.",
                        by=SEAT_A, surface="WEB", scope="shared", now=1000.0)
        p = ns.add_note(cart, KEY_A, "why", "A's private aside.",
                        by=SEAT_A, surface="WEB", now=1001.0)

        theirs = ns.list_notes(cart, seat=SEAT_B)
        check("4.1 seat B sees the shared note via list_notes",
              [n["id"] for n in theirs] == [s["id"]], repr(theirs))
        theirs_r = ns.for_results(cart, [KEY_A], seat=SEAT_B)
        check("4.2 seat B sees the shared note via for_results",
              [n["id"] for n in theirs_r.get(KEY_A, [])] == [s["id"]], repr(theirs_r))
        check("4.3 and the shared note did NOT drag A's private note along with it",
              p["id"] not in [n["id"] for n in theirs_r.get(KEY_A, [])])
        check("4.4 the author sees both their private note and the shared one",
              sorted(n["id"] for n in ns.list_notes(cart, seat=SEAT_A))
              == sorted([s["id"], p["id"]]))
        # for_results keys on the content key, and only on keys that were asked for.
        check("4.5 for_results returns nothing for a key that was not a search hit",
              ns.for_results(cart, [KEY_B], seat=SEAT_B) == {})
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- 5 · share is the author's
def t_share_is_author_only():
    cart, d = fresh_cart()
    try:
        a = ns.add_note(cart, KEY_A, "why", "A's reasoning.",
                        by=SEAT_A, surface="CLI", now=1000.0)

        did, msg = raises(PermissionError, ns.share, cart, a["id"], by=SEAT_B, now=1002.0)
        check("5.1 share by a non-author is refused", did, msg)
        check("5.2 and the refused note is STILL private on disk",
              ns.list_notes(cart, seat=SEAT_A)[0]["scope"] == ns.SCOPE_PRIVATE)
        check("5.3 and still invisible to the seat that tried",
              ns.list_notes(cart, seat=SEAT_B) == [])

        # CONTROL: share is not simply broken for everyone.
        shared = ns.share(cart, a["id"], by=SEAT_A, now=1003.0)
        check("5.4 CONTROL: the author can share their own note",
              shared["scope"] == ns.SCOPE_SHARED, repr(shared))
        check("5.5 the sharing act is itself attributed",
              shared["shared_by"] == SEAT_A and shared["shared_at"] == 1003.0,
              repr(shared))
        check("5.6 after sharing, seat B sees it on both read paths",
              len(ns.list_notes(cart, seat=SEAT_B)) == 1
              and len(ns.for_results(cart, [KEY_A], seat=SEAT_B).get(KEY_A, [])) == 1)
        did, msg = raises(ValueError, ns.share, cart, a["id"], by=SEAT_A, now=1004.0)
        check("5.7 re-sharing an already shared note is refused, not silently re-stamped",
              did, msg)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- 6 · supersede, not edit
def t_supersede_keeps_the_original():
    cart, d = fresh_cart()
    try:
        first = ns.add_note(cart, KEY_A, "why", "Corrected because the PO was duplicated.",
                            by=SEAT_A, surface="CLI", now=1000.0)
        before = json.loads(json.dumps(first))          # deep snapshot

        second = ns.supersede(cart, first["id"], "Actually the vendor rebilled; not a dupe.",
                              by=SEAT_A, surface="CLI", now=1005.0)
        check("6.1 the correction is a NEW note with its own id",
              second["id"] != first["id"])
        check("6.2 the new note points at the old one",
              second["supersedes"] == first["id"], repr(second.get("supersedes")))
        check("6.3 the correction inherits target and kind",
              second["target_key"] == first["target_key"]
              and second["kind"] == first["kind"])
        check("6.4 the correction inherits scope, so correcting cannot widen visibility",
              second["scope"] == first["scope"] == ns.SCOPE_PRIVATE)

        all_notes = ns.list_notes(cart, seat=SEAT_A)
        check("6.5 both notes are stored -- nothing was deleted", len(all_notes) == 2,
              repr([n["id"] for n in all_notes]))
        stored_first = [n for n in all_notes if n["id"] == first["id"]][0]
        check("6.6 the ORIGINAL is byte-identical to how it was written",
              stored_first == before,
              f"was {before}\n        now {stored_first}")

        # Delivery shows the current why; the audit view above shows both.
        delivered = ns.for_results(cart, [KEY_A], seat=SEAT_A).get(KEY_A, [])
        check("6.7 for_results delivers only the correction, not the superseded note",
              [n["id"] for n in delivered] == [second["id"]], repr(delivered))
        both = ns.for_results(cart, [KEY_A], seat=SEAT_A, include_superseded=True)
        check("6.8 ...and include_superseded=True brings the old one back",
              len(both.get(KEY_A, [])) == 2)

        # Another seat may not correct a note it cannot even see.
        did, msg = raises(PermissionError, ns.supersede, cart, first["id"], "sneaky",
                          SEAT_B, "CLI", 1006.0)
        check("6.9 a seat cannot supersede another seat's private note", did, msg)
        check("6.10 and that refusal stored nothing",
              len(ns.list_notes(cart, seat=SEAT_A)) == 2)

        did, msg = raises(KeyError, ns.supersede, cart, "nte_doesnotexist00", "x",
                          SEAT_A, "CLI", 1007.0)
        check("6.11 superseding an unknown id raises", did, msg)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- 7 · validate, don't store junk
def t_body_cap_is_a_refusal():
    cart, d = fresh_cart()
    try:
        # DECISION, asserted here so it is not ambiguous: an over-long body is REFUSED,
        # never truncated. A note's body is the only copy of the reasoning, so a silent cut
        # stores a mutilated record that still looks complete.
        did, msg = raises(ValueError, ns.add_note, cart, KEY_A, "why", "x" * 2001,
                          SEAT_A, "CLI")
        check("7.1 a body over 2000 chars is REFUSED (not truncated)", did, msg)
        check("7.2 and nothing was stored", ns.stats(cart)["total"] == 0)

        ok = ns.add_note(cart, KEY_A, "why", "y" * 2000, by=SEAT_A, surface="CLI",
                         now=1000.0)
        check("7.3 CONTROL: exactly 2000 chars is accepted and kept whole",
              len(ok["body"]) == 2000
              and len(ns.list_notes(cart, seat=SEAT_A)[0]["body"]) == 2000)
        check("7.4 the cap constant is 2000", ns.MAX_BODY_CHARS == 2000)

        did, _ = raises(ValueError, ns.add_note, cart, KEY_A, "why", "   ", SEAT_A, "CLI")
        check("7.5 an empty/whitespace body is refused", did)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def t_bad_input_raises():
    cart, d = fresh_cart()
    try:
        did, msg = raises(ValueError, ns.add_note, cart, KEY_A, "sticky", "body",
                          SEAT_A, "CLI")
        check("8.1 an invalid kind raises rather than storing junk", did, msg)
        check("8.2 and stored nothing", ns.stats(cart)["total"] == 0)

        for kind in ns.VALID_KINDS:
            ns.add_note(cart, KEY_A, kind, f"a {kind} note", by=SEAT_A, surface="CLI",
                        now=1000.0 + ns.VALID_KINDS.index(kind))
        check("8.3 CONTROL: every documented kind is accepted",
              ns.stats(cart)["total"] == len(ns.VALID_KINDS),
              repr(ns.stats(cart)["by_kind"]))

        # target_key is a CONTENT key. An index-keyed note points at a stranger after the
        # next reindex, so an ordinal must not get in.
        did, msg = raises(ValueError, ns.add_note, cart, "7", "why", "body", SEAT_A, "CLI")
        check("8.4 an index-shaped target_key is refused", did, msg)
        did, _ = raises(ValueError, ns.add_note, cart, "", "why", "body", SEAT_A, "CLI")
        check("8.5 an empty target_key is refused", did)

        # The attribution triple cannot be left blank: an unattributed note looks
        # attributed to every downstream provenance filter.
        did, msg = raises(ValueError, ns.add_note, cart, KEY_A, "why", "body", "", "CLI")
        check("8.6 an empty `by` is refused", did, msg)
        did, msg = raises(ValueError, ns.add_note, cart, KEY_A, "why", "body", SEAT_A, "")
        check("8.7 an empty surface is refused", did, msg)
        did, msg = raises(ValueError, ns.add_note, cart, KEY_A, "why", "body", SEAT_A,
                          "MOBILE")
        check("8.8 an unknown surface is refused", did, msg)
        did, _ = raises(ValueError, ns.add_note, cart, KEY_A, "why", "body", SEAT_A, "CLI",
                        "public")
        check("8.9 an unknown scope is refused", did)
        check("8.10 nothing junk slipped in during 8.4-8.9",
              ns.stats(cart)["total"] == len(ns.VALID_KINDS))
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------- 9 · unreadable != "no notes"
def t_unreadable_sidecar():
    cart, d = fresh_cart()
    try:
        # The baseline fact we must stay distinguishable from: no file at all.
        clean = ns.load(cart)
        check("9.1 a cart with no sidecar is readable with zero notes",
              clean["readable"] is True and clean["notes"] == [])
        check("9.2 stats agrees it is readable", ns.stats(cart)["readable"] is True)

        with open(cart + ".notes.json", "w", encoding="utf-8") as fh:
            fh.write("{ this is not json")
        broken = ns.load(cart)
        check("9.3 a corrupt sidecar is reported UNREADABLE, not as 'no notes'",
              broken["readable"] is False and bool(broken.get("error")), repr(broken))
        check("9.4 stats reports unreadable too", ns.stats(cart)["readable"] is False)
        did, msg = raises(ValueError, ns.add_note, cart, KEY_A, "why", "body",
                          SEAT_A, "CLI")
        check("9.5 appending to an unreadable sidecar is refused, not overwritten", did, msg)
        check("9.6 the corrupt file was left on disk for a human to look at",
              open(cart + ".notes.json", encoding="utf-8").read() == "{ this is not json")

        # Valid JSON, wrong document: still unreadable rather than "no notes".
        with open(cart + ".notes.json", "w", encoding="utf-8") as fh:
            json.dump({"format": "something-else", "notes": []}, fh)
        check("9.7 a foreign format is unreadable, not empty",
              ns.load(cart)["readable"] is False)
        with open(cart + ".notes.json", "w", encoding="utf-8") as fh:
            json.dump({"format": ns.FORMAT, "version": 1, "notes": "nope"}, fh)
        check("9.8 notes-not-a-list is unreadable, not empty",
              ns.load(cart)["readable"] is False)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------- 10 · no destructive verb exists at all
def t_no_destructive_verb():
    public = [n for n in dir(ns) if not n.startswith("_") and callable(getattr(ns, n))]
    bad = [n for n in public
           if any(w in n.lower() for w in ("delete", "remove", "purge", "drop", "erase",
                                           "clear", "wipe"))]
    check("10.1 the module exposes no destructive verb", not bad, f"found {bad}")
    for verb in ("add_note", "list_notes", "for_results", "supersede", "share", "stats"):
        check(f"10.2 {verb} exists", hasattr(ns, verb))


def t_stats_shape():
    cart, d = fresh_cart()
    try:
        ns.add_note(cart, KEY_A, "why", "one", by=SEAT_A, surface="CLI", now=1000.0)
        n2 = ns.add_note(cart, KEY_B, "caution", "two", by=SEAT_B, surface="WEB",
                         scope="shared", now=1001.0)
        ns.supersede(cart, n2["id"], "two, corrected", by=SEAT_B, surface="WEB", now=1002.0)
        s = ns.stats(cart)
        check("11.1 stats counts every note regardless of scope", s["total"] == 3, repr(s))
        check("11.2 stats splits private and shared",
              s["private"] == 1 and s["shared"] == 2, repr(s))
        check("11.3 stats counts the superseded note", s["superseded"] == 1, repr(s))
        check("11.4 stats lists both authors", s["authors"] == sorted([SEAT_A, SEAT_B]))
        check("11.5 stats leaks no bodies or targets",
              not any(k in s for k in ("notes", "bodies", "targets_list")))
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    for fn in (t_round_trip, t_default_private, t_private_is_not_visible_to_another_seat,
               t_shared_is_visible_to_another_seat, t_share_is_author_only,
               t_supersede_keeps_the_original, t_body_cap_is_a_refusal, t_bad_input_raises,
               t_unreadable_sidecar, t_no_destructive_verb, t_stats_shape):
        fn()
    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILED: " + ", ".join(failures))
        sys.exit(1)
    print("note_store: OK")
