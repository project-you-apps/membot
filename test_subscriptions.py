#!/usr/bin/env python3
"""subscriptions: the watch list, the rich payload, and the two boundaries that matter.

HOUSE RULE THIS FILE IS WRITTEN AGAINST: "EXISTENCE IS NOT OPERATION -- a check identical in
both states is not a check." Every case below is built so that it FAILS if the mechanism it
names were removed. For a filter that means each case is PAIRED: "seat B sees nothing"
passes just as happily against a feed that returns nothing to anybody, so every negative
half is accompanied by a positive control asserting that the entitled seat DOES see it.

The two most likely bugs in `subscriptions.py`, and what covers them here:

  1. A SUBSCRIPTION BECOMING A READ CHANNEL FOR SOMEBODY ELSE'S PRIVATE NOTE. Covered in
     §5, and covered THROUGH `what_changed` rather than through note_store -- note_store's
     own boundary is already tested in test_note_store.py, and the failure being hunted here
     is a new delivery path that forgot to ask it.
  2. A FEED THAT RETURNS EVERYTHING. §2 asserts both halves: the watched passage's events
     ARE returned and the unwatched passage's events are NOT. A naive test that only checked
     "my passage is in there" would pass against a function that returned the whole journal.

Also covered: the integrity gate (§8 -- a tampered journal must not quietly hand back rows),
the per-seat high-water mark and its refusal to rewind (§6, §7), missing-vs-empty journal as
different facts (§11), an unreadable sidecar reported as unreadable rather than as "nobody is
subscribed" (§9), and a scope guard asserting NO DELIVERY CHANNEL was smuggled in (§13).

No network, no GPU, no server. Writes only inside temp dirs it removes.

Run from membot/:  python test_subscriptions.py
"""
import json
import os
import re
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_MEMBOT_DIR = _HERE if os.path.exists(os.path.join(_HERE, "subscriptions.py")) \
    else os.path.dirname(_HERE)
sys.path.insert(0, _MEMBOT_DIR)

import journal                                           # noqa: E402
import note_store as ns                                  # noqa: E402
import subscriptions as sub                              # noqa: E402

failures = []
checks = 0

# Content keys: sha1-shaped, never ordinals. The module refuses anything under 8 chars for
# exactly the reason these are written out longhand.
KEY_A = "a1b2c3d4e5f60718293a4b5c"
KEY_B = "ffeeddccbbaa99887766554433"
KEY_OLD = "0011223344556677889900aa"
SEAT_A = "seat-a"
SEAT_B = "seat-b"


def check(label, ok, detail=""):
    global checks
    checks += 1
    if not ok:
        failures.append(label)
    print(f"{'PASS' if ok else 'FAIL'}  {label}"
          + (f"\n        {detail}" if detail and not ok else ""))


def raises(exc, fn, *a, **kw):
    """Returns (did_raise, message). Catches only `exc`, so an unexpected error still blows
    up loudly instead of being scored as a correct refusal."""
    try:
        fn(*a, **kw)
    except exc as e:
        return True, str(e)
    return False, ""


def fresh_cart():
    d = tempfile.mkdtemp(prefix="subscriptions-test-")
    return os.path.join(d, "demo.cart"), d


def jev(cart, op, key="", writer=SEAT_B, payload=None, idx=None, surface="MCP",
        event_ts=0.0):
    """One journal entry, shaped the way `membot_server._journal_event` shapes them."""
    target = {"cart": "demo.cart"}
    if key:
        target["target_key"] = key
    if idx is not None:
        target["idx"] = idx
    return journal.append_to_cart(cart, writer=writer, surface=surface, op=op,
                                  target=target, payload=dict(payload or {}),
                                  event_ts=event_ts)


def tamper_middle_line(cart):
    """Edit an interior entry in place, which is what breaks a hash chain.

    Deliberately leaves valid JSON behind: the interesting failure is a record that still
    PARSES and no longer matches its own hash, because that is what an edit looks like. A
    garbled line would be caught by the JSON parser and would prove much less.
    """
    jp = str(journal.journal_path_for(cart))
    with open(jp, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    mid = len(lines) // 2
    obj = json.loads(lines[mid])
    obj["payload"] = dict(obj.get("payload") or {}, tampered_by="somebody with an editor")
    lines[mid] = json.dumps(obj, sort_keys=True, ensure_ascii=False) + "\n"
    with open(jp, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    return mid + 1


def note_bodies(result):
    """Every note body anywhere in a what_changed result -- `why` AND `why_prior`.

    Checking only `why` would let a leak through the other field, and `why_prior` is the
    field that deliberately reaches onto a DIFFERENT passage (the superseded one), so it is
    the likelier of the two to have skipped the visibility filter.
    """
    out = []
    for c in result.get("changes", []):
        for n in list(c.get("why", [])) + list(c.get("why_prior", [])):
            out.append(n.get("body", ""))
    return out


def note_rows(result):
    out = []
    for c in result.get("changes", []):
        out.extend(list(c.get("why", [])) + list(c.get("why_prior", [])))
    return out


# ------------------------------------------------- 1 · round trip across a reload
def t_round_trip():
    cart, d = fresh_cart()
    try:
        entry = sub.subscribe(cart, SEAT_A, KEY_A, now=1000.0)
        check("1.1 sidecar lands beside the cart as <cart>.subscriptions.json",
              os.path.exists(cart + ".subscriptions.json"),
              f"looked for {cart + '.subscriptions.json'}")
        check("1.2 id is sub_ + 16 hex",
              entry["id"].startswith("sub_") and len(entry["id"]) == 20, entry["id"])
        check("1.3 a new row reports created=True", entry.get("created") is True,
              repr(entry))

        # A RELOAD, not the in-memory dict: read the file back through a fresh call.
        got = sub.list_subscriptions(cart, SEAT_A)
        check("1.4 exactly one subscription survives the reload", len(got) == 1, repr(got))
        row = got[0] if got else {}
        for field in ("id", "seat", "scope", "target_key", "at", "active"):
            check(f"1.5 reloaded row carries `{field}`", field in row, repr(row))
        check("1.6 reloaded values match what was written",
              row.get("seat") == SEAT_A and row.get("scope") == "passage"
              and row.get("target_key") == KEY_A and row.get("at") == 1000.0
              and row.get("active") is True, repr(row))

        # Idempotent: a second subscribe to the same target must not mint a duplicate, or
        # the same change would be reported twice and unsubscribe would look broken.
        again = sub.subscribe(cart, SEAT_A, KEY_A, now=1001.0)
        check("1.7 re-subscribing is idempotent (same id, created=False)",
              again["id"] == entry["id"] and again.get("created") is False, repr(again))
        check("1.8 and no duplicate row landed on disk",
              len(sub.list_subscriptions(cart, SEAT_A)) == 1)

        off = sub.unsubscribe(cart, SEAT_A, entry["id"], now=1002.0)
        check("1.9 unsubscribe flips active to False", off.get("active") is False, repr(off))
        check("1.10 and attributes the stop",
              off.get("unsubscribed_by") == SEAT_A and off.get("unsubscribed_at") == 1002.0,
              repr(off))
        check("1.11 the inactive row is gone from the default list",
              sub.list_subscriptions(cart, SEAT_A) == [], repr(sub.list_subscriptions(cart, SEAT_A)))
        check("1.12 but the row is RETAINED, not deleted (append-only)",
              len(sub.list_subscriptions(cart, SEAT_A, include_inactive=True)) == 1)
        did, msg = raises(ValueError, sub.unsubscribe, cart, SEAT_A, entry["id"])
        check("1.13 unsubscribing twice is refused, not silently repeated", did, msg)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------- 2 · the watched / unwatched split
def t_watched_vs_unwatched():
    """BOTH HALVES. Returning the whole journal would pass the first half on its own."""
    cart, d = fresh_cart()
    try:
        sub.subscribe(cart, SEAT_A, KEY_A, now=1000.0)
        jev(cart, "store", KEY_A, payload={"identity_source": "verified"})       # seq 1
        jev(cart, "store", KEY_B, payload={"identity_source": "verified"})       # seq 2
        jev(cart, "corroborate", KEY_A, payload={"identity_source": "claimed"})  # seq 3

        res = sub.what_changed(cart, SEAT_A, now=2000.0)
        seqs = sorted(c["seq"] for c in res["changes"])
        check("2.1 events on the WATCHED passage are returned", seqs == [1, 3], repr(seqs))
        keys = {c["where"]["target_key"] for c in res["changes"]}
        check("2.2 events on an UNWATCHED passage are NOT returned "
              "(this is the half a naive test skips)", KEY_B not in keys, repr(keys))
        check("2.3 the result reports the head of the journal, so 'quiet' is checkable",
              res["head_seq"] == 3, repr(res["head_seq"]))
        check("2.4 matched_by names the passage subscription",
              res["changes"] and all(c["where"]["matched_by"] == "passage"
                                     for c in res["changes"]))
        check("2.5 order is newest-first and declared",
              res["order"] == "seq_desc"
              and [c["seq"] for c in res["changes"]] == [3, 1],
              repr([c["seq"] for c in res["changes"]]))
        check("2.6 verdict_line is computed and leads with the count",
              res["verdict_line"].startswith("2 change(s)"), res["verdict_line"])

        # A seat with NO subscription must get a different answer from a seat whose
        # documents are quiet -- otherwise a wiring bug reads as reassurance.
        other = sub.what_changed(cart, SEAT_B, now=2000.0)
        check("2.7 a seat watching nothing gets 0 changes AND is told why",
              other["change_count"] == 0 and "watches nothing" in other["verdict_line"],
              other["verdict_line"])

        # No notes exist on this cart, so the headline must say the why is MISSING rather
        # than render as though the change were explained.
        check("2.8 a change with no recorded why says so instead of looking explained",
              res["changes"] and all(c["why_count"] == 0
                                     and "no recorded why" in c["headline"]
                                     for c in res["changes"]),
              repr([c["headline"] for c in res["changes"]]))
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------- 3 · cart scope
def t_cart_scope():
    cart, d = fresh_cart()
    try:
        sub.subscribe(cart, SEAT_A, scope="cart", now=1000.0)
        jev(cart, "store", KEY_A, payload={"identity_source": "verified"})
        jev(cart, "store", KEY_B, payload={"identity_source": "verified"})
        jev(cart, "tombstone", "", idx=7, payload={"identity_source": "verified"})

        res = sub.what_changed(cart, SEAT_A, now=2000.0)
        check("3.1 a cart-scope subscription sees changes to ANY passage",
              res["change_count"] == 3, repr([c["seq"] for c in res["changes"]]))
        check("3.2 including an event that names no content key at all",
              any(c["op"] == "tombstone" for c in res["changes"]))
        check("3.3 matched_by says cart, so the caller can tell how it matched",
              res["changes"] and all(c["where"]["matched_by"] == "cart"
                                     for c in res["changes"]))
        check("3.4 cart_scope is reported on the result", res["cart_scope"] is True)

        did, msg = raises(ValueError, sub.subscribe, cart, SEAT_A, KEY_A, "cart")
        check("3.5 a cart-scope subscription refuses a target_key (it would be a lie "
              "about narrowing)", did, msg)
        did, msg = raises(ValueError, sub.subscribe, cart, SEAT_A, "7", "passage")
        check("3.6 an INDEX-shaped target_key is refused", did, msg)
        did, msg = raises(ValueError, sub.subscribe, cart, "", KEY_A)
        check("3.7 an empty seat is refused -- the caller does not pick its own identity",
              did, msg)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------- 4 · the rich payload
def t_payload_is_rich():
    """The differentiator: who, whether that identity was VERIFIED, what it descends from,
    and WHY the old one stopped being current."""
    cart, d = fresh_cart()
    try:
        sub.subscribe(cart, SEAT_A, KEY_A, now=1000.0)
        # Why the NEW one is current, and why the OLD one stopped. Shared, so SEAT_A sees
        # both -- the private case is §5's job.
        ns.add_note(cart, KEY_A, "decision", "Replaces the 2025 form; legal signed off.",
                    by=SEAT_B, surface="CLI", scope="shared", now=1100.0)
        ns.add_note(cart, KEY_OLD, "why",
                    "The old form cited a rescinded FDICIA threshold, so it stopped being "
                    "current the day the threshold moved.",
                    by=SEAT_B, surface="CLI", scope="shared", now=1101.0)

        jev(cart, "supersede", KEY_A, writer=SEAT_B, event_ts=1762000100.0,
            payload={"identity_source": "verified", "superseded_key": KEY_OLD})
        jev(cart, "corroborate", KEY_A, writer="agent-7", event_ts=1762000200.0,
            payload={"identity_source": "claimed"})
        jev(cart, "flag", KEY_A, writer="legacy-path", event_ts=1762000300.0, payload={})

        res = sub.what_changed(cart, SEAT_A, now=2000.0)
        by_seq = {c["seq"]: c for c in res["changes"]}
        check("4.0 all three events came back", len(by_seq) == 3, repr(sorted(by_seq)))

        c1 = by_seq.get(1, {})
        check("4.1 the payload names the author",
              c1.get("who", {}).get("writer") == SEAT_B, repr(c1.get("who")))
        check("4.2 and the surface it was written on",
              c1.get("who", {}).get("surface") == "MCP", repr(c1.get("who")))
        check("4.3 a VERIFIED identity is reported as verified",
              c1.get("who", {}).get("identity_verified") is True
              and c1["who"]["identity_source"] == "verified", repr(c1.get("who")))
        check("4.4 the event carries WHEN, as both epoch and ISO",
              c1.get("when", {}).get("event_ts") == 1762000100.0
              and c1["when"]["event_iso"].startswith("20"), repr(c1.get("when")))
        check("4.5 what it supersedes is read off the journal",
              (c1.get("supersedes") or {}).get("key") == KEY_OLD, repr(c1.get("supersedes")))
        check("4.6 the explaining note on the NEW passage rides along",
              any("legal signed off" in b for b in
                  [n.get("body", "") for n in c1.get("why", [])]),
              repr(c1.get("why")))
        check("4.7 * and the note saying why the OLD one stopped being current rides "
              "along too -- the clause nobody else can send",
              any("rescinded FDICIA threshold" in n.get("body", "")
                  for n in c1.get("why_prior", [])), repr(c1.get("why_prior")))
        check("4.8 the entry hash travels with the change, so it is checkable",
              str(c1.get("entry_hash", "")).startswith("sha256:"),
              repr(c1.get("entry_hash")))
        check("4.9 headline is computed and names the verification state",
              "verified" in c1.get("headline", "")
              and "supersedes" in c1.get("headline", ""), c1.get("headline"))

        c2 = by_seq.get(2, {})
        check("4.10 a CLAIMED identity is NOT reported as verified",
              c2.get("who", {}).get("identity_verified") is False
              and c2["who"]["identity_recorded"] is True
              and c2["who"]["identity_source"] == "claimed", repr(c2.get("who")))
        check("4.11 and the headline says so in words",
              "unverified claimed" in c2.get("headline", ""), c2.get("headline"))

        c3 = by_seq.get(3, {})
        check("4.12 an ABSENT identity stamp is 'unrecorded', not folded into either "
              "verified or claimed",
              c3.get("who", {}).get("identity_recorded") is False
              and c3["who"]["identity_source"] == "unrecorded"
              and c3["who"]["identity_verified"] is False, repr(c3.get("who")))
        check("4.13 the passage's notes ride along with EVERY event on that passage, not "
              "only the superseding one",
              c2.get("why_count") == 1 and c3.get("why_count") == 1,
              f"seq2={c2.get('why_count')} seq3={c3.get('why_count')}")

        # Nothing was inferred: an event that does not claim a lineage must not grow one.
        check("4.14 supersedes is None when the journal does not say (never inferred)",
              c2.get("supersedes") is None, repr(c2.get("supersedes")))
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------- 5 · THE LEAK TEST
def t_private_note_never_crosses_seats():
    """⭐⭐ Seat B's private note must never appear in seat A's what_changed.

    Tested THROUGH what_changed, because the boundary note_store already enforces is not the
    thing at risk -- a new delivery path that forgot to ask it is. Every negative half below
    is paired with a positive control, so "A sees nothing" cannot pass against a feed that
    shows nobody anything.
    """
    cart, d = fresh_cart()
    try:
        sub.subscribe(cart, SEAT_A, KEY_A, now=1000.0)
        sub.subscribe(cart, SEAT_B, KEY_A, now=1000.0)

        secret = ns.add_note(cart, KEY_A, "caution",
                             "PRIVATE-B: I think the controller approved this under duress.",
                             by=SEAT_B, surface="CLI", now=1100.0)   # private by default
        ns.add_note(cart, KEY_A, "why", "SHARED-A: superseded by the Q3 revision.",
                    by=SEAT_A, surface="CLI", scope="shared", now=1101.0)
        # A private note on the SUPERSEDED passage too, because `why_prior` reaches onto a
        # different key and is the likelier field to have skipped the filter.
        ns.add_note(cart, KEY_OLD, "why", "PRIVATE-B-OLD: the old one was my mistake.",
                    by=SEAT_B, surface="CLI", now=1102.0)

        jev(cart, "supersede", KEY_A, writer=SEAT_B,
            payload={"identity_source": "verified", "superseded_key": KEY_OLD})
        # The other reach into notes: an event that names a note by id.
        jev(cart, "note", KEY_A, writer=SEAT_B,
            payload={"identity_source": "verified", "note_id": secret["id"],
                     "kind": "caution", "scope": "private"})

        a = sub.what_changed(cart, SEAT_A, now=2000.0)
        b = sub.what_changed(cart, SEAT_B, now=2000.0)

        a_bodies = note_bodies(a)
        b_bodies = note_bodies(b)

        check("5.1 ** seat B's PRIVATE note body does not appear in seat A's feed",
              not any("PRIVATE-B:" in x for x in a_bodies), repr(a_bodies))
        check("5.2 ** nor does B's private note on the SUPERSEDED passage "
              "(the why_prior path)",
              not any("PRIVATE-B-OLD" in x for x in a_bodies), repr(a_bodies))
        check("5.3 nor does it arrive via the note_id on the `note` event",
              not any(n.get("id") == secret["id"] for n in note_rows(a)),
              repr([n.get("id") for n in note_rows(a)]))
        check("5.4 no row attributed to seat B with scope private reaches seat A at all",
              not any(n.get("by") == SEAT_B and n.get("scope") == "private"
                      for n in note_rows(a)),
              repr([(n.get("by"), n.get("scope")) for n in note_rows(a)]))

        # POSITIVE CONTROLS. Without these, a feed that returned no notes to anyone would
        # score four passes above.
        check("5.5 POSITIVE CONTROL: seat A DOES see the shared note",
              any("SHARED-A" in x for x in a_bodies), repr(a_bodies))
        check("5.6 POSITIVE CONTROL: seat B DOES see its OWN private note",
              any("PRIVATE-B:" in x for x in b_bodies), repr(b_bodies))
        check("5.7 POSITIVE CONTROL: seat B sees its own private note on the superseded "
              "passage", any("PRIVATE-B-OLD" in x for x in b_bodies), repr(b_bodies))
        check("5.8 POSITIVE CONTROL: seat B also sees seat A's shared note",
              any("SHARED-A" in x for x in b_bodies), repr(b_bodies))
        check("5.9 POSITIVE CONTROL: both seats got the same EVENTS -- only the notes "
              "differ, so the filter is on reasoning and not on the feed",
              sorted(c["seq"] for c in a["changes"])
              == sorted(c["seq"] for c in b["changes"]),
              f"{[c['seq'] for c in a['changes']]} vs {[c['seq'] for c in b['changes']]}")
        check("5.10 and the why counts genuinely differ, so the filter did something",
              sum(c["why_count"] for c in a["changes"])
              < sum(c["why_count"] for c in b["changes"]),
              f"A={sum(c['why_count'] for c in a['changes'])} "
              f"B={sum(c['why_count'] for c in b['changes'])}")

        # A subscription must not be a way to read another seat's watch list either.
        check("5.11 seat A cannot list seat B's subscriptions",
              all(r.get("seat") == SEAT_A for r in sub.list_subscriptions(cart, SEAT_A)))
        check("5.12 an unidentified caller gets an empty watch list, not everyone's",
              sub.list_subscriptions(cart, "") == [])
        did, msg = raises(PermissionError, sub.unsubscribe, cart, SEAT_A,
                          sub.list_subscriptions(cart, SEAT_B)[0]["id"])
        check("5.13 seat A cannot cancel seat B's subscription (silent un-notification)",
              did, msg)
        check("5.14 and B's subscription is still active after the attempt",
              len(sub.list_subscriptions(cart, SEAT_B)) == 1)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------- 6 · the mark advances the window
def t_mark_seen_advances():
    cart, d = fresh_cart()
    try:
        sub.subscribe(cart, SEAT_A, KEY_A, now=1000.0)
        for i in range(3):
            jev(cart, "store", KEY_A, payload={"identity_source": "verified", "n": i})

        before = sub.what_changed(cart, SEAT_A, now=2000.0)
        check("6.1 POSITIVE CONTROL: with no mark, all three changes come back",
              before["change_count"] == 3, repr([c["seq"] for c in before["changes"]]))
        check("6.2 and since_seq_source says there was no mark to use",
              before["since_seq_source"] == "none", before["since_seq_source"])

        mark = sub.mark_seen(cart, SEAT_A, 2, now=2001.0)
        check("6.3 mark_seen advances", mark["advanced"] is True and mark["seq"] == 2,
              repr(mark))

        after = sub.what_changed(cart, SEAT_A, now=2002.0)
        check("6.4 what_changed now returns only the NEWER event",
              [c["seq"] for c in after["changes"]] == [3],
              repr([c["seq"] for c in after["changes"]]))
        check("6.5 and says it used the stored mark, so no client cursor is needed",
              after["since_seq"] == 2 and after["since_seq_source"] == "mark",
              f"{after['since_seq']} / {after['since_seq_source']}")

        # Reading is not acknowledging: a look must not move the mark, or a client that
        # crashed while rendering would lose those changes with nothing able to tell.
        check("6.6 what_changed did NOT advance the mark by itself",
              sub.last_seen(cart, SEAT_A) == 2, sub.last_seen(cart, SEAT_A))

        full = sub.what_changed(cart, SEAT_A, since_seq=sub.FROM_BEGINNING, now=2003.0)
        check("6.7 FROM_BEGINNING replays past the mark on request",
              full["change_count"] == 3 and full["since_seq_source"] == "from_beginning",
              repr(full["since_seq_source"]))

        # The mark is PER SEAT. One global mark would tell the second person nothing changed.
        sub.subscribe(cart, SEAT_B, KEY_A, now=1000.0)
        b = sub.what_changed(cart, SEAT_B, now=2004.0)
        check("6.8 ! the mark is per seat: seat B's window is untouched by seat A's mark",
              b["change_count"] == 3 and b["last_seen_seq"] == 0,
              f"count={b['change_count']} mark={b['last_seen_seq']}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------- 7 · the mark cannot be rewound
def t_mark_seen_cannot_rewind():
    cart, d = fresh_cart()
    try:
        sub.subscribe(cart, SEAT_A, KEY_A, now=1000.0)
        for i in range(3):
            jev(cart, "store", KEY_A, payload={"identity_source": "verified", "n": i})

        sub.mark_seen(cart, SEAT_A, 3, now=2000.0)
        check("7.0 POSITIVE CONTROL: the mark really is at 3",
              sub.last_seen(cart, SEAT_A) == 3)

        back = sub.mark_seen(cart, SEAT_A, 1, now=2001.0)
        check("7.1 a rewind is refused and SAYS so in the return value",
              back["advanced"] is False and back["refused_rewind"] is True, repr(back))
        check("7.2 the returned seq is the mark as it still stands, not the request",
              back["seq"] == 3 and back["requested_seq"] == 1, repr(back))
        check("7.3 and the stored mark did not move",
              sub.last_seen(cart, SEAT_A) == 3, sub.last_seen(cart, SEAT_A))
        check("7.4 ! so the already-answered changes stay hidden, which is the point",
              sub.what_changed(cart, SEAT_A, now=2002.0)["change_count"] == 0)

        same = sub.mark_seen(cart, SEAT_A, 3, now=2003.0)
        check("7.5 re-marking the same seq is a no-op, not a rewind",
              same["advanced"] is False and same["refused_rewind"] is False, repr(same))
        fwd = sub.mark_seen(cart, SEAT_A, 4, now=2004.0)
        check("7.6 POSITIVE CONTROL: forward still works after a refused rewind",
              fwd["advanced"] is True and fwd["seq"] == 4, repr(fwd))
        did, msg = raises(ValueError, sub.mark_seen, cart, SEAT_A, -1)
        check("7.7 a negative seq is refused", did, msg)
        did, msg = raises(ValueError, sub.mark_seen, cart, SEAT_A, "3")
        check("7.8 a non-int seq is refused rather than coerced", did, msg)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------- 8 · THE INTEGRITY GATE
def t_tampered_journal_is_reported_not_served():
    """⭐ A tampered journal must report the integrity problem rather than hand back rows.

    Changes read out of a journal that may have been edited are not evidence, and a caller
    that cannot tell the difference will render them as a notification.
    """
    cart, d = fresh_cart()
    try:
        sub.subscribe(cart, SEAT_A, KEY_A, now=1000.0)
        for i in range(3):
            jev(cart, "store", KEY_A, payload={"identity_source": "verified",
                                               "note": f"entry {i}"})

        clean = sub.what_changed(cart, SEAT_A, now=2000.0)
        check("8.0 POSITIVE CONTROL: on an intact chain the rows ARE served",
              clean["ok"] is True and clean["change_count"] == 3
              and clean["integrity"]["ok"] is True
              and clean["changes_withheld"] is False, clean["verdict_line"])
        check("8.1 POSITIVE CONTROL: and every row is marked trustworthy",
              clean["changes"] and all(c["trustworthy"] is True
                                       for c in clean["changes"]))

        line_no = tamper_middle_line(cart)
        res = sub.what_changed(cart, SEAT_A, now=2001.0)

        check("8.2 * the integrity problem is reported",
              res["integrity"]["ok"] is False and res["integrity"]["checked"] is True,
              repr(res["integrity"]))
        check("8.3 and the verdict line LEADS with it rather than burying it",
              res["verdict_line"].startswith("INTEGRITY FAILURE"), res["verdict_line"])
        check("8.4 * the matching rows are WITHHELD, not returned as if trustworthy",
              res["changes"] == [] and res["changes_withheld"] is True, repr(res["changes"]))
        check("8.5 and the count of withheld rows is reported, so this is not confusable "
              "with 'nothing changed'", res["withheld_count"] == 3, repr(res["withheld_count"]))
        check("8.6 ok is False", res["ok"] is False)
        check("8.7 the break names the line, so a human can look at it",
              (res["integrity"]["first_break"] or {}).get("line_no") == line_no,
              f"expected line {line_no}, got "
              f"{(res['integrity']['first_break'] or {}).get('line_no')}")
        check("8.8 the break names hash_mismatch -- a record edited in place",
              "hash_mismatch" in (res["integrity"]["first_break"] or {}).get("reasons", []),
              repr((res["integrity"]["first_break"] or {}).get("reasons")))
        check("8.9 a notice tells the caller not to render these as a notification",
              any("WITHHELD" in n for n in res["notices"]), repr(res["notices"]))

        forensic = sub.what_changed(cart, SEAT_A, now=2002.0, allow_broken_chain=True)
        check("8.10 the forensics escape hatch returns rows",
              forensic["change_count"] == 3, repr(forensic["change_count"]))
        check("8.11 but every one of them is stamped trustworthy=False",
              forensic["changes"] and all(c["trustworthy"] is False
                                          for c in forensic["changes"]))
        check("8.12 and the verdict line says UNTRUSTED",
              forensic["verdict_line"].startswith("UNTRUSTED"), forensic["verdict_line"])
        check("8.13 ok is still False under the escape hatch", forensic["ok"] is False)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------- 9 · an unreadable sidecar
def t_corrupt_sidecar_is_unreadable_not_empty():
    cart, d = fresh_cart()
    try:
        sub.subscribe(cart, SEAT_A, KEY_A, now=1000.0)
        jev(cart, "store", KEY_A, payload={"identity_source": "verified"})
        good = sub.what_changed(cart, SEAT_A, now=2000.0)
        check("9.0 POSITIVE CONTROL: readable sidecar, one change",
              good["subscriptions_readable"] is True and good["change_count"] == 1,
              good["verdict_line"])

        with open(cart + ".subscriptions.json", "w", encoding="utf-8") as fh:
            fh.write('{"format": "vps-subscriptions", "subscriptions": ')  # truncated

        data = sub.load(cart)
        check("9.1 load() reports readable=False with the parse error",
              data["readable"] is False and data.get("error"), repr(data))

        res = sub.what_changed(cart, SEAT_A, now=2001.0)
        check("9.2 what_changed reports the sidecar as unreadable",
              res["subscriptions_readable"] is False and bool(res["error"]), repr(res["error"]))
        check("9.3 ! it does NOT report 'no subscriptions' / 0 changes as an answer",
              res["verdict_line"].startswith("CANNOT ANSWER")
              and "0 change" not in res["verdict_line"], res["verdict_line"])
        check("9.4 ok is False -- the caller must not treat this as a clean look",
              res["ok"] is False)
        check("9.5 a notice spells out that this is not 'watching nothing'",
              any("NOT the same" in n for n in res["notices"]), repr(res["notices"]))

        did, msg = raises(ValueError, sub.subscribe, cart, SEAT_A, KEY_B)
        check("9.6 a write refuses rather than overwriting the unparseable file",
              did, msg)
        check("9.7 stats also reports unreadable rather than zero",
              sub.stats(cart)["readable"] is False, repr(sub.stats(cart)))
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------- 10 · the index/key seam
def t_index_only_events_are_not_silent():
    """The journal's target is an index; a subscription is a content key. When the bridge is
    missing the answer must SAY SO, because a silent zero reads as 'nothing changed'."""
    cart, d = fresh_cart()
    try:
        sub.subscribe(cart, SEAT_A, KEY_A, now=1000.0)
        jev(cart, "store", "", idx=0, payload={"identity_source": "verified"})
        jev(cart, "store", "", idx=1, payload={"identity_source": "verified"})

        bare = sub.what_changed(cart, SEAT_A, now=2000.0)
        check("10.1 without a resolver nothing matches the passage subscription",
              bare["change_count"] == 0)
        check("10.2 ! but the unmatched events are COUNTED, not swallowed",
              bare["unresolved_index_events"] == 2, repr(bare["unresolved_index_events"]))
        check("10.3 and a notice says the zero is not evidence of quiet",
              any("NOT evidence" in n for n in bare["notices"]), repr(bare["notices"]))

        res = sub.what_changed(cart, SEAT_A, now=2001.0,
                               key_resolver={0: KEY_A, 1: KEY_B})
        check("10.4 POSITIVE CONTROL: with a resolver the watched index matches",
              [c["seq"] for c in res["changes"]] == [1],
              repr([c["seq"] for c in res["changes"]]))
        check("10.5 and the unwatched index still does not",
              res["changes"] and all(c["where"]["target_key"] == KEY_A
                                     for c in res["changes"]))
        check("10.6 nothing is left unresolved once the bridge is supplied",
              res["unresolved_index_events"] == 0)

        callable_res = sub.what_changed(
            cart, SEAT_A, now=2002.0,
            key_resolver=lambda t: KEY_A if t.get("idx") == 0 else "")
        check("10.7 a callable resolver works too", callable_res["change_count"] == 1)

        def boom(_t):
            raise RuntimeError("resolver exploded")

        safe = sub.what_changed(cart, SEAT_A, now=2003.0, key_resolver=boom)
        check("10.8 a resolver that raises does not take down the feed",
              safe["change_count"] == 0 and len(safe["resolver_errors"]) == 2,
              repr(safe["resolver_errors"]))
        check("10.9 and the failure is reported rather than looking like quiet",
              any("key_resolver raised" in n for n in safe["notices"]), repr(safe["notices"]))
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------- 11 · missing vs empty journal
def t_missing_journal_is_not_empty_journal():
    cart, d = fresh_cart()
    try:
        sub.subscribe(cart, SEAT_A, KEY_A, now=1000.0)
        res = sub.what_changed(cart, SEAT_A, now=2000.0)
        check("11.1 a missing journal does not crash",
              isinstance(res, dict) and res["change_count"] == 0)
        check("11.2 it is reported as NO JOURNAL, a louder fact than 'no changes'",
              res["journal_exists"] is False
              and res["verdict_line"].startswith("NO JOURNAL"), res["verdict_line"])
        check("11.3 ok is True -- nothing has happened yet is not an error",
              res["ok"] is True)
        check("11.4 and a notice names it as what an unwired mutation path looks like",
              any("unwired" in n for n in res["notices"]), repr(res["notices"]))

        open(str(journal.journal_path_for(cart)), "w", encoding="utf-8").close()
        res2 = sub.what_changed(cart, SEAT_A, now=2001.0)
        check("11.5 ! an EXISTING but empty journal is a DIFFERENT answer",
              res2["journal_exists"] is True and res2["journal_empty"] is True
              and not res2["verdict_line"].startswith("NO JOURNAL"), res2["verdict_line"])
        check("11.6 and it verifies as an intact chain of nothing",
              res2["integrity"]["ok"] is True and res2["ok"] is True)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------- 12 · limit is reported, never silent
def t_truncation_is_reported():
    cart, d = fresh_cart()
    try:
        sub.subscribe(cart, SEAT_A, scope="cart", now=1000.0)
        for i in range(5):
            jev(cart, "store", KEY_A, payload={"identity_source": "verified", "n": i})
        res = sub.what_changed(cart, SEAT_A, now=2000.0, limit=2)
        check("12.1 the limit is applied", res["change_count"] == 2)
        check("12.2 and truncation is reported with the count dropped",
              res["truncated"] is True and res["withheld_older"] == 3, repr(res))
        check("12.3 the newest are kept, and the notice says the OLDEST were dropped",
              [c["seq"] for c in res["changes"]] == [5, 4]
              and any("OLDER" in n for n in res["notices"]), repr(res["notices"]))
        did, msg = raises(ValueError, sub.what_changed, cart, SEAT_A, 0, 0.0, None, 0)
        check("12.4 a non-positive limit is refused", did, msg)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------- 13 · scope guard: NO DELIVERY CHANNEL
def t_no_delivery_channel():
    """The scope decision was that push delivery is out. A module that grew a mailer would
    still pass every functional test above, so the guard has to be its own case."""
    src = open(os.path.join(_MEMBOT_DIR, "subscriptions.py"), encoding="utf-8").read()
    # Only the executable part: the docstring explains WHY these are absent and would
    # otherwise trip the scan on its own.
    body = src.split('"""', 2)[-1]
    banned = ("smtplib", "requests", "urllib", "http.client", "httpx", "socket",
              "threading", "multiprocessing", "sched", "subprocess", "asyncio",
              "smtp", "webhook", "Timer(")
    found = [b for b in banned if b in body]
    check("13.1 ! no transport, scheduler or mailer anywhere in the module body",
          not found, f"found {found}")
    verbs = [n for n in dir(sub) if not n.startswith("_") and callable(getattr(sub, n))]
    bad = [n for n in verbs if any(w in n.lower() for w in
                                   ("send", "notify", "email", "mail", "push", "deliver",
                                    "poll", "watch_loop", "delete", "purge", "wipe"))]
    check("13.2 and no send/notify/poll/delete verb is exposed", not bad, f"found {bad}")
    for verb in ("subscribe", "unsubscribe", "list_subscriptions", "what_changed",
                 "mark_seen", "last_seen", "stats", "load"):
        check(f"13.3 {verb} exists", hasattr(sub, verb))

    # ⛔ THE STRUCTURAL HALF OF THE PRIVACY BOUNDARY. Every functional test in §5 would
    # still pass if someone later added a "quick" unfiltered read for one extra field, so
    # the shape of the access is asserted directly: every note read goes through
    # note_store's seat-filtered API, and this module never opens the sidecar itself.
    # One level of nesting allowed, so `sorted(...)` as an argument does not end the match
    # early and score a filtered call as unfiltered.
    note_reads = re.findall(
        r"_notes\.(?:for_results|list_notes)\((?:[^()]|\([^()]*\))*\)", body)
    check("13.4 * every note read passes seat=seat -- the validated seat and no other",
          note_reads and all("seat=seat" in c for c in note_reads), repr(note_reads))
    check("13.5 and the module never names, opens or walks the notes sidecar itself",
          ".notes.json" not in body and "_notes.SIDECAR_SUFFIX" not in body
          and "_notes.sidecar_path" not in body and '.get("notes"' not in body
          and '["notes"]' not in body,
          "a direct notes-sidecar read would bypass note_store._visible()")


# ------------------------------------------------- 14 · owner stats leak nothing
def t_stats_shape():
    cart, d = fresh_cart()
    try:
        s1 = sub.subscribe(cart, SEAT_A, KEY_A, now=1000.0)
        sub.subscribe(cart, SEAT_B, scope="cart", now=1001.0)
        sub.unsubscribe(cart, SEAT_A, s1["id"], now=1002.0)
        sub.mark_seen(cart, SEAT_B, 5, now=1003.0)
        st = sub.stats(cart)
        check("14.1 stats counts every row regardless of state", st["total"] == 2, repr(st))
        check("14.2 and splits active from inactive",
              st["active"] == 1 and st["inactive"] == 1, repr(st))
        check("14.3 by_scope is reported",
              st["by_scope"] == {"passage": 1, "cart": 1}, repr(st["by_scope"]))
        check("14.4 marks are summarized", st["marked_seats"] == 1
              and st["highest_mark"] == 5, repr(st))
        check("14.5 ! stats leaks no target keys and no per-seat watch list",
              not any(k in st for k in ("subscriptions", "targets", "target_keys",
                                        "watch_lists")) and KEY_A not in json.dumps(st),
              repr(st))
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    for fn in (t_round_trip, t_watched_vs_unwatched, t_cart_scope, t_payload_is_rich,
               t_private_note_never_crosses_seats, t_mark_seen_advances,
               t_mark_seen_cannot_rewind, t_tampered_journal_is_reported_not_served,
               t_corrupt_sidecar_is_unreadable_not_empty,
               t_index_only_events_are_not_silent,
               t_missing_journal_is_not_empty_journal, t_truncation_is_reported,
               t_no_delivery_channel, t_stats_shape):
        fn()
    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILED: " + ", ".join(failures))
        sys.exit(1)
    print("subscriptions: OK")
