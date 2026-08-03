"""Edit succession: an edited passage must not vanish from clusters and attention.

THE BUG THIS CLOSES. Passages are never edited in place — an edit appends a new passage and
tombstones the old one. Until 2026-08-03 those were two independent calls with nothing
recording that they were related, so an edit changed the content key and every sub-cart
manifest holding the old key had a dangling reference. A curated cluster would quietly shed
members as they were edited.

WHERE THE WRITE SIDE LIVES. In `succession_store` (a sidecar), not here. The link-based
writers that were originally in `subcart.py` were removed on 2026-08-03: studio carts have
no links at all, so a sidecar is the only mechanism that works for every cart, and two
writers for one fact is a maintenance trap. This module keeps only the READ side —
`follow_succession` and the `resolve()` integration.

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_succession.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import overlay as ov     # noqa: E402
import subcart as sc     # noqa: E402


ORIGINAL = "the frustration policy is a free parameter"
EDITED = "the frustration RESOLUTION policy is a free parameter"
EDITED_AGAIN = "frustration resolution policy: a free parameter we never varied"


def link(tombstone=False):
    return {"tombstone": tombstone, "access_count": 1, "accessed_at": 1.0,
            "_reserved": {"flags": {"agency": "reflective"},
                          "canonical": {"truth_status": "active"}}}


def cart_after_one_edit():
    """Cart + succession map as they stand after ORIGINAL is edited to EDITED."""
    texts = ["unrelated one", ORIGINAL, "unrelated two", EDITED]
    links = [link(), link(tombstone=True), link(), link()]
    succession = {sc.content_key(ORIGINAL): sc.content_key(EDITED)}
    return texts, links, succession


# ------------------------------------------------------------------ follow

def test_follow_walks_a_multi_hop_chain():
    assert sc.follow_succession("a", {"a": "b", "b": "c"}) == ("c", 2)


def test_follow_stops_at_the_head():
    assert sc.follow_succession("c", {"a": "b", "b": "c"}) == ("c", 0)


def test_follow_survives_a_cycle():
    """A -> B -> A happens the moment an edit is reverted. An unguarded walk hangs the
    search path rather than failing visibly."""
    final, hops = sc.follow_succession("a", {"a": "b", "b": "a"})
    assert final in ("a", "b") and hops <= 2


def test_follow_survives_a_self_loop():
    assert sc.follow_succession("a", {"a": "a"}) == ("a", 0)


def test_follow_caps_a_pathological_chain():
    chain = {str(i): str(i + 1) for i in range(1000)}
    final, hops = sc.follow_succession("0", chain, max_hops=8)
    assert hops == 8 and final == "8"


def test_follow_with_no_succession_map_is_identity():
    assert sc.follow_succession("a", {}) == ("a", 0)
    assert sc.follow_succession("a", None) == ("a", 0)


# ------------------------------------------------------------------ resolve

def _manifest(*keys):
    return sc.new_manifest("cluster", "c.pkl", keys=list(keys), created_at=1.0)


def test_without_succession_the_old_key_resolves_to_the_old_passage():
    """Today's behaviour when no edit was recorded — and now at least VISIBLE, because
    resolve reports that it landed on a tombstone."""
    texts, links, _ = cart_after_one_edit()
    r = sc.resolve(_manifest(sc.content_key(ORIGINAL)), texts, links=links)
    assert r["indices"] == [1] and r["stats"]["tombstoned"] == 1


def test_with_succession_the_cluster_follows_the_edit():
    texts, links, succession = cart_after_one_edit()
    r = sc.resolve(_manifest(sc.content_key(ORIGINAL)), texts, links=links,
                   succession=succession)
    assert r["indices"] == [3]                        # the EDITED passage
    assert r["stats"]["tombstoned"] == 0 and r["stats"]["followed"] == 1


def test_the_follow_is_reported_never_silent():
    """A caller must be able to say 'this is the current version of what you selected'
    rather than quietly hand back different content than was asked for."""
    texts, links, succession = cart_after_one_edit()
    r = sc.resolve(_manifest(sc.content_key(ORIGINAL)), texts, links=links,
                   succession=succession)
    assert r["followed"] == {sc.content_key(ORIGINAL): sc.content_key(EDITED)}


def test_a_two_edit_chain_lands_on_the_newest():
    texts = [ORIGINAL, EDITED, EDITED_AGAIN]
    succession = {sc.content_key(ORIGINAL): sc.content_key(EDITED),
                  sc.content_key(EDITED): sc.content_key(EDITED_AGAIN)}
    r = sc.resolve(_manifest(sc.content_key(ORIGINAL)), texts, succession=succession)
    assert r["indices"] == [2]


def test_untouched_members_are_unaffected():
    texts, links, succession = cart_after_one_edit()
    m = _manifest(sc.content_key("unrelated one"), sc.content_key(ORIGINAL))
    r = sc.resolve(m, texts, links=links, succession=succession)
    assert r["indices"] == [0, 3] and r["stats"]["followed"] == 1


def test_a_deletion_with_no_successor_stays_deleted():
    """A real removal must NOT be rescued. Succession follows edits, not deletions."""
    texts, links = ["kept", "deleted"], [link(), link(tombstone=True)]
    r = sc.resolve(_manifest(sc.content_key("deleted")), texts, links=links, succession={})
    assert r["indices"] == [1] and r["stats"]["tombstoned"] == 1
    assert r["stats"]["followed"] == 0


def test_succession_to_a_passage_not_in_this_cart_does_not_follow():
    """The successor must actually be here. Trading a resolvable key for an unresolvable
    one is strictly worse than not following."""
    texts, links = ["only me"], [link(tombstone=True)]
    succession = {sc.content_key("only me"): "f" * 40}
    r = sc.resolve(_manifest(sc.content_key("only me")), texts, links=links,
                   succession=succession)
    assert r["indices"] == [0] and r["stats"]["followed"] == 0


def test_resolve_without_links_or_succession_behaves_as_before():
    """Every pre-existing caller passes neither; nothing may change for them."""
    texts, _, _ = cart_after_one_edit()
    r = sc.resolve(_manifest(sc.content_key(ORIGINAL)), texts)
    assert r["indices"] == [1] and r["stats"]["tombstoned"] == 0


def test_mode_all_reports_tombstones_too():
    texts, links, _ = cart_after_one_edit()
    m = sc.new_manifest("all", "c.pkl", mode="all", authored_by="access", created_at=1.0)
    r = sc.resolve(m, texts, links=links)
    assert r["stats"]["tombstoned"] == 1 and len(r["indices"]) == 4


# ------------------------------------------------------------------ attention

def test_a_seats_attention_survives_an_edit():
    """The point of the whole exercise: 40 reads must not evaporate because someone fixed
    a typo."""
    _, _, succession = cart_after_one_edit()
    seat = ov.new_overlay("user_a", "c.pkl")
    old_key = sc.content_key(ORIGINAL)
    ov.touch(seat, old_key, count=40)

    head, hops = sc.follow_succession(old_key, succession)
    assert hops == 1 and head == sc.content_key(EDITED)
    # Read THROUGH the chain rather than rewritten — no per-edit writes per seat.
    assert ov.get(seat, old_key)["access_count"] == 40


def test_attention_read_through_the_chain_finds_nothing_on_the_successor():
    """Documents the trade: following at READ time means the successor's own row stays
    empty until this seat touches it. Migrating rows would cost one write per seat per
    edit, which is the thing being avoided."""
    seat = ov.new_overlay("user_a", "c.pkl")
    ov.touch(seat, sc.content_key(ORIGINAL), count=40)
    assert ov.get(seat, sc.content_key(EDITED))["access_count"] == 0
