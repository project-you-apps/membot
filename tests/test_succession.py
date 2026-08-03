"""Edit succession: an edited passage must not vanish from clusters and attention.

THE BUG THIS CLOSES. Passages are never edited in place -- an edit appends a new passage and
tombstones the old one. In the studio those are two independent calls (`POST /api/patterns`
then `DELETE /api/patterns/{idx}`) with nothing recording that they are related. Since an
edit changes the body it changes the content key, so:

  * every sub-cart manifest holding the old key had a dangling reference
  * every seat's overlay row for it went unresolved

A curated cluster would quietly shed members as they were edited, and a person's attention
history would lose exactly the passages they had worked on hardest. Nothing raised.

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


def link(tombstone=False, superseded_by=None):
    canonical = {"truth_status": "active"}
    if superseded_by is not None:
        canonical[sc.SUCCESSION_FIELD] = superseded_by
    return {"tombstone": tombstone, "access_count": 1, "accessed_at": 1.0,
            "_reserved": {"flags": {"agency": "reflective"}, "canonical": canonical}}


def cart_after_one_edit():
    """What the cart looks like after ORIGINAL is edited to EDITED."""
    texts = ["unrelated one", ORIGINAL, "unrelated two", EDITED]
    links = [link(), link(tombstone=True, superseded_by=sc.content_key(EDITED)),
             link(), link()]
    return texts, links


# ------------------------------------------------------------------ record

def test_record_succession_points_old_at_new():
    updated = sc.record_succession(link(), EDITED, now=100.0)
    canonical = updated["_reserved"]["canonical"]
    assert canonical[sc.SUCCESSION_FIELD] == sc.content_key(EDITED)
    assert canonical["superseded_at"] == 100.0


def test_record_succession_does_not_tombstone():
    """Supersession and deletion are separate facts."""
    assert sc.record_succession(link(), EDITED)["tombstone"] is False


def test_record_succession_does_not_mutate_the_original_link():
    original = link()
    sc.record_succession(original, EDITED)
    assert sc.SUCCESSION_FIELD not in original["_reserved"]["canonical"]


def test_record_succession_preserves_other_canonical_fields():
    base = link()
    base["_reserved"]["canonical"]["duplicate_of_key"] = "abc"
    updated = sc.record_succession(base, EDITED)
    assert updated["_reserved"]["canonical"]["duplicate_of_key"] == "abc"
    assert updated["_reserved"]["canonical"]["truth_status"] == "active"


def test_record_succession_rejects_garbage():
    with pytest.raises(ValueError):
        sc.record_succession("not a link", EDITED)


# ------------------------------------------------------------------ build + follow

def test_build_succession_maps_old_key_to_new():
    texts, links = cart_after_one_edit()
    succession = sc.build_succession(texts, links)
    assert succession == {sc.content_key(ORIGINAL): sc.content_key(EDITED)}


def test_build_succession_ignores_passages_with_no_successor():
    texts, links = cart_after_one_edit()
    assert sc.content_key("unrelated one") not in sc.build_succession(texts, links)


def test_follow_walks_a_multi_hop_chain():
    succession = {"a": "b", "b": "c"}
    assert sc.follow_succession("a", succession) == ("c", 2)


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


def test_without_succession_an_edited_passage_is_kept_but_flagged_deleted():
    """Today's behaviour, now at least VISIBLE: the old key still resolves, to a tombstone."""
    texts, links = cart_after_one_edit()
    m = _manifest(sc.content_key(ORIGINAL))
    r = sc.resolve(m, texts, links=links)
    assert r["indices"] == [1]
    assert r["stats"]["tombstoned"] == 1


def test_with_succession_the_cluster_follows_the_edit():
    texts, links = cart_after_one_edit()
    m = _manifest(sc.content_key(ORIGINAL))
    r = sc.resolve(m, texts, links=links,
                   succession=sc.build_succession(texts, links))
    assert r["indices"] == [3]                       # the EDITED passage
    assert r["stats"]["tombstoned"] == 0
    assert r["stats"]["followed"] == 1


def test_the_follow_is_reported_never_silent():
    """A caller must be able to say 'this is the current version of what you selected'
    rather than quietly hand back different content than was asked for."""
    texts, links = cart_after_one_edit()
    m = _manifest(sc.content_key(ORIGINAL))
    r = sc.resolve(m, texts, links=links, succession=sc.build_succession(texts, links))
    assert r["followed"] == {sc.content_key(ORIGINAL): sc.content_key(EDITED)}


def test_a_two_edit_chain_lands_on_the_newest():
    texts = [ORIGINAL, EDITED, EDITED_AGAIN]
    links = [link(tombstone=True, superseded_by=sc.content_key(EDITED)),
             link(tombstone=True, superseded_by=sc.content_key(EDITED_AGAIN)),
             link()]
    m = _manifest(sc.content_key(ORIGINAL))
    r = sc.resolve(m, texts, links=links, succession=sc.build_succession(texts, links))
    assert r["indices"] == [2]


def test_untouched_members_are_unaffected():
    texts, links = cart_after_one_edit()
    m = _manifest(sc.content_key("unrelated one"), sc.content_key(ORIGINAL))
    r = sc.resolve(m, texts, links=links, succession=sc.build_succession(texts, links))
    assert r["indices"] == [0, 3] and r["stats"]["followed"] == 1


def test_a_tombstone_with_no_successor_stays_deleted():
    """A real deletion must NOT be rescued. Succession follows edits, not removals."""
    texts = ["kept", "deleted"]
    links = [link(), link(tombstone=True)]
    m = _manifest(sc.content_key("deleted"))
    r = sc.resolve(m, texts, links=links, succession=sc.build_succession(texts, links))
    assert r["indices"] == [1] and r["stats"]["tombstoned"] == 1
    assert r["stats"]["followed"] == 0


def test_succession_to_a_passage_not_in_this_cart_does_not_follow():
    """The successor must actually be here. Otherwise a resolvable key is traded for an
    unresolvable one, which is strictly worse than not following."""
    texts, links = ["only me"], [link(tombstone=True, superseded_by="f" * 40)]
    m = _manifest(sc.content_key("only me"))
    r = sc.resolve(m, texts, links=links, succession=sc.build_succession(texts, links))
    assert r["indices"] == [0] and r["stats"]["followed"] == 0


def test_resolve_without_links_behaves_exactly_as_before():
    """Every existing caller passes no links and no succession; nothing may change."""
    texts, _ = cart_after_one_edit()
    r = sc.resolve(_manifest(sc.content_key(ORIGINAL)), texts)
    assert r["indices"] == [1] and r["stats"]["tombstoned"] == 0


def test_mode_all_reports_tombstones_too():
    texts, links = cart_after_one_edit()
    m = sc.new_manifest("all", "c.pkl", mode="all", authored_by="access", created_at=1.0)
    r = sc.resolve(m, texts, links=links)
    assert r["stats"]["tombstoned"] == 1 and len(r["indices"]) == 4


# ------------------------------------------------------------------ attention

def test_a_seats_attention_survives_an_edit():
    """The point of the whole exercise: 40 reads of a passage must not evaporate because
    someone fixed a typo in it."""
    texts, links = cart_after_one_edit()
    seat = ov.new_overlay("user_a", "c.pkl")
    ov.touch(seat, sc.content_key(ORIGINAL), count=40)

    succession = sc.build_succession(texts, links)
    old_key = sc.content_key(ORIGINAL)
    head, hops = sc.follow_succession(old_key, succession)

    assert hops == 1 and head == sc.content_key(EDITED)
    # Attention is read through the chain rather than rewritten -- no per-edit writes.
    assert ov.get(seat, old_key)["access_count"] == 40


def test_attention_read_through_the_chain_finds_nothing_new_on_the_successor():
    """Documents the trade: following at READ time means the successor's own row is still
    empty until this seat touches it. The alternative -- migrating rows on every edit --
    costs one write per seat per edit, which is the thing we are avoiding."""
    seat = ov.new_overlay("user_a", "c.pkl")
    ov.touch(seat, sc.content_key(ORIGINAL), count=40)
    assert ov.get(seat, sc.content_key(EDITED))["access_count"] == 0
