"""Materializing a sub-cartridge into a standalone cart.

The load-bearing tests are the link-remap ones. `prev`/`next`/`parent`/`related` and
`_reserved.canonical.duplicate_of` are POSITIONS, and a materialized cart renumbers
everything -- so copying them verbatim points every reference at whatever now sits at the
old offset. That is the 2026-07-29 failure (five thread flags onto five unrelated passages)
waiting to happen again, and nothing about it raises.

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_materialize.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subcart as sc                       # noqa: E402
from materialize import materialize, verify  # noqa: E402


TEXTS = ["alpha body", "beta body", "gamma body", "delta body", "epsilon body"]


def link(prev=None, nxt=None, parent=None, related=None, dup=None):
    return {
        "prev": prev, "next": nxt, "parent": parent, "related": related or [],
        "access_count": 1, "accessed_at": 5.0,
        "_reserved": {"flags": {"agency": "reflective"},
                      "canonical": {"truth_status": "active", "duplicate_of": dup,
                                    "duplicate_of_key": "abc123"}},
    }


def cart():
    return {
        "version": 1,
        "passages": list(TEXTS),
        "embeddings": np.arange(len(TEXTS) * 4, dtype=np.float32).reshape(len(TEXTS), 4),
        "links": [link(nxt=1), link(prev=0, nxt=2), link(prev=1, nxt=3, related=[0, 4]),
                  link(prev=2, nxt=4), link(prev=3, dup=0)],
        "compressed_lens": [10, 20, 30, 40, 50],
        "metadata": {"name": "parent"},
    }


def man(*idx, name="slice"):
    return sc.new_manifest(name, "parent.pkl",
                           keys=[sc.content_key(TEXTS[i]) for i in idx], created_at=1.0)


# ------------------------------------------------------------------ basics

def test_only_selected_passages_are_carried():
    out = materialize(man(0, 2, 4), cart())
    assert out["passages"] == ["alpha body", "gamma body", "epsilon body"]


def test_embeddings_track_passage_order_exactly():
    """Any other order silently pairs every vector with the wrong text."""
    parent = cart()
    out = materialize(man(3, 1), parent)
    assert np.array_equal(out["embeddings"][0], parent["embeddings"][3])
    assert np.array_equal(out["embeddings"][1], parent["embeddings"][1])


def test_compressed_lens_follow_too():
    assert materialize(man(1, 3), cart())["compressed_lens"] == [20, 40]


def test_mode_all_materializes_everything():
    m = sc.new_manifest("everything", "parent.pkl", mode="all", authored_by="access",
                        created_at=1.0)
    assert len(materialize(m, cart())["passages"]) == len(TEXTS)


# ------------------------------------------------------------------ the trap

def test_indices_are_remapped_not_copied():
    """passages 1,2,3 -> new 0,1,2. Link 2's prev was 1, must become 0."""
    out = materialize(man(1, 2, 3), cart())
    assert out["links"][1]["prev"] == 0
    assert out["links"][1]["next"] == 2


def test_references_outside_the_slice_are_severed_and_counted():
    out = materialize(man(1, 2, 3), cart())
    assert out["links"][0]["prev"] is None      # pointed at 0, not included
    assert out["links"][2]["next"] is None      # pointed at 4, not included
    assert out["stats"]["links_severed"] >= 2


def test_related_lists_are_remapped_and_pruned():
    out = materialize(man(0, 2, 4), cart())
    # gamma's related was [0, 4]; both included, now [0, 2]
    assert out["links"][1]["related"] == [0, 2]


def test_related_drops_excluded_targets():
    out = materialize(man(2, 3), cart())
    assert out["links"][0]["related"] == []     # 0 and 4 both excluded


def test_duplicate_of_index_is_remapped_but_the_key_survives():
    """duplicate_of is an INDEX; duplicate_of_key is content-addressed. Exactly why
    the 2026-07-30 migration was worth doing."""
    out = materialize(man(0, 4), cart())
    canonical = out["links"][1]["_reserved"]["canonical"]
    assert canonical["duplicate_of"] == 0
    assert canonical["duplicate_of_key"] == "abc123"


def test_duplicate_of_is_severed_when_the_survivor_is_excluded():
    out = materialize(man(3, 4), cart())
    assert out["links"][1]["_reserved"]["canonical"]["duplicate_of"] is None


def test_materialize_does_not_mutate_the_parent():
    parent = cart()
    materialize(man(1, 2), parent)
    assert parent["links"][1]["prev"] == 0
    assert parent["links"][2]["related"] == [0, 4]
    assert len(parent["passages"]) == 5


# ------------------------------------------------------------------ provenance

def test_result_can_explain_where_it_came_from():
    """An artifact that cannot state its own provenance is one nobody can audit later."""
    m = man(0, 1, name="q3")
    out = materialize(m, cart(), now=999.0)
    prov = out["materialized_from"]
    assert prov["subcart_name"] == "q3" and prov["subcart_id"] == m["id"]
    assert prov["parent_cart"] == "parent.pkl" and prov["parent_passages"] == 5
    assert prov["materialized_at"] == 999.0


def test_unresolved_keys_are_reported_in_provenance():
    m = sc.new_manifest("ghosts", "parent.pkl",
                        keys=[sc.content_key(TEXTS[0]), "f" * 40], created_at=1.0)
    out = materialize(m, cart())
    assert out["materialized_from"]["unresolved"] == 1
    assert len(out["passages"]) == 1


# ------------------------------------------------------------------ verify

def test_verify_passes_on_a_good_materialization():
    m = man(0, 2, 4)
    assert verify(cart(), materialize(m, cart()), m) == []


def test_verify_catches_an_embedding_row_mismatch():
    m = man(0, 2)
    out = materialize(m, cart())
    out["embeddings"] = out["embeddings"][:1]
    assert any("embeddings" in p for p in verify(cart(), out, m))


def test_verify_catches_a_dangling_index():
    m = man(0, 2)
    out = materialize(m, cart())
    out["links"][0]["next"] = 99
    assert any("out of range" in p for p in verify(cart(), out, m))


def test_verify_catches_a_passage_that_is_not_in_the_manifest():
    m = man(0, 2)
    out = materialize(m, cart())
    out["passages"][0] = "something else entirely"
    assert any("not in the manifest" in p for p in verify(cart(), out, m))


def test_verify_catches_a_link_count_mismatch():
    m = man(0, 2)
    out = materialize(m, cart())
    out["links"].append(link())
    assert any("links" in p for p in verify(cart(), out, m))


# ------------------------------------------------------------------ edge cases

def test_empty_selection_produces_an_empty_cart_not_an_error():
    m = sc.new_manifest("nothing", "parent.pkl", keys=[], created_at=1.0)
    out = materialize(m, cart())
    assert out["passages"] == [] and out["stats"]["passages"] == 0


def test_a_cart_with_no_links_materializes_fine():
    parent = cart()
    parent["links"] = []
    out = materialize(man(0, 1), parent)
    assert out["links"] == [] and len(out["passages"]) == 2


def test_a_cart_with_no_embeddings_materializes_fine():
    parent = cart()
    parent.pop("embeddings")
    out = materialize(man(0, 1), parent)
    assert "embeddings" not in out and len(out["passages"]) == 2
