"""Tests for sub-cartridge manifests (docs/SPEC-SUBCARTRIDGE-MANIFEST.md v1).

The load-bearing test is `test_unsalted_key_matches_shipped_implementation`: it imports the
REAL content_key from tools/build_origin_dates.py and asserts byte equality. Two copies of a
hash function that must agree is the setup for a silent divergence, and a divergence here
detaches every manifest, overlay and duplicate mark from its passages at once.

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_subcart.py -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subcart  # noqa: E402


def _load_shipped_content_key():
    """Import tools/build_origin_dates.py by path -- it is not an installed package."""
    path = Path(__file__).resolve().parents[2] / "tools" / "build_origin_dates.py"
    if not path.exists():
        pytest.skip(f"shipped implementation not present at {path}")
    spec = importlib.util.spec_from_file_location("_shipped_bod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.content_key


PASSAGES = [
    "SESSION: 2026-07-29 | live-store\nKIND: thread\nTHREAD: Heartbeat corpus ingestion",
    "SESSION: 2026-07-29 | live-store\nSESSION: 2026-05-17 | f88e4546\nANDY: what about spans?",
    "no header at all, just a body",
    "   leading and    collapsed\n\n\nwhitespace   ",
]


# ------------------------------------------------------------------ key parity

def test_unsalted_key_matches_shipped_implementation():
    shipped = _load_shipped_content_key()
    for p in PASSAGES:
        assert subcart.content_key(p) == shipped(p), f"divergence on {p[:40]!r}"


def test_all_leading_session_lines_are_stripped():
    one = "SESSION: a\nbody text"
    two = "SESSION: a\nSESSION: b\nbody text"
    assert subcart.content_key(one) == subcart.content_key(two) == \
        subcart.content_key("body text")


def test_whitespace_is_normalised():
    assert subcart.content_key("a   b\n\nc") == subcart.content_key("a b c")


def test_empty_and_none_do_not_raise():
    assert subcart.content_key("") == subcart.content_key(None)


# ------------------------------------------------------------------ salting

def test_salt_changes_the_key():
    assert subcart.content_key("body", "pepper") != subcart.content_key("body")


def test_empty_salt_is_identical_to_no_salt():
    # Existing carts and the 2026-07-30 duplicate_of_key migration depend on this exactly.
    for p in PASSAGES:
        assert subcart.content_key(p, "") == subcart.content_key(p)


def test_salt_is_length_delimited():
    # Without the delimiter, salt "ab" + body "c" and salt "a" + body "bc" would collide.
    assert subcart.content_key("c", "ab") != subcart.content_key("bc", "a")


def test_salt_id_is_stable_and_not_the_salt():
    assert subcart.salt_id("pepper") == subcart.salt_id("pepper")
    assert subcart.salt_id("pepper") != subcart.salt_id("salt")
    assert "pepper" not in subcart.salt_id("pepper")


# ------------------------------------------------------------------ key index

def test_duplicate_passages_map_to_every_index():
    texts = ["same body", "other", "same body"]
    index = subcart.build_key_index(texts)
    assert index[subcart.content_key("same body")] == [0, 2]


# ------------------------------------------------------------------ validation

def _valid(**kw):
    return subcart.new_manifest("cluster", "parent.pkl", keys=["deadbeef"], **kw)


def test_new_manifest_validates():
    assert subcart.validate_manifest(_valid()) == []


def test_manifest_never_carries_the_salt():
    m = _valid()
    m["parent"]["key_salt"] = "pepper"
    problems = subcart.validate_manifest(m)
    assert any("key_salt must NEVER" in p for p in problems)


def test_salt_id_is_present_but_salt_is_not():
    m = subcart.new_manifest("c", "p.pkl", keys=[], parent_salt="pepper")
    assert m["parent"]["key_salt_id"] == subcart.salt_id("pepper")
    assert "pepper" not in json.dumps(m)


@pytest.mark.parametrize("mutate,fragment", [
    (lambda m: m.update(format="nope"), "format"),
    (lambda m: m.update(version=99), "version"),
    (lambda m: m.update(id=""), "id"),
    (lambda m: m.update(name=""), "name"),
    (lambda m: m["parent"].update(cart=""), "parent.cart"),
    (lambda m: m["membership"].update(mode="rule"), "mode"),
    (lambda m: m["membership"].update(authored_by="vibes"), "authored_by"),
    (lambda m: m.update(revocation_policy="shred"), "revocation_policy"),
])
def test_validation_catches(mutate, fragment):
    m = _valid()
    mutate(m)
    assert any(fragment in p for p in subcart.validate_manifest(m))


def test_rule_mode_is_reserved_not_accepted():
    # v2 slot. Accepting it early would silently re-select a user's curated cluster.
    m = _valid()
    m["membership"]["mode"] = "rule"
    assert subcart.validate_manifest(m)


def test_validate_never_raises_on_garbage():
    for junk in (None, [], "manifest", 7, {}):
        assert subcart.validate_manifest(junk)


def test_explicit_mode_requires_keys():
    m = _valid()
    del m["membership"]["keys"]
    assert any("requires membership.keys" in p for p in subcart.validate_manifest(m))


# ------------------------------------------------------------------ resolution

def test_resolve_explicit_maps_keys_to_indices():
    m = subcart.new_manifest("c", "p.pkl",
                             keys=[subcart.content_key(PASSAGES[2]),
                                   subcart.content_key(PASSAGES[0])])
    r = subcart.resolve(m, PASSAGES)
    assert r["indices"] == [2, 0]          # manifest order, not cart order
    assert r["unresolved"] == []


def test_missing_keys_are_reported_not_dropped():
    m = subcart.new_manifest("c", "p.pkl",
                             keys=[subcart.content_key(PASSAGES[0]), "0" * 40])
    r = subcart.resolve(m, PASSAGES)
    assert r["indices"] == [0]
    assert r["unresolved"] == ["0" * 40]
    assert r["stats"]["unresolved"] == 1


def test_revocation_is_just_an_unresolved_key():
    """A revoked passage needs no new mechanism -- it is a dangling reference on purpose."""
    key = subcart.content_key(PASSAGES[1])
    m = subcart.new_manifest("c", "p.pkl", keys=[key])
    assert subcart.resolve(m, PASSAGES)["stats"]["resolved"] == 1
    revoked_parent = [p for i, p in enumerate(PASSAGES) if i != 1]
    r = subcart.resolve(m, revoked_parent)
    assert r["indices"] == [] and r["unresolved"] == [key]


def test_repeated_key_resolves_once_and_is_counted():
    key = subcart.content_key(PASSAGES[0])
    m = subcart.new_manifest("c", "p.pkl", keys=[key, key])
    r = subcart.resolve(m, PASSAGES)
    assert r["indices"] == [0]
    assert r["stats"]["dropped_repeats"] == 1


def test_duplicate_passages_all_resolve():
    texts = ["dup", "x", "dup"]
    m = subcart.new_manifest("c", "p.pkl", keys=[subcart.content_key("dup")])
    r = subcart.resolve(m, texts)
    assert r["indices"] == [0, 2]
    assert r["stats"]["duplicate_keys"] == 1


def test_mode_all_returns_everything_without_a_key_list():
    m = subcart.new_manifest("c", "p.pkl", mode="all", authored_by="access")
    r = subcart.resolve(m, PASSAGES)
    assert r["indices"] == list(range(len(PASSAGES)))
    assert "keys" not in m["membership"]


def test_salted_keys_do_not_resolve_against_an_unsalted_index():
    m = subcart.new_manifest("c", "p.pkl",
                             keys=[subcart.content_key(PASSAGES[0], "pepper")])
    assert subcart.resolve(m, PASSAGES, salt="pepper")["indices"] == [0]
    assert subcart.resolve(m, PASSAGES)["indices"] == []


def test_resolve_refuses_an_invalid_manifest():
    m = _valid()
    m["format"] = "nope"
    with pytest.raises(ValueError):
        subcart.resolve(m, PASSAGES)


# ------------------------------------------------------------------ nesting

def test_chain_intersects_levels():
    root = subcart.new_manifest("root", "p.pkl", mode="all", authored_by="access")
    child = subcart.new_manifest("child", "p.pkl",
                                 keys=[subcart.content_key(PASSAGES[1]),
                                       subcart.content_key(PASSAGES[2])])
    r = subcart.resolve_chain([root, child], PASSAGES)
    assert sorted(r["indices"]) == [1, 2]


def test_chain_cannot_widen_beyond_its_parent():
    """A child manifest must not grant reach its parent does not have."""
    narrow = subcart.new_manifest("narrow", "p.pkl", keys=[subcart.content_key(PASSAGES[0])])
    wide = subcart.new_manifest("wide", "p.pkl", mode="all", authored_by="access")
    assert subcart.resolve_chain([narrow, wide], PASSAGES)["indices"] == [0]


def test_chain_reports_unresolved_per_level():
    root = subcart.new_manifest("root", "p.pkl", mode="all", authored_by="access")
    child = subcart.new_manifest("child", "p.pkl", keys=["f" * 40])
    r = subcart.resolve_chain([root, child], PASSAGES)
    assert r["stats"]["unresolved"] == 1
    assert child["id"] in r["unresolved_by_level"]


# ------------------------------------------------------------------ round trip

def test_save_load_round_trip(tmp_path):
    m = _valid()
    path = tmp_path / "c.subcart.json"
    subcart.save_manifest(m, path)
    assert subcart.load_manifest(path) == m


def test_save_refuses_invalid(tmp_path):
    m = _valid()
    m["parent"]["key_salt"] = "pepper"
    with pytest.raises(ValueError):
        subcart.save_manifest(m, tmp_path / "bad.json")
