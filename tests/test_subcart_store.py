"""Sub-cartridge storage: create, list, get, rename, delete.

    PYTHONIOENCODING=utf-8 python -m pytest membot/tests/test_subcart_store.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import subcart as sc                      # noqa: E402
from subcart_store import SubcartStore, SUFFIX, TRASH_DIRNAME  # noqa: E402


def store(tmp_path):
    cart = tmp_path / "brain.pkl"
    cart.write_bytes(b"stub")
    return SubcartStore(cart)


def manifest(name="notes", keys=("a" * 40,), created_at=1.0):
    return sc.new_manifest(name, "brain.pkl", keys=list(keys), created_at=created_at)


# ------------------------------------------------------------------ location

def test_manifests_live_beside_the_cart(tmp_path):
    s = store(tmp_path)
    assert s.dir == tmp_path / "subcarts"


def test_accepts_a_directory_as_well_as_a_cart_file(tmp_path):
    """Callers have both, and guessing wrong yields a silently empty listing."""
    assert SubcartStore(tmp_path).dir == tmp_path / "subcarts"


def test_listing_an_absent_directory_is_empty_not_an_error(tmp_path):
    assert store(tmp_path).list() == []


# ------------------------------------------------------------------ create

def test_create_then_list(tmp_path):
    s = store(tmp_path)
    s.create(manifest("notes"))
    rows = s.list()
    assert len(rows) == 1
    assert rows[0]["name"] == "notes" and rows[0]["keys"] == 1
    assert rows[0]["problems"] == []


def test_file_is_named_by_id_not_by_name(tmp_path):
    """The name is mutable, so it can never be identity."""
    s = store(tmp_path)
    m = s.create(manifest("my notes"))
    assert (s.dir / f"{m['id']}{SUFFIX}").exists()


def test_duplicate_name_is_refused_with_a_reason(tmp_path):
    s = store(tmp_path)
    s.create(manifest("notes", created_at=1.0))
    with pytest.raises(ValueError, match="already exists"):
        s.create(manifest("notes", created_at=2.0))


def test_duplicate_name_refusal_beats_silent_suffixing(tmp_path):
    """Auto-renaming would leave a user with `notes` and `notes-2` and no idea which
    one they just built."""
    s = store(tmp_path)
    s.create(manifest("notes", created_at=1.0))
    try:
        s.create(manifest("notes", created_at=2.0))
    except ValueError:
        pass
    assert [r["name"] for r in s.list()] == ["notes"]


def test_invalid_manifest_is_refused(tmp_path):
    s = store(tmp_path)
    m = manifest()
    m["format"] = "nope"
    with pytest.raises(ValueError, match="invalid manifest"):
        s.create(m)


def test_a_manifest_carrying_the_salt_cannot_be_created(tmp_path):
    """The salt must never reach an artifact a revoked holder keeps."""
    s = store(tmp_path)
    m = manifest()
    m["parent"]["key_salt"] = "pepper"
    with pytest.raises(ValueError):
        s.create(m)


# ------------------------------------------------------------------ get

def test_get_by_id_and_by_name(tmp_path):
    s = store(tmp_path)
    m = s.create(manifest("notes"))
    assert s.get(m["id"])["name"] == "notes"
    assert s.get("notes")["id"] == m["id"]


def test_get_missing_raises(tmp_path):
    with pytest.raises(KeyError):
        store(tmp_path).get("nope")


# ------------------------------------------------------------------ corruption

def test_unreadable_manifest_is_reported_not_skipped(tmp_path):
    """A corrupted sub-cart must not look identical to one that never existed."""
    s = store(tmp_path)
    s.create(manifest("good"))
    s.dir.mkdir(parents=True, exist_ok=True)
    (s.dir / f"broken{SUFFIX}").write_text("{ not json", encoding="utf-8")
    rows = s.list()
    assert len(rows) == 2
    broken = [r for r in rows if r["file"].startswith("broken")][0]
    assert broken["problems"] and "unreadable" in broken["problems"][0]


def test_schema_problems_are_surfaced_in_the_listing(tmp_path):
    s = store(tmp_path)
    s.create(manifest("good"))
    bad = manifest("bad", created_at=2.0)
    bad["membership"]["mode"] = "rule"        # reserved for v2, not accepted
    (s.dir / f"bad{SUFFIX}").write_text(json.dumps(bad), encoding="utf-8")
    rows = [r for r in s.list() if r["file"].startswith("bad")]
    assert rows[0]["problems"]


# ------------------------------------------------------------------ rename

def test_rename_keeps_the_id_and_the_file(tmp_path):
    s = store(tmp_path)
    m = s.create(manifest("old"))
    renamed = s.rename("old", "new")
    assert renamed["id"] == m["id"]
    assert (s.dir / f"{m['id']}{SUFFIX}").exists()
    assert s.get("new")["id"] == m["id"]


def test_rename_onto_a_taken_name_is_refused(tmp_path):
    s = store(tmp_path)
    s.create(manifest("one", created_at=1.0))
    s.create(manifest("two", created_at=2.0))
    with pytest.raises(ValueError, match="already exists"):
        s.rename("one", "two")


def test_rename_to_its_own_name_is_allowed(tmp_path):
    s = store(tmp_path)
    s.create(manifest("same"))
    assert s.rename("same", "same")["name"] == "same"


# ------------------------------------------------------------------ delete

def test_delete_moves_to_trash_and_never_unlinks(tmp_path):
    """Deleting the JSON destroys a hand-made SELECTION -- recovery should be a file
    move, not a re-curation."""
    s = store(tmp_path)
    m = s.create(manifest("notes"))
    out = s.delete("notes")
    assert not (s.dir / f"{m['id']}{SUFFIX}").exists()
    assert s.list() == []
    trashed = list((s.dir / TRASH_DIRNAME).glob(f"*{SUFFIX}"))
    assert len(trashed) == 1 and Path(out["moved_to"]).exists()


def test_trashed_manifest_is_still_valid_and_recoverable(tmp_path):
    s = store(tmp_path)
    s.create(manifest("notes"))
    out = s.delete("notes")
    recovered = json.loads(Path(out["moved_to"]).read_text(encoding="utf-8"))
    assert sc.validate_manifest(recovered) == []
    assert recovered["name"] == "notes"


def test_trash_does_not_appear_in_listings(tmp_path):
    s = store(tmp_path)
    s.create(manifest("notes"))
    s.delete("notes")
    s.create(manifest("notes", created_at=9.0))       # name is free again
    assert [r["name"] for r in s.list()] == ["notes"]


def test_deleting_twice_raises_rather_than_silently_succeeding(tmp_path):
    s = store(tmp_path)
    s.create(manifest("notes"))
    s.delete("notes")
    with pytest.raises(KeyError):
        s.delete("notes")


def test_delete_leaves_the_parent_cart_alone(tmp_path):
    s = store(tmp_path)
    s.create(manifest("notes"))
    s.delete("notes")
    assert (tmp_path / "brain.pkl").read_bytes() == b"stub"


# ------------------------------------------------------------------ round trip

def test_created_manifest_resolves_against_a_parent(tmp_path):
    """End to end: the thing we store is the thing the resolver consumes."""
    texts = ["alpha body", "beta body"]
    s = store(tmp_path)
    s.create(sc.new_manifest("alpha-only", "brain.pkl",
                             keys=[sc.content_key(texts[0])], created_at=1.0))
    resolved = sc.resolve(s.get("alpha-only"), texts)
    assert resolved["indices"] == [0] and resolved["unresolved"] == []
