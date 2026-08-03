"""Turn a by-reference sub-cartridge into a standalone cart.

Step 6 of `docs/SPEC-SUBCARTRIDGE-MANIFEST.md`.

WHY THIS EXISTS
---------------
A sub-cartridge is a parent pointer plus content keys. That is exactly what makes it cheap,
and exactly what makes it useless the moment the parent is unreachable -- which fights
local-first ("customer stacks need NO server"). Materializing resolves the references into a
real cart with the passages and embeddings copied in.

**Export must ALWAYS materialize.** Handing someone a manifest that points at a cart they
cannot read is a broken artifact that looks like a working one. This is also the operation the
marketplace idea needs: sell a slice, ship it standalone.

THE TRAP: LINKS ARE INDEX-BASED
-------------------------------
`prev`, `next`, `parent`, `related` and `duplicate_of` are POSITIONS, not identities. A
materialized cart renumbers everything, so copying links verbatim would leave every one of
them pointing at whatever passage happens to sit at the old offset in the new, smaller cart.

That is not a hypothetical: on 2026-07-29 an index-keyed join applied five thread flags to
five unrelated passages, and both carts reading 38,202 made them look aligned. Materialize
walks the same knife edge, so it REMAPS every index-bearing field and severs the ones whose
target was not included -- and reports how many it severed rather than doing it quietly.

numpy is required here (embeddings), unlike `subcart.py`, which stays stdlib-only.
"""

from __future__ import annotations

import time

import numpy as np

import subcart as _sc

# Link fields holding a single passage index, and those holding a list of them.
_INDEX_FIELDS = ("prev", "next", "parent")
_INDEX_LIST_FIELDS = ("related",)


def _remap_links(links, indices, report):
    """Rebuild links for the materialized subset, renumbering every index-bearing field.

    `indices` is the parent-order list of included passages; position in that list is the new
    index. A reference whose target was not included becomes None (or is dropped from a list)
    and is COUNTED -- a severed link is information, and silently keeping a stale integer is
    how a passage ends up pointing at a stranger.
    """
    old_to_new = {old: new for new, old in enumerate(indices)}
    out = []
    for old in indices:
        link = links[old] if old < len(links) and isinstance(links[old], dict) else {}
        new_link = dict(link)

        for field in _INDEX_FIELDS:
            value = new_link.get(field)
            if isinstance(value, int):
                mapped = old_to_new.get(value)
                if mapped is None:
                    report["severed"] += 1
                new_link[field] = mapped

        for field in _INDEX_LIST_FIELDS:
            value = new_link.get(field)
            if isinstance(value, list):
                kept = [old_to_new[v] for v in value
                        if isinstance(v, int) and v in old_to_new]
                report["severed"] += sum(1 for v in value
                                         if isinstance(v, int) and v not in old_to_new)
                new_link[field] = kept

        # `duplicate_of` lives inside _reserved.canonical and is ALSO an index. Its
        # content-keyed twin `duplicate_of_key` survives untouched, which is precisely why
        # that migration was worth doing.
        reserved = new_link.get("_reserved")
        if isinstance(reserved, dict):
            reserved = dict(reserved)
            canonical = dict(reserved.get("canonical") or {})
            dup = canonical.get("duplicate_of")
            if isinstance(dup, int):
                mapped = old_to_new.get(dup)
                if mapped is None:
                    report["severed"] += 1
                canonical["duplicate_of"] = mapped
            reserved["canonical"] = canonical
            new_link["_reserved"] = reserved

        out.append(new_link)
    return out


def materialize(manifest: dict, parent: dict, salt: str = "",
                key_index: dict = None, now: float = 0.0) -> dict:
    """Resolve a manifest against a loaded cart dict and return a standalone cart dict.

    `parent` is the cart as loaded from a PKL: {passages, embeddings, links, metadata, ...}.
    Returns a cart of the same shape containing only the manifest's passages, with embeddings
    sliced to match and links renumbered.

    The result carries a `materialized_from` block so a cart that came from a slice can always
    say so. An artifact that cannot explain its own provenance is one nobody can audit later.
    """
    now = now or time.time()
    passages = parent.get("passages") or parent.get("texts") or []
    resolved = _sc.resolve(manifest, passages, salt, key_index)
    indices = resolved["indices"]

    report = {"severed": 0}
    links = parent.get("links") or []
    new_links = _remap_links(links, indices, report) if links else []

    embeddings = parent.get("embeddings")
    new_embeddings = None
    if embeddings is not None and len(embeddings):
        # Row order must track passage order exactly; anything else silently mismatches
        # every vector with the wrong text.
        new_embeddings = np.asarray(embeddings)[indices]

    out = {
        "version": parent.get("version", 1),
        "passages": [passages[i] for i in indices],
        "links": new_links,
        "metadata": dict(parent.get("metadata") or {}),
        "materialized_from": {
            "subcart_id": manifest.get("id"),
            "subcart_name": manifest.get("name"),
            "parent_cart": (manifest.get("parent") or {}).get("cart"),
            "parent_passages": len(passages),
            "materialized_at": now,
            "unresolved": len(resolved["unresolved"]),
            "links_severed": report["severed"],
        },
    }
    if new_embeddings is not None:
        out["embeddings"] = new_embeddings

    compressed = parent.get("compressed_lens")
    if compressed is not None and len(compressed) == len(passages):
        out["compressed_lens"] = [compressed[i] for i in indices]

    out["stats"] = {
        "passages": len(out["passages"]),
        "requested": resolved["stats"]["requested"],
        "unresolved": len(resolved["unresolved"]),
        "links_severed": report["severed"],
    }
    return out


def verify(original: dict, materialized: dict, manifest: dict, salt: str = "") -> list:
    """Problems with a materialized cart. Empty list means it holds together.

    Checked rather than assumed, because every failure mode here is silent: an embedding
    matrix one row short, a link pointing into the void, a passage that is not what its key
    says it is. None of those raise; they just quietly return the wrong answer forever.
    """
    problems = []
    passages = materialized.get("passages") or []
    embeddings = materialized.get("embeddings")
    links = materialized.get("links") or []

    if embeddings is not None and len(embeddings) != len(passages):
        problems.append(f"embeddings {len(embeddings)} != passages {len(passages)}")
    if links and len(links) != len(passages):
        problems.append(f"links {len(links)} != passages {len(passages)}")

    n = len(passages)
    for i, link in enumerate(links):
        for field in _INDEX_FIELDS:
            value = link.get(field)
            if value is not None and not (isinstance(value, int) and 0 <= value < n):
                problems.append(f"link[{i}].{field} = {value!r} out of range 0..{n-1}")
        for value in (link.get("related") or []):
            if not (isinstance(value, int) and 0 <= value < n):
                problems.append(f"link[{i}].related contains {value!r} out of range")

    # Every materialized passage must still hash to a key the manifest asked for.
    membership = manifest.get("membership") or {}
    if membership.get("mode") == "explicit":
        wanted = set(membership.get("keys") or [])
        for i, text in enumerate(passages):
            if _sc.content_key(text, salt) not in wanted:
                problems.append(f"passage {i} is not in the manifest's key set")
                break

    return problems
