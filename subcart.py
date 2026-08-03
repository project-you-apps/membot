"""Sub-cartridge manifests: a named selection of a parent cart, stored BY REFERENCE.

Implements `docs/SPEC-SUBCARTRIDGE-MANIFEST.md` v1, steps 1-2 of §9 (keys + resolver).
No passage text and no embeddings live here. Resolving a manifest yields a set of parent
indices; callers mask the parent's own arrays with them.

WHY CONTENT KEYS AND NOT INDICES
--------------------------------
On 2026-07-29 an index-keyed join applied five thread flags to five unrelated passages --
local and droplet carts both read 38,202 passages, which made them look aligned. A sub-cart
performs that same join on every resolve, so index-keying would make that failure permanent
and silent instead of a one-time incident.

Stdlib only, so both the VPS API and the Membot server can import it without dragging in
either one's dependencies.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FORMAT = "vps-subcart"
VERSION = 1

VALID_MODES = ("explicit", "all")
VALID_AUTHORED_BY = ("user", "access", "query", "import")
VALID_REVOCATION = ("redact", "erase")


# --------------------------------------------------------------------------- keys

def content_key(passage: str, salt: str = "") -> str:
    """Stable identity for a passage across carts, hosts and rebuilds.

    MUST match `memory_server.content_key()` / `build_origin_dates.content_key()` exactly
    when `salt` is empty -- `test_subcart.py` asserts that against the real implementation
    rather than trusting this docstring. Leading SESSION: lines are stripped because /store
    prepends a fresh one on arrival, so a synced passage accumulates a second header.

    SALT IS A SECURITY PROPERTY, NOT A NAMESPACE. Unsalted, this is a plain hash of the body,
    so anyone holding stale keys can confirm whether a document is in a cart by guessing its
    text and hashing it -- and "a file about Client Y exists" can itself be the confidential
    fact. The salt lives in the PARENT CART, never in a manifest: a manifest is precisely the
    artifact a revoked user still holds.

    salt="" reproduces the unsalted digest byte-for-byte, so existing carts and the
    2026-07-30 duplicate_of_key migration keep resolving without a rehash.
    """
    lines = (passage or "").splitlines()
    i = 0
    while i < len(lines) and lines[i].startswith("SESSION:"):
        i += 1
    body = " ".join("\n".join(lines[i:]).split())

    h = hashlib.sha1()
    if salt:
        h.update(salt.encode("utf-8"))
        h.update(b"\x00")          # length-delimit so salt+body can't be re-split
    h.update(body.encode("utf-8"))
    return h.hexdigest()


def salt_id(salt: str) -> str:
    """Public fingerprint OF a salt. Safe to put in a manifest; useless for making keys."""
    return "sha1:" + hashlib.sha1(("salt-id\x00" + (salt or "")).encode("utf-8")).hexdigest()


def build_key_index(texts, salt: str = "") -> dict:
    """content_key -> [parent indices]. A LIST, because carts really do hold duplicates.

    38,633 passages carry 8,582 duplicate marks, so collapsing to one index here would
    silently drop passages a user explicitly selected. Presentation-time dedup is a separate,
    deliberate feature; it must not happen by accident inside a resolver.
    """
    index: dict[str, list] = {}
    for i, t in enumerate(texts):
        index.setdefault(content_key(t, salt), []).append(i)
    return index


# --------------------------------------------------------------------- manifests

def new_manifest(name: str, parent_cart: str, *, title: str = "",
                 mode: str = "explicit", authored_by: str = "user",
                 keys=None, parent_salt: str = "", fingerprint: str = "",
                 passages_at_authoring: int = 0,
                 revocation_policy: str = "redact",
                 writer: dict = None, created_at: float = 0.0) -> dict:
    """Build a manifest dict. `parent_salt` is used ONLY to derive the public salt_id."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "id": "sc_" + hashlib.sha1(
            f"{parent_cart}\x00{name}\x00{created_at}".encode("utf-8")).hexdigest()[:24],
        "name": name,
        "title": title or name,
        "parent": {
            "cart": parent_cart,
            "fingerprint": fingerprint,
            "passages_at_authoring": int(passages_at_authoring),
            "key_salt_id": salt_id(parent_salt),
        },
        "membership": {"mode": mode, "authored_by": authored_by},
        "revocation_policy": revocation_policy,
        "created_at": created_at,
        "event_ts": created_at,
        "writer": writer or {},
    }
    if mode == "explicit":
        manifest["membership"]["keys"] = list(keys or [])
    return manifest


def validate_manifest(manifest: dict) -> list:
    """Return a list of problems. Empty list means valid. Never raises on bad input."""
    problems = []
    if not isinstance(manifest, dict):
        return ["manifest is not an object"]

    if manifest.get("format") != FORMAT:
        problems.append(f"format must be {FORMAT!r}, got {manifest.get('format')!r}")
    if manifest.get("version") != VERSION:
        problems.append(f"unsupported version {manifest.get('version')!r}")
    if not manifest.get("id"):
        problems.append("missing id (references point at id, not name)")
    if not manifest.get("name"):
        problems.append("missing name")

    parent = manifest.get("parent")
    if not isinstance(parent, dict) or not parent.get("cart"):
        problems.append("parent.cart is required")

    if isinstance(parent, dict) and "key_salt" in parent:
        # Hard failure, not a warning: this is the exact leak the salt exists to close.
        problems.append(
            "parent.key_salt must NEVER appear in a manifest -- a revoked holder would "
            "gain the salt and with it the confirmation oracle. Use parent.key_salt_id."
        )

    membership = manifest.get("membership")
    if not isinstance(membership, dict):
        problems.append("membership is required")
        return problems

    mode = membership.get("mode")
    if mode not in VALID_MODES:
        problems.append(f"membership.mode must be one of {VALID_MODES}, got {mode!r}")
    elif mode == "explicit" and not isinstance(membership.get("keys"), list):
        problems.append("membership.mode 'explicit' requires membership.keys as a list")

    authored_by = membership.get("authored_by")
    if authored_by is not None and authored_by not in VALID_AUTHORED_BY:
        problems.append(f"membership.authored_by {authored_by!r} not in {VALID_AUTHORED_BY}")

    policy = manifest.get("revocation_policy", "redact")
    if policy not in VALID_REVOCATION:
        problems.append(f"revocation_policy must be one of {VALID_REVOCATION}, got {policy!r}")

    return problems


def load_manifest(path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError(f"invalid manifest {path}: " + "; ".join(problems))
    return manifest


def save_manifest(manifest: dict, path) -> None:
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError("refusing to write invalid manifest: " + "; ".join(problems))
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# --------------------------------------------------------------------- resolution

def resolve(manifest: dict, texts, salt: str = "", key_index: dict = None) -> dict:
    """Map a manifest onto a parent's passages.

    Returns {indices, unresolved, stats}. `indices` is ordered by the manifest's key order
    (order is meaningful for a curated cluster, meaningless for an access view -- consumers
    that do not care ignore it).

    UNRESOLVED KEYS ARE RETURNED, NEVER DROPPED. A key with no passage behind it means the
    passage was tombstoned, revoked, or not yet synced. Dropping it silently shrinks a user's
    curated set with no signal, and a tombstoned-then-restored passage would leave the cluster
    permanently. Revocation is exactly this case induced on purpose, which is why it needs no
    separate mechanism.

    Callers MUST surface a non-zero `unresolved` count somewhere. This is the consumer being
    named before the flag exists -- the inverse of the 8,592 duplicate marks that ranked the
    Hot Stack and drove nothing.
    """
    problems = validate_manifest(manifest)
    if problems:
        raise ValueError("cannot resolve invalid manifest: " + "; ".join(problems))

    membership = manifest["membership"]
    mode = membership["mode"]

    if mode == "all":
        n = len(texts)
        return {
            "indices": list(range(n)),
            "unresolved": [],
            "stats": {"mode": "all", "requested": n, "resolved": n,
                      "unresolved": 0, "duplicate_keys": 0, "dropped_repeats": 0},
        }

    if key_index is None:
        key_index = build_key_index(texts, salt)

    indices: list = []
    unresolved: list = []
    seen_keys = set()
    duplicate_keys = 0
    dropped_repeats = 0

    for key in membership.get("keys", []):
        if key in seen_keys:
            dropped_repeats += 1        # same key listed twice; resolve once, keep order
            continue
        seen_keys.add(key)

        hits = key_index.get(key)
        if not hits:
            unresolved.append(key)
            continue
        if len(hits) > 1:
            duplicate_keys += 1
        indices.extend(hits)

    return {
        "indices": indices,
        "unresolved": unresolved,
        "stats": {
            "mode": "explicit",
            "requested": len(membership.get("keys", [])),
            "resolved": len(indices),
            "unresolved": len(unresolved),
            "duplicate_keys": duplicate_keys,
            "dropped_repeats": dropped_repeats,
        },
    }


def resolve_chain(manifests, texts, salt: str = "") -> dict:
    """Nested sub-carts: resolve EAGERLY AGAINST THE ROOT, never walk the chain.

    Users author their own carts, so depth cannot meaningfully be capped. It costs nothing --
    a mask of a mask is just a smaller mask. `manifests` runs root-most first; each level
    intersects the previous while preserving the deeper level's ordering, and every level's
    unresolved keys are reported under its own id.
    """
    if not manifests:
        raise ValueError("resolve_chain requires at least one manifest")

    key_index = build_key_index(texts, salt)
    allowed = None
    unresolved_by_level: dict = {}
    last: dict = {}

    for manifest in manifests:
        last = resolve(manifest, texts, salt, key_index)
        if last["unresolved"]:
            unresolved_by_level[manifest.get("id", manifest.get("name", "?"))] = \
                last["unresolved"]
        current = last["indices"]
        allowed = current if allowed is None else [i for i in current if i in set(allowed)]

    return {
        "indices": allowed or [],
        "unresolved_by_level": unresolved_by_level,
        "stats": {"levels": len(manifests), "resolved": len(allowed or []),
                  "unresolved": sum(len(v) for v in unresolved_by_level.values())},
    }
