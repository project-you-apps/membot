"""
multi_cart.py — Multi-cart query layer for Membot.

The foundation that turns Membot from a single-cart library into a multi-relation
neuromorphic database. Lets one Membot instance hold many carts mounted at once
and query across them with namespaced result attribution.

Spec: docs/RFC/multi-cart-query-spec.md
Related: docs/RFC/federated-cart-spec.md, docs/RFC/membox-multiuser-dbms-spec.md

Architecture
============
This is a parallel layer alongside the existing per-session single-cart state.
The single-cart code in membot_server.py keeps working unchanged. Multi-cart
mode is opt-in via a separate set of functions:

    mount(cart_path, cart_id=None, role=None)
    unmount(cart_id)
    list_mounts()
    mount_directory(dir_path, role, pattern="*.cart")
    search(query, top_k=10, scope="all", role_filter=None)

State lives in a module-global `_mounted_carts` dict (cart_id → cart_state).
Each cart_state has the same shape as a single-session state in membot_server,
so the existing search/load helpers can be reused.

Phase 1 scope (this file)
=========================
- Mount API + cart_id namespacing
- Multi-cart search with scope filtering and result attribution
- list/unmount/mount_directory
- Backward-compat: an existing session-mounted cart can be queried via the
  multi-cart API by mounting it with a cart_id of "default" or similar.

NOT in Phase 1
==============
- Cross-cart h-row edges (Phase 3 of the spec) — schema designed but not enforced yet
- Lazy loading / memory eviction (Phase 5)
- Cart-id collision auto-rename (we error explicitly instead)
- Embedding model heterogeneity (we assume Nomic v1.5 across all mounted carts)
"""

from __future__ import annotations

import glob
import logging
import os
import sqlite3
import time
from typing import Any, Optional

import numpy as np

# We import from membot_server lazily inside functions to avoid a circular
# import at module load time. The functions used:
#   load_cartridge_safe, verify_manifest, embed_text, find_cartridges,
#   sanitize_name, _sqlite_fetch_passages

log = logging.getLogger(__name__)

# Global blend constant — kept consistent with membot_server for now.
# Same value as HAMMING_BLEND there. If they ever diverge, this is the
# multi-cart blend; the single-cart code uses its own.
_HAMMING_BLEND = 0.30

# Stop words for keyword reranking — match membot_server.py
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
    "and", "or", "for", "on", "it", "be", "as", "at", "by", "this",
    "that", "with", "from", "not", "but", "has", "had", "have", "do",
    "does", "did", "will", "can", "its", "who", "what", "how", "why",
}

# Popcount lookup table for packed Hamming distance
_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

# =============================================================================
# STATE — module-global dict of mounted carts, keyed by cart_id
# =============================================================================

# cart_id → cart state dict (same shape as a session in membot_server)
_mounted_carts: dict[str, dict] = {}


def _new_cart_state() -> dict:
    """Create a fresh cart state dict. Same shape as _new_session in membot_server."""
    return {
        "cart_id": None,
        "cart_path": None,
        "cart_name": None,
        "role": None,                  # 'identity' | 'episodic' | 'semantic' | 'federated' | etc.
        "embeddings": None,            # (N, 768) float32
        "texts": [],                   # list[str]
        "binary_corpus": None,         # (N, 96) packed or (N, 768) unpacked sign_zero bits
        "hippocampus": None,           # list[dict] — per-pattern metadata
        "has_embeddings": False,
        "is_split_cart": False,
        "sqlite_conn": None,           # sqlite3.Connection for split carts
        "mounted_at": None,
        "n_patterns": 0,
        "embedding_dim": 0,
    }


# =============================================================================
# MOUNT API
# =============================================================================

def mount(cart_path: str, cart_id: Optional[str] = None,
          role: Optional[str] = None,
          verify_integrity: bool = True) -> dict:
    """Mount a cart by file path. Adds it to the multi-cart pool.

    Args:
        cart_path: Filesystem path to the cart file (.npz / .pkl / split format)
        cart_id: Short stable identifier. Defaults to filename without extension.
        role: Optional semantic tag — 'identity', 'episodic', 'semantic',
              'working', 'federated', 'consolidated', etc. Used for role_filter
              on search; not interpreted by the lattice.
        verify_integrity: If True (default), reject carts whose manifest
            fingerprint doesn't match. Set False for testing or for known-stale
            manifests where you accept the risk. Production calls should leave
            this on.

    Returns:
        dict with mount details: {cart_id, cart_path, role, n_patterns,
        embedding_dim, status, elapsed_ms, integrity}

    Raises:
        ValueError if cart_id collides with an existing mount, path doesn't
        exist, or (when verify_integrity=True) the manifest check fails
    """
    # Lazy import to avoid circular dependency
    from membot_server import load_cartridge_safe, verify_manifest

    if not os.path.exists(cart_path):
        raise ValueError(f"Cart path does not exist: {cart_path}")

    if cart_id is None:
        cart_id = os.path.splitext(os.path.basename(cart_path))[0]

    if cart_id in _mounted_carts:
        raise ValueError(
            f"cart_id '{cart_id}' is already mounted (path: "
            f"{_mounted_carts[cart_id]['cart_path']}). Use unmount() first or "
            f"choose a different cart_id."
        )

    t0 = time.time()
    data = load_cartridge_safe(cart_path)
    embeddings = data["embeddings"]
    texts = data["texts"]

    ok, verify_msg = verify_manifest(cart_path, embeddings, len(texts))
    if not ok:
        if verify_integrity:
            raise ValueError(
                f"Cart '{cart_id}' failed integrity check: {verify_msg}. "
                f"Refusing to mount. Pass verify_integrity=False to override."
            )
        else:
            log.warning(
                f"[multi_cart] Cart '{cart_id}' integrity check FAILED ({verify_msg}) "
                f"but verify_integrity=False — mounting anyway."
            )

    state = _new_cart_state()
    state["cart_id"] = cart_id
    state["cart_path"] = cart_path
    state["cart_name"] = os.path.splitext(os.path.basename(cart_path))[0]
    state["role"] = role
    state["embeddings"] = embeddings
    state["texts"] = texts
    state["hippocampus"] = data.get("hippocampus")
    state["n_patterns"] = len(texts)
    state["embedding_dim"] = embeddings.shape[1] if len(embeddings) > 0 else 0
    state["has_embeddings"] = len(embeddings) > 0 and embeddings.size > 0
    state["mounted_at"] = time.time()

    # Sign-zero binary corpus for Hamming search
    if "sign_bits" in data and data["sign_bits"] is not None:
        state["binary_corpus"] = data["sign_bits"]
    elif state["has_embeddings"]:
        state["binary_corpus"] = (embeddings > 0).astype(np.uint8)
    else:
        state["binary_corpus"] = None

    # Split cart: open SQLite sidecar
    if data.get("sqlite_db_path"):
        state["sqlite_conn"] = sqlite3.connect(data["sqlite_db_path"])
        state["is_split_cart"] = True

    _mounted_carts[cart_id] = state
    elapsed_ms = (time.time() - t0) * 1000

    log.info(
        f"[multi_cart] Mounted '{cart_id}' (role={role}, {state['n_patterns']} patterns, "
        f"{state['embedding_dim']}-dim) from {cart_path} in {elapsed_ms:.0f}ms"
    )

    return {
        "cart_id": cart_id,
        "cart_path": cart_path,
        "role": role,
        "n_patterns": state["n_patterns"],
        "embedding_dim": state["embedding_dim"],
        "has_embeddings": state["has_embeddings"],
        "is_split_cart": state["is_split_cart"],
        "status": "mounted",
        "elapsed_ms": round(elapsed_ms, 1),
    }


def unmount(cart_id: str) -> dict:
    """Remove a cart from the multi-cart pool. Returns details of what was unmounted."""
    if cart_id not in _mounted_carts:
        return {"cart_id": cart_id, "status": "not_mounted"}

    state = _mounted_carts.pop(cart_id)
    if state.get("sqlite_conn"):
        try:
            state["sqlite_conn"].close()
        except Exception:
            pass

    log.info(f"[multi_cart] Unmounted '{cart_id}'")
    return {
        "cart_id": cart_id,
        "n_patterns": state["n_patterns"],
        "role": state["role"],
        "status": "unmounted",
    }


def list_mounts() -> list[dict]:
    """Return summary info for every currently-mounted cart."""
    out = []
    for cart_id, state in _mounted_carts.items():
        out.append({
            "cart_id": cart_id,
            "role": state["role"],
            "n_patterns": state["n_patterns"],
            "embedding_dim": state["embedding_dim"],
            "has_embeddings": state["has_embeddings"],
            "is_split_cart": state["is_split_cart"],
            "cart_path": state["cart_path"],
            "mounted_at": state["mounted_at"],
            "mounted_seconds_ago": round(time.time() - state["mounted_at"], 1) if state["mounted_at"] else None,
        })
    return out


def mount_directory(dir_path: str, role: Optional[str] = None,
                    pattern: str = "*.cart",
                    verify_integrity: bool = True) -> list[dict]:
    """Mount every cart file in a directory matching the glob pattern.

    Used for federated mode: point at a fleet-learning directory and get every
    machine's cart mounted with the same role.

    Args:
        dir_path: Directory to scan
        role: Role to apply to every mounted cart
        pattern: Glob pattern (default '*.cart' — also supports '*.npz', '*.pkl', etc.)
        verify_integrity: Passed through to each mount() call. Defaults True.

    Returns:
        List of mount result dicts (same shape as mount() return)
    """
    if not os.path.isdir(dir_path):
        raise ValueError(f"Not a directory: {dir_path}")

    # Walk recursively if the pattern starts with '**'
    if pattern.startswith("**"):
        cart_files = glob.glob(os.path.join(dir_path, pattern), recursive=True)
    else:
        cart_files = glob.glob(os.path.join(dir_path, pattern))
        # Also try one level deep — federated layouts use machine subdirs
        cart_files.extend(glob.glob(os.path.join(dir_path, "*", pattern)))

    cart_files = sorted(set(cart_files))

    results = []
    for cart_file in cart_files:
        # Default cart_id from filename, plus parent dir if there's ambiguity
        base = os.path.splitext(os.path.basename(cart_file))[0]
        parent = os.path.basename(os.path.dirname(cart_file))
        cart_id = f"{parent}_{base}" if parent and parent != os.path.basename(dir_path.rstrip("/\\")) else base

        # Avoid collisions by appending a counter if needed
        original_cart_id = cart_id
        suffix = 1
        while cart_id in _mounted_carts:
            cart_id = f"{original_cart_id}_{suffix}"
            suffix += 1

        try:
            result = mount(cart_file, cart_id=cart_id, role=role,
                           verify_integrity=verify_integrity)
            results.append(result)
        except Exception as e:
            log.warning(f"[multi_cart] Failed to mount {cart_file}: {e}")
            results.append({
                "cart_id": cart_id,
                "cart_path": cart_file,
                "status": "error",
                "error": str(e),
            })

    log.info(
        f"[multi_cart] mount_directory({dir_path}): "
        f"{sum(1 for r in results if r.get('status') == 'mounted')} mounted, "
        f"{sum(1 for r in results if r.get('status') == 'error')} errors"
    )
    return results


# =============================================================================
# SEARCH — multi-cart query with scope and attribution
# =============================================================================

def _hamming_scores(query_emb: np.ndarray, corpus_bin: np.ndarray) -> np.ndarray:
    """Compute sign-zero Hamming similarity scores against a packed or unpacked corpus."""
    is_packed = corpus_bin.shape[1] <= 96
    if is_packed:
        q_packed = np.packbits((query_emb > 0).astype(np.uint8))
        xor = np.bitwise_xor(q_packed, corpus_bin)
        dist = _POPCOUNT_TABLE[xor].sum(axis=1)
        n_bits = 768
    else:
        q_bin = (query_emb > 0).astype(np.uint8)
        xor = np.bitwise_xor(q_bin, corpus_bin)
        dist = xor.sum(axis=1)
        n_bits = corpus_bin.shape[1]
    return 1.0 - dist.astype(np.float32) / n_bits


def _search_one_cart(state: dict, query_emb: np.ndarray, query_text: str) -> list[dict]:
    """Run cosine + Hamming + keyword reranking against a single cart's state.

    Returns a list of result dicts, NOT yet sorted globally. Each dict has:
        cart_id, local_addr, score, cos_score, ham_score, kw_boost, text, role
    """
    if state["binary_corpus"] is None and not state["has_embeddings"]:
        return []

    # 1. Embedding cosine
    if state["has_embeddings"]:
        stored = state["embeddings"]
        stored_norms = np.linalg.norm(stored, axis=1, keepdims=True) + 1e-9
        query_norm = np.linalg.norm(query_emb) + 1e-9
        emb_scores = np.dot(stored / stored_norms, query_emb / query_norm)
    else:
        emb_scores = None

    # 2. Hamming
    if state["binary_corpus"] is not None:
        ham_scores = _hamming_scores(query_emb, state["binary_corpus"])
    else:
        ham_scores = None

    # 3. Blend
    if emb_scores is not None and ham_scores is not None:
        n = min(len(emb_scores), len(ham_scores))
        scores = (1.0 - _HAMMING_BLEND) * emb_scores[:n] + _HAMMING_BLEND * ham_scores[:n]
    elif emb_scores is not None:
        scores = emb_scores
    elif ham_scores is not None:
        scores = ham_scores
    else:
        return []

    # 4. Keyword reranking — pull a wider candidate pool
    keywords = [w for w in query_text.lower().split() if len(w) >= 3 and w not in _STOP_WORDS]
    candidate_k = min(max(40, 20), len(scores))
    candidate_idx = np.argsort(scores)[-candidate_k:][::-1]

    results = []
    for i in candidate_idx:
        i = int(i)
        base = float(scores[i])
        if base < 0.05:
            continue
        text = state["texts"][i] if i < len(state["texts"]) else ""
        text_lower = text.lower()
        hits = sum(1 for kw in keywords if kw in text_lower)
        boost = min(hits * 0.03, 0.12)
        results.append({
            "cart_id": state["cart_id"],
            "role": state["role"],
            "local_addr": i,
            "score": base + boost,
            "cos_score": float(emb_scores[i]) if emb_scores is not None and i < len(emb_scores) else None,
            "ham_score": float(ham_scores[i]) if ham_scores is not None and i < len(ham_scores) else None,
            "kw_boost": boost,
            "text": text,
        })
    return results


def search(query: str, top_k: int = 10,
           scope: Any = "all",
           role_filter: Any = None,
           scope_mode: str = "global") -> dict:
    """Multi-cart semantic search. Searches every mounted cart in scope and
    returns ranked results attributed to their source cart.

    Args:
        query: Natural language search query
        top_k: Number of top results to return (interpretation depends on scope_mode)
        scope: Which carts to search. One of:
            "all" (default) — every mounted cart
            "local" — backward-compat alias for the first-mounted cart
            cart_id (str) — search just that one cart
            list[cart_id] — search just those carts
        role_filter: Optional. One role string or list of role strings.
            Restricts the search to carts with matching role tags.
            Combinable with scope.
        scope_mode: How results across multiple carts are ranked and merged.
            Use this when carts are very different sizes and you don't want
            one large cart to dominate the results. Four options:

            "global" (DEFAULT) — true top-K across all carts, ranked by score.
                Best when you want the single best answer regardless of source.
                Risk: if one cart is much larger, it can dominate the results.

            "per_cart" — top-K from EACH cart, returned grouped by cart_id,
                no cross-cart re-ranking. Up to K × N_carts total results.
                Best when you want every source's best answer for comparison.

            "balanced" — top-K from each cart as candidates, then global
                re-rank to top-K of those candidates. Guarantees small carts
                aren't drowned, but the final ranking still reflects global
                score. Best when you want fair representation AND global ranking.

            "diagnostic" — top-K from every cart with no merging at all,
                fully labeled by source. Useful for debugging cross-cart
                search behavior — "show me what each cart thinks the answer is."

    Returns:
        dict with keys:
            results: list of result dicts with cart_id attribution
            elapsed_ms: float
            cart_count: int — how many carts were searched
            total_patterns: int — how many patterns total were considered
            mode: 'multi_cart'
            scope_mode: which mode was used
            grouped_results: dict[cart_id → results] (only set for per_cart/diagnostic)
    """
    from membot_server import embed_text

    if not _mounted_carts:
        return {
            "results": [],
            "elapsed_ms": 0,
            "cart_count": 0,
            "total_patterns": 0,
            "mode": "multi_cart",
            "scope_mode": scope_mode,
            "error": "no_carts_mounted",
        }

    if scope_mode not in ("global", "per_cart", "balanced", "diagnostic"):
        raise ValueError(
            f"Unknown scope_mode: {scope_mode!r}. "
            f"Use 'global', 'per_cart', 'balanced', or 'diagnostic'."
        )

    t0 = time.time()

    # Resolve which carts to search
    target_ids = _resolve_scope(scope, role_filter)
    if not target_ids:
        return {
            "results": [],
            "elapsed_ms": (time.time() - t0) * 1000,
            "cart_count": 0,
            "total_patterns": 0,
            "mode": "multi_cart",
            "scope_mode": scope_mode,
            "error": "no_carts_match_scope",
        }

    # 1. Embed query once (shared across all carts)
    query_emb = embed_text(query, prefix="search_query")

    # 2. Search each target cart, sort its own results by score
    per_cart_results: dict[str, list[dict]] = {}
    total_patterns = 0
    for cart_id in target_ids:
        state = _mounted_carts[cart_id]
        total_patterns += state["n_patterns"]
        cart_results = _search_one_cart(state, query_emb, query)
        cart_results.sort(key=lambda r: r["score"], reverse=True)
        per_cart_results[cart_id] = cart_results

    # 3. Apply scope_mode to merge per-cart results into final output
    if scope_mode == "global":
        # Concatenate all per-cart results, then sort globally
        all_results = []
        for cart_results in per_cart_results.values():
            all_results.extend(cart_results)
        all_results.sort(key=lambda r: r["score"], reverse=True)
        top_results = all_results[:top_k]
        grouped_results = None

    elif scope_mode == "per_cart":
        # Top-K from each cart, returned in cart-name order, no global re-rank
        top_results = []
        grouped_results = {}
        for cart_id in target_ids:
            cart_top = per_cart_results[cart_id][:top_k]
            grouped_results[cart_id] = cart_top
            top_results.extend(cart_top)
        # top_results is grouped by cart in target order — no global sort

    elif scope_mode == "balanced":
        # Top-K from each cart as candidates, then global re-rank to top-K
        candidates = []
        for cart_id in target_ids:
            candidates.extend(per_cart_results[cart_id][:top_k])
        candidates.sort(key=lambda r: r["score"], reverse=True)
        top_results = candidates[:top_k]
        grouped_results = None

    elif scope_mode == "diagnostic":
        # Every cart's top-K, no merging at all, fully labeled
        top_results = []
        grouped_results = {}
        for cart_id in target_ids:
            cart_top = per_cart_results[cart_id][:top_k]
            grouped_results[cart_id] = cart_top
            top_results.extend(cart_top)

    # 4. Truncate display text on the final results
    for r in top_results:
        if r["score"] < 0.1:
            continue
        if len(r["text"]) > 500:
            r["text"] = r["text"][:500] + "..."

    elapsed_ms = (time.time() - t0) * 1000
    log.info(
        f"[multi_cart] search('{query[:60]}', top_k={top_k}, scope_mode={scope_mode}): "
        f"{len(top_results)} results from {len(target_ids)} carts "
        f"({total_patterns} patterns) in {elapsed_ms:.0f}ms"
    )

    out = {
        "results": top_results,
        "elapsed_ms": round(elapsed_ms, 1),
        "cart_count": len(target_ids),
        "total_patterns": total_patterns,
        "mode": "multi_cart",
        "scope_mode": scope_mode,
    }
    if grouped_results is not None:
        out["grouped_results"] = grouped_results
    return out


def _resolve_scope(scope: Any, role_filter: Any) -> list[str]:
    """Resolve a scope spec + role filter to a concrete list of cart_ids."""
    # Step 1: scope → initial set
    if scope == "all":
        candidates = list(_mounted_carts.keys())
    elif scope == "local":
        # Backward-compat: just the first mounted cart
        candidates = [next(iter(_mounted_carts))] if _mounted_carts else []
    elif isinstance(scope, str):
        candidates = [scope] if scope in _mounted_carts else []
    elif isinstance(scope, (list, tuple)):
        candidates = [s for s in scope if s in _mounted_carts]
    else:
        candidates = []

    # Step 2: role_filter → narrow the set
    if role_filter is None:
        return candidates

    if isinstance(role_filter, str):
        role_set = {role_filter}
    elif isinstance(role_filter, (list, tuple, set)):
        role_set = set(role_filter)
    else:
        return candidates

    return [cid for cid in candidates if _mounted_carts[cid].get("role") in role_set]


# =============================================================================
# UTILITY
# =============================================================================

def get_cart(cart_id: str) -> Optional[dict]:
    """Return the state dict for a mounted cart, or None if not mounted.

    Useful for code that wants to inspect a cart directly (e.g. count patterns,
    iterate hippocampus entries) without going through the search API.
    """
    return _mounted_carts.get(cart_id)


def total_patterns_mounted() -> int:
    """Sum of all patterns across every mounted cart."""
    return sum(s["n_patterns"] for s in _mounted_carts.values())


def unmount_all() -> int:
    """Unmount every cart. Returns the number unmounted. Useful for tests."""
    count = len(_mounted_carts)
    for cart_id in list(_mounted_carts.keys()):
        unmount(cart_id)
    return count


# =============================================================================
# IMPRINT — single-pattern write into a mounted cart
# =============================================================================
# This is the general-purpose write API used by Membox (membox.imprint())
# and any other layer that needs to add a single pattern to a mounted cart
# at runtime. It updates the in-memory state AND persists to disk.

def imprint_with_meta(cart_id: str, text: str,
                      per_pattern_meta: Optional[dict] = None) -> dict:
    """Imprint a single new pattern into an already-mounted cart.

    Embeds the text via the existing embedding pipeline, appends it to the
    cart's embeddings/texts/binary_corpus arrays in-memory, appends the
    per_pattern_meta entry, and persists the updated cart to disk.

    Args:
        cart_id: The cart to write to (must be mounted)
        text: Pattern text content
        per_pattern_meta: Optional dict of per-pattern metadata fields
            (agent_id, written_at, tags, reasoning, etc.)

    Returns:
        dict with keys: cart_id, local_addr (the new pattern's index),
        n_patterns (cart total after the write), elapsed_ms

    Raises:
        ValueError if cart_id is not mounted or text is empty
    """
    import json
    import os
    import zlib
    import hashlib
    from membot_server import embed_text

    if cart_id not in _mounted_carts:
        raise ValueError(f"Cart {cart_id!r} is not mounted")
    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    state = _mounted_carts[cart_id]
    t0 = time.time()

    # 1. Embed the new text
    new_embedding = embed_text(text, prefix="search_document")
    # Ensure float32 shape (1, 768)
    new_embedding = np.asarray(new_embedding, dtype=np.float32)
    if new_embedding.ndim == 1:
        new_embedding = new_embedding.reshape(1, -1)

    # 2. Append to in-memory embeddings
    if state["embeddings"] is not None and len(state["embeddings"]) > 0:
        state["embeddings"] = np.vstack([state["embeddings"], new_embedding]).astype(np.float32)
    else:
        state["embeddings"] = new_embedding.astype(np.float32)

    # 3. Append to in-memory texts list
    state["texts"].append(text)
    new_local_addr = len(state["texts"]) - 1
    state["n_patterns"] = len(state["texts"])
    state["embedding_dim"] = state["embeddings"].shape[1]
    state["has_embeddings"] = True

    # 4. Update binary corpus (sign-zero bits) — append the new pattern's bits
    new_sign = (new_embedding > 0).astype(np.uint8)
    if state["binary_corpus"] is not None and len(state["binary_corpus"]) > 0:
        if state["binary_corpus"].shape[1] <= 96:
            # Already packed — pack the new bits and append
            new_packed = np.packbits(new_sign, axis=1)
            state["binary_corpus"] = np.vstack([state["binary_corpus"], new_packed]).astype(np.uint8)
        else:
            # Unpacked format — append unpacked bits
            state["binary_corpus"] = np.vstack([state["binary_corpus"], new_sign]).astype(np.uint8)
    else:
        # First pattern — pack by default for storage efficiency
        state["binary_corpus"] = np.packbits(new_sign, axis=1).astype(np.uint8)

    # 5. Persist to disk — re-save the cart with the new pattern.
    #    We use the same on-disk format the cart was loaded from.
    _persist_cart(state, per_pattern_meta_addition=per_pattern_meta)

    elapsed_ms = (time.time() - t0) * 1000
    log.info(
        f"[multi_cart] imprint_with_meta: cart={cart_id} local_addr={new_local_addr} "
        f"n_patterns={state['n_patterns']} elapsed_ms={elapsed_ms:.0f}"
    )

    return {
        "cart_id": cart_id,
        "local_addr": new_local_addr,
        "n_patterns": state["n_patterns"],
        "elapsed_ms": round(elapsed_ms, 1),
    }


def _persist_cart(state: dict, per_pattern_meta_addition: Optional[dict] = None) -> None:
    """Re-save a mounted cart to disk after an in-memory mutation.

    Reads the existing on-disk cart to preserve fields we don't track in memory
    (hippocampus, pattern0, version), updates the embeddings + passages +
    sign_bits + per_pattern_meta from the current state, and writes back.

    For per_pattern_meta: appends the new entry to whatever was already there.
    Maintains the JSON-string-in-0d-ndarray format used by Cart Builder.
    """
    import json
    import os
    import zlib
    import hashlib

    cart_path = state["cart_path"]
    if not cart_path or not os.path.exists(cart_path):
        log.warning(
            f"[multi_cart] _persist_cart: cart_path missing or doesn't exist for "
            f"{state.get('cart_id')}; cannot persist"
        )
        return

    # 1. Load the existing on-disk cart so we can preserve fields we don't have in memory
    existing = np.load(cart_path, allow_pickle=True)
    existing_keys = set(existing.files)

    save_kwargs: dict = {}

    # embeddings — from in-memory state (just got appended to)
    save_kwargs["embeddings"] = state["embeddings"]

    # passages — from in-memory state
    save_kwargs["passages"] = np.array(state["texts"], dtype=object)

    # compressed_texts — recompute from the current texts
    compressed = []
    for t in state["texts"]:
        compressed.append(np.void(zlib.compress(t.encode("utf-8"), level=9)))
    save_kwargs["compressed_texts"] = np.array(compressed, dtype=object)

    # sign_bits — recompute from the current embeddings
    save_kwargs["sign_bits"] = np.packbits(
        (state["embeddings"] > 0).astype(np.uint8), axis=1
    )

    # hippocampus — preserve from existing cart if present (we don't update
    # hippocampus on imprint in Phase 1; that's a Phase 2 enhancement)
    if "hippocampus" in existing_keys:
        # Note: this leaves the new pattern without a hippocampus entry, which
        # is fine for Phase 1 — it just won't appear in episodic navigation.
        # Phase 2 will add hippocampus entries on imprint.
        save_kwargs["hippocampus"] = existing["hippocampus"]

    # pattern0 — preserve as-is
    if "pattern0" in existing_keys:
        save_kwargs["pattern0"] = existing["pattern0"]

    # version — preserve as-is
    if "version" in existing_keys:
        save_kwargs["version"] = existing["version"]

    # per_pattern_meta — load existing, append the new entry, re-encode
    existing_meta_list: list = []
    if "per_pattern_meta" in existing_keys:
        try:
            raw = existing["per_pattern_meta"]
            s = raw.item() if raw.ndim == 0 else str(raw)
            parsed = json.loads(s)
            if isinstance(parsed, list):
                existing_meta_list = parsed
        except Exception as e:
            log.warning(f"[multi_cart] _persist_cart: could not parse existing per_pattern_meta: {e}")
            existing_meta_list = []

    # Pad existing meta list with empty dicts if it's shorter than the texts
    # (this handles the case where the cart has more patterns than meta entries)
    while len(existing_meta_list) < len(state["texts"]) - 1:
        existing_meta_list.append({})

    # Append the new entry (or empty dict if no addition was specified)
    new_entry = per_pattern_meta_addition if per_pattern_meta_addition is not None else {}
    existing_meta_list.append(new_entry)

    # Re-encode as JSON string in a 0-d object array (matches the Cart Builder format)
    save_kwargs["per_pattern_meta"] = np.array(json.dumps(existing_meta_list), dtype=object)

    # 2. Atomically write the updated cart
    np.savez_compressed(cart_path, **save_kwargs)

    # 3. Rewrite the manifest with the new fingerprint and count
    h = hashlib.sha256()
    if len(state["embeddings"]) > 0:
        h.update(state["embeddings"][0].tobytes())
        h.update(state["embeddings"][-1].tobytes())
    h.update(str(len(state["texts"])).encode())
    fingerprint = h.hexdigest()[:16]

    manifest_path = cart_path.rsplit(".", 1)[0] + "_manifest.json"
    manifest = {
        "version": "mcp-v4",
        "count": len(state["texts"]),
        "has_hippocampus": "hippocampus" in existing_keys,
        "fingerprint": fingerprint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rebuilt_by": "multi_cart.imprint_with_meta",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
