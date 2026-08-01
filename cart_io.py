"""Cartridge loading + integrity verification. NO MCP DEPENDENCY, DELIBERATELY.

WHY THIS MODULE EXISTS (2026-08-01)
------------------------------------
Companion to `embed.py`, and the second half of the same fix. This code lived in
`membot_server.py`, which begins `from fastmcp import FastMCP` — so `multi_cart.mount()`
reaching in for `load_cartridge_safe` / `verify_manifest` made the MOUNT path require the
MCP server package in the caller's interpreter.

HUB reported it on 2026-07-28, measured on the box that PUBLISHES the fleet cartridge:

    | interpreter                  | mount            | search                       |
    | membot/.venv/bin/python      | 80 patterns      | 3 hits                       |
    | system python3               | status: "error"  | ModuleNotFoundError: fastmcp |
    | ~/.web4/venv/bin/python      | status: "error"  | ModuleNotFoundError: fastmcp |

Reproduced locally before this change, at `multi_cart.py:140`.

Moving `embed_text` alone fixed SEARCH and left MOUNT broken — a half-fix that would have
looked complete from the changelog. Both paths had to come out.

A rejected alternative, recorded so it is not re-proposed: make `fastmcp` optional in
`membot_server` with no-op decorator stand-ins. That is ~10 lines instead of ~280, but it
makes the server importable and HOLLOW — `@mcp.tool` silently registers nothing. A mechanism
that appears present and does nothing is the exact failure class this codebase spent
2026-07-30/31 removing.

**Keep this module free of `membot_server` imports.** If one appears, the read path silently
reacquires the server's dependency tree and the HUB table goes red again.

Requires: numpy + stdlib. No MCP, no fastmcp, no sentence-transformers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import struct
import zlib

import numpy as np

log = logging.getLogger("membot")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories searched when resolving a cartridge by bare name. `is_trusted_directory`
# gates loading to these, so a pickle cannot be loaded from an arbitrary path.
CARTRIDGE_DIRS = [
    os.path.join(BASE_DIR, "cartridges"),
    os.path.join(BASE_DIR, "data"),
    os.path.join(BASE_DIR, "..", "vector-benchmark-demo", "cuda", "webgpu", "cartridges"),
    os.path.join(BASE_DIR, "..", "vector-benchmark-demo", "cuda", "cartridges"),
    os.path.join(BASE_DIR, "..", "vector-benchmark-demo", "cuda", "self_contained_cart_test"),
]

_HIPPO_FORMAT = '<I B B I I I I H I B B 34s'


def compute_fingerprint(embeddings: np.ndarray, n_texts: int) -> str:
    """Compute SHA256 fingerprint from embedding content + count."""
    h = hashlib.sha256()
    if embeddings is not None and len(embeddings) > 0:
        h.update(embeddings[0].tobytes())
        h.update(embeddings[-1].tobytes())
    h.update(str(n_texts).encode())
    return h.hexdigest()[:16]


def verify_manifest(cart_path: str, embeddings: np.ndarray, n_texts: int) -> tuple[bool, str]:
    """Verify cartridge against its manifest. Returns (ok, message)."""
    manifest_path = cart_path.rsplit(".", 1)[0] + "_manifest.json"
    if not os.path.exists(manifest_path):
        return True, "no manifest (legacy cartridge)"

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        expected_fp = manifest.get("fingerprint", "")
        actual_fp = compute_fingerprint(embeddings, n_texts)

        if expected_fp != actual_fp:
            return False, f"FINGERPRINT MISMATCH: expected {expected_fp}, got {actual_fp}"

        expected_count = manifest.get("count", -1)
        if expected_count != n_texts:
            return False, f"COUNT MISMATCH: expected {expected_count}, got {n_texts}"

        return True, f"verified ({actual_fp})"
    except Exception as e:
        return False, f"manifest read error: {e}"


def is_trusted_directory(path: str) -> bool:
    """Check if a file is within one of the trusted cartridge directories."""
    real_path = os.path.realpath(path)
    for d in CARTRIDGE_DIRS:
        real_dir = os.path.realpath(d)
        if real_path.startswith(real_dir + os.sep) or real_path.startswith(real_dir):
            return True
    return False


def load_cartridge_safe(path: str) -> dict:
    """Load a cartridge using the safest available method.

    NPZ files are always safe (no code execution).
    PKL files are only loaded from trusted directories.
    """
    if path.endswith(".npz"):
        return load_npz_cartridge(path)

    if path.endswith(".pkl"):
        if not is_trusted_directory(path):
            raise PermissionError(
                f"Refusing to load .pkl from untrusted path: {path}. "
                f"Only files in {CARTRIDGE_DIRS} are allowed. "
                f"Convert to .npz format for unrestricted loading."
            )
        log.warning(f"Loading legacy .pkl (trusted path): {path}")
        return load_pkl_cartridge(path)

    raise ValueError(f"Unknown cartridge format: {path}")


def load_pkl_cartridge(path: str) -> dict:
    """Load a legacy .pkl cartridge. ONLY call from trusted directories."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Handle versioned cartridges (v7.0-8.3) where embeddings are nested in data["data"]
    version = data.get("version", "legacy")
    if version in ("7.0", "8.0", "8.1", "8.2", "8.3"):
        core = data.get("data", data)
        embeddings = core.get("embeddings", data.get("embeddings", np.array([])))
        passages = core.get("passages", data.get("passages", []))
    elif "data" in data and isinstance(data["data"], dict):
        core = data["data"]
        embeddings = core.get("embeddings", np.array([]))
        passages = core.get("passages", [])
    else:
        embeddings = data.get("embeddings", np.array([]))
        passages = data.get("passages", data.get("texts", []))

    if isinstance(embeddings, list):
        embeddings = np.array(embeddings, dtype=np.float32)
    elif isinstance(embeddings, np.ndarray) and embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    result = {
        "embeddings": embeddings,
        "texts": list(passages),
        "version": version,
        "format": "pkl",
    }
    # Pre-computed sign bits (packed uint8, 96 bytes per pattern)
    if "sign_bits" in data:
        result["sign_bits"] = data["sign_bits"]
    # Extra metadata fields
    for key in ("paper_ids", "titles", "metadata"):
        if key in data:
            result[key] = data[key]
    return result


def load_npz_cartridge(path: str) -> dict:
    """Load an .npz cartridge (safe — no code execution).

    Supports split carts: if the index contains has_sqlite=True,
    texts are loaded as snippets from the index and full passages
    are fetched on demand from a SQLite sidecar file.
    """
    data = np.load(path, allow_pickle=True)

    result = {"format": "npz", "version": "unknown"}

    # Check for split cart (index + SQLite sidecar)
    has_sqlite = bool(data["has_sqlite"]) if "has_sqlite" in data else False
    if has_sqlite:
        db_name = str(data["text_db"]) if "text_db" in data else None
        if db_name:
            db_path = os.path.join(os.path.dirname(path), db_name)
            if os.path.exists(db_path):
                result["sqlite_db_path"] = db_path
                log.info(f"Split cart: SQLite sidecar at {db_path}")
            else:
                log.warning(f"Split cart: SQLite sidecar not found at {db_path}")

        # Snippets serve as texts for keyword reranking
        if "snippets" in data:
            snippets = data["snippets"]
            if isinstance(snippets, np.ndarray) and snippets.dtype == object:
                result["texts"] = list(snippets)
            else:
                result["texts"] = list(snippets)
            count = int(data["count"]) if "count" in data else len(result["texts"])
            log.info(f"Split cart: {count:,} snippets loaded, full text in SQLite")
        else:
            result["texts"] = []

        result["embeddings"] = np.array([])
        result["is_split_cart"] = True

        # Sign bits
        if "sign_bits" in data:
            result["sign_bits"] = data["sign_bits"]

        return result

    # Handle various NPZ layouts (standard carts)
    if "embeddings" in data:
        result["embeddings"] = data["embeddings"]
    elif "embs" in data:
        embs = data["embs"]
        if isinstance(embs, np.ndarray) and embs.dtype == object:
            result["embeddings"] = np.array(list(embs), dtype=np.float32)
        else:
            result["embeddings"] = embs
    else:
        result["embeddings"] = np.array([])

    if "passages" in data:
        result["texts"] = list(data["passages"])
    elif "memories" in data:
        mems = data["memories"]
        if isinstance(mems, np.ndarray) and mems.dtype == object:
            result["texts"] = [m.get("text", str(m)) if isinstance(m, dict) else str(m) for m in mems]
        else:
            result["texts"] = list(mems)
    elif "compressed_texts" in data:
        texts = []
        for ct in data["compressed_texts"]:
            try:
                texts.append(zlib.decompress(bytes(ct)).decode("utf-8"))
            except Exception:
                texts.append("[decompress error]")
        result["texts"] = texts
    else:
        result["texts"] = []

    # Load hippocampus metadata if present (mcp-v4+)
    if "hippocampus" in data:
        raw_hippo = data["hippocampus"]
        log.info(f"[HIPPO] Found hippocampus in NPZ: shape={raw_hippo.shape}, dtype={raw_hippo.dtype}")
        result["hippocampus"] = _unpack_hippocampus(raw_hippo)
        log.info(f"[HIPPO] Unpacked {len(result['hippocampus'])} entries. First: {result['hippocampus'][0] if result['hippocampus'] else 'EMPTY'}")
        # Log a sample entry that should have links
        for h in result["hippocampus"][:20]:
            if h["prev"] is not None or h["next"] is not None:
                log.info(f"[HIPPO] Sample linked entry: {h}")
                break
        else:
            log.warning("[HIPPO] No linked entries found in first 20 hippocampus records!")
    else:
        # Expected for Supabase-backed Mempacks (H-block array lives in the
        # mempack_patterns Postgres rows, not in the NPZ blob) and for legacy
        # carts that pre-date the hippocampus array. Not an error condition.
        log.debug(f"[HIPPO] No hippocampus array in NPZ (Mempack-style: H-blocks in Postgres). Keys: {list(data.keys())}")
        result["hippocampus"] = None

    # Load per-pattern structured metadata (federate-v1+ format; used by
    # fleet-learning carts where each pattern carries a JSON dict of
    # game/machine/event/confidence/support_count/beat_dummy_pp/etc.).
    # Parsed eagerly so the filter API can query fields without re-parsing.
    #
    # Storage variants seen in the wild:
    #   - 1-d object array of dicts (federate-v1 CBP/Nomad/etc.)
    #   - 0-d object array wrapping a Python list of dicts
    #   - 0-d object array wrapping a JSON-encoded STRING (heartbeat cart)
    #   - 0-d object array wrapping None (meta field reserved but empty)
    # Also tolerate each element being either a dict or a JSON string.
    if "per_pattern_meta" in data:
        raw_meta = data["per_pattern_meta"]
        # Unwrap 0-d scalar
        if hasattr(raw_meta, "ndim") and raw_meta.ndim == 0:
            raw_meta = raw_meta.item()
        # Decode JSON-string-encoded lists
        if isinstance(raw_meta, (str, bytes)):
            try:
                raw_meta = json.loads(raw_meta)
            except Exception as e:
                log.warning(f"[META] per_pattern_meta is a string but not valid JSON: {e}")
                raw_meta = None
        if raw_meta is None:
            result["per_pattern_meta"] = None
        elif isinstance(raw_meta, dict):
            # Single dict for the whole cart — unusual but tolerate by broadcasting
            log.info(f"[META] per_pattern_meta is a single dict; not broadcasting (treating as absent)")
            result["per_pattern_meta"] = None
        else:
            try:
                iterable_meta = list(raw_meta)
            except TypeError:
                log.warning(f"[META] per_pattern_meta not iterable after unwrap: type={type(raw_meta).__name__}")
                result["per_pattern_meta"] = None
                return result
            parsed = []
            for entry in iterable_meta:
                if isinstance(entry, dict):
                    parsed.append(entry)
                elif isinstance(entry, (str, bytes)):
                    try:
                        obj = json.loads(entry)
                        parsed.append(obj if isinstance(obj, dict) else {})
                    except Exception:
                        parsed.append({})
                else:
                    parsed.append({})
            result["per_pattern_meta"] = parsed
            # Find first non-empty dict for log sample
            sample_keys = []
            for p in parsed:
                if p:
                    sample_keys = list(p.keys())
                    break
            log.info(f"[META] Loaded per_pattern_meta: {len(parsed)} entries, sample keys={sample_keys}")
    else:
        result["per_pattern_meta"] = None

    return result


def _unpack_hippocampus(raw: np.ndarray) -> list[dict]:
    """Unpack hippocampus metadata array into list of dicts.

    Args:
        raw: (N, 64) uint8 array from NPZ 'hippocampus' key

    Returns:
        List of dicts with: pattern_id, format_version, cartridge_type,
        prev, next, sibling, source_hash, sequence_num, timestamp, flags,
        perms_byte (canonical format only)
    """
    result = []
    for row in raw:
        vals = struct.unpack(_HIPPO_FORMAT, row.tobytes())
        result.append({
            "pattern_id":     vals[0],
            "format_version": vals[1],
            "cartridge_type": vals[2],
            "prev":           vals[3] if vals[3] > 0 else None,
            "next":           vals[4] if vals[4] > 0 else None,
            "sibling":        vals[5] if vals[5] > 0 else None,
            "source_hash":    vals[6],
            "sequence_num":   vals[7],
            "timestamp":      vals[8],
            "flags":          vals[9],
            "perms_byte":     vals[10],
        })
    return result
