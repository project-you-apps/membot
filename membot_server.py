"""
Membot — Brain Cartridge Server for AI Agents
==============================================
MCP server that gives AI agents swappable, physics-enhanced memory.
Built on the Vector+ Lattice Engine.

Architecture:
  - SentenceTransformer (nomic-ai/nomic-embed-text-v1.5) for 768-dim embeddings
  - Sign-zero Hamming search: 70% embedding cosine + 30% Hamming similarity on binary codes
  - Binary corpus: sign_zero encoding (bit_i = 1 if embedding_i > 0), 768 bits = 96 bytes/pattern
  - Keyword reranking on top of blended scores
  - GPU/lattice available for recall but not used for search ranking
  - Compatible with Vector Plus Studio v8.3 cartridge format (.npz/.pkl)

Security:
  - New cartridges saved as NPZ (no code execution risk)
  - Legacy .pkl loading restricted to trusted directories only
  - SHA256 manifest for cartridge integrity verification
  - Cartridge name sanitization (no path traversal)
  - Resource caps (max entries, max text length)

Tools:
  list_cartridges  — Browse available brain cartridges
  mount_cartridge  — Load a cartridge into memory
  memory_search    — Semantic search across mounted cartridge
  memory_store     — Store new text into current cartridge
  save_cartridge   — Persist current cartridge to disk
  unmount          — Free memory and unload cartridge
  get_status       — Server diagnostics

Transport:
  stdio  — Local pipe (OpenClaw, Claude Desktop)
  http   — Streamable HTTP (remote agents, Claude Code)

Usage:
  python membot_server.py                              # stdio (default)
  python membot_server.py --transport http --port 8000  # HTTP server
  MEMBOT_API_KEY=secret python membot_server.py --transport http  # HTTP + auth

See README.md for agent configuration.

https://github.com/project-you-apps/membot
"""

from fastmcp import FastMCP
import os
import sys
import re
import struct
import hashlib
import pickle
import sqlite3
import zlib
import time
import base64
import json
import logging
import argparse
import collections
import math
import numpy as np

# --- Logging (stderr only — stdout is reserved for MCP JSON-RPC) ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("membot")

# Also log to file for post-mortem debugging
_file_handler = logging.FileHandler(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "membot.log"),
    mode="w",
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
log.addHandler(_file_handler)

# --- Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARTRIDGE_DIRS = [
    os.path.join(BASE_DIR, "cartridges"),
    os.path.join(BASE_DIR, "data"),
    os.path.join(BASE_DIR, "..", "vector-benchmark-demo", "cuda", "webgpu", "cartridges"),
    os.path.join(BASE_DIR, "..", "vector-benchmark-demo", "cuda", "cartridges"),
    os.path.join(BASE_DIR, "..", "vector-benchmark-demo", "cuda", "self_contained_cart_test"),
]
# Mempack storage: per-user writable carts at cartridges/users/<owner_id>/<name>.cart.npz.
# Walked recursively by find_mempacks() (separate from find_cartridges which only does
# top-level CARTRIDGE_DIRS). Andy 2026-05-12 — Block F revival as Mempack storage.
MEMPACK_BASE_DIR = os.path.join(BASE_DIR, "cartridges", "users")
# UUID format pattern for owner_id directory names (defense in depth — Supabase UUIDs).
_UUID_RE = re.compile(r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$")
HAMMING_BLEND = 0.3           # 70% cosine + 30% sign_zero Hamming (replaces physics L2)
RECENCY_HALF_LIFE = 86400.0   # 24h — recency weight halves per day (0 = disabled)

# --- Security Limits ---
MAX_ENTRIES = 3_000_000      # Max memories per cartridge (supports 2.4M arXiv)
MAX_TEXT_LENGTH = 10_000     # Max characters per memory_store call
MAX_QUERY_LENGTH = 2_000    # Max characters per search query
SAFE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_\- \.]*$')


# ============================================================
# SECURITY HELPERS
# ============================================================

def sanitize_name(name: str) -> str:
    """Sanitize cartridge name: strip path separators, validate characters."""
    name = os.path.basename(name)
    name = name.replace("..", "").replace("/", "").replace("\\", "")
    if not name or not SAFE_NAME_PATTERN.match(name):
        raise ValueError(f"Invalid cartridge name: '{name}'. Use alphanumeric, hyphens, underscores, spaces, dots only.")
    return name


def compute_fingerprint(embeddings: np.ndarray, n_texts: int) -> str:
    """Compute SHA256 fingerprint from embedding content + count."""
    h = hashlib.sha256()
    if embeddings is not None and len(embeddings) > 0:
        h.update(embeddings[0].tobytes())
        h.update(embeddings[-1].tobytes())
    h.update(str(n_texts).encode())
    return h.hexdigest()[:16]


# Pattern 0 v2 permission helpers (Andy 2026-05-12).
# Permission semantics from spec docs/RFC/pattern-0-v2-spec.md:
#   r = read (search the cart)        bit 0 = 1
#   w = write (add/edit patterns)     bit 1 = 2
#   d = delete (tombstone patterns)   bit 2 = 4
#   a = admin (manage perms/delete)   bit 3 = 8
# Stored in the manifest as a readable string ("rw", "rwda", "" for none).
_PERM_BITS = {"r": 1, "w": 2, "d": 4, "a": 8}


def parse_perms(perm_str: str | None) -> int:
    """Convert a permission string like 'rwd' into a bitmask. Empty/None = 0."""
    if not perm_str:
        return 0
    mask = 0
    for ch in perm_str.lower():
        mask |= _PERM_BITS.get(ch, 0)
    return mask


def format_perms(mask: int) -> str:
    """Convert a permission bitmask back into a string like 'rwd'."""
    return "".join(ch for ch, bit in _PERM_BITS.items() if mask & bit)


# Pattern 0 v2 cart-type marker (Andy 2026-05-12). Mempack-shaped carts have
# cart_type="agent-memory" and reserve pattern index 1 ("Pattern I") for the
# agent's behavioral instructions. Other types are treated as knowledge carts.
CART_TYPE_KNOWLEDGE = "knowledge"
CART_TYPE_AGENT_MEMORY = "agent-memory"
PATTERN_I_IDX = 1  # The reserved slot for Pattern I in agent-memory carts


def save_manifest(cart_path: str, embeddings: np.ndarray, n_texts: int,
                  briefing: str | None = None,
                  owner_id: str | None = None,
                  owner_perms: str | None = None,
                  group_perms: str | None = None,
                  world_perms: str | None = None,
                  group_id: str | None = None,
                  max_patterns: int | None = None,
                  cart_type: str | None = None):
    """Save integrity manifest alongside cartridge.

    Pattern 0 v2 fields (Andy 2026-05-12) — see docs/RFC/pattern-0-v2-spec.md:
      briefing      — UTF-8 text presented to agents on mount (Phase 1)
      owner_id      — agent or user ID that owns this cart (Phase 2)
      owner_perms   — string of {r,w,d,a} flags for the owner
      group_perms   — flags for group members (if group_id set)
      world_perms   — flags for everyone else
      group_id      — team/org identifier; pairs with group_perms
      max_patterns  — capacity limit (0 or omitted = unlimited)

    All Pattern 0 v2 fields are optional and backward-compatible. Carts without
    them mount with world_perms=r (treat legacy carts as publicly readable) so
    existing flows keep working unchanged.
    """
    manifest_path = cart_path.rsplit(".", 1)[0] + "_manifest.json"
    manifest = {
        "version": "mcp-v3",
        "count": n_texts,
        "fingerprint": compute_fingerprint(embeddings, n_texts),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # Optional Pattern 0 v2 fields — only emit when caller supplied them so legacy
    # manifests stay minimal.
    if briefing:
        manifest["briefing"] = briefing
    if owner_id is not None:
        manifest["owner_id"] = owner_id
    if owner_perms is not None:
        manifest["owner_perms"] = owner_perms
    if group_perms is not None:
        manifest["group_perms"] = group_perms
    if world_perms is not None:
        manifest["world_perms"] = world_perms
    if group_id is not None:
        manifest["group_id"] = group_id
    if max_patterns is not None:
        manifest["max_patterns"] = int(max_patterns)
    if cart_type is not None:
        manifest["cart_type"] = cart_type

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    extra = []
    if briefing:
        extra.append(f"briefing({len(briefing)})")
    if owner_id is not None:
        extra.append(f"owner={owner_id}")
    extra_str = (" +" + " +".join(extra)) if extra else ""
    log.info(f"Manifest saved: {manifest['fingerprint']} ({n_texts} entries){extra_str}")
    return manifest


def load_manifest(cart_path: str) -> dict | None:
    """Read the manifest sidecar for a cart and return as dict, or None if missing/malformed.

    Used to extract Pattern 0 fields (briefing, eventually ownership, capabilities)
    on cart mount. Does NOT verify integrity — that's verify_manifest's job. This
    one is for reading metadata once integrity is established.
    """
    manifest_path = cart_path.rsplit(".", 1)[0] + "_manifest.json"
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"manifest read error for {cart_path}: {e}")
        return None


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


# ============================================================
# STATE (per-session)
# ============================================================

SESSION_TIMEOUT_SEC = 1800   # 30 minutes idle → session expires
MAX_SESSIONS = 50            # Max concurrent sessions

def _new_session() -> dict:
    """Create a fresh session state dict."""
    return {
        "cartridge_name": None,
        "cartridge_path": None,
        "briefing": None,         # Pattern 0 v2 Phase 1 — cart's agent-facing introduction
        # Pattern 0 v2 Phase 2 — ownership/access control. Defaults preserve
        # legacy behavior: no owner, world readable, no caps.
        "owner_id": None,
        "owner_perms": "rwda",
        "group_perms": "",
        "world_perms": "r",       # legacy carts treated as world-readable
        "group_id": None,
        "max_patterns": 0,        # 0 = unlimited
        "embeddings": None,       # (N, 768) float32
        "texts": [],              # list[str]
        "binary_corpus": None,    # (N, 768) uint8 — sign_zero encoding for Hamming search
        "signatures": None,       # (N, 4096) float32 or None (legacy, not used in search)
        "hippocampus": None,      # list[dict] — per-passage metadata (prev/next/flags/etc.)
        "lattice": None,          # CUDA wrapper or None
        "gpu_available": False,
        "modified": False,        # True if memory_store was called since last save
        "last_access": time.time(),
        "created_at": time.time(),
        "query_count": 0,
        "mount_count": 0,
        "last_action": None,      # e.g. "search 'earthquake'" or "mount wiki-10k"
        "last_action_time": None,
        "sqlite_conn": None,      # sqlite3.Connection for split carts (text on disk)
        "is_split_cart": False,   # True if using index + SQLite sidecar
        "has_embeddings": False,
    }


def _sqlite_fetch_passages(conn: sqlite3.Connection, indices: list) -> dict:
    """Fetch full passages from SQLite sidecar by index. Returns {idx: passage}."""
    if not conn or not indices:
        return {}
    placeholders = ",".join("?" for _ in indices)
    rows = conn.execute(
        f"SELECT idx, passage, title, paper_id FROM passages WHERE idx IN ({placeholders})",
        indices
    ).fetchall()
    return {r[0]: {"passage": r[1], "title": r[2], "paper_id": r[3]} for r in rows}


def _soft_truncate(text: str, target_min: int = 250, hard_max: int = 550) -> str:
    """Truncate at first sentence-end past target_min, falling back to last
    word boundary, then hard cap. Preserves provenance-card aesthetic without
    mid-word cuts."""
    if len(text) <= hard_max:
        return text
    for marker in (". ", "? ", "! ", "\n"):
        idx = text.find(marker, target_min)
        if 0 < idx <= hard_max - len(marker):
            return text[: idx + len(marker)].rstrip() + " …"
    cut = text.rfind(" ", target_min, hard_max)
    if cut > target_min:
        return text[:cut].rstrip() + " …"
    return text[:hard_max].rstrip() + " …"


# --- Depot Activity Log (ring buffer) ---
DEPOT_ACTIVITY_MAX = 200
_depot_activity: collections.deque = collections.deque(maxlen=DEPOT_ACTIVITY_MAX)
_depot_start_time = time.time()


def _log_activity(session_id: str, action: str, detail: str = "", latency_ms: float = 0):
    """Append an entry to the depot activity log and update session state."""
    entry = {
        "time": time.time(),
        "session": session_id,
        "action": action,
        "detail": detail,
        "latency_ms": round(latency_ms, 1),
    }
    _depot_activity.append(entry)
    # Update session last_action
    if session_id in _sessions:
        _sessions[session_id]["last_action"] = f"{action} {detail}".strip()
        _sessions[session_id]["last_action_time"] = time.time()

# session_id → session state dict
_sessions: dict[str, dict] = {}

# Global config (not per-session)
_server_config = {
    "read_only": False,
}


def _get_session(session_id: str) -> dict:
    """Get or create a session by ID. Evicts expired sessions."""
    now = time.time()

    # Evict expired sessions
    expired = [sid for sid, s in _sessions.items()
               if now - s["last_access"] > SESSION_TIMEOUT_SEC]
    for sid in expired:
        log.info(f"Session expired: {sid} (idle {now - _sessions[sid]['last_access']:.0f}s)")
        del _sessions[sid]

    if session_id not in _sessions:
        if len(_sessions) >= MAX_SESSIONS:
            # Evict oldest session
            oldest = min(_sessions, key=lambda sid: _sessions[sid]["last_access"])
            log.info(f"Session evicted (max reached): {oldest}")
            del _sessions[oldest]
        _sessions[session_id] = _new_session()
        log.info(f"New session created: {session_id}")

    _sessions[session_id]["last_access"] = now
    return _sessions[session_id]


def _resolve_session_id(session_id: str) -> str:
    """Resolve empty session_id to 'default', return as-is otherwise."""
    return session_id.strip() if session_id.strip() else "default"


# ============================================================
# EMBEDDING — Ollama backend (resource-constrained) or SentenceTransformer
# ============================================================

_embed_model = None  # lazy-loaded SentenceTransformer (if not using Ollama)
_embed_backend = os.environ.get("MEMBOT_EMBED_BACKEND", "auto")  # "ollama", "st", or "auto"
_ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_ollama_embed_model = os.environ.get("MEMBOT_OLLAMA_EMBED_MODEL", "nomic-embed-text")

def _ollama_available() -> bool:
    """Check if Ollama is running and has the embedding model."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{_ollama_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            models = [m["name"].split(":")[0] for m in data.get("models", [])]
            return _ollama_embed_model.split(":")[0] in models
    except Exception:
        return False

def _embed_via_ollama(text: str) -> np.ndarray:
    """Get embedding from Ollama API (no extra RAM — reuses running Ollama process)."""
    import urllib.request
    payload = json.dumps({"model": _ollama_embed_model, "input": text}).encode()
    req = urllib.request.Request(
        f"{_ollama_url}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
        return np.array(data["embeddings"][0], dtype=np.float32)

def _get_embed_model():
    """Lazy-load SentenceTransformer (same model Studio uses to build cartridges)."""
    global _embed_model
    if _embed_model is None:
        log.info("Loading SentenceTransformer nomic-embed-text-v1.5 ...")
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
        log.info("SentenceTransformer ready")
    return _embed_model

def _resolve_embed_backend() -> str:
    """Resolve which embedding backend to use."""
    if _embed_backend == "ollama":
        return "ollama"
    if _embed_backend == "st":
        return "st"
    # auto: prefer Ollama if available (saves ~2GB RAM)
    if _ollama_available():
        log.info(f"Embedding backend: Ollama ({_ollama_embed_model}) — saves ~2GB RAM")
        return "ollama"
    log.info("Embedding backend: SentenceTransformer (Ollama not available)")
    return "st"

_resolved_backend = None

def embed_text(text: str, prefix: str = "search_query") -> np.ndarray:
    """Get 768-dim Nomic embedding via Ollama or SentenceTransformer."""
    global _resolved_backend
    if _resolved_backend is None:
        _resolved_backend = _resolve_embed_backend()

    if _resolved_backend == "ollama":
        return _embed_via_ollama(f"{prefix}: {text}")
    else:
        model = _get_embed_model()
        vec = model.encode(f"{prefix}: {text}", convert_to_numpy=True)
        return vec.astype(np.float32)


# ============================================================
# GPU INIT (OPTIONAL — NON-FATAL IF UNAVAILABLE)
# ============================================================

_gpu_state = {
    "lattice": None,
    "available": False,
}

def init_gpu():
    """Try to load CUDA LatticeRunner. Returns True if successful."""
    if _gpu_state["lattice"] is not None:
        return _gpu_state["available"]
    try:
        sys.path.insert(0, BASE_DIR)
        from multi_lattice_wrapper_v7 import MultiLatticeCUDAv7
        _gpu_state["lattice"] = MultiLatticeCUDAv7(lattice_size=4096, verbose=0)
        _gpu_state["available"] = True
        log.info("CUDA LatticeRunner V7 ready (4096x4096)")
        return True
    except Exception as e:
        log.warning(f"CUDA not available (CPU-only mode): {e}")
        _gpu_state["available"] = False
        return False


# ============================================================
# CARTRIDGE I/O
# ============================================================

def find_mempacks(owner_id: str | None = None) -> list[dict]:
    """Walk MEMPACK_BASE_DIR recursively for per-user .cart.npz Mempacks.

    Separate from find_cartridges so the public catalog stays clean and per-user
    discovery is opt-in. If owner_id is supplied, only return Mempacks under that
    user's directory; otherwise return all (admin/debug use).

    Returns the same shape as find_cartridges entries, plus owner_id pulled
    from the directory layout. Each Mempack's manifest sidecar holds the
    authoritative ownership block (Pattern 0 Phase 2).
    """
    results = []
    if not os.path.isdir(MEMPACK_BASE_DIR):
        return results
    base_real = os.path.realpath(MEMPACK_BASE_DIR)

    if owner_id is not None:
        if not _UUID_RE.match(owner_id):
            return results  # bad input -> empty, not error
        user_dirs = [os.path.join(MEMPACK_BASE_DIR, owner_id)]
    else:
        user_dirs = [
            os.path.join(MEMPACK_BASE_DIR, d)
            for d in os.listdir(MEMPACK_BASE_DIR)
            if _UUID_RE.match(d) and os.path.isdir(os.path.join(MEMPACK_BASE_DIR, d))
        ]

    for user_dir in user_dirs:
        if not os.path.isdir(user_dir):
            continue
        # path-traversal defense: resolved user_dir must be under the resolved base
        try:
            user_real = os.path.realpath(user_dir)
            if not (user_real == base_real or user_real.startswith(base_real + os.sep)):
                log.warning(f"find_mempacks: rejected {user_dir} (escapes base)")
                continue
        except OSError:
            continue

        this_uid = os.path.basename(user_dir)
        for f in os.listdir(user_dir):
            if f.endswith("_signatures.npz") or f.endswith("_brain.npy") or f.endswith("_manifest.json"):
                continue
            if not (f.endswith(".cart.npz") or f.endswith(".pkl") or f.endswith(".npz")):
                continue
            name = f.replace(".cart.npz", "").replace(".pkl", "").replace(".npz", "")
            path = os.path.join(user_dir, f)
            try:
                size_mb = os.path.getsize(path) / (1024 * 1024)
            except OSError:
                continue
            # save_manifest uses cart_path.rsplit(".", 1)[0] + "_manifest.json",
            # which preserves the ".cart" middle-extension for .cart.npz files.
            # Check both filename patterns for robustness.
            manifest_candidates = [
                os.path.join(user_dir, f"{name}.cart_manifest.json"),
                os.path.join(user_dir, f"{name}_manifest.json"),
            ]
            has_manifest = any(os.path.exists(p) for p in manifest_candidates)
            results.append({
                "name": name,
                "path": path,
                "owner_id": this_uid,
                "format": "npz" if f.endswith(".npz") else "pkl",
                "size_mb": round(size_mb, 1),
                "has_manifest": has_manifest,
            })
    return results


def find_cartridges() -> list[dict]:
    """Scan cartridge directories for available .pkl and .npz cartridge files."""
    carts = []
    seen_names = set()

    for d in CARTRIDGE_DIRS:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            # Support both .pkl (legacy) and .npz cartridges
            # Skip signature/brain companion files
            if f.endswith("_signatures.npz") or f.endswith("_brain.npy") or f.endswith("_manifest.json"):
                continue

            is_pkl = f.endswith(".pkl")
            is_npz = f.endswith(".cart.npz") or (f.endswith(".npz") and not f.endswith("_signatures.npz"))

            if not is_pkl and not is_npz:
                continue

            name = f.replace(".cart.npz", "").replace(".pkl", "").replace(".npz", "")
            if name in seen_names:
                continue
            seen_names.add(name)

            path = os.path.join(d, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)

            has_brain = os.path.exists(os.path.join(d, f"{name}_brain.npy"))
            has_sigs = os.path.exists(os.path.join(d, f"{name}_signatures.npz"))
            has_manifest = os.path.exists(os.path.join(d, f"{name}_manifest.json"))

            carts.append({
                "name": name,
                "path": path,
                "format": "npz" if is_npz else "pkl",
                "size_mb": round(size_mb, 1),
                "has_brain": has_brain,
                "has_signatures": has_sigs,
                "has_manifest": has_manifest,
                "dir": os.path.basename(d),
            })
    return carts


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
        log.info(f"[HIPPO] No hippocampus key in NPZ. Keys present: {list(data.keys())}")
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


def load_signatures(path: str) -> dict:
    """Load .npz signature file with optional compressed texts."""
    data = np.load(path, allow_pickle=True)
    result = {
        "signatures": data["signatures"],
        "n_patterns": int(data.get("n_patterns", len(data["signatures"]))),
        "method": str(data.get("signature_method", "unknown")),
    }

    if "compressed_texts" in data:
        texts = []
        for ct in data["compressed_texts"]:
            try:
                texts.append(zlib.decompress(bytes(ct)).decode("utf-8"))
            except Exception:
                texts.append("[decompress error]")
        result["texts"] = texts

    if "titles" in data:
        result["titles"] = [str(t) for t in data["titles"]]

    return result


# --- Hippocampus metadata struct (matches cartridge_builder.py) ---
_HIPPO_FORMAT = '<I B B I I I I H I B 35s'
_HIPPO_SIZE = 64

def _unpack_hippocampus(raw: np.ndarray) -> list[dict]:
    """Unpack hippocampus metadata array into list of dicts.

    Args:
        raw: (N, 64) uint8 array from NPZ 'hippocampus' key

    Returns:
        List of dicts with: pattern_id, format_version, cartridge_type,
        prev, next, sibling, source_hash, sequence_num, timestamp, flags
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
        })
    return result


def save_as_npz(path: str, embeddings: np.ndarray, texts: list[str]):
    """Save cartridge in safe NPZ format (no pickle, no code execution)."""
    compressed_texts = []
    for t in texts:
        compressed_texts.append(np.void(zlib.compress(t.encode("utf-8"), level=9)))

    np.savez_compressed(
        path,
        embeddings=embeddings,
        passages=np.array(texts, dtype=object),
        compressed_texts=np.array(compressed_texts, dtype=object),
        version="mcp-v3",
    )


# ============================================================
# MCP SERVER
# ============================================================

mcp = FastMCP("Membot")


# ============================================================
# REST API — Simple HTTP endpoints for browser extensions, etc.
# These bypass MCP protocol overhead for direct tool access.
# ============================================================

from starlette.requests import Request
from starlette.responses import JSONResponse


def _log_rest(request: Request, endpoint: str, extra: str = ""):
    """Log REST API access with client identification."""
    client = request.headers.get("x-client", "unknown")
    ua = request.headers.get("user-agent", "")
    ip = request.client.host if request.client else "?"
    sid = "(no body)"
    msg = f"[REST] {endpoint} | client={client} | ip={ip}"
    if extra:
        msg += f" | {extra}"
    if ua and client == "unknown":
        # Show UA only if no explicit client header (helps identify curl vs browser vs extension)
        msg += f" | ua={ua[:80]}"
    log.info(msg)


@mcp.custom_route("/api/store", methods=["POST", "OPTIONS"])
async def rest_store(request: Request) -> JSONResponse:
    """REST wrapper for memory_store — used by Heartbeat browser extension."""
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    try:
        data = await request.json()
        content = data.get("content", "")
        tags = data.get("tags", "")
        _log_rest(request, "STORE", f"tags={tags} len={len(content)}")
        _call = getattr(memory_store, 'fn', memory_store)  # FastMCP 2.x vs 3.x
        result = _call(
            content=content,
            tags=tags,
            session_id=data.get("session_id", "")
        )
        return JSONResponse({"status": "ok", "result": result}, headers=_cors_headers())
    except Exception as e:
        log.error(f"REST /api/store error: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500, headers=_cors_headers())


@mcp.custom_route("/api/search", methods=["POST", "OPTIONS"])
async def rest_search(request: Request) -> JSONResponse:
    """REST search returning structured JSON for browser apps."""
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    try:
        data = await request.json()
        query = data.get("query", "")
        top_k = data.get("top_k", 5)
        _log_rest(request, "SEARCH", f"q=\"{query[:60]}\" top_k={top_k}")
        session_id = _resolve_session_id(data.get("session_id", ""))
        state = _get_session(session_id)

        if state["cartridge_name"] is None:
            return JSONResponse({"status": "ok", "results": [], "error": "No cartridge mounted"}, headers=_cors_headers())

        has_emb = state.get("has_embeddings", True) and state["embeddings"] is not None and len(state["embeddings"]) > 0
        has_ham = state["binary_corpus"] is not None
        if not has_emb and not has_ham:
            return JSONResponse({"status": "ok", "results": []}, headers=_cors_headers())

        t0 = time.time()

        # Embed query
        query_emb = embed_text(query, prefix="search_query")

        # Helper: Hamming scores (handles packed and unpacked).
        # Uses popcount lookup + chunked XOR. Avoids both the 1.7 GB unpackbits
        # 768-expansion AND the 220 MB full-matrix XOR allocation.
        _POPCOUNT_TABLE_LOCAL = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
        def _ham_scores(q_emb, corpus_bin):
            is_packed = corpus_bin.shape[1] <= 96
            n = len(corpus_bin)
            n_bits = 768 if is_packed else corpus_bin.shape[1]
            dist = np.empty(n, dtype=np.uint32)
            BATCH = 100_000
            if is_packed:
                q_packed = np.packbits((q_emb > 0).astype(np.uint8))
                for start in range(0, n, BATCH):
                    end = min(start + BATCH, n)
                    xor = np.bitwise_xor(q_packed, corpus_bin[start:end])
                    dist[start:end] = _POPCOUNT_TABLE_LOCAL[xor].sum(axis=1)
            else:
                q_bin = (q_emb > 0).astype(np.uint8)
                for start in range(0, n, BATCH):
                    end = min(start + BATCH, n)
                    xor = np.bitwise_xor(q_bin, corpus_bin[start:end])
                    dist[start:end] = xor.sum(axis=1)
            return 1.0 - dist.astype(np.float32) / n_bits

        if has_emb:
            # Cosine similarity
            stored = state["embeddings"]
            stored_norms = np.linalg.norm(stored, axis=1, keepdims=True) + 1e-9
            query_norm = np.linalg.norm(query_emb) + 1e-9
            emb_scores = np.dot(stored / stored_norms, query_emb / query_norm)

            # Hamming blend
            scores = emb_scores
            if HAMMING_BLEND > 0 and has_ham:
                try:
                    corpus_bin = state["binary_corpus"]
                    n_bin = min(len(corpus_bin), len(emb_scores))
                    ham = _ham_scores(query_emb, corpus_bin[:n_bin])
                    blended = np.copy(emb_scores)
                    blended[:n_bin] = (1.0 - HAMMING_BLEND) * emb_scores[:n_bin] + HAMMING_BLEND * ham
                    scores = blended
                except Exception:
                    pass
        else:
            # Hamming-only
            scores = _ham_scores(query_emb, state["binary_corpus"])

        # Optional pre-filter: if a filter block is provided in the request,
        # restrict candidates to matching indices BEFORE ranking. This is the
        # shape Dennis's mechanics encoder + Query Hierarchy Phase 1 both want:
        # narrow by structural constraints first, rank by relevance second.
        # Filter spec shape documented in forum/filter-api-spec.md.
        filter_spec = data.get("filter")
        filter_matched = None
        if filter_spec:
            allowed_indices = _apply_filter(state, filter_spec)
            filter_matched = len(allowed_indices)
            if not allowed_indices:
                elapsed_ms = (time.time() - t0) * 1000
                _log_activity(session_id, "search+filter", f"'{query[:40]}' -> 0 (filter eliminated all)", elapsed_ms)
                return JSONResponse(
                    {
                        "status": "ok",
                        "results": [],
                        "filter_matched": 0,
                        "elapsed_ms": round(elapsed_ms),
                    },
                    headers=_cors_headers(),
                )
            # Vectorized mask: excluded indices get -inf, included keep score
            allowed_mask = np.zeros(len(scores), dtype=bool)
            for idx in allowed_indices:
                if idx < len(allowed_mask):
                    allowed_mask[idx] = True
            scores = np.where(allowed_mask, scores, -np.inf)

        # Keyword reranking
        STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
                      "and", "or", "for", "on", "it", "be", "as", "at", "by", "this",
                      "that", "with", "from", "not", "but", "has", "had", "have", "do",
                      "does", "did", "will", "can", "its", "who", "what", "how", "why"}
        keywords = [w for w in query.lower().split() if len(w) >= 3 and w not in STOP_WORDS]

        candidate_k = min(max(top_k * 4, 20), len(scores))
        candidate_idx = np.argsort(scores)[-candidate_k:][::-1]

        # Recency weighting: recent memories rank higher when similarity is close.
        # Uses timestamps from hippocampus metadata if available.
        # Formula: score * (0.5 + 0.5 * exp(-ln2 * age / half_life))
        # At age=0: 1.0, at half_life: 0.75, at infinity: 0.5 floor.
        _ln2 = math.log(2)
        now_ts = time.time()
        hippo_meta = state.get("hippocampus_meta")  # list of parsed header dicts
        has_recency = RECENCY_HALF_LIFE > 0 and hippo_meta is not None
        hippo = state.get("hippocampus")  # per-pattern structured meta: pattern_id, prev, next, flags

        boosted = []
        for i in candidate_idx:
            base_score = float(scores[i])
            if base_score < 0.05:
                continue
            # Skip tombstoned patterns (hippocampus flags bit 0x01 = TOMBSTONE)
            if hippo and i < len(hippo):
                flags = hippo[i].get("flags", 0) or 0
                if flags & 0x01:
                    continue
            text_lower = state["texts"][i].lower()
            hits = sum(1 for kw in keywords if kw in text_lower)
            boost = min(hits * 0.03, 0.12)
            final = base_score + boost
            # Apply recency weight
            if has_recency and i < len(hippo_meta):
                entry_ts = hippo_meta[i].get("timestamp", 0)
                if entry_ts > 0:
                    age = max(0.0, now_ts - entry_ts)
                    recency = 0.5 + 0.5 * math.exp(-_ln2 * age / RECENCY_HALF_LIFE)
                    final *= recency
            boosted.append((i, final))

        boosted.sort(key=lambda x: x[1], reverse=True)

        # Dedupe chunks that share a parent exchange. Long exchanges are chunked into
        # multiple patterns at cart-build time; each chunk is indexed independently
        # and all can match a query. Without dedup, top-k fills up with sibling chunks
        # of a single source exchange and wastes slots. Strategy: parse the tag prefix
        # for (url, turn); keep the highest-scoring chunk per source. Patterns without
        # a parseable prefix fall through (no dedup key = can't collide).
        seen_chunks = set()
        deduped = []
        dup_count = 0
        for i, score in boosted:
            text = state["texts"][i] if i < len(state["texts"]) else ""
            key = _chunk_dedup_key(text)
            if key is not None:
                if key in seen_chunks:
                    dup_count += 1
                    continue
                seen_chunks.add(key)
            deduped.append((i, score))
        if dup_count:
            log.info(f"[search] dedup removed {dup_count} sibling chunks")
        boosted = deduped

        elapsed_ms = (time.time() - t0) * 1000

        # Build structured results.
        # IMPORTANT: search returns the in-cart preview ONLY. The split-cart source
        # database is consulted on user demand via /api/passage, not here — keeps the
        # cart/source-DB distinction visible in the UX (every DB hit = a labeled user
        # action) and keeps search fast (no per-result SQLite query). The client gets
        # source_db so it knows the cart is split and can render the "load source"
        # CTA on the modal.
        is_split = bool(state.get("is_split_cart") and state.get("sqlite_conn"))
        source_db_label = (
            os.path.basename(state["sqlite_db_path"])
            if is_split and state.get("sqlite_db_path") else None
        )

        results = []
        for i, final_score in boosted[:top_k]:
            if final_score < 0.1:
                continue
            ram_text = state["texts"][i]
            # Extract tags if text starts with [TAGS] prefix (limit 300 chars for
            # URL-bearing tag prefixes used by Heartbeat).
            tags = ""
            body = ram_text
            if body.startswith("[") and "]" in body[:300]:
                tag_end = body.index("]")
                tags = body[1:tag_end]
                body = body[tag_end + 1:].strip()

            text = _soft_truncate(body)
            entry = {"text": text, "full_text": body, "score": round(final_score, 4), "tags": tags, "index": int(i)}
            # Provenance hint for split carts: client uses presence of source_db to
            # decide whether to render the "load source" CTA on the modal.
            if source_db_label:
                entry["source_db"] = source_db_label
            # Include hippocampus nav if available (hippo was hoisted earlier for tombstone check)
            if hippo and i < len(hippo):
                meta = hippo[i]
                log.info(f"[HIPPO-SEARCH] idx={i} pattern_id={meta['pattern_id']} prev_raw={meta['prev']} next_raw={meta['next']}")
                if meta["prev"] is not None:
                    entry["prev_idx"] = meta["prev"] - 1  # pattern_id 1-based → text 0-based
                if meta["next"] is not None:
                    entry["next_idx"] = meta["next"] - 1
            elif not hippo:
                log.info(f"[HIPPO-SEARCH] No hippocampus in session state!")
            else:
                log.info(f"[HIPPO-SEARCH] idx={i} out of range, hippo len={len(hippo)}")
            results.append(entry)

        state["query_count"] = state.get("query_count", 0) + 1
        log_label = "search+filter" if filter_matched is not None else "search"
        _log_activity(session_id, log_label, f"'{query[:40]}' -> {len(results)} results", elapsed_ms)

        resp = {"status": "ok", "results": results, "elapsed_ms": round(elapsed_ms)}
        if filter_matched is not None:
            resp["filter_matched"] = filter_matched
        return JSONResponse(resp, headers=_cors_headers())
    except Exception as e:
        log.error(f"REST /api/search error: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500, headers=_cors_headers())


# ---------------------------------------------------------------------------
# Filter API (shipped 2026-04-19 for Dennis's Phase 4 sprint + general use)
# ---------------------------------------------------------------------------
# Returns indices + text + tags + meta for patterns matching a filter spec.
# Filter spec supports:
#   tags_all: ["ns:val", ...]  AND semantics across tags
#   tags_any: ["ns:val", ...]  OR semantics within a namespace
#   meta:     {field: value}   exact-match predicates on per_pattern_meta fields
#   quality:  {predicate: N}   support_count_min / beat_dummy_pp_min /
#                              uplift_over_weights_only_min — read from
#                              per_pattern_meta fields of matching name
#   limit:    int              max results (default 500, cap 10000)
#
# See forum/canonical-tags.md for the tag namespace vocabulary.

def _extract_tags_from_text(text: str) -> list[str]:
    """Extract canonical + free-form tags from a passage's bracketed prefix.

    Supports both 'memory_store'-style `[tag1,tag2] content` and federate-v1
    `[machine/player game L0 event @timestamp]` headers (the latter becomes
    a single tag we then split on whitespace/slash for canonical matches).
    """
    if not text or not text.startswith("["):
        return []
    end = text.find("]")
    if end < 0 or end > 200:  # defensive: real tag blocks are short
        return []
    raw = text[1:end]
    pieces = [p.strip() for p in raw.replace("/", " ").replace(",", " ").split()]
    return [p for p in pieces if p]


def _match_tag(tag_entry: str, filter_tag: str) -> bool:
    """Match a single filter tag against a single entry tag.

    Handles namespace:value canonical form (colon-separated) plus plain
    free-form tags. Case-sensitive on both sides (canonical values are
    lowercase by convention).
    """
    # Exact match
    if tag_entry == filter_tag:
        return True
    # Namespace prefix match: filter "game:" matches entry "game:ft09"
    if filter_tag.endswith(":") and tag_entry.startswith(filter_tag):
        return True
    return False


def _chunk_dedup_key(text: str):
    """Extract (url, turn) from a tag-prefix for chunk-dedup.

    Long exchanges split into multiple patterns at cart-build, all sharing the
    same (url, turn) but differing by chunk index. Returning that composite key
    lets the search result path dedupe siblings.

    Handles two Heartbeat tag-prefix formats (mixed carts contain both):
      v24-comma: [HEARTBEAT,CLAUDE,turn-119,url=claude.ai/chat/abc]
      v25-pipe:  [Claude | Turn 119 | 2026-03-09T22:08:21.046Z | claude.ai/chat/abc]

    Returns None if no prefix or the required tokens are absent (falls through
    to no-dedup — safer than a false-positive collision).
    """
    if not text or not text.startswith("[") or "]" not in text[:300]:
        return None
    prefix = text[1:text.index("]")]
    turn = None
    url = None

    # v24 comma format — extract turn-N and url=X tokens
    if "," in prefix:
        for tok in (t.strip() for t in prefix.split(",")):
            if tok.startswith("turn-") and turn is None:
                turn = tok
            elif tok.startswith("url=") and url is None:
                url = tok[4:]

    # v25 pipe format — fallback if v24 parse didn't yield both
    if (turn is None or url is None) and "|" in prefix:
        tokens = [t.strip() for t in prefix.split("|")]
        for tok in tokens:
            # Match "Turn 119" (case-insensitive, whitespace-tolerant)
            if turn is None:
                low = tok.lower()
                if low.startswith("turn "):
                    parts = tok.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        turn = f"turn-{parts[1]}"
        # URL token: trailing token that looks like a domain path
        if url is None and tokens and "/" in tokens[-1] and "." in tokens[-1]:
            url = tokens[-1]

    if turn and url:
        return (url, turn)
    return None


# Known filter-spec keys; anything else at top level is logged as a warning
# so bad client shapes (e.g. `{"tags": {"any": [...]}}` — nested style) don't
# silently no-op. Doesn't reject — filter still runs on recognized keys only.
_FILTER_SPEC_KNOWN_KEYS = {"tags_all", "tags_any", "meta", "quality", "limit", "session_id"}


def _apply_filter(state: dict, spec: dict) -> list[int]:
    """Apply a filter spec to the mounted cart, return matching indices.

    Missing state or empty cart returns [].
    """
    if not state or not state.get("texts"):
        return []

    # Warn on unknown top-level keys (likely a client mis-shape, e.g. nested
    # `{"tags": {"any": [...]}}` when spec requires flat `{"tags_any": [...]}`).
    if isinstance(spec, dict):
        unknown = [k for k in spec.keys() if k not in _FILTER_SPEC_KNOWN_KEYS]
        if unknown:
            log.warning(
                f"[filter] unknown spec keys ignored: {unknown}. "
                f"Valid: {sorted(_FILTER_SPEC_KNOWN_KEYS)}. "
                f"See shared-context/forum/filter-api-spec.md for flat shape."
            )

    texts = state["texts"]
    meta_list = state.get("per_pattern_meta") or [{}] * len(texts)
    n = len(texts)

    tags_all = spec.get("tags_all") or []
    tags_any = spec.get("tags_any") or []
    meta_filter = spec.get("meta") or {}
    quality = spec.get("quality") or {}

    out = []
    for i in range(n):
        text = texts[i] if i < len(texts) else ""
        meta = meta_list[i] if i < len(meta_list) else {}

        # tags_all: every filter tag must match some entry tag
        if tags_all:
            entry_tags = _extract_tags_from_text(text)
            # Also expose meta fields as pseudo-tags (game:ft09 from meta.game=ft09)
            for k, v in meta.items():
                if isinstance(v, (str, int)) and v != "":
                    entry_tags.append(f"{k}:{v}")
            ok = all(any(_match_tag(et, ft) for et in entry_tags) for ft in tags_all)
            if not ok:
                continue

        # tags_any: at least one filter tag must match some entry tag
        if tags_any:
            entry_tags = _extract_tags_from_text(text)
            for k, v in meta.items():
                if isinstance(v, (str, int)) and v != "":
                    entry_tags.append(f"{k}:{v}")
            ok = any(any(_match_tag(et, ft) for et in entry_tags) for ft in tags_any)
            if not ok:
                continue

        # meta exact-match predicates
        if meta_filter:
            ok = True
            for field, expected in meta_filter.items():
                if meta.get(field) != expected:
                    ok = False
                    break
            if not ok:
                continue

        # quality predicates (router PRD names; absent field = pass, not fail,
        # so carts that don't yet track these don't get zero-filtered)
        if quality:
            ok = True
            sc_min = quality.get("support_count_min")
            if sc_min is not None and meta.get("support_count") is not None:
                if meta.get("support_count", 0) < sc_min:
                    ok = False
            bd_min = quality.get("beat_dummy_pp_min")
            if ok and bd_min is not None and meta.get("beat_dummy_pp") is not None:
                if meta.get("beat_dummy_pp", 0.0) < bd_min:
                    ok = False
            uw_min = quality.get("uplift_over_weights_only_min")
            if ok and uw_min is not None and meta.get("uplift_over_weights_only") is not None:
                if meta.get("uplift_over_weights_only", 0.0) < uw_min:
                    ok = False
            if not ok:
                continue

        out.append(i)

    return out


@mcp.custom_route("/api/filter", methods=["POST", "OPTIONS"])
async def rest_filter(request: Request) -> JSONResponse:
    """Filter cart entries by tags, metadata predicates, and/or quality gates.

    Returns matching entries (index + text + tags + meta) with no similarity
    ranking. Complement to /api/search: search ranks by relevance, filter
    enumerates by structural match.

    Request body:
        {
          "tags_all": ["game:ft09", "machine:cbp"],  // AND within tags_all
          "tags_any": ["game:ft09", "game:lf52"],     // OR within tags_any
          "meta":    {"event": "level_solved"},       // exact-match on meta fields
          "quality": {"support_count_min": 5, "beat_dummy_pp_min": 10.0},
          "limit":    500,
          "session_id": ""
        }

    Response:
        {"status": "ok", "count": N, "results": [{index, text, tags, meta}, ...]}
    """
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    try:
        data = await request.json()
        _log_rest(request, "FILTER", f"spec={list(data.keys())}")
        session_id = _resolve_session_id(data.get("session_id", ""))
        state = _get_session(session_id)

        if state["cartridge_name"] is None:
            return JSONResponse(
                {"status": "ok", "count": 0, "results": [], "error": "No cartridge mounted"},
                headers=_cors_headers(),
            )

        t0 = time.time()
        indices = _apply_filter(state, data)

        # Respect limit (default 500, cap 10000)
        limit = min(int(data.get("limit", 500)), 10000)
        indices = indices[:limit]

        # Build response entries
        texts = state["texts"]
        meta_list = state.get("per_pattern_meta") or [{}] * len(texts)
        out = []
        for i in indices:
            text = texts[i] if i < len(texts) else ""
            tags = _extract_tags_from_text(text)
            meta = meta_list[i] if i < len(meta_list) else {}
            out.append({
                "index": int(i),
                "text": text if len(text) <= 1200 else text[:1200] + "...",
                "full_text": text,
                "tags": tags,
                "meta": meta,
            })

        elapsed_ms = (time.time() - t0) * 1000
        _log_activity(session_id, "filter", f"spec -> {len(out)} results", elapsed_ms)

        return JSONResponse(
            {
                "status": "ok",
                "count": len(out),
                "total_scanned": len(texts),
                "results": out,
                "elapsed_ms": round(elapsed_ms),
            },
            headers=_cors_headers(),
        )
    except Exception as e:
        log.error(f"REST /api/filter error: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500, headers=_cors_headers())


@mcp.custom_route("/api/status", methods=["GET", "OPTIONS"])
async def rest_status(request: Request) -> JSONResponse:
    """REST health check — returns structured cartridge status."""
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    _log_rest(request, "STATUS")
    try:
        sid = request.query_params.get("session_id", "")
        sid = _resolve_session_id(sid)
        state = _get_session(sid)
        n = len(state["texts"]) if state["texts"] else 0
        import socket
        return JSONResponse({
            "status": "ok",
            "cartridge": state["cartridge_name"],
            "briefing": state.get("briefing"),  # Pattern 0 v2 Phase 1
            # Pattern 0 v2 Phase 2 — ownership block. owner_id None = legacy cart.
            "owner_id": state.get("owner_id"),
            "owner_perms": state.get("owner_perms"),
            "group_perms": state.get("group_perms"),
            "world_perms": state.get("world_perms"),
            "group_id": state.get("group_id"),
            "max_patterns": state.get("max_patterns"),
            # Mempack — cart_type and Pattern I presence flag
            "cart_type": state.get("cart_type"),
            "has_pattern_i": (
                state.get("cart_type") == CART_TYPE_AGENT_MEMORY
                and state.get("texts") is not None
                and len(state["texts"]) > PATTERN_I_IDX
            ),
            "memories": n,
            "gpu": _gpu_state["available"],
            "hamming": state["binary_corpus"] is not None,
            "search_mode": "hamming+embedding" if (state.get("has_embeddings") and state["binary_corpus"] is not None) else "hamming-only" if state["binary_corpus"] is not None else "embedding" if state.get("has_embeddings") else "none",
            "session_id": sid,
            "read_only": _server_config.get("read_only", False),
            "hostname": socket.gethostname(),
        }, headers=_cors_headers())
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500, headers=_cors_headers())


@mcp.custom_route("/api/sync", methods=["POST", "OPTIONS"])
async def rest_sync(request: Request) -> JSONResponse:
    """Heartbeat JSONL sync — append harvested exchanges to a local flat file.

    POST body: { "exchanges": [...], "machine": "hostname" }
    Each exchange: { turn, userMessage, assistantMessage, timestamp, url, platform, platformName }
    Appends to cartridges/heartbeat_<machine>.jsonl (one JSON object per line).
    """
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    try:
        data = await request.json()
        exchanges = data.get("exchanges", [])
        machine = re.sub(r'[^a-zA-Z0-9_\-]', '_', data.get("machine", "unknown"))

        if not exchanges:
            return JSONResponse({"status": "ok", "appended": 0}, headers=_cors_headers())

        # Ensure cartridges directory exists
        cart_dir = os.path.join(BASE_DIR, "cartridges")
        os.makedirs(cart_dir, exist_ok=True)

        sync_file = os.path.join(cart_dir, f"heartbeat_{machine}.jsonl")
        appended = 0

        with open(sync_file, "a", encoding="utf-8") as f:
            for ex in exchanges:
                line = json.dumps(ex, ensure_ascii=False)
                f.write(line + "\n")
                appended += 1

        file_size = os.path.getsize(sync_file)
        _log_rest(request, "SYNC", f"machine={machine} appended={appended} file_size={file_size}")

        return JSONResponse({
            "status": "ok",
            "appended": appended,
            "file": f"heartbeat_{machine}.jsonl",
            "file_size": file_size,
        }, headers=_cors_headers())
    except Exception as e:
        log.error(f"REST /api/sync error: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500, headers=_cors_headers())


@mcp.custom_route("/api/cartridges", methods=["GET", "OPTIONS"])
async def rest_cartridges(request: Request) -> JSONResponse:
    """List available cartridges on disk."""
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    try:
        carts = find_cartridges()
        return JSONResponse({
            "status": "ok",
            "cartridges": [
                {"name": c["name"], "size_mb": c["size_mb"], "format": c["format"],
                 "has_brain": c["has_brain"]}
                for c in carts
            ],
        }, headers=_cors_headers())
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500, headers=_cors_headers())


@mcp.custom_route("/api/mount", methods=["POST", "OPTIONS"])
async def rest_mount(request: Request) -> JSONResponse:
    """Mount a cartridge by name for the app session."""
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    try:
        data = await request.json()
        name = data.get("name", "")
        session_id = data.get("session_id", "")
        if not name:
            return JSONResponse({"status": "error", "error": "name required"}, status_code=400, headers=_cors_headers())
        _call = getattr(mount_cartridge, 'fn', mount_cartridge)
        result = _call(name=name, session_id=session_id)
        return JSONResponse({"status": "ok", "result": result}, headers=_cors_headers())
    except Exception as e:
        log.error(f"REST /api/mount error: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500, headers=_cors_headers())


@mcp.custom_route("/api/passage", methods=["GET", "OPTIONS"])
async def rest_passage(request: Request) -> JSONResponse:
    """REST endpoint to fetch any passage by index, with hippocampus nav links."""
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    try:
        idx = int(request.query_params.get("idx", -1))
        session_id = _resolve_session_id(request.query_params.get("session_id", ""))
        state = _get_session(session_id)

        if state["cartridge_name"] is None:
            return JSONResponse({"status": "error", "error": "No cartridge mounted"}, headers=_cors_headers())

        texts = state["texts"]
        if idx < 0 or idx >= len(texts):
            return JSONResponse({"status": "error", "error": f"Index {idx} out of range"}, headers=_cors_headers())

        # For split carts, fetch full passage from the SQLite source database.
        # This is the user-driven "load source" path: only hit on click, never as
        # a side effect of search. Falls back to in-cart preview when SQLite is
        # unavailable or the row isn't there.
        full_text = texts[idx]
        paper_id = None
        source_db = None
        if state.get("is_split_cart") and state.get("sqlite_conn"):
            sqlite_row = _sqlite_fetch_passages(state["sqlite_conn"], [int(idx)]).get(int(idx))
            if sqlite_row:
                full_text = sqlite_row.get("passage") or full_text
                paper_id = sqlite_row.get("paper_id")
            if state.get("sqlite_db_path"):
                source_db = os.path.basename(state["sqlite_db_path"])

        entry = {"index": idx, "full_text": full_text}
        if paper_id:
            entry["paper_id"] = paper_id
        if source_db:
            entry["source_db"] = source_db

        hippo = state.get("hippocampus")
        if hippo and idx < len(hippo):
            meta = hippo[idx]
            if meta["prev"] is not None:
                entry["prev_idx"] = meta["prev"] - 1
            if meta["next"] is not None:
                entry["next_idx"] = meta["next"] - 1

        return JSONResponse({"status": "ok", "passage": entry}, headers=_cors_headers())
    except Exception as e:
        log.error(f"REST /api/passage error: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500, headers=_cors_headers())


@mcp.custom_route("/api/embed", methods=["POST", "OPTIONS"])
async def rest_embed(request: Request) -> JSONResponse:
    """Local embedding via sentence-transformers — replaces Nomic API calls from browser extension."""
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    try:
        data = await request.json()
        texts = data.get("texts", [])
        task_type = data.get("task_type", "search_document")
        if not texts:
            return JSONResponse({"status": "error", "error": "No texts provided"}, status_code=400, headers=_cors_headers())

        # Map Nomic task_type to sentence-transformers prefix
        prefix = "search_query" if task_type == "search_query" else "search_document"

        embeddings = []
        for text in texts:
            vec = embed_text(text, prefix=prefix)
            embeddings.append(vec.tolist())

        _log_rest(request, "EMBED", f"n={len(texts)} task={task_type}")
        return JSONResponse({
            "status": "ok",
            "embeddings": embeddings,
            "model": "nomic-ai/nomic-embed-text-v1.5"
        }, headers=_cors_headers())
    except Exception as e:
        log.error(f"REST /api/embed error: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500, headers=_cors_headers())


@mcp.custom_route("/api/save", methods=["POST", "OPTIONS"])
async def rest_save(request: Request) -> JSONResponse:
    """REST wrapper for save_cartridge — persist to disk."""
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    try:
        _call = getattr(save_cartridge, 'fn', save_cartridge)
        result = _call()
        return JSONResponse({"status": "ok", "result": result}, headers=_cors_headers())
    except Exception as e:
        log.error(f"REST /api/save error: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500, headers=_cors_headers())


def _cors_headers():
    """CORS headers for browser extension access."""
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }


# ============================================================
# DEPOT DASHBOARD — Operator status board
# ============================================================

from starlette.responses import HTMLResponse


@mcp.custom_route("/depot/status", methods=["GET", "OPTIONS"])
async def depot_status(request: Request) -> JSONResponse:
    """JSON snapshot of all sessions, cartridges, and recent activity."""
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())

    now = time.time()
    uptime_sec = now - _depot_start_time

    # Sessions summary
    sessions_out = []
    for sid, s in _sessions.items():
        idle_sec = now - s["last_access"]
        if idle_sec > SESSION_TIMEOUT_SEC:
            status = "expired"
        elif s.get("last_action_time") and (now - s["last_action_time"]) < 10:
            status = "active"
        elif s["cartridge_name"] is None:
            status = "idle"
        else:
            status = "idle" if idle_sec > 30 else "active"
        sessions_out.append({
            "session_id": sid,
            "cartridge": s["cartridge_name"],
            "status": status,
            "idle_sec": round(idle_sec, 1),
            "query_count": s.get("query_count", 0),
            "mount_count": s.get("mount_count", 0),
            "last_action": s.get("last_action"),
            "created_at": s.get("created_at", s["last_access"]),
        })

    # Cartridge refcounts (how many sessions have each cart mounted)
    cart_refs = {}
    for s in _sessions.values():
        cn = s["cartridge_name"]
        if cn:
            if cn not in cart_refs:
                cart_refs[cn] = {"name": cn, "agents": 0, "entries": len(s["texts"]) if s["texts"] else 0}
            cart_refs[cn]["agents"] += 1

    # Available cartridges on disk
    disk_carts = []
    try:
        for c in find_cartridges():
            disk_carts.append({
                "name": c["name"],
                "size_mb": c.get("size_mb", 0),
                "mounted": c["name"] in cart_refs,
                "agents": cart_refs.get(c["name"], {}).get("agents", 0),
            })
    except Exception:
        pass

    # Recent activity (last 50)
    activity_out = []
    for entry in list(_depot_activity)[-50:]:
        activity_out.append({
            "time": entry["time"],
            "ago_sec": round(now - entry["time"], 1),
            "session": entry["session"],
            "action": entry["action"],
            "detail": entry["detail"],
            "latency_ms": entry["latency_ms"],
        })
    activity_out.reverse()  # newest first

    return JSONResponse({
        "uptime_sec": round(uptime_sec, 0),
        "active_sessions": len([s for s in sessions_out if s["status"] == "active"]),
        "total_sessions": len(sessions_out),
        "sessions": sessions_out,
        "cartridges": disk_carts,
        "mounted_carts": list(cart_refs.values()),
        "activity": activity_out,
        "gpu": _gpu_state["available"],
    }, headers=_cors_headers())


@mcp.custom_route("/depot", methods=["GET"])
async def depot_dashboard(request: Request) -> HTMLResponse:
    """Membot Depot — server rack dashboard."""
    return HTMLResponse(_DEPOT_HTML)


_DEPOT_HTML = r"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Membot Depot</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0a0e14; color: #c8d0da; font-family: 'Segoe UI', system-ui, sans-serif;
    font-size: 13px; overflow: hidden; height: 100vh;
  }
  .header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 20px; background: #111820; border-bottom: 1px solid #1e2a38;
  }
  .header h1 { font-size: 16px; font-weight: 600; letter-spacing: 1px; }
  .header h1 span { color: #3b82f6; }
  .header-stats { display: flex; gap: 20px; font-size: 12px; color: #6b7b8d; }
  .header-stats .val { color: #e2e8f0; font-weight: 600; }

  .panels { display: flex; flex-direction: column; height: calc(100vh - 49px); }

  /* Rack panel */
  .rack-panel {
    padding: 16px 20px; border-bottom: 1px solid #1e2a38;
    flex-shrink: 0; position: relative;
  }
  .rack-panel h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: #4a5568; margin-bottom: 12px; }
  .rack-grid { display: flex; flex-wrap: wrap; gap: 12px; }

  .cart-block {
    width: 130px; background: #141c26; border: 1px solid #1e2a38;
    border-radius: 6px; padding: 10px; cursor: pointer; transition: all 0.2s;
    position: relative;
  }
  .cart-block:hover { border-color: #3b82f6; background: #182030; }
  .cart-block.hot { border-color: #22c55e33; box-shadow: 0 0 12px #22c55e11; }
  .cart-block.cold { opacity: 0.5; }
  .cart-name { font-size: 12px; font-weight: 600; color: #e2e8f0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .cart-meta { font-size: 10px; color: #4a5568; margin-top: 2px; }
  .cart-leds { display: flex; gap: 4px; margin-top: 8px; min-height: 10px; }
  .led {
    width: 10px; height: 10px; border-radius: 50%; position: relative;
    cursor: pointer; transition: all 0.3s;
  }
  .led.active { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
  .led.idle { background: #eab308; box-shadow: 0 0 4px #eab30844; }
  .led.expired { background: #4a5568; }
  .led-tooltip {
    display: none; position: absolute; bottom: 18px; left: 50%; transform: translateX(-50%);
    background: #1e293b; border: 1px solid #334155; border-radius: 4px; padding: 4px 8px;
    font-size: 10px; white-space: nowrap; z-index: 10; color: #e2e8f0;
  }
  .led:hover .led-tooltip { display: block; }

  .empty-slot {
    width: 130px; height: 72px; border: 1px dashed #1e2a38; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; color: #2a3544;
  }

  /* Expand button on panels */
  .expand-btn {
    position: absolute; top: 8px; right: 12px; background: none; border: none;
    color: #4a5568; cursor: pointer; font-size: 14px; padding: 4px;
  }
  .expand-btn:hover { color: #e2e8f0; }

  /* Activity log */
  .activity-panel { flex: 1; overflow: hidden; display: flex; flex-direction: column; padding: 0 20px 12px; }
  .activity-panel h2 {
    font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px;
    color: #4a5568; padding: 12px 0 8px; flex-shrink: 0; position: relative;
  }
  .activity-list { flex: 1; overflow-y: auto; }
  .activity-list::-webkit-scrollbar { width: 4px; }
  .activity-list::-webkit-scrollbar-thumb { background: #1e2a38; border-radius: 2px; }

  .activity-row {
    display: grid; grid-template-columns: 70px 110px 60px 1fr 60px;
    gap: 8px; padding: 4px 0; font-size: 11px; font-family: 'Cascadia Code', 'Fira Code', monospace;
    border-bottom: 1px solid #0d1219;
  }
  .activity-row .time { color: #4a5568; }
  .activity-row .session { color: #3b82f6; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .activity-row .action { color: #22c55e; }
  .activity-row .action.mount { color: #eab308; }
  .activity-row .action.unmount { color: #ef4444; }
  .activity-row .detail { color: #8892a0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .activity-row .latency { color: #4a5568; text-align: right; }

  /* Detail pane (click cart or LED) */
  .detail-pane {
    display: none; position: fixed; right: 0; top: 0; width: 340px; height: 100vh;
    background: #111820; border-left: 1px solid #1e2a38; padding: 16px; z-index: 100;
    overflow-y: auto;
  }
  .detail-pane.open { display: block; }
  .detail-pane h3 { font-size: 14px; margin-bottom: 12px; }
  .detail-pane .close-btn {
    position: absolute; top: 12px; right: 12px; background: none; border: none;
    color: #6b7b8d; cursor: pointer; font-size: 16px;
  }
  .detail-field { margin-bottom: 8px; }
  .detail-field .label { font-size: 10px; text-transform: uppercase; color: #4a5568; }
  .detail-field .value { font-size: 13px; color: #e2e8f0; }

  /* Expanded panel */
  .panels.rack-expanded .rack-panel { flex: 1; overflow-y: auto; }
  .panels.rack-expanded .activity-panel { display: none; }
  .panels.activity-expanded .rack-panel { display: none; }
  .panels.activity-expanded .activity-panel { flex: 1; }

  /* No data state */
  .no-data { color: #2a3544; font-style: italic; padding: 20px; text-align: center; }
</style>
</head>
<body>

<div class="header">
  <h1><span>MEMBOT</span> DEPOT</h1>
  <div class="header-stats">
    <div><a href="#" onclick="showCartList();return false" style="color:#4fc3f7;text-decoration:none" title="View cartridge list">Carts: <span class="val" id="hdr-carts">0</span></a></div>
    <div>Sessions: <span class="val" id="hdr-sessions">0</span></div>
    <div>Queries: <span class="val" id="hdr-queries">0</span></div>
    <div>GPU: <span class="val" id="hdr-gpu">—</span></div>
    <div>Uptime: <span class="val" id="hdr-uptime">—</span></div>
  </div>
</div>

<div class="panels" id="panels">
  <div class="rack-panel" id="rack-panel">
    <h2>Cartridge Rack</h2>
    <button class="expand-btn" onclick="toggleExpand('rack')" title="Expand">&#x2922;</button>
    <div class="rack-grid" id="rack-grid"></div>
  </div>

  <div class="activity-panel" id="activity-panel">
    <h2>
      Recent Activity
      <button class="expand-btn" onclick="toggleExpand('activity')" title="Expand">&#x2922;</button>
    </h2>
    <div class="activity-list" id="activity-list"></div>
  </div>
</div>

<div class="detail-pane" id="detail-pane">
  <button class="close-btn" onclick="closeDetail()">&#x2715;</button>
  <div id="detail-content"></div>
</div>

<script>
const POLL_MS = 2000;
let lastData = null;

function fmt_time(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString('en-US', {hour12: false, hour:'2-digit', minute:'2-digit', second:'2-digit'});
}

function fmt_uptime(sec) {
  const d = Math.floor(sec / 86400);
  const h = Math.floor((sec % 86400) / 3600);
  const m = Math.floor((sec % 3600) / 60);
  if (d > 0) return d + 'd ' + h + 'h';
  if (h > 0) return h + 'h ' + m + 'm';
  return m + 'm';
}

function fmt_ago(sec) {
  if (sec < 5) return '<5s';
  if (sec < 60) return Math.round(sec) + 's';
  if (sec < 3600) return Math.round(sec/60) + 'm';
  return Math.round(sec/3600) + 'h';
}

function toggleExpand(panel) {
  const el = document.getElementById('panels');
  if (el.classList.contains(panel + '-expanded')) {
    el.className = 'panels';
  } else {
    el.className = 'panels ' + panel + '-expanded';
  }
}

function closeDetail() {
  document.getElementById('detail-pane').classList.remove('open');
}

function showCartList() {
  const pane = document.getElementById('detail-pane');
  const content = document.getElementById('detail-content');
  const carts = lastData?.cartridges || [];
  const sessions = lastData?.sessions || [];
  const totalMB = carts.reduce((a, c) => a + (parseFloat(c.size_mb) || 0), 0).toFixed(1);
  content.innerHTML = '<h3>Available Cartridges (' + carts.length + ')</h3>' +
    '<div style="color:#8892a0;font-size:12px;margin-bottom:12px">Total: ' + totalMB + ' MB on disk</div>' +
    (carts.length === 0 ? '<div class="no-data">No cartridges found</div>' :
      carts.map(c => {
        const agents = sessions.filter(s => s.cartridge === c.name);
        const status = c.mounted ? '<span style="color:#4caf50">mounted (' + agents.length + ' agent' + (agents.length !== 1 ? 's' : '') + ')</span>' : '<span style="color:#4a5568">on disk</span>';
        return '<div style="margin:6px 0;padding:8px;background:#0d1219;border-radius:4px;cursor:pointer" onclick="showCartDetail(' + esc_attr(JSON.stringify(c)) + ')">' +
          '<div style="display:flex;justify-content:space-between;align-items:center">' +
            '<span style="color:#e2e8f0;font-weight:600">' + esc(c.name) + '</span>' +
            '<span style="color:#8892a0;font-size:11px">' + c.size_mb + ' MB</span>' +
          '</div>' +
          '<div style="font-size:11px;margin-top:4px">' + status + '</div>' +
        '</div>';
      }).join('')
    );
  pane.classList.add('open');
}

function showCartDetail(cart) {
  const pane = document.getElementById('detail-pane');
  const content = document.getElementById('detail-content');
  // Find sessions on this cart
  const sessions = (lastData?.sessions || []).filter(s => s.cartridge === cart.name);
  content.innerHTML = '<h3>' + esc(cart.name) + '</h3>' +
    field('Size', cart.size_mb + ' MB') +
    field('Agents', cart.agents) +
    field('Status', cart.mounted ? 'Mounted' : 'On disk') +
    '<h3 style="margin-top:16px">Connected Agents</h3>' +
    (sessions.length === 0 ? '<div class="no-data">None</div>' :
      sessions.map(s =>
        '<div style="margin:8px 0;padding:8px;background:#0d1219;border-radius:4px">' +
          field('Session', s.session_id) +
          field('Queries', s.query_count) +
          field('Idle', fmt_ago(s.idle_sec)) +
          field('Last', s.last_action || '—') +
        '</div>'
      ).join('')
    );
  pane.classList.add('open');
}

function showSessionDetail(session) {
  const pane = document.getElementById('detail-pane');
  const content = document.getElementById('detail-content');
  const activities = (lastData?.activity || []).filter(a => a.session === session.session_id).slice(0, 20);
  content.innerHTML = '<h3>' + esc(session.session_id) + '</h3>' +
    field('Cartridge', session.cartridge || 'None') +
    field('Status', session.status) +
    field('Queries', session.query_count) +
    field('Mounts', session.mount_count) +
    field('Idle', fmt_ago(session.idle_sec)) +
    field('Last Action', session.last_action || '—') +
    '<h3 style="margin-top:16px">Recent Activity</h3>' +
    (activities.length === 0 ? '<div class="no-data">None</div>' :
      activities.map(a =>
        '<div style="font-size:11px;font-family:monospace;padding:2px 0;color:#8892a0">' +
          fmt_time(a.time) + ' ' + a.action + ' ' + esc(a.detail) +
          (a.latency_ms > 0 ? ' <span style="color:#4a5568">' + a.latency_ms + 'ms</span>' : '') +
        '</div>'
      ).join('')
    );
  pane.classList.add('open');
}

function field(label, val) {
  return '<div class="detail-field"><div class="label">' + label + '</div><div class="value">' + esc(String(val)) + '</div></div>';
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function render(data) {
  lastData = data;

  // Header
  const carts_all = data.cartridges || [];
  document.getElementById('hdr-carts').textContent = carts_all.length;
  document.getElementById('hdr-sessions').textContent = data.active_sessions + ' active';
  const totalQ = (data.sessions || []).reduce((a, s) => a + s.query_count, 0);
  document.getElementById('hdr-queries').textContent = totalQ;
  document.getElementById('hdr-gpu').textContent = data.gpu ? 'Ready' : 'CPU only';
  document.getElementById('hdr-uptime').textContent = fmt_uptime(data.uptime_sec);

  // Rack grid — one block per disk cartridge + empty slots
  const grid = document.getElementById('rack-grid');
  const carts = data.cartridges || [];
  const sessions = data.sessions || [];

  let html = '';
  for (const cart of carts) {
    const agents = sessions.filter(s => s.cartridge === cart.name);
    const isHot = agents.some(a => a.status === 'active');
    const cls = cart.mounted ? (isHot ? 'hot' : '') : 'cold';
    html += '<div class="cart-block ' + cls + '" onclick="showCartDetail(' + esc_attr(JSON.stringify(cart)) + ')">';
    html += '<div class="cart-name">' + esc(cart.name) + '</div>';
    html += '<div class="cart-meta">' + cart.size_mb + ' MB</div>';
    html += '<div class="cart-leds">';
    for (const agent of agents) {
      html += '<div class="led ' + agent.status + '" onclick="event.stopPropagation();showSessionDetail(' + esc_attr(JSON.stringify(agent)) + ')">';
      html += '<div class="led-tooltip">' + esc(agent.session_id) + '<br>' + (agent.last_action || 'idle') + '</div>';
      html += '</div>';
    }
    html += '</div></div>';
  }
  // Empty slots to fill the row
  const empties = Math.max(0, 6 - carts.length);
  for (let i = 0; i < empties; i++) {
    html += '<div class="empty-slot">empty</div>';
  }
  grid.innerHTML = html;

  // Activity log
  const actList = document.getElementById('activity-list');
  let actHtml = '';
  const acts = data.activity || [];
  if (acts.length === 0) {
    actHtml = '<div class="no-data">No activity yet — agents will appear here when they connect</div>';
  }
  for (const a of acts) {
    const actionCls = a.action === 'mount' ? 'mount' : a.action === 'unmount' ? 'unmount' : '';
    actHtml += '<div class="activity-row">';
    actHtml += '<div class="time">' + fmt_time(a.time) + '</div>';
    actHtml += '<div class="session">' + esc(a.session) + '</div>';
    actHtml += '<div class="action ' + actionCls + '">' + a.action + '</div>';
    actHtml += '<div class="detail">' + esc(a.detail) + '</div>';
    actHtml += '<div class="latency">' + (a.latency_ms > 0 ? a.latency_ms + 'ms' : '') + '</div>';
    actHtml += '</div>';
  }
  actList.innerHTML = actHtml;
}

function esc_attr(s) {
  return s.replace(/&/g,'&amp;').replace(/'/g,'&#39;').replace(/"/g,'&quot;').replace(/</g,'&lt;');
}

async function poll() {
  try {
    const base = location.pathname.replace(/\/depot\/?$/, '');
    const resp = await fetch(base + '/depot/status');
    if (resp.ok) {
      const data = await resp.json();
      render(data);
    }
  } catch (e) {
    // Network error — will retry next cycle
  }
}

// Initial load + polling
poll();
setInterval(poll, POLL_MS);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# App frontend — user-facing search & store interface
# ---------------------------------------------------------------------------

# Load icon at import time (once)
_icon_path = os.path.join(BASE_DIR, "membot-icon.png")
if os.path.exists(_icon_path):
    _app_icon_b64 = base64.b64encode(open(_icon_path, "rb").read()).decode()
else:
    _app_icon_b64 = ""

@mcp.custom_route("/app", methods=["GET"])
async def app_frontend(request: Request) -> HTMLResponse:
    """Membot App — user-facing search & store interface."""
    html = _APP_HTML.replace("{{ICON_DATA_URI}}",
        f"data:image/png;base64,{_app_icon_b64}" if _app_icon_b64 else "")
    return HTMLResponse(html)

_APP_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Membot — Brain Cartridge Memory</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :root {
    --bg: #0c1017; --surface: #131a24; --surface-2: #1a2332;
    --border: #1e2d3d; --border-hover: #2a3f54;
    --text: #d1dae6; --text-dim: #6b7f95; --text-bright: #f0f4f8;
    --accent: #3b82f6; --accent-dim: #2563eb; --accent-glow: #3b82f620;
    --green: #22c55e; --green-dim: #16a34a; --green-glow: #22c55e18;
    --amber: #eab308; --red: #ef4444;
    --mono: 'Cascadia Code','Fira Code','JetBrains Mono','Consolas',monospace;
    --sans: 'Inter','Segoe UI',system-ui,-apple-system,sans-serif;
  }
  [data-theme="light"] {
    --bg: #f5f7fa; --surface: #ffffff; --surface-2: #f0f2f5;
    --border: #e2e8f0; --border-hover: #cbd5e1;
    --text: #334155; --text-dim: #94a3b8; --text-bright: #0f172a;
    --accent: #3b82f6; --accent-dim: #2563eb; --accent-glow: #3b82f615;
    --green: #16a34a; --green-dim: #15803d; --green-glow: #22c55e12;
    --amber: #ca8a04; --red: #dc2626;
  }
  body { background:var(--bg); color:var(--text); font-family:var(--sans); font-size:14px; line-height:1.6; min-height:100vh; }
  .app { max-width:960px; margin:0 auto; padding:20px; }
  .brand { display:flex; align-items:center; gap:14px; padding:24px 0 20px; border-bottom:1px solid var(--border); margin-bottom:24px; }
  .brand-icon { width:48px; height:48px; border-radius:12px; overflow:hidden; flex-shrink:0; box-shadow:0 4px 20px #3b82f630; }
  .brand-icon img { width:100%; height:100%; object-fit:cover; }
  .brand-text h1 { font-size:22px; font-weight:700; color:var(--text-bright); letter-spacing:-0.5px; }
  .brand-text h1 span { color:var(--accent); }
  .brand-text p { font-size:12px; color:var(--text-dim); margin-top:-2px; }
  .intro { font-size:13px; color:var(--text-dim); line-height:1.5; margin:-8px 0 18px; }
  .intro strong { color:var(--text); font-weight:600; }
  .theme-toggle { background:var(--surface); border:1px solid var(--border); color:var(--text-dim); width:36px; height:36px; border-radius:8px; cursor:pointer; font-size:18px; display:flex; align-items:center; justify-content:center; transition:all 0.2s; flex-shrink:0; margin-left:auto; }
  .theme-toggle:hover { border-color:var(--border-hover); color:var(--text); }
  [data-theme="light"] .result-card, [data-theme="light"] .cart-chip { box-shadow:0 1px 3px rgba(0,0,0,0.08); }
  .status-bar { display:flex; align-items:center; gap:8px; font-size:12px; color:var(--text-dim); }
  .status-dot { width:10px; height:10px; border-radius:50%; background:var(--text-dim); transition:all 0.3s; }
  .status-dot.connected { background:var(--green); box-shadow:0 0 8px var(--green-glow); }
  .status-dot.error { background:var(--red); }
  .cart-bar { display:flex; gap:10px; margin-bottom:20px; overflow-x:auto; padding-bottom:4px; align-items:center; }
  .cart-chip { flex-shrink:0; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:10px 16px; min-width:140px; transition:all 0.2s; cursor:pointer; }
  .cart-chip:hover { border-color:var(--border-hover); transform:translateY(-1px); }
  .cart-chip.active { border-color:var(--accent); background:var(--accent-glow); cursor:default; }
  .cart-chip.mounting { opacity:0.6; pointer-events:none; }
  .cart-chip .name { font-size:13px; font-weight:600; color:var(--text-bright); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .cart-chip .meta { font-size:11px; color:var(--text-dim); margin-top:2px; }
  .cart-chip .meta .count { color:var(--accent); font-weight:600; }
  .search-box { position:relative; display:flex; gap:8px; margin-bottom:8px; }
  .search-box input { flex:1; background:var(--surface); border:1px solid var(--border); color:var(--text-bright); font-size:15px; padding:12px 16px 12px 42px; border-radius:10px; outline:none; transition:all 0.2s; font-family:var(--sans); }
  .search-box input:focus { border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-glow); }
  .search-box input::placeholder { color:var(--text-dim); }
  .search-icon { position:absolute; left:14px; top:50%; transform:translateY(-50%); color:var(--text-dim); font-size:16px; pointer-events:none; }
  .search-box button { background:var(--accent); color:white; border:none; padding:12px 24px; border-radius:10px; font-size:14px; font-weight:600; cursor:pointer; transition:all 0.2s; white-space:nowrap; }
  .search-box button:hover { background:var(--accent-dim); }
  .search-meta { display:flex; justify-content:space-between; font-size:12px; color:var(--text-dim); margin-bottom:16px; }
  .results { display:flex; flex-direction:column; gap:12px; }
  .result-card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:16px 20px; transition:all 0.2s; }
  .result-card:hover { border-color:var(--border-hover); }
  .result-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
  .result-rank { font-size:11px; font-weight:700; color:var(--accent); background:var(--accent-glow); padding:2px 8px; border-radius:4px; font-family:var(--mono); }
  .result-score { font-size:11px; color:var(--text-dim); font-family:var(--mono); }
  .result-text { font-size:13px; line-height:1.7; color:var(--text); white-space:pre-wrap; word-break:break-word; }
  .result-text mark { background:#eab30830; color:var(--amber); border-radius:2px; padding:0 2px; }
  .result-footer { margin-top:10px; padding-top:8px; border-top:1px solid var(--border); display:flex; gap:12px; font-size:11px; color:var(--text-dim); font-family:var(--mono); }
  .result-footer .tag { background:var(--surface-2); padding:1px 6px; border-radius:3px; }
  .result-footer .prov { margin-left:auto; font-size:10px; opacity:0.7; font-style:italic; }
  .result-text a, .passage-text a { color:var(--accent); text-decoration:none; border-bottom:1px dotted var(--accent); }
  .result-text a:hover, .passage-text a:hover { border-bottom-style:solid; }
  .passage-modal-source { padding:8px 20px; border-top:1px solid var(--border); font-size:11px; color:var(--text-dim); font-family:var(--mono); flex-shrink:0; }
  .passage-modal-cta { padding:12px 20px; border-top:1px solid var(--border); text-align:center; flex-shrink:0; }
  .modal-cta-btn { background:var(--accent); color:white; border:none; padding:10px 18px; border-radius:6px; cursor:pointer; font-size:13px; font-weight:600; font-family:var(--sans); transition:background 0.2s; }
  .modal-cta-btn:hover { background:var(--accent-dim); }
  .modal-cta-btn:disabled { opacity:0.6; cursor:wait; }
  .modal-cta-loading { font-size:12px; color:var(--text-dim); font-style:italic; }
  .result-card { cursor:pointer; }
  .passage-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.7); z-index:900; display:flex; align-items:center; justify-content:center; opacity:0; transition:opacity 0.2s; pointer-events:none; }
  .passage-overlay.open { opacity:1; pointer-events:auto; }
  .passage-modal { background:var(--bg); border:1px solid var(--border); border-radius:14px; width:min(800px,90vw); max-height:80vh; display:flex; flex-direction:column; box-shadow:0 24px 80px #00000080; }
  .passage-modal-header { display:flex; justify-content:space-between; align-items:center; padding:16px 20px; border-bottom:1px solid var(--border); flex-shrink:0; }
  .passage-modal-header .modal-rank { font-size:12px; font-weight:700; color:var(--accent); font-family:var(--mono); }
  .passage-modal-header .modal-score { font-size:12px; color:var(--text-dim); font-family:var(--mono); margin-left:12px; }
  .passage-modal-header .modal-close { background:none; border:1px solid var(--border); color:var(--text-dim); width:32px; height:32px; border-radius:6px; cursor:pointer; font-size:18px; display:flex; align-items:center; justify-content:center; transition:all 0.2s; }
  .passage-modal-header .modal-close:hover { border-color:var(--red); color:var(--red); background:rgba(239,68,68,0.1); }
  .passage-modal-body { padding:20px; overflow-y:auto; flex:1; }
  .passage-modal-body .passage-text { font-size:14px; line-height:1.8; color:var(--text); white-space:pre-wrap; word-break:break-word; }
  .passage-modal-body .passage-text mark { background:#eab30830; color:var(--amber); border-radius:2px; padding:0 2px; }
  .passage-modal-footer { padding:12px 20px; border-top:1px solid var(--border); display:flex; justify-content:center; gap:8px; flex-shrink:0; }
  .passage-nav-btn { background:var(--surface); border:1px solid var(--border); color:var(--text-dim); padding:6px 16px; border-radius:6px; font-size:12px; cursor:not-allowed; opacity:0.4; transition:all 0.2s; font-family:var(--mono); }
  .passage-nav-btn.enabled { cursor:pointer; opacity:1; }
  .passage-nav-btn.enabled:hover { border-color:var(--accent); color:var(--accent); }
  .store-section { margin-top:32px; padding-top:24px; border-top:1px solid var(--border); }
  .store-section h2 { font-size:14px; font-weight:600; color:var(--text-bright); margin-bottom:12px; }
  .store-box textarea { width:100%; background:var(--surface); border:1px solid var(--border); color:var(--text); font-size:13px; font-family:var(--sans); padding:12px 16px; border-radius:10px; resize:vertical; min-height:100px; outline:none; transition:border-color 0.2s; margin-bottom:8px; }
  .store-box textarea:focus { border-color:var(--green); }
  .store-box textarea::placeholder { color:var(--text-dim); }
  .store-row { display:flex; gap:8px; align-items:center; }
  .store-row input { flex:1; background:var(--surface); border:1px solid var(--border); color:var(--text); font-size:12px; padding:8px 12px; border-radius:8px; outline:none; font-family:var(--mono); }
  .store-row input:focus { border-color:var(--green); }
  .store-row input::placeholder { color:var(--text-dim); }
  .store-row button { background:var(--green); color:white; border:none; padding:8px 20px; border-radius:8px; font-size:13px; font-weight:600; cursor:pointer; transition:all 0.2s; }
  .store-row button:hover { background:var(--green-dim); }
  .toast { position:fixed; bottom:24px; right:24px; background:var(--surface-2); border:1px solid var(--border); border-radius:8px; padding:12px 20px; font-size:13px; color:var(--text); box-shadow:0 8px 32px #00000060; transform:translateY(80px); opacity:0; transition:all 0.3s; z-index:1000; }
  .toast.show { transform:translateY(0); opacity:1; }
  .toast.success { border-left:3px solid var(--green); }
  .toast.error { border-left:3px solid var(--red); }
  .empty-state { text-align:center; padding:60px 20px; color:var(--text-dim); }
  .empty-state .icon { font-size:48px; margin-bottom:12px; opacity:0.3; }
  .loading { display:none; text-align:center; padding:40px; color:var(--text-dim); }
  .loading.active { display:block; }
  .spinner { display:inline-block; width:20px; height:20px; border:2px solid var(--border); border-top-color:var(--accent); border-radius:50%; animation:spin 0.8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .app-footer { margin-top:40px; padding:20px 0; border-top:1px solid var(--border); text-align:center; font-size:11px; color:var(--text-dim); }
  .app-footer a { color:var(--accent); text-decoration:none; }
  .app-footer a:hover { text-decoration:underline; }
  @media (max-width:640px) { .app { padding:12px; } .brand { flex-wrap:wrap; } .search-box { flex-direction:column; } .search-box button { width:100%; } }
</style>
</head>
<body>
<div class="app">
  <div class="brand">
    <div class="brand-icon"><img src="{{ICON_DATA_URI}}" alt="Membot"></div>
    <div class="brand-text">
      <h1>Mem<span>bot</span></h1>
      <p>Brain Cartridge Memory System</p>
    </div>
    <button class="theme-toggle" id="themeBtn" onclick="toggleTheme()" title="Toggle light/dark">&#x2600;</button>
    <div class="status-bar" id="statusBar">
      <div class="status-dot" id="statusDot"></div>
      <span id="statusText">Connecting...</span>
    </div>
  </div>
  <p class="intro"><strong>Self-contained semantic search without the LLM.</strong> Click a cart to mount it, then search with plain English &mdash; no generation, no hallucinations.</p>
  <div class="cart-bar" id="cartBar"><div class="cart-chip"><div class="name" style="color:var(--text-dim)">Loading...</div></div></div>
  <div class="search-box">
    <span class="search-icon">&#x1F50D;</span>
    <input type="text" id="searchInput" placeholder="Search your memory..." onkeydown="if(event.key==='Enter')doSearch()">
    <button onclick="doSearch()">Search</button>
  </div>
  <div class="search-meta" id="searchMeta"></div>
  <div class="loading" id="loadingEl"><div class="spinner"></div> Searching...</div>
  <div class="results" id="resultsEl">
    <div class="empty-state"><div class="icon">&#x1F9E0;</div><p>Search your brain cartridge</p><div class="hint" style="font-size:12px;margin-top:8px">Type a query to find memories by meaning</div></div>
  </div>
  <div class="passage-overlay" id="passageOverlay" onclick="if(event.target===this)closePassage()">
    <div class="passage-modal">
      <div class="passage-modal-header">
        <div><span class="modal-rank" id="modalRank"></span><span class="modal-score" id="modalScore"></span></div>
        <button class="modal-close" onclick="closePassage()" title="Close">&#x2715;</button>
      </div>
      <div class="passage-modal-body"><div class="passage-text" id="modalText"></div></div>
      <div class="passage-modal-cta" id="modalCta" style="display:none"></div>
      <div class="passage-modal-source" id="modalSource" style="display:none"></div>
      <div class="passage-modal-footer">
        <button class="passage-nav-btn" id="btnPrev" title="Previous chunk (when metadata wired)">&#x25C0; Prev</button>
        <button class="passage-nav-btn" id="btnNext" title="Next chunk (when metadata wired)">Next &#x25B6;</button>
      </div>
    </div>
  </div>
  <div class="store-section">
    <h2>Store a Memory</h2>
    <div class="store-box">
      <textarea id="storeContent" placeholder="Paste or type content to store..." rows="4"></textarea>
      <div class="store-row">
        <input type="text" id="storeTags" placeholder="Tags (optional): ARCHITECTURE, JOURNAL, ...">
        <button onclick="doStore()">Store</button>
      </div>
    </div>
  </div>
  <div class="app-footer">Membot &mdash; Neuromorphic memory for AI agents &mdash; <a href="https://github.com/project-you-apps/membot">GitHub</a></div>
</div>
<div class="toast" id="toastEl"></div>
<script>
const $=s=>document.querySelector(s);
let _mounted=null, _memories=0;
function applyReadOnly(ro){
  const ta=$('#storeContent'),btn=document.querySelector('.store-row button'),ti=$('#storeTags');
  if(ro){
    if(ta){ta.value='THIS SERVICE NOT YET AVAILABLE';ta.disabled=true;ta.style.opacity='0.5';}
    if(ti){ti.disabled=true;ti.style.opacity='0.5';}
    if(btn){btn.disabled=true;btn.style.opacity='0.5';btn.style.cursor='not-allowed';}
  }else{
    if(ta){ta.value='';ta.disabled=false;ta.style.opacity='1';}
    if(ti){ti.disabled=false;ti.style.opacity='1';}
    if(btn){btn.disabled=false;btn.style.opacity='1';btn.style.cursor='pointer';}
  }
}
const BASE=()=>{
  const loc=location.pathname.replace(/\\/app\\/?$/,'');
  return location.protocol==='file:'?'http://137.184.227.79:8000':location.origin+loc;
};
async function checkStatus(){
  const dot=$('#statusDot'),txt=$('#statusText');
  try{
    const r=await fetch(BASE()+'/api/status',{signal:AbortSignal.timeout(5000)});
    const d=await r.json();
    dot.className='status-dot connected';
    _mounted=d.cartridge; _memories=d.memories||0;
    txt.textContent=(_mounted||'No cart')+' ('+_memories.toLocaleString()+' memories)';
    applyReadOnly(d.read_only);
  }catch(e){ dot.className='status-dot error'; txt.textContent='Disconnected'; }
}
async function loadCartridges(){
  const bar=$('#cartBar');
  try{
    const [cr,sr]=await Promise.all([
      fetch(BASE()+'/api/cartridges',{signal:AbortSignal.timeout(5000)}).then(r=>r.json()),
      fetch(BASE()+'/api/status',{signal:AbortSignal.timeout(5000)}).then(r=>r.json())
    ]);
    _mounted=sr.cartridge; _memories=sr.memories||0;
    const dot=$('#statusDot'),txt=$('#statusText');
    dot.className='status-dot connected';
    txt.textContent=(_mounted||'No cart')+' ('+_memories.toLocaleString()+' memories)';
    applyReadOnly(sr.read_only);
    const carts=cr.cartridges||[];
    if(carts.length===0){bar.innerHTML='<div class="cart-chip"><div class="name" style="color:var(--text-dim)">No cartridges found</div></div>';return;}
    bar.innerHTML=carts.map(c=>{
      const active=_mounted&&c.name===_mounted;
      const cls='cart-chip'+(active?' active':'');
      const mem=active?' &middot; <span class="count">'+_memories.toLocaleString()+' memories</span>':'';
      return '<div class="'+cls+'" data-name="'+esc(c.name)+'"><div class="name">'+esc(c.name)+'</div><div class="meta">'+c.size_mb+' MB &middot; '+c.format.toUpperCase()+(c.has_brain?' &middot; GPU':'')+mem+'</div></div>';
    }).join('');
    bar.querySelectorAll('.cart-chip').forEach(el=>el.addEventListener('click',()=>mountCart(el.dataset.name)));
  }catch(e){bar.innerHTML='<div class="cart-chip"><div class="name" style="color:var(--red)">Connection failed</div></div>';}
}
async function mountCart(name){
  if(name===_mounted)return;
  $('#resultsEl').innerHTML='<div class="empty-state"><div class="icon">&#x1F9E0;</div><p>Search your brain cartridge</p><div class="hint" style="font-size:12px;margin-top:8px">Type a query to find memories by meaning</div></div>';
  $('#searchMeta').textContent=''; _lastResults=[]; _lastQuery='';
  const chips=document.querySelectorAll('.cart-chip');
  chips.forEach(c=>{if(c.dataset.name===name)c.classList.add('mounting');});
  toast('Mounting '+name+'...');
  try{
    const r=await fetch(BASE()+'/api/mount',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});
    const d=await r.json();
    if(d.error){toast(d.error,'error');}
    else{toast('Mounted '+name,'success');}
    await loadCartridges();
  }catch(e){toast('Mount failed: '+e.message,'error');chips.forEach(c=>c.classList.remove('mounting'));}
}
var _lastResults=[], _lastQuery='';
async function doSearch(){
  const query=$('#searchInput').value.trim();
  if(!query)return;
  _lastQuery=query;
  const el=$('#resultsEl'),loading=$('#loadingEl'),meta=$('#searchMeta');
  el.innerHTML=''; loading.className='loading active'; meta.textContent='';
  const t0=performance.now();
  try{
    const r=await fetch(BASE()+'/api/search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query,top_k:8})});
    const data=await r.json();
    const elapsed=Math.round(performance.now()-t0);
    loading.className='loading';
    var items=data.results;
    if(!items&&typeof data.result==='string'){
      items=[];
      var lines=data.result.split(/\\n\\n/);
      for(var li=0;li<lines.length;li++){
        var m=lines[li].match(/^#\\d+\\s+\\[([\\d.]+)\\]\\s+(.*)/s);
        if(m)items.push({score:parseFloat(m[1]),text:m[2],full_text:m[2],tags:''});
      }
    }
    _lastResults=items||[];
    console.log('[HIPPO] Search results:', JSON.stringify(items?.map(r=>({idx:r.index,prev:r.prev_idx,next:r.next_idx}))));
    const haystack=_memories?' of '+_memories.toLocaleString():'';
    if(!items||items.length===0){
      el.innerHTML='<div class="empty-state"><div class="icon">&#x1F914;</div><p>No results found</p></div>';
      meta.textContent='0 results'+haystack+' in '+elapsed+'ms'; return;
    }
    meta.textContent=items.length+haystack+' results in '+elapsed+'ms';
    el.innerHTML=items.map((item,i)=>{
      const text=esc(item.text||item.content||'');
      const score=item.score!=null?item.score.toFixed(4):'';
      const tags=item.tags?item.tags.split(',').map(t=>'<span class="tag">'+esc(t.trim())+'</span>').join(''):'';
      const provBits=[];
      if(item.source_db)provBits.push(esc(item.source_db));
      if(item.paper_id)provBits.push('id: '+esc(item.paper_id));
      const prov=provBits.length?'<span class="prov">'+provBits.join(' · ')+'</span>':'';
      return '<div class="result-card" data-idx="'+i+'"><div class="result-header"><span class="result-rank">#'+(i+1)+'</span>'+(score?'<span class="result-score">score: '+score+'</span>':'')+'</div><div class="result-text">'+highlight(linkify(text),query)+'</div>'+((tags||prov)?'<div class="result-footer">'+tags+prov+'</div>':'')+'</div>';
    }).join('');
    el.querySelectorAll('.result-card').forEach(c=>c.addEventListener('click',()=>openPassage(parseInt(c.dataset.idx))));
  }catch(e){ loading.className='loading'; el.innerHTML='<div class="empty-state"><div class="icon">&#x26A0;</div><p>Search failed</p><div style="font-size:12px;margin-top:8px">'+esc(e.message)+'</div></div>'; }
}
var _currentPassage=null;
function openPassage(resultIdx){
  if(resultIdx<0||resultIdx>=_lastResults.length)return;
  const item=_lastResults[resultIdx];
  console.log('[HIPPO] openPassage resultIdx='+resultIdx+' item.index='+item.index+' prev_idx='+item.prev_idx+' next_idx='+item.next_idx, item);
  _showPassage({index:item.index,full_text:item.full_text||item.text||'',prev_idx:item.prev_idx,next_idx:item.next_idx,score:item.score,rank:resultIdx+1,paper_id:item.paper_id,source_db:item.source_db});
}
function _showPassage(p){
  _currentPassage=p;
  $('#modalRank').textContent=p.rank?'#'+p.rank:'idx:'+p.index;
  $('#modalScore').textContent=p.score!=null?'score: '+p.score.toFixed(4):'';
  $('#modalText').innerHTML=highlight(linkify(esc(p.full_text)),_lastQuery);
  // Provenance state machine: split-cart preview (source_db, no paper_id) shows the
  // load CTA; loaded state (paper_id present) shows the source line; non-split shows
  // neither.
  const isSplit=!!p.source_db;
  const isLoaded=!!p.paper_id;
  const cta=$('#modalCta'),src=$('#modalSource');
  if(cta){
    if(isSplit&&!isLoaded){
      cta.innerHTML='<button class="modal-cta-btn" onclick="loadSource()">&#x1F4C2; Load full passage from '+esc(p.source_db)+' &rarr;</button>';
      cta.style.display='block';
    } else { cta.style.display='none'; cta.innerHTML=''; }
  }
  if(src){
    if(isLoaded){
      const bits=['source: '+p.source_db];
      if(p.paper_id)bits.push('id: '+p.paper_id);
      src.textContent=bits.join(' · ');
      src.style.display='block';
    } else { src.style.display='none'; }
  }
  const btnP=$('#btnPrev'),btnN=$('#btnNext');
  if(p.prev_idx!=null){btnP.className='passage-nav-btn enabled';btnP.onclick=()=>navigatePassage(p.prev_idx);}
  else{btnP.className='passage-nav-btn';btnP.onclick=null;}
  if(p.next_idx!=null){btnN.className='passage-nav-btn enabled';btnN.onclick=()=>navigatePassage(p.next_idx);}
  else{btnN.className='passage-nav-btn';btnN.onclick=null;}
  $('#passageOverlay').classList.add('open');
  document.addEventListener('keydown',_passageKeys);
}
async function loadSource(){
  if(!_currentPassage)return;
  const cta=$('#modalCta'),idx=_currentPassage.index;
  if(cta)cta.innerHTML='<span class="modal-cta-loading">Loading from source database&hellip;</span>';
  try{
    const r=await fetch(BASE()+'/api/passage?idx='+idx);
    const data=await r.json();
    if(data.error){toast(data.error,'error');return;}
    const p=data.passage;
    _showPassage({index:p.index,full_text:p.full_text,prev_idx:p.prev_idx,next_idx:p.next_idx,score:_currentPassage.score,rank:_currentPassage.rank,source_db:p.source_db,paper_id:p.paper_id});
  }catch(e){
    toast('Source load failed: '+e.message,'error');
    if(cta)cta.innerHTML='<button class="modal-cta-btn" onclick="loadSource()">Retry</button>';
  }
}
async function navigatePassage(idx){
  try{
    const r=await fetch(BASE()+'/api/passage?idx='+idx);
    const data=await r.json();
    if(data.error){toast(data.error,'error');return;}
    const p=data.passage;
    _showPassage({index:p.index,full_text:p.full_text,prev_idx:p.prev_idx,next_idx:p.next_idx,score:null,rank:null,source_db:p.source_db,paper_id:p.paper_id});
  }catch(e){toast('Navigation failed: '+e.message,'error');}
}
function closePassage(){
  $('#passageOverlay').classList.remove('open');
  _currentPassage=null;
  document.removeEventListener('keydown',_passageKeys);
}
function _passageKeys(e){
  if(e.key==='Escape')closePassage();
  if(e.key==='ArrowLeft'&&_currentPassage&&_currentPassage.prev_idx!=null)navigatePassage(_currentPassage.prev_idx);
  if(e.key==='ArrowRight'&&_currentPassage&&_currentPassage.next_idx!=null)navigatePassage(_currentPassage.next_idx);
}
async function doStore(){
  const content=$('#storeContent').value.trim();
  const tags=$('#storeTags').value.trim();
  if(!content){toast('Enter content to store','error');return;}
  try{
    const r=await fetch(BASE()+'/api/store',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({content,tags})});
    const data=await r.json();
    if(data.error){toast(data.error,'error');}
    else{toast('Memory stored','success');$('#storeContent').value='';$('#storeTags').value='';checkStatus();}
  }catch(e){toast('Store failed: '+e.message,'error');}
}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
const _STOP=new Set(['the','and','are','was','were','for','that','this','with','from','not','but','has','had','have','does','did','will','can','its','who','what','how','why','about','tell','you','your','some','than','them','then','they','been','more','also','into','would','could','should','just','like','very','much','many','only','other','over','such','after','before','between','through','where','when','which','while','each','there','their','these','those','being','because','during','both','same','own','most','well','way','all','out','one','two','may']);
function highlight(html,query){
  if(!query)return html;
  const words=query.split(/\\s+/).filter(w=>w.length>2&&!_STOP.has(w.toLowerCase()));
  let r=html;
  // Negative lookahead (?![^<]*>) skips matches inside open tags / attribute values
  // (e.g. inside href="https://arxiv.org") so URL hrefs don't break.
  for(const w of words){const re=new RegExp('('+w.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&')+')(?![^<]*>)','gi');r=r.replace(re,'<mark>$1</mark>');}
  return r;
}
function linkify(escapedText){
  // Wrap http(s) URLs in clickable links. Input must already be HTML-escaped.
  return escapedText.replace(/(https?:\\/\\/[^\\s<]+?)([.,;:!?)\\]]?(?=\\s|$|<))/g,'<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>$2');
}
function toast(msg,type='success'){
  const el=$('#toastEl');el.textContent=msg;el.className='toast '+type+' show';
  setTimeout(()=>el.className='toast',3000);
}
function toggleTheme(){
  const html=document.documentElement;
  const next=html.getAttribute('data-theme')==='light'?'dark':'light';
  html.setAttribute('data-theme',next);
  $('#themeBtn').innerHTML=next==='light'?'&#x1F319;':'&#x2600;';
  localStorage.setItem('membot-theme',next);
}
(function(){const s=localStorage.getItem('membot-theme');if(s==='light'){document.documentElement.setAttribute('data-theme','light');setTimeout(()=>$('#themeBtn').innerHTML='&#x1F319;',0);}})();
loadCartridges();
</script>
</body>
</html>
"""


@mcp.tool()
def list_cartridges() -> str:
    """List available brain cartridges with size and capabilities.
    Scans the cartridges/ and data/ directories.
    """
    log.info("list_cartridges")
    carts = find_cartridges()

    if not carts:
        return "No cartridges found. Place .npz or .pkl files in cartridges/ or data/ directories."

    lines = [f"Available cartridges ({len(carts)}):\n"]
    for c in carts:
        flags = [c["format"].upper()]
        if c["has_brain"]:
            flags.append("GPU brain")
        if c["has_signatures"]:
            flags.append("L2 sigs")
        if c["has_manifest"]:
            flags.append("verified")
        flag_str = f"  [{', '.join(flags)}]"
        lines.append(f"  {c['name']}  ({c['size_mb']} MB, {c['dir']}/){flag_str}")

    return "\n".join(lines)


@mcp.tool()
def mount_cartridge(name: str, session_id: str = "") -> str:
    """Mount a brain cartridge by name (or partial name).
    Loads embeddings and text into memory for searching.
    Verifies integrity against manifest if available.

    Args:
        name: Cartridge name (exact or partial match)
        session_id: Session identifier (auto-assigned if empty). Each session has its own mounted cartridge.
    """
    session_id = _resolve_session_id(session_id)
    state = _get_session(session_id)
    log.info(f"mount_cartridge({name}, session={session_id})")

    try:
        clean_name = sanitize_name(name)
    except ValueError as e:
        return str(e)

    carts = find_cartridges()
    match = [c for c in carts if c["name"] == clean_name]

    if not match:
        match = [c for c in carts if clean_name.lower() in c["name"].lower()]

    # Mempack fallback (Andy 2026-05-12): if the standard catalog doesn't have
    # this name, check user-segmented Mempacks. Exact-name match only here —
    # substring matching against Mempacks risks cross-user collision.
    if not match:
        mempacks = find_mempacks()
        mempack_match = [m for m in mempacks if m["name"] == clean_name]
        if len(mempack_match) > 1:
            owners = ", ".join(m.get("owner_id", "?")[:8] for m in mempack_match)
            return (
                f"Multiple Mempacks named '{clean_name}' across users ({owners}). "
                f"Pass the cart's full path or specify owner_id to disambiguate."
            )
        match = mempack_match

    if not match:
        available = ", ".join(c["name"] for c in carts[:10])
        return f"Cartridge '{clean_name}' not found. Available: {available}"

    cart = match[0]

    try:
        t0 = time.time()
        data = load_cartridge_safe(cart["path"])

        embeddings = data["embeddings"]
        texts = data["texts"]

        if len(texts) > MAX_ENTRIES:
            return f"Cartridge too large: {len(texts)} entries exceeds limit of {MAX_ENTRIES}."

        # Verify integrity
        ok, verify_msg = verify_manifest(cart["path"], embeddings, len(texts))
        if not ok:
            log.error(f"INTEGRITY CHECK FAILED for {cart['name']}: {verify_msg}")
            return f"SECURITY: Cartridge '{cart['name']}' failed integrity check: {verify_msg}. Refusing to mount."

        state["embeddings"] = embeddings
        state["texts"] = texts
        state["cartridge_name"] = cart["name"]
        state["cartridge_path"] = cart["path"]
        state["signatures"] = None
        state["hippocampus"] = data.get("hippocampus")
        state["per_pattern_meta"] = data.get("per_pattern_meta")
        state["modified"] = False

        # Pattern 0 v2 (Andy 2026-05-12): extract briefing + ownership block + cart
        # type from manifest sidecar. Defaults preserve legacy behavior (world-
        # readable, no owner, no caps, knowledge type). See pattern-0-v2-spec.md.
        _manifest = load_manifest(cart["path"])
        if _manifest:
            state["briefing"] = _manifest.get("briefing")
            state["owner_id"] = _manifest.get("owner_id")
            state["owner_perms"] = _manifest.get("owner_perms", "rwda")
            state["group_perms"] = _manifest.get("group_perms", "")
            state["world_perms"] = _manifest.get("world_perms", "r")
            state["group_id"] = _manifest.get("group_id")
            state["max_patterns"] = int(_manifest.get("max_patterns", 0) or 0)
            state["cart_type"] = _manifest.get("cart_type", CART_TYPE_KNOWLEDGE)
        else:
            # Legacy cart with no manifest at all — assume world-readable knowledge.
            state["briefing"] = None
            state["owner_id"] = None
            state["owner_perms"] = "rwda"
            state["group_perms"] = ""
            state["world_perms"] = "r"
            state["group_id"] = None
            state["max_patterns"] = 0
            state["cart_type"] = CART_TYPE_KNOWLEDGE

        # Sign-zero binary corpus for Hamming search
        # Priority: pre-computed sign_bits > computed from embeddings > None
        if "sign_bits" in data and data["sign_bits"] is not None:
            state["binary_corpus"] = data["sign_bits"]
            log.info(f"Loaded pre-computed sign bits: {data['sign_bits'].shape}")
        elif len(embeddings) > 0:
            state["binary_corpus"] = (embeddings > 0).astype(np.uint8)
        else:
            state["binary_corpus"] = None

        # Track whether we have full embeddings (for search mode selection)
        state["has_embeddings"] = len(embeddings) > 0 and embeddings.size > 0

        # Split cart: open SQLite sidecar for on-demand text retrieval
        if data.get("sqlite_db_path"):
            # Close any previous connection
            if state.get("sqlite_conn"):
                try: state["sqlite_conn"].close()
                except: pass
            state["sqlite_conn"] = sqlite3.connect(data["sqlite_db_path"])
            state["sqlite_db_path"] = data["sqlite_db_path"]
            state["is_split_cart"] = True
            log.info(f"Split cart: SQLite connection opened to {data['sqlite_db_path']}")
        else:
            state["sqlite_conn"] = None
            state["sqlite_db_path"] = None
            state["is_split_cart"] = data.get("is_split_cart", False)

        n = len(texts)
        dim = embeddings.shape[1] if len(embeddings) > 0 else 0
        elapsed_ms = (time.time() - t0) * 1000

        split_label = " (split: snippets in RAM, full text in SQLite)" if state["is_split_cart"] else ""
        log.info(f"Loaded {n} entries, dim={dim}, format={data['format']}, integrity={verify_msg}, sign_zero={n} codes, in {elapsed_ms:.0f}ms{split_label}")

        # Load L2 signatures if available
        sig_base = cart["path"].rsplit(".", 1)[0]
        sig_path = f"{sig_base}_signatures.npz"
        if os.path.exists(sig_path):
            try:
                sig_data = load_signatures(sig_path)
                state["signatures"] = sig_data["signatures"]
                log.info(f"Loaded {sig_data['n_patterns']} L2 signatures ({sig_data['method']})")
            except Exception as e:
                log.warning(f"Signature load failed: {e}")

        # Load brain weights to GPU if available
        brain_path = f"{sig_base}_brain.npy"
        gpu_msg = ""
        if os.path.exists(brain_path) and init_gpu():
            try:
                _gpu_state["lattice"].load_brain(brain_path)
                gpu_msg = ", brain loaded to GPU"
                log.info("Brain weights loaded to GPU")
            except Exception as e:
                gpu_msg = f", brain load failed: {e}"
                log.warning(f"Brain load failed: {e}")

        hippo_msg = ""
        if state["hippocampus"]:
            n_linked = sum(1 for h in state["hippocampus"] if h.get("prev") or h.get("next"))
            hippo_msg = f", hippocampus={len(state['hippocampus'])} ({n_linked} linked)"

        state["mount_count"] = state.get("mount_count", 0) + 1
        _log_activity(session_id, "mount", cart['name'], elapsed_ms)
        result = (
            f"Mounted '{cart['name']}': {n} memories, {dim}-dim, "
            f"{data['format'].upper()}, integrity={verify_msg}{gpu_msg}{hippo_msg}. "
            f"Session: {session_id}"
        )
        # Pattern 0 v2 Phase 1: surface the cart's briefing to the agent on
        # mount. Agents see the introduction immediately without needing a
        # separate get_status call. None for legacy carts; no change to output.
        if state.get("briefing"):
            result += f"\n\n--- CART BRIEFING ---\n{state['briefing']}\n--- END BRIEFING ---"
        # Pattern I (= Pattern 1) for Mempack-shaped carts: surface the agent's
        # behavioral text on mount so it bootstraps from the cart it's standing
        # in. Only fires for cart_type == agent-memory; ordinary knowledge carts
        # don't have a reserved Pattern I.
        if state.get("cart_type") == CART_TYPE_AGENT_MEMORY and len(texts) > PATTERN_I_IDX:
            pattern_i_text = texts[PATTERN_I_IDX]
            if pattern_i_text:
                result += (
                    f"\n\n--- PATTERN I (your behavior, idx={PATTERN_I_IDX}) ---\n"
                    f"{pattern_i_text}\n--- END PATTERN I ---"
                )
        return result

    except PermissionError as e:
        log.error(f"Security block: {e}")
        return str(e)
    except Exception as e:
        log.error(f"Mount failed: {e}")
        return f"Failed to mount '{clean_name}': {e}"


@mcp.tool()
def memory_search(query: str, top_k: int = 5, session_id: str = "", verbose: bool = False) -> str:
    """Search the mounted cartridge using lattice physics + embedding similarity.
    Runs the query through the neural lattice (settle → L2 signature) and blends
    physics-based similarity with embedding cosine for ranked results.

    Falls back to embedding-only search if GPU or signatures are unavailable.

    Args:
        query: Natural language search query
        top_k: Number of results to return (default 5)
        session_id: Session identifier (uses default session if empty)
        verbose: Show per-result signal breakdown (cosine, hamming, keyword boost)
    """  # noqa: docstring kept generic for MCP schema — actual impl uses sign_zero Hamming
    if len(query) > MAX_QUERY_LENGTH:
        return f"Query too long ({len(query)} chars). Max is {MAX_QUERY_LENGTH}."

    session_id = _resolve_session_id(session_id)
    state = _get_session(session_id)
    log.info(f"memory_search('{query[:60]}', top_k={top_k}, session={session_id})")

    if state["cartridge_name"] is None:
        return "No cartridge mounted. Use mount_cartridge first."

    has_embeddings = state.get("has_embeddings", True) and state["embeddings"] is not None and len(state["embeddings"]) > 0
    has_hamming = state["binary_corpus"] is not None

    if not has_embeddings and not has_hamming:
        return "Cartridge is empty (no embeddings or sign bits)."

    try:
        t0 = time.time()

        # 1. Embed query
        query_emb = embed_text(query, prefix="search_query")

        # Helper: compute Hamming scores against binary corpus
        # Handles both packed (N, 96) and unpacked (N, 768) formats
        # Popcount lookup table for packed bytes (avoids unpackbits OOM at scale)
        _POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

        def hamming_scores(query_emb, corpus_bin):
            is_packed = corpus_bin.shape[1] <= 96
            if is_packed:
                q_packed = np.packbits((query_emb > 0).astype(np.uint8))
                xor = np.bitwise_xor(q_packed, corpus_bin)
                # Popcount via lookup table — O(N*96) uint8, no expansion to 768
                dist = _POPCOUNT_TABLE[xor].sum(axis=1)
                n_bits = 768
            else:
                q_bin = (query_emb > 0).astype(np.uint8)
                xor = np.bitwise_xor(q_bin, corpus_bin)
                dist = xor.sum(axis=1)
                n_bits = corpus_bin.shape[1]
            return 1.0 - dist.astype(np.float32) / n_bits

        if has_embeddings:
            # 2a. Embedding cosine similarity
            stored = state["embeddings"]
            stored_norms = np.linalg.norm(stored, axis=1, keepdims=True) + 1e-9
            query_norm = np.linalg.norm(query_emb) + 1e-9
            emb_scores = np.dot(stored / stored_norms, query_emb / query_norm)

            # 3a. Blend with Hamming if available
            search_mode = "embedding"
            ham_scores = None

            if HAMMING_BLEND > 0 and has_hamming:
                try:
                    corpus_bin = state["binary_corpus"]
                    n_bin = min(len(corpus_bin), len(emb_scores))
                    ham = hamming_scores(query_emb, corpus_bin[:n_bin])
                    ham_scores = ham  # Expose raw Hamming scores for verbose display

                    blended = np.copy(emb_scores)
                    blended[:n_bin] = (1.0 - HAMMING_BLEND) * emb_scores[:n_bin] + HAMMING_BLEND * ham
                    scores = blended
                    search_mode = "hamming+embedding"
                    log.info(f"Hamming search: sign_zero blended {1.0 - HAMMING_BLEND:.0%}/{HAMMING_BLEND:.0%}, {n_bin} patterns")
                except Exception as e:
                    log.warning(f"Hamming search failed, falling back to embedding: {e}")
                    scores = emb_scores
            else:
                scores = emb_scores
                if not has_hamming:
                    log.info("No binary corpus — embedding-only search")

        else:
            # 2b. HAMMING-ONLY search (no full embeddings in cart)
            search_mode = "hamming-only"
            corpus_bin = state["binary_corpus"]
            scores = hamming_scores(query_emb, corpus_bin)
            emb_scores = scores  # For verbose display compatibility
            ham_scores = scores  # Raw Hamming scores are the only scores
            log.info(f"Hamming-only search: {len(corpus_bin)} patterns, packed={corpus_bin.shape[1] <= 96}")

        # 4. Keyword reranking — pull wider candidate pool, boost by keyword hits
        STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
                      "and", "or", "for", "on", "it", "be", "as", "at", "by", "this",
                      "that", "with", "from", "not", "but", "has", "had", "have", "do",
                      "does", "did", "will", "can", "its", "who", "what", "how", "why"}
        keywords = [w for w in query.lower().split() if len(w) >= 3 and w not in STOP_WORDS]

        candidate_k = min(max(top_k * 4, 20), len(scores))
        candidate_idx = np.argsort(scores)[-candidate_k:][::-1]

        boosted = []
        for i in candidate_idx:
            base_score = float(scores[i])
            if base_score < 0.05:
                continue
            text_lower = state["texts"][i].lower()
            hits = sum(1 for kw in keywords if kw in text_lower)
            boost = min(hits * 0.03, 0.12)  # up to 0.12 boost (4+ keyword hits)
            boosted.append((i, base_score + boost, boost))

        boosted.sort(key=lambda x: x[1], reverse=True)

        elapsed_ms = (time.time() - t0) * 1000

        # 5. Format results (include passage index + nav hints if hippocampus present)
        # For split carts, fetch full passages from SQLite for display.
        # int() cast is load-bearing: sqlite3 binds numpy.int64 silently as no-match.
        top_indices = [int(i) for i, _, _ in boosted[:top_k]]
        full_texts = {}
        if state.get("is_split_cart") and state.get("sqlite_conn"):
            full_texts = _sqlite_fetch_passages(state["sqlite_conn"], top_indices)

        hippo = state.get("hippocampus")
        results = []
        for rank, (i, final_score, kw_boost) in enumerate(boosted[:top_k], 1):
            if final_score < 0.1:
                continue
            # Use full text from SQLite if available, otherwise snippet from RAM
            if i in full_texts:
                text = full_texts[i]["passage"]
            else:
                text = state["texts"][i]
            if len(text) > 500:
                text = text[:500] + "..."

            # Nav hint from hippocampus
            nav = ""
            if hippo and i < len(hippo):
                h = hippo[i]
                parts = []
                if h.get("prev"): parts.append(f"prev=#{h['prev']-1}")
                if h.get("next"): parts.append(f"next=#{h['next']-1}")
                if parts:
                    nav = f" [{' '.join(parts)}]"

            if verbose:
                cos_s = f"{float(emb_scores[i]):.3f}"
                ham_s = f"{float(ham_scores[i]):.3f}" if ham_scores is not None and i < len(ham_scores) else "—"
                kw_s = f"+{kw_boost:.3f}" if kw_boost > 0 else "—"
                results.append(f"#{rank} (idx:{i}) [{final_score:.3f}] cos={cos_s} ham={ham_s} kw={kw_s}{nav}\n{text}")
            else:
                results.append(f"#{rank} (idx:{i}) [{final_score:.3f}]{nav} {text}")

        state["query_count"] = state.get("query_count", 0) + 1
        _log_activity(session_id, "search", f"'{query[:40]}' → {len(boosted[:top_k])} results", elapsed_ms)

        if not results:
            return f"No relevant matches for '{query}' (searched {len(state['texts'])} memories, {elapsed_ms:.0f}ms)"

        kw_label = f"+kw" if keywords else ""
        split_label = "+sqlite" if state.get("is_split_cart") else ""
        if search_mode == "hamming+embedding":
            mode_label = f"hamming+embedding 70/30{kw_label}{split_label}"
        elif search_mode == "hamming-only":
            mode_label = f"hamming-only{kw_label}{split_label}"
        else:
            mode_label = f"embedding-only{kw_label}{split_label}"
        header = f"Search [{mode_label}]: {len(results)} results from '{state['cartridge_name']}' ({elapsed_ms:.0f}ms)\n"
        return header + "\n\n".join(results)

    except Exception as e:
        log.error(f"Search error: {e}")
        return f"Search error: {e}"


@mcp.tool()
def mempack_read_pattern_i(session_id: str = "") -> str:
    """Read Pattern I (agent behavior) from the currently mounted Mempack.

    Pattern I is reserved at index 1 of any cart with cart_type='agent-memory'.
    Returns the text content of that pattern, or an explanatory message if the
    mounted cart isn't a Mempack or doesn't yet have Pattern I populated.

    Use this when you first mount a Mempack to load your behavioral instructions
    into context. Re-read periodically across long sessions to refresh.

    Args:
        session_id: Session identifier (uses default session if empty).
    """
    session_id = _resolve_session_id(session_id)
    state = _get_session(session_id)

    if state["cartridge_name"] is None:
        return "No cartridge mounted. Use mount_cartridge first."

    if state.get("cart_type") != CART_TYPE_AGENT_MEMORY:
        return (
            f"This cart is type '{state.get('cart_type', 'knowledge')}', not a "
            f"Mempack (cart_type='agent-memory'). Pattern I only exists in "
            f"agent-memory carts."
        )

    texts = state.get("texts") or []
    if len(texts) <= PATTERN_I_IDX:
        return (
            f"Mempack has no Pattern I yet — cart has {len(texts)} pattern(s), "
            f"index {PATTERN_I_IDX} is empty. Use the upcoming "
            f"mempack_update_pattern_i tool to populate it."
        )

    return texts[PATTERN_I_IDX]


@mcp.tool()
def memory_store(content: str, tags: str = "", session_id: str = "") -> str:
    """Store new text in the currently mounted cartridge.
    The text is embedded via Nomic and added to the searchable memory.
    If GPU is available, the pattern is also imprinted into the lattice.

    Args:
        content: Text content to memorize (max 10,000 chars)
        tags: Optional metadata tags (prepended to stored text)
        session_id: Session identifier (uses default session if empty)
    """
    if _server_config.get("read_only"):
        return "Server is in read-only mode. memory_store is disabled."

    if len(content) > MAX_TEXT_LENGTH:
        return f"Text too long ({len(content)} chars). Max is {MAX_TEXT_LENGTH}."

    session_id = _resolve_session_id(session_id)
    state = _get_session(session_id)

    if state["cartridge_name"] is None:
        return "No cartridge mounted. Use mount_cartridge first."

    n_current = len(state["texts"]) if state["texts"] else 0
    if n_current >= MAX_ENTRIES:
        return f"Cartridge full ({n_current}/{MAX_ENTRIES}). Save and start a new cartridge."

    log.info(f"memory_store('{content[:60]}...')")

    try:
        t0 = time.time()

        # 1. Embed via Nomic
        emb = embed_text(content, prefix="search_document")

        # 2. Add to embedding matrix
        if state["embeddings"] is None or len(state["embeddings"]) == 0:
            state["embeddings"] = emb.reshape(1, -1)
        else:
            state["embeddings"] = np.vstack([state["embeddings"], emb.reshape(1, -1)])

        # 3. Extend binary corpus (sign_zero for Hamming search)
        new_bin = (emb > 0).astype(np.uint8).reshape(1, -1)
        if state["binary_corpus"] is None or len(state["binary_corpus"]) == 0:
            state["binary_corpus"] = new_bin
        else:
            state["binary_corpus"] = np.vstack([state["binary_corpus"], new_bin])

        # 4. Store text
        stored_text = f"[{tags}] {content}" if tags else content
        state["texts"].append(stored_text)
        state["modified"] = True

        # 5. GPU lattice imprint (if available)
        gpu_msg = ""
        if _gpu_state["available"] and _gpu_state["lattice"] is not None:
            try:
                ml = _gpu_state["lattice"]
                ml.reset()
                ml.imprint_vector(emb)
                ml.settle(frames=10, learn=True)
                gpu_msg = " + lattice imprint"
            except Exception as e:
                gpu_msg = f" (lattice failed: {e})"
                log.warning(f"Lattice imprint failed: {e}")

        elapsed_ms = (time.time() - t0) * 1000
        n = len(state["texts"])

        return f"Stored memory #{n}{gpu_msg} ({elapsed_ms:.0f}ms)"

    except Exception as e:
        log.error(f"Store error: {e}")
        return f"Store error: {e}"


@mcp.tool()
def save_cartridge(name: str = "", session_id: str = "") -> str:
    """Save the current cartridge to disk as a secure .npz file.
    Generates a SHA256 manifest for integrity verification.

    Args:
        name: Optional name for the saved cartridge (default: use mounted name)
        session_id: Session identifier (uses default session if empty)
    """
    session_id = _resolve_session_id(session_id)
    state = _get_session(session_id)
    log.info(f"save_cartridge('{name}', session={session_id})")

    if _server_config.get("read_only"):
        return "Server is in read-only mode. save_cartridge is disabled."

    if state["cartridge_name"] is None and not name:
        return "No cartridge to save. Mount one first or provide a name."

    try:
        save_name = sanitize_name(name or state["cartridge_name"])
    except ValueError as e:
        return str(e)

    save_dir = os.path.join(BASE_DIR, "cartridges")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{save_name}.cart.npz")

    try:
        save_as_npz(save_path, state["embeddings"], state["texts"])
        size_mb = os.path.getsize(save_path) / (1024 * 1024)

        # Save integrity manifest
        manifest = save_manifest(save_path, state["embeddings"], len(state["texts"]))
        state["modified"] = False

        # Also save brain weights if GPU is active
        brain_msg = ""
        if _gpu_state["available"] and _gpu_state["lattice"] is not None:
            brain_path = os.path.join(save_dir, f"{save_name}_brain.npy")
            try:
                _gpu_state["lattice"].save_brain(brain_path)
                brain_msg = " + brain weights"
            except Exception as e:
                brain_msg = f" (brain save failed: {e})"

        return (
            f"Saved '{save_name}': {len(state['texts'])} memories, {size_mb:.1f} MB, "
            f"fingerprint={manifest['fingerprint']}{brain_msg}"
        )

    except Exception as e:
        log.error(f"Save error: {e}")
        return f"Save error: {e}"


@mcp.tool()
def unmount(session_id: str = "") -> str:
    """Unmount the current cartridge and free memory.

    Args:
        session_id: Session identifier (uses default session if empty)
    """
    session_id = _resolve_session_id(session_id)
    state = _get_session(session_id)
    log.info(f"unmount(session={session_id})")

    if state["cartridge_name"] is None:
        return "No cartridge mounted."

    name = state["cartridge_name"]
    warning = ""
    if state["modified"]:
        warning = " WARNING: Unsaved changes were discarded."

    # Close SQLite sidecar if open
    if state.get("sqlite_conn"):
        try: state["sqlite_conn"].close()
        except: pass

    # Reset session state but keep it alive
    state["cartridge_name"] = None
    state["cartridge_path"] = None
    state["embeddings"] = None
    state["texts"] = []
    state["binary_corpus"] = None
    state["signatures"] = None
    state["hippocampus"] = None
    state["modified"] = False
    state["sqlite_conn"] = None
    state["is_split_cart"] = False

    _log_activity(session_id, "unmount", name)
    return f"Unmounted '{name}'.{warning}"


# =============================================================================
# MULTI-CART API — query across many mounted carts at once
# =============================================================================
# Spec: docs/RFC/multi-cart-query-spec.md
# Implementation: multi_cart.py (parallel layer alongside the single-cart code above)
#
# The single-cart tools above (mount_cartridge, memory_search, etc.) keep
# working unchanged — they use per-session state. The multi-cart tools below
# share a process-global pool of mounted carts so one process can hold many
# carts and query across them. Both layers can coexist.

import multi_cart as _mc


@mcp.tool()
def multi_mount(cart_path: str, cart_id: str = "", role: str = "",
                verify_integrity: bool = True) -> str:
    """Mount a cart by file path into the multi-cart pool. Multiple carts can
    be mounted at once and queried across via multi_search.

    Args:
        cart_path: Filesystem path to the cart file (.npz, .pkl, or split format)
        cart_id: Short stable identifier for this mount (defaults to filename)
        role: Optional semantic tag — 'identity', 'episodic', 'semantic',
              'working', 'federated', 'consolidated', etc. Used to filter
              searches by role.
        verify_integrity: If True (default), reject carts with stale or
              mismatched manifests. Set False to override for known-stale
              carts (testing or migration).
    """
    log.info(f"multi_mount({cart_path}, cart_id={cart_id!r}, role={role!r}, verify={verify_integrity})")
    try:
        result = _mc.mount(
            cart_path,
            cart_id=cart_id or None,
            role=role or None,
            verify_integrity=verify_integrity,
        )
        return (
            f"Mounted '{result['cart_id']}' (role={result['role']}, "
            f"{result['n_patterns']} patterns, {result['embedding_dim']}-dim, "
            f"{result['elapsed_ms']}ms)"
        )
    except Exception as e:
        return f"multi_mount failed: {e}"


@mcp.tool()
def multi_unmount(cart_id: str) -> str:
    """Remove a cart from the multi-cart pool by cart_id."""
    log.info(f"multi_unmount({cart_id})")
    result = _mc.unmount(cart_id)
    if result["status"] == "not_mounted":
        return f"Cart '{cart_id}' is not mounted."
    return (
        f"Unmounted '{cart_id}' (was role={result['role']}, "
        f"{result['n_patterns']} patterns)"
    )


@mcp.tool()
def multi_list() -> str:
    """List every cart currently mounted in the multi-cart pool."""
    log.info("multi_list()")
    mounts = _mc.list_mounts()
    if not mounts:
        return "No carts mounted in the multi-cart pool."
    lines = [f"Multi-cart pool: {len(mounts)} carts, {_mc.total_patterns_mounted()} total patterns"]
    for m in mounts:
        role_label = f" [role={m['role']}]" if m["role"] else ""
        split_label = " (split)" if m["is_split_cart"] else ""
        lines.append(
            f"  '{m['cart_id']}'{role_label}: {m['n_patterns']} patterns, "
            f"{m['embedding_dim']}-dim{split_label}"
        )
    return "\n".join(lines)


@mcp.tool()
def multi_mount_directory(dir_path: str, role: str = "", pattern: str = "*.cart") -> str:
    """Mount every cart file in a directory matching the glob pattern.
    Used for federated mode — point at a fleet-learning directory and get
    every machine's cart mounted with the same role.

    Args:
        dir_path: Directory to scan
        role: Role to apply to every mounted cart (e.g. 'federated')
        pattern: Glob pattern (default '*.cart' — also try '*.npz', '*.pkl')
    """
    log.info(f"multi_mount_directory({dir_path}, role={role!r}, pattern={pattern!r})")
    try:
        results = _mc.mount_directory(dir_path, role=role or None, pattern=pattern)
        ok = sum(1 for r in results if r.get("status") == "mounted")
        err = sum(1 for r in results if r.get("status") == "error")
        lines = [f"mount_directory({dir_path}): {ok} mounted, {err} errors"]
        for r in results:
            if r.get("status") == "mounted":
                lines.append(f"  ✓ {r['cart_id']}: {r['n_patterns']} patterns")
            else:
                lines.append(f"  ✗ {r.get('cart_id', '?')}: {r.get('error', 'unknown')}")
        return "\n".join(lines)
    except Exception as e:
        return f"multi_mount_directory failed: {e}"


@mcp.tool()
def multi_search(query: str, top_k: int = 10, scope: str = "all",
                 role_filter: str = "", scope_mode: str = "global") -> str:
    """Search across every mounted cart in the multi-cart pool. Results are
    ranked and attributed to their source cart.

    Args:
        query: Natural language search query
        top_k: Number of top results to return (interpretation depends on scope_mode)
        scope: 'all' (default), 'local' (first cart only), or a specific cart_id
        role_filter: Optional role tag to restrict the search to matching carts
                     (e.g. 'federated' to search only federated-mode carts)
        scope_mode: How results across carts are ranked. Use this when carts
                    are very different sizes and you don't want one large cart
                    to dominate the results. Options:
                    - 'global' (default): true top-K across all carts. Best
                      for "what's the single best answer regardless of source?"
                    - 'per_cart': top-K from EACH cart, no cross-cart re-ranking.
                      Returns up to K × N_carts results. Best for "show me each
                      source's best answer for comparison."
                    - 'balanced': top-K candidates per cart, then globally rerank
                      to top-K. Guarantees small carts aren't drowned but the
                      final ranking still reflects global score. Best when
                      you want fair representation AND global ranking.
                    - 'diagnostic': top-K from every cart, no merging at all,
                      fully labeled. Useful for debugging.
    """
    if len(query) > MAX_QUERY_LENGTH:
        return f"Query too long ({len(query)} chars). Max is {MAX_QUERY_LENGTH}."
    log.info(f"multi_search('{query[:60]}', top_k={top_k}, scope={scope!r}, role_filter={role_filter!r}, scope_mode={scope_mode!r})")

    try:
        result = _mc.search(
            query,
            top_k=top_k,
            scope=scope,
            role_filter=role_filter or None,
            scope_mode=scope_mode,
        )
    except ValueError as e:
        return f"multi_search error: {e}"
    except Exception as e:
        return f"multi_search error: {e}"

    if result.get("error"):
        return f"multi_search: {result['error']}"

    results = result["results"]
    if not results:
        return (
            f"No relevant matches for '{query}' "
            f"(searched {result['cart_count']} carts, "
            f"{result['total_patterns']} patterns, {result['elapsed_ms']}ms, "
            f"scope_mode={scope_mode})"
        )

    header = (
        f"multi_search [scope={scope}, role_filter={role_filter or 'any'}, "
        f"scope_mode={scope_mode}]: "
        f"{len(results)} results from {result['cart_count']} carts "
        f"({result['total_patterns']} patterns, {result['elapsed_ms']}ms)\n"
    )
    lines = [header]

    # For per_cart and diagnostic modes, group output by cart_id for readability
    grouped = result.get("grouped_results")
    if grouped and scope_mode in ("per_cart", "diagnostic"):
        for cart_id, cart_results in grouped.items():
            if not cart_results:
                continue
            role_tag = ""
            if cart_results[0].get("role"):
                role_tag = f"/{cart_results[0]['role']}"
            lines.append(f"\n=== {cart_id}{role_tag} ===")
            for rank, r in enumerate(cart_results, 1):
                if r["score"] < 0.1:
                    continue
                lines.append(f"#{rank} [#{r['local_addr']}] [{r['score']:.3f}] {r['text']}")
    else:
        # Global / balanced — flat list with full attribution per result
        for rank, r in enumerate(results, 1):
            if r["score"] < 0.1:
                continue
            cart_label = f"[{r['cart_id']}"
            if r.get("role"):
                cart_label += f"/{r['role']}"
            cart_label += f"#{r['local_addr']}]"
            lines.append(f"#{rank} {cart_label} [{r['score']:.3f}] {r['text']}")
    return "\n\n".join(lines)


# =============================================================================
# FEDERATE — Federated cart mode (drop-in for Dennis's federated learning)
# =============================================================================
# Spec: docs/RFC/federated-cart-spec.md
# Implementation: federate.py (built on multi_cart.py)
#
# Use these tools to manage a fleet of machine carts that share a git directory
# the way Dennis's SAGE fleet shares its JSONL learning data. publish_session()
# appends to a machine's cart; consolidate() finds cross-machine matches and
# writes a consolidated cart; load_fleet() mounts the whole fleet for solver use.

import federate as _fed


@mcp.tool()
def federate_publish(session_file: str, machine_id: str, fleet_dir: str) -> str:
    """Append session learning entries to a machine's federated cart.
    Drop-in replacement for Dennis Palatov's publish_learning.py.

    Args:
        session_file: Path to a .jsonl file (one JSON object per line) or a
            .json file with a 'learning_entries' or 'entries' key.
        machine_id: Machine identifier (e.g. 'cbp', 'sprout', 'mcnugget')
        fleet_dir: Root of the fleet-learning directory (parent of machine dirs)
    """
    log.info(f"federate_publish({session_file}, {machine_id}, {fleet_dir})")
    try:
        result = _fed.publish_session(session_file, machine_id, fleet_dir)
        return (
            f"Published to '{machine_id}': added {result['added']}, "
            f"skipped {result['skipped']} dedup, "
            f"total {result['total_in_cart']} → {result['cart_path']}"
        )
    except Exception as e:
        return f"federate_publish failed: {e}"


@mcp.tool()
def federate_consolidate(fleet_dir: str, output_dir: str = "",
                         similarity_threshold: float = 0.85,
                         mode: str = "preserve") -> str:
    """Mount every machine cart in fleet_dir, find cross-machine pattern
    matches, and write a consolidated cart with cross-machine consensus
    captured as confirming-machine metadata. Drop-in replacement for
    Dennis's consolidate.py.

    Args:
        fleet_dir: Root of the fleet-learning directory
        output_dir: Where to write the consolidated cart (defaults to
            fleet_dir/../consolidated)
        similarity_threshold: Patterns with cross-machine cosine+hamming
            score above this count as CONFIRMED_BY (default 0.85)
        mode: Consolidation strategy. "preserve" (default, recommended for
            federated fleets) keeps ALL variants from every machine and
            stores cross-cart edges as metadata so the solver can weight
            trust contextually — aligns with the Web4 trust model. "collapse"
            picks one representative per CONFIRMED_BY component (smaller
            cart, loses individual machine voices).
    """
    log.info(f"federate_consolidate({fleet_dir}, output_dir={output_dir!r}, threshold={similarity_threshold}, mode={mode!r})")
    try:
        result = _fed.consolidate(
            fleet_dir,
            output_dir=output_dir or None,
            similarity_threshold=similarity_threshold,
            mode=mode,
        )
        if result.get("error"):
            return f"federate_consolidate: {result['error']}"
        return (
            f"Consolidated {result['n_machines']} machines, "
            f"{result['total_input_patterns']} input patterns → "
            f"{result['n_consolidated_patterns']} consolidated "
            f"({result['n_confirmed_pairs']} confirmed pairs, "
            f"{result['n_contradicted_pairs']} contradicted pairs, "
            f"mode={result['mode']}) "
            f"in {result['elapsed_seconds']}s. "
            f"Output: {result['output_path']}"
        )
    except Exception as e:
        return f"federate_consolidate failed: {e}"


@mcp.tool()
def federate_migrate_jsonl(jsonl_dir: str, output_dir: str = "",
                            in_place: bool = False) -> str:
    """One-time migration from JSONL learning files to brain carts.
    Walks a directory of fleet-learning/{machine}/*_learning.jsonl files and
    builds a kb.cart.npz for each machine.

    Args:
        jsonl_dir: Directory with per-machine subdirs containing JSONL files
        output_dir: Target directory for the new carts (defaults to jsonl_dir
            if in_place=True or jsonl_dir if not specified)
        in_place: Write carts alongside the JSONL files in their original
            machine directories. Non-destructive — JSONL files are not removed.
    """
    log.info(f"federate_migrate_jsonl({jsonl_dir}, output_dir={output_dir!r}, in_place={in_place})")
    try:
        result = _fed.migrate_jsonl(
            jsonl_dir,
            output_dir=output_dir or None,
            in_place=in_place,
        )
        lines = [
            f"Migration: {result['carts_built']} carts built from "
            f"{result['machines_processed']} machine dirs, "
            f"{result['total_entries']} entries in {result['elapsed_seconds']}s"
        ]
        if result["errors"]:
            lines.append(f"  {len(result['errors'])} errors:")
            for err in result["errors"][:5]:
                lines.append(f"    ! {err}")
        return "\n".join(lines)
    except Exception as e:
        return f"federate_migrate_jsonl failed: {e}"


@mcp.tool()
def federate_load(fleet_dir: str) -> str:
    """Mount every machine's federated cart in fleet_dir into the multi-cart
    pool with role='federated'. Used by solvers at session start to make the
    fleet's accumulated learning available for cross-machine search via
    multi_search(scope='all', role_filter='federated').

    Args:
        fleet_dir: Root of the fleet-learning directory (parent of machine dirs)
    """
    log.info(f"federate_load({fleet_dir})")
    try:
        result = _fed.load_fleet(fleet_dir)
        machine_list = ", ".join(result["machines"]) if result["machines"] else "none"
        out = (
            f"Loaded fleet from {fleet_dir}: {len(result['mounted'])} machines mounted "
            f"({machine_list}), {result['total_patterns']} total patterns"
        )
        if result["errors"]:
            out += f"\n  {len(result['errors'])} errors:"
            for err in result["errors"][:3]:
                out += f"\n    ! {err.get('machine', '?')}: {err.get('error', '?')}"
        return out
    except Exception as e:
        return f"federate_load failed: {e}"


# =============================================================================
# MEMBOX — Multi-user shared cart with locking + agent attribution
# =============================================================================
# Spec: docs/RFC/membox-phase1-implementation.md
# Implementation: membox.py (built on multi_cart.py)
#
# Membox is the third mode of the three-mode framework (single-user / federated
# / multiuser). Multiple agents safely write to the same cart with per-agent
# attribution and a write mutex that guarantees serialization without blocking
# reads. Phase 1 ships locking + tagging only — version chains, dispute
# detection, and permissions come in Phases 2-4.

import membox as _mb


@mcp.tool()
def membox_mount(cart_path: str, cart_id: str = "", role: str = "",
                 lease_seconds: int = 30, verify_integrity: bool = True) -> str:
    """Mount a brain cart in Membox mode (multi-user shared with locking).

    Multiple agents can write to the cart safely via membox_imprint, with
    each write attributed to the calling agent_id. Reads via membox_search
    never block on the write lock.

    Args:
        cart_path: Filesystem path to the cart file (.npz, .pkl, or split format)
        cart_id: Stable identifier for the mount (defaults to filename)
        role: Optional semantic tag (e.g. 'team_kb', 'project_notes')
        lease_seconds: Auto-release timeout if a holder crashes (default 30)
        verify_integrity: Reject carts with stale manifests (default True)
    """
    log.info(f"membox_mount({cart_path}, cart_id={cart_id!r}, role={role!r}, lease={lease_seconds}s)")
    try:
        result = _mb.mount(
            cart_path,
            cart_id=cart_id or None,
            role=role or None,
            lease_seconds=lease_seconds,
            verify_integrity=verify_integrity,
        )
        return (
            f"Mounted '{result['cart_id']}' in Membox mode "
            f"(role={result.get('role')}, n_patterns={result['n_patterns']}, "
            f"lease={result['lease_seconds']}s)"
        )
    except Exception as e:
        return f"membox_mount failed: {e}"


@mcp.tool()
def membox_unmount(cart_id: str) -> str:
    """Unmount a Membox cart. Releases the lock if held."""
    log.info(f"membox_unmount({cart_id})")
    try:
        result = _mb.unmount(cart_id)
        return f"Unmounted Membox cart '{cart_id}' (was {result.get('n_patterns', '?')} patterns)"
    except Exception as e:
        return f"membox_unmount failed: {e}"


@mcp.tool()
def membox_list() -> str:
    """List every Membox-mounted cart with current lock state and write stats."""
    log.info("membox_list()")
    try:
        mounts = _mb.list_mounts()
        if not mounts:
            return "No carts mounted in Membox mode."
        lines = [f"Membox pool: {len(mounts)} carts"]
        for m in mounts:
            lock = m["lock"]
            holder = lock.get("holder") or "(idle)"
            lines.append(
                f"  '{m['cart_id']}' role={m.get('role')} "
                f"n_patterns={m['n_patterns']} "
                f"lock_holder={holder} "
                f"acquires={lock.get('acquire_count', 0)}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"membox_list failed: {e}"


@mcp.tool()
def membox_imprint(cart_id: str, text: str, agent_id: str,
                   tags: str = "", reasoning: str = "",
                   origin: str = "agent", timeout_ms: int = 5000) -> str:
    """Write a new pattern to a Membox cart with agent_id attribution.

    Acquires the write lock, writes the pattern with full per-agent metadata
    (agent_id, written_at, origin, tags, reasoning), releases the lock.
    Returns the new local_addr or an error if the lock can't be acquired
    within timeout_ms.

    Args:
        cart_id: Membox-mounted cart to write to
        text: Pattern text content (the thing to embed and store)
        agent_id: Who's writing — REQUIRED, no anonymous writes in Membox
        tags: Optional comma-separated tags
        reasoning: Optional explanation of WHY this is being written (audit trail)
        origin: 'agent' (default) | 'human' | 'system'
        timeout_ms: How long to wait for the write lock (default 5000)
    """
    log.info(f"membox_imprint(cart_id={cart_id}, agent={agent_id}, text_len={len(text)})")
    try:
        result = _mb.imprint(
            cart_id, text=text, agent_id=agent_id,
            tags=tags, reasoning=reasoning, origin=origin,
            timeout_ms=timeout_ms,
        )
    except ValueError as e:
        return f"membox_imprint error: {e}"
    except Exception as e:
        return f"membox_imprint failed: {e}"

    if not result.get("ok"):
        if result.get("error") == "lock_timeout":
            return (
                f"Lock timeout — cart '{cart_id}' is currently held by "
                f"{result.get('current_holder')!r}. Try again or increase timeout_ms."
            )
        return f"membox_imprint failed: {result.get('error')}"

    return (
        f"Imprinted to '{cart_id}' as agent {agent_id!r} "
        f"at local_addr={result['local_addr']} (written_at={result['written_at']})"
    )


@mcp.tool()
def membox_search(cart_id: str, query: str, top_k: int = 10,
                  agent_id: str = "") -> str:
    """Search a Membox cart. Never blocks on the write lock — reads always
    succeed even while another agent is writing.

    Returns ranked results with per-pattern Membox metadata visible
    (agent_id, written_at, origin, reasoning, tags) so the consumer can
    see who wrote what.

    Args:
        cart_id: Membox-mounted cart to search
        query: Natural language query
        top_k: Max results to return (default 10)
        agent_id: Optional — who's reading (for audit logging in Phase 4)
    """
    if len(query) > MAX_QUERY_LENGTH:
        return f"Query too long ({len(query)} chars). Max is {MAX_QUERY_LENGTH}."
    log.info(f"membox_search(cart_id={cart_id}, query={query[:60]!r})")

    try:
        result = _mb.search(cart_id, query, top_k=top_k, agent_id=agent_id or None)
    except ValueError as e:
        return f"membox_search error: {e}"
    except Exception as e:
        return f"membox_search failed: {e}"

    results = result.get("results", [])
    if not results:
        return f"No results for '{query}' in '{cart_id}' ({result.get('elapsed_ms', '?')}ms)"

    lines = [f"membox_search '{cart_id}': {len(results)} results in {result.get('elapsed_ms', '?')}ms\n"]
    for rank, r in enumerate(results, 1):
        if r.get("score", 0) < 0.1:
            continue
        membox_meta = r.get("membox_meta") or {}
        attrib = ""
        if isinstance(membox_meta, dict) and membox_meta.get("agent_id"):
            written = membox_meta.get("written_at", "")
            attrib = f"  [agent={membox_meta['agent_id']} @ {written[:19]}]"
        lines.append(f"#{rank} [#{r['local_addr']}] [{r['score']:.3f}]{attrib} {r['text'][:300]}")
    return "\n\n".join(lines)


@mcp.tool()
def membox_acquire_lock(cart_id: str, agent_id: str, timeout_ms: int = 5000) -> str:
    """Manually acquire the write lock on a Membox cart. Most callers should
    use membox_imprint instead, which handles acquire+write+release atomically.

    Use this directly only when you need to do multiple writes in sequence
    without releasing the lock between them.

    Args:
        cart_id: Membox-mounted cart
        agent_id: Who's acquiring (required)
        timeout_ms: How long to wait if the lock is held (default 5000)
    """
    log.info(f"membox_acquire_lock(cart_id={cart_id}, agent={agent_id})")
    try:
        ok = _mb.acquire_lock(cart_id, agent_id, timeout_ms=timeout_ms)
    except ValueError as e:
        return f"membox_acquire_lock error: {e}"
    if ok:
        return f"Lock acquired on '{cart_id}' by agent {agent_id!r}"
    holder = _mb.lock_holder(cart_id)
    return f"Lock timeout — '{cart_id}' is held by {holder!r}, try again or increase timeout_ms"


@mcp.tool()
def membox_release_lock(cart_id: str, agent_id: str) -> str:
    """Release the write lock on a Membox cart. Only the current holder may release.

    Args:
        cart_id: Membox-mounted cart
        agent_id: The agent currently holding the lock
    """
    log.info(f"membox_release_lock(cart_id={cart_id}, agent={agent_id})")
    try:
        _mb.release_lock(cart_id, agent_id)
        return f"Lock released on '{cart_id}' by agent {agent_id!r}"
    except (ValueError, PermissionError, RuntimeError) as e:
        return f"membox_release_lock error: {e}"


@mcp.tool()
def membox_lock_holder(cart_id: str) -> str:
    """Return the agent_id currently holding the write lock, or '(idle)'."""
    log.info(f"membox_lock_holder(cart_id={cart_id})")
    try:
        holder = _mb.lock_holder(cart_id)
    except ValueError as e:
        return f"membox_lock_holder error: {e}"
    if holder is None:
        return f"'{cart_id}': lock is idle"
    return f"'{cart_id}': lock held by agent {holder!r}"


@mcp.tool()
def membox_status(cart_id: str) -> str:
    """Return Membox status for a cart: lock state, write counts per agent,
    pattern count, recent write log.
    """
    log.info(f"membox_status(cart_id={cart_id})")
    try:
        s = _mb.status(cart_id)
    except ValueError as e:
        return f"membox_status error: {e}"
    lines = [
        f"Membox status for '{cart_id}':",
        f"  Patterns: {s['n_patterns']}",
        f"  Lock holder: {s['lock'].get('holder') or '(idle)'}",
        f"  Lock acquires: {s['lock'].get('acquire_count', 0)}",
        f"  Lock waits: {s['lock'].get('wait_count', 0)}",
        f"  Writes by agent:",
    ]
    for agent, count in sorted(s["writes_by_agent"].items()):
        lines.append(f"    {agent}: {count}")
    if s.get("recent_writes"):
        lines.append(f"  Recent writes (last {len(s['recent_writes'])}):")
        for w in s["recent_writes"]:
            lines.append(
                f"    [{w['written_at'][:19]}] {w['agent_id']} → addr={w['local_addr']}: "
                f"{w['text_preview'][:80]}"
            )
    return "\n".join(lines)


@mcp.tool()
def get_status(session_id: str = "") -> str:
    """Get server status: mounted cartridge, memory count, GPU availability.

    Args:
        session_id: Session identifier (uses default session if empty)
    """
    session_id = _resolve_session_id(session_id)
    state = _get_session(session_id)
    log.info(f"get_status(session={session_id})")

    cart = state["cartridge_name"] or "None"
    n = len(state["texts"]) if state["texts"] else 0
    dim = state["embeddings"].shape[1] if state["embeddings"] is not None and len(state["embeddings"]) > 0 else 0
    has_hamming = state["binary_corpus"] is not None
    ham_bytes = f"{n * 96:,} bytes" if has_hamming else "N/A"
    gpu = "Ready" if _gpu_state["available"] else "Not available"
    modified = " (unsaved changes)" if state["modified"] else ""
    active_sessions = len(_sessions)

    hippo = state.get("hippocampus")
    hippo_str = "No"
    if hippo:
        n_linked = sum(1 for h in hippo if h.get("prev") or h.get("next"))
        hippo_str = f"Yes ({n_linked} linked)"

    return (
        f"Cartridge: {cart}{modified} | "
        f"Memories: {n}/{MAX_ENTRIES} | "
        f"Embedding dim: {dim} | "
        f"Hamming index: {'Yes' if has_hamming else 'No'} ({ham_bytes}) | "
        f"Hippocampus: {hippo_str} | "
        f"GPU: {gpu} | "
        f"Sessions: {active_sessions} | "
        f"Session: {session_id} | "
        f"Search: cosine+hamming 70/30+kw | "
        f"Embed: nomic-embed-text-v1.5 via SentenceTransformer"
    )


@mcp.tool()
def passage_links(idx: int, session_id: str = "") -> str:
    """Get hippocampus navigation links for a passage by index.

    Returns prev/next pointers, source hash, sequence position, flags,
    and the text of linked passages for easy traversal.

    Args:
        idx: Passage index (0-based, from search results)
        session_id: Session identifier (uses default session if empty)
    """
    session_id = _resolve_session_id(session_id)
    state = _get_session(session_id)
    log.info(f"passage_links(idx={idx}, session={session_id})")

    if state["cartridge_name"] is None:
        return "No cartridge mounted."

    hippo = state.get("hippocampus")
    if not hippo:
        return "No hippocampus metadata in this cartridge."

    if idx < 0 or idx >= len(hippo):
        return f"Index {idx} out of range (0-{len(hippo)-1})."

    meta = hippo[idx]
    texts = state["texts"]

    lines = [f"Passage #{idx} metadata:"]
    lines.append(f"  pattern_id:  {meta['pattern_id']}")
    lines.append(f"  seq:         {meta['sequence_num']}")
    lines.append(f"  source_hash: {meta['source_hash']:08x}")
    lines.append(f"  timestamp:   {meta['timestamp']}")

    flags = []
    if meta["flags"] & 0x01: flags.append("TOMBSTONE")
    if meta["flags"] & 0x02: flags.append("PINNED")
    if meta["flags"] & 0x04: flags.append("HAS_PARENT")
    if meta["flags"] & 0x08: flags.append("HAS_CHILD")
    if meta["flags"] & 0x10: flags.append("HAS_SIBLING")
    lines.append(f"  flags:       {', '.join(flags) if flags else 'none'}")

    # Prev/Next navigation with text previews
    if meta["prev"] is not None:
        pi = meta["prev"] - 1  # pattern_id is 1-based, texts are 0-based
        if 0 <= pi < len(texts):
            preview = texts[pi][:120].replace("\n", " ")
            lines.append(f"  PREV (#{pi}): {preview}...")
        else:
            lines.append(f"  PREV: pattern {meta['prev']} (out of range)")
    else:
        lines.append(f"  PREV: none (start of document)")

    if meta["next"] is not None:
        ni = meta["next"] - 1
        if 0 <= ni < len(texts):
            preview = texts[ni][:120].replace("\n", " ")
            lines.append(f"  NEXT (#{ni}): {preview}...")
        else:
            lines.append(f"  NEXT: pattern {meta['next']} (out of range)")
    else:
        lines.append(f"  NEXT: none (end of document)")

    if meta["sibling"] is not None:
        si = meta["sibling"] - 1
        if 0 <= si < len(texts):
            preview = texts[si][:120].replace("\n", " ")
            lines.append(f"  SIBLING (#{si}): {preview}...")

    return "\n".join(lines)


@mcp.tool()
def get_passage(idx: int, session_id: str = "") -> str:
    """Retrieve a passage by index with full text and navigation links.

    Use this after memory_search to navigate to prev/next passages
    within the same document. Search results include [prev=#N next=#N]
    hints — pass those indices here to read adjacent passages.

    Args:
        idx: Passage index (0-based, from search results or prev/next hints)
        session_id: Session identifier (uses default session if empty)
    """
    session_id = _resolve_session_id(session_id)
    state = _get_session(session_id)
    log.info(f"get_passage(idx={idx}, session={session_id})")

    if state["cartridge_name"] is None:
        return "No cartridge mounted."

    texts = state["texts"]
    if idx < 0 or idx >= len(texts):
        return f"Index {idx} out of range (0-{len(texts)-1})."

    full_text = texts[idx]

    # Navigation from hippocampus
    nav_parts = []
    hippo = state.get("hippocampus")
    if hippo and idx < len(hippo):
        meta = hippo[idx]
        if meta["prev"] is not None:
            pi = meta["prev"] - 1
            nav_parts.append(f"prev=#{pi}")
        if meta["next"] is not None:
            ni = meta["next"] - 1
            nav_parts.append(f"next=#{ni}")

    nav = f" [{' '.join(nav_parts)}]" if nav_parts else ""
    header = f"Passage #{idx}{nav} from '{state['cartridge_name']}':\n\n"

    _log_activity(session_id, "get_passage", f"idx={idx}", 0)
    return header + full_text


# ============================================================
# HTTP MIDDLEWARE (auth + rate limiting)
# ============================================================

# --- Rate Limiter (sliding window, per-IP) ---
_rate_window: dict[str, collections.deque] = {}
RATE_LIMIT = 60          # requests per window
RATE_WINDOW_SEC = 60     # window size in seconds


def _check_rate_limit(client_id: str) -> bool:
    """Returns True if request is allowed, False if rate-limited."""
    now = time.time()
    if client_id not in _rate_window:
        _rate_window[client_id] = collections.deque()
    window = _rate_window[client_id]
    # Evict expired entries
    while window and window[0] < now - RATE_WINDOW_SEC:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        return False
    window.append(now)
    return True


def _setup_http_middleware(api_key: str | None):
    """Add auth + rate limiting middleware when running in HTTP mode."""
    try:
        from fastmcp.server.middleware import Middleware, MiddlewareContext
        from fastmcp.server.dependencies import get_http_request
    except ImportError:
        log.warning("FastMCP middleware not available — running HTTP without auth/rate-limiting")
        return

    class AuthAndRateLimitMiddleware(Middleware):
        async def on_call_tool(self, context: MiddlewareContext, call_next):
            try:
                request = get_http_request()
            except Exception:
                # Not an HTTP request (shouldn't happen in HTTP mode)
                return await call_next(context)

            # Rate limiting by client IP
            client_ip = request.client.host if request.client else "unknown"
            if not _check_rate_limit(client_ip):
                from fastmcp.exceptions import ToolError
                log.warning(f"Rate limited: {client_ip}")
                raise ToolError(f"Rate limited. Max {RATE_LIMIT} requests per {RATE_WINDOW_SEC}s.")

            # API key auth (if configured)
            if api_key:
                auth_header = request.headers.get("authorization", "")
                if not auth_header.startswith("Bearer "):
                    from fastmcp.exceptions import ToolError
                    raise ToolError("Access denied: missing Bearer token")
                token = auth_header.removeprefix("Bearer ").strip()
                if token != api_key:
                    from fastmcp.exceptions import ToolError
                    log.warning(f"Auth failed from {client_ip}")
                    raise ToolError("Access denied: invalid API key")

            return await call_next(context)

    mcp.add_middleware(AuthAndRateLimitMiddleware())
    if api_key:
        log.info("HTTP middleware: auth (Bearer token) + rate limiting enabled")
    else:
        log.info("HTTP middleware: rate limiting enabled (no auth — set MEMBOT_API_KEY to require)")


# ============================================================
# MEMPACK ENDPOINTS (Andy 2026-05-12)
# ============================================================
# Per-user writable carts. Pattern 0 in manifest sidecar, Pattern I at idx=1.
# Storage: cartridges/users/<owner_id>/<name>.cart.npz
# Discovery: find_mempacks(owner_id) walks MEMPACK_BASE_DIR recursively.
# Auth model (MVP): existing write API key required; owner_id passed in body.
# Long-term: verify Supabase JWT directly, derive owner_id from `sub` claim.

_DEFAULT_PATTERN_I_TEMPLATE = """# Pattern I — Default Mempack Behavior

You are an AI agent using a Mempack provisioned for {owner_id_short}.

## Identity
- Bound to user: {owner_id}
- Created: {created_at}
- Mempack version: 1.0

## Behavior
- Read this Pattern I first on every session to remind yourself who you are.
- Store findings worth keeping into your Mempack via `memory_store`.
- Update this Pattern I (re-write idx=1) as your behavior evolves over time.
- Search your Mempack before falling back to external sources.

## Specialization
(none yet — accumulates with use)

## Active threads
(none yet — track in-flight investigations here so they survive across sessions)
"""

_DEFAULT_BRIEFING_TEMPLATE = (
    "This is {owner_id_short}'s personal Mempack — a writable cart at "
    "cartridges/users/{owner_id}/{name}.cart.npz. Pattern 0 (this manifest) holds "
    "ownership/perms/cart-type metadata. Pattern I (idx=1) holds the agent's "
    "behavioral instructions; read it first on every session. Patterns 2+ are "
    "accumulated findings. Add new content via memory_store; update Pattern I "
    "to reflect evolved behavior. cart_type='agent-memory'."
)


@mcp.custom_route("/api/mempack/create", methods=["POST", "OPTIONS"])
async def rest_mempack_create(request: Request) -> JSONResponse:
    """Provision a new Mempack for a user.

    Body:
        name        (str, required)  cart name (e.g. "primary")
        owner_id    (str, required)  Supabase user UUID
        briefing    (str, optional)  override default briefing template
        pattern_i   (str, optional)  override default Pattern I template

    Creates cartridges/users/<owner_id>/<name>.cart.npz with two starter
    patterns: idx=0 = marker, idx=1 = Pattern I (behavioral text). Manifest
    carries cart_type='agent-memory', owner_id, owner_perms='rwda', plus the
    briefing.
    """
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    if _server_config.get("read_only"):
        return JSONResponse(
            {"status": "error", "error": "Server is read-only; cannot create Mempacks."},
            status_code=403, headers=_cors_headers(),
        )
    try:
        data = await request.json()
        name = data.get("name", "")
        owner_id = data.get("owner_id", "")

        if not name:
            return JSONResponse({"status": "error", "error": "name required"},
                                status_code=400, headers=_cors_headers())
        if not owner_id or not _UUID_RE.match(owner_id):
            return JSONResponse({"status": "error", "error": "owner_id must be a UUID"},
                                status_code=400, headers=_cors_headers())
        try:
            clean_name = sanitize_name(name)
        except ValueError as e:
            return JSONResponse({"status": "error", "error": str(e)},
                                status_code=400, headers=_cors_headers())

        user_dir = os.path.join(MEMPACK_BASE_DIR, owner_id)
        os.makedirs(user_dir, exist_ok=True)
        cart_path = os.path.join(user_dir, f"{clean_name}.cart.npz")

        if os.path.exists(cart_path):
            return JSONResponse(
                {"status": "error", "error": f"Mempack '{clean_name}' already exists for this user."},
                status_code=409, headers=_cors_headers(),
            )

        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        owner_id_short = owner_id[:8]

        briefing = data.get("briefing") or _DEFAULT_BRIEFING_TEMPLATE.format(
            owner_id=owner_id, owner_id_short=owner_id_short, name=clean_name,
        )
        pattern_i_text = data.get("pattern_i") or _DEFAULT_PATTERN_I_TEMPLATE.format(
            owner_id=owner_id, owner_id_short=owner_id_short, created_at=created_at,
        )
        # idx=0 = short marker; the manifest carries the real Pattern 0 metadata.
        idx0_marker = (
            f"Mempack header — Pattern 0 manifest in {clean_name}_manifest.json sidecar. "
            f"Pattern I (agent behavior) at idx=1. owner_id={owner_id}."
        )

        texts = [idx0_marker, pattern_i_text]
        embeddings = np.stack([
            embed_text(idx0_marker, prefix="search_document"),
            embed_text(pattern_i_text, prefix="search_document"),
        ]).astype(np.float32)

        save_as_npz(cart_path, embeddings, texts)
        save_manifest(
            cart_path, embeddings, len(texts),
            briefing=briefing,
            owner_id=owner_id,
            owner_perms="rwda",     # owner has full control over their Mempack
            group_perms="",
            world_perms="",         # private by default
            cart_type=CART_TYPE_AGENT_MEMORY,
        )

        size_mb = os.path.getsize(cart_path) / (1024 * 1024)
        log.info(f"Mempack created: {cart_path} ({size_mb:.2f} MB) owner={owner_id}")
        return JSONResponse({
            "status": "ok",
            "name": clean_name,
            "owner_id": owner_id,
            "path": cart_path,
            "size_mb": round(size_mb, 2),
            "cart_type": CART_TYPE_AGENT_MEMORY,
            "briefing": briefing,
            "pattern_i": pattern_i_text,
        }, headers=_cors_headers())
    except Exception as e:
        log.error(f"REST /api/mempack/create error: {e}")
        return JSONResponse({"status": "error", "error": str(e)},
                            status_code=500, headers=_cors_headers())


@mcp.custom_route("/api/mempacks", methods=["GET", "OPTIONS"])
async def rest_mempacks_list(request: Request) -> JSONResponse:
    """List Mempacks for a given owner_id (query param) or all (no param, admin use)."""
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    try:
        owner_id = request.query_params.get("owner_id", "") or None
        if owner_id is not None and not _UUID_RE.match(owner_id):
            return JSONResponse(
                {"status": "error", "error": "owner_id must be a UUID"},
                status_code=400, headers=_cors_headers(),
            )
        mempacks = find_mempacks(owner_id=owner_id)
        return JSONResponse({
            "status": "ok",
            "count": len(mempacks),
            "mempacks": mempacks,
        }, headers=_cors_headers())
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)},
                            status_code=500, headers=_cors_headers())


@mcp.custom_route("/api/copy", methods=["POST", "OPTIONS"])
async def rest_copy(request: Request) -> JSONResponse:
    """Copy a passage from any mounted cart into the user's Mempack.

    Body:
        src_cart    (str, required)  source cart name (must be findable)
        src_idx     (int, required)  passage index in src_cart
        dst_path    (str, required)  destination Mempack absolute path
                                     (typically cartridges/users/<uid>/<name>.cart.npz)
        note        (str, optional)  free-text reason ("why I kept this")

    Embeds the copied passage in the destination Mempack with provenance
    metadata appended to the text body so future searches preserve attribution.
    """
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    if _server_config.get("read_only"):
        return JSONResponse(
            {"status": "error", "error": "Server is read-only; cannot copy."},
            status_code=403, headers=_cors_headers(),
        )
    try:
        data = await request.json()
        src_cart_name = data.get("src_cart", "")
        src_idx = int(data.get("src_idx", -1))
        dst_path = data.get("dst_path", "")
        note = data.get("note", "")

        if not src_cart_name or src_idx < 0 or not dst_path:
            return JSONResponse(
                {"status": "error", "error": "src_cart + src_idx + dst_path all required"},
                status_code=400, headers=_cors_headers(),
            )

        # Resolve src
        carts = find_cartridges() + find_mempacks()
        src_match = [c for c in carts if c["name"] == src_cart_name]
        if not src_match:
            return JSONResponse(
                {"status": "error", "error": f"src_cart '{src_cart_name}' not found"},
                status_code=404, headers=_cors_headers(),
            )
        src_path = src_match[0]["path"]
        src_data = load_cartridge_safe(src_path)
        src_texts = src_data["texts"]
        if src_idx >= len(src_texts):
            return JSONResponse(
                {"status": "error", "error": f"src_idx {src_idx} out of range (cart has {len(src_texts)} patterns)"},
                status_code=400, headers=_cors_headers(),
            )
        src_text = src_texts[src_idx]

        # Resolve dst (must exist and be a Mempack — defense against arbitrary path writes)
        dst_real = os.path.realpath(dst_path)
        base_real = os.path.realpath(MEMPACK_BASE_DIR)
        if not dst_real.startswith(base_real + os.sep):
            return JSONResponse(
                {"status": "error", "error": "dst_path must be under MEMPACK_BASE_DIR"},
                status_code=403, headers=_cors_headers(),
            )
        if not os.path.exists(dst_path):
            return JSONResponse(
                {"status": "error", "error": f"dst Mempack not found at {dst_path}"},
                status_code=404, headers=_cors_headers(),
            )

        # Load destination cart, append the new pattern with provenance, save back
        dst_data = load_cartridge_safe(dst_path)
        dst_texts = list(dst_data["texts"])
        dst_embeddings = dst_data["embeddings"]

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        provenance_footer = (
            f"\n\n---\n[provenance] from {src_cart_name}#{src_idx} on {timestamp}"
            + (f" — {note}" if note else "")
        )
        new_text = src_text + provenance_footer
        new_embedding = embed_text(new_text, prefix="search_document").astype(np.float32)

        dst_texts.append(new_text)
        new_embeddings = np.vstack([dst_embeddings, new_embedding[None, :]])

        save_as_npz(dst_path, new_embeddings, dst_texts)

        # Re-save manifest preserving Pattern 0 fields
        dst_manifest = load_manifest(dst_path) or {}
        save_manifest(
            dst_path, new_embeddings, len(dst_texts),
            briefing=dst_manifest.get("briefing"),
            owner_id=dst_manifest.get("owner_id"),
            owner_perms=dst_manifest.get("owner_perms"),
            group_perms=dst_manifest.get("group_perms"),
            world_perms=dst_manifest.get("world_perms"),
            group_id=dst_manifest.get("group_id"),
            max_patterns=dst_manifest.get("max_patterns"),
            cart_type=dst_manifest.get("cart_type"),
        )

        log.info(f"Copy: {src_cart_name}#{src_idx} -> {dst_path} (now {len(dst_texts)} patterns)")
        return JSONResponse({
            "status": "ok",
            "src_cart": src_cart_name,
            "src_idx": src_idx,
            "dst_path": dst_path,
            "dst_new_idx": len(dst_texts) - 1,
            "preview": new_text[:200],
        }, headers=_cors_headers())
    except Exception as e:
        log.error(f"REST /api/copy error: {e}")
        return JSONResponse({"status": "error", "error": str(e)},
                            status_code=500, headers=_cors_headers())


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Membot — Brain Cartridge Server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "http", "sse"],
                        help="Transport mode (default: stdio)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="HTTP bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="HTTP port (default: 8000)")
    parser.add_argument("--read-only", action="store_true", default=True,
                        help="Disable memory_store and save_cartridge (default: on)")
    parser.add_argument("--writable", action="store_true",
                        help="Enable memory_store and save_cartridge (override read-only)")
    parser.add_argument("--mount", type=str, default=None,
                        help="Auto-mount a cartridge on startup (creates empty cart if not found)")
    args = parser.parse_args()

    _server_config["read_only"] = not args.writable

    log.info("Starting Membot — Brain Cartridge Server...")
    log.info(f"Transport: {args.transport}")
    if _server_config["read_only"]:
        log.info("READ-ONLY mode (default): memory_store and save_cartridge disabled. Use --writable to enable.")
    else:
        log.info("WRITABLE mode: memory_store and save_cartridge enabled")
    log.info(f"Cartridge dirs: {CARTRIDGE_DIRS}")
    log.info(f"Security: max_entries={MAX_ENTRIES}, max_text={MAX_TEXT_LENGTH}, pkl=trusted-dirs-only")

    # Try GPU at startup (non-fatal)
    init_gpu()

    # Count available cartridges
    carts = find_cartridges()
    log.info(f"Found {len(carts)} cartridges")

    # Auto-mount cartridge if requested
    if args.mount:
        cart_name = args.mount
        # Check if cartridge exists; if not, create empty one
        matching = [c for c in carts if cart_name.lower() in c["name"].lower()]
        if not matching:
            log.info(f"Cartridge '{cart_name}' not found — creating empty cartridge")
            save_dir = os.path.join(BASE_DIR, "cartridges")
            os.makedirs(save_dir, exist_ok=True)
            empty_path = os.path.join(save_dir, f"{cart_name}.cart.npz")
            empty_emb = np.zeros((0, 768), dtype=np.float32)
            save_as_npz(empty_path, empty_emb, [])
            save_manifest(empty_path, empty_emb, 0)
            log.info(f"Created empty cartridge: {empty_path}")

        # Mount into default session (bypass @mcp.tool decorator)
        sid = _resolve_session_id("")
        state = _get_session(sid)
        mount_carts = find_cartridges()
        mount_match = [c for c in mount_carts if cart_name.lower() in c["name"].lower()]
        if mount_match:
            cart = mount_match[0]
            data = load_cartridge_safe(cart["path"])
            state["embeddings"] = data["embeddings"]
            state["texts"] = data["texts"]
            state["cartridge_name"] = cart["name"]
            state["cartridge_path"] = cart["path"]
            state["signatures"] = None
            state["modified"] = False
            if len(data["embeddings"]) > 0:
                state["binary_corpus"] = (data["embeddings"] > 0).astype(np.uint8)
            else:
                state["binary_corpus"] = None
            ok, verify_msg = verify_manifest(cart["path"], data["embeddings"], len(data["texts"]))
            log.info(f"Auto-mounted '{cart['name']}': {len(data['texts'])} memories, integrity={verify_msg}")
        else:
            log.error(f"Auto-mount failed: '{cart_name}' still not found after creation")

    # Setup HTTP middleware if not in stdio mode
    if args.transport != "stdio":
        api_key = os.environ.get("MEMBOT_API_KEY")
        _setup_http_middleware(api_key)
        log.info(f"HTTP server starting on {args.host}:{args.port}")
        if _server_config["read_only"]:
            log.info("Public server mode: write operations blocked")

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)
