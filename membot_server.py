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
import hashlib
import pickle
import zlib
import time
import json
import logging
import argparse
import collections
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
]
HAMMING_BLEND = 0.3           # 70% cosine + 30% sign_zero Hamming (replaces physics L2)

# --- Security Limits ---
MAX_ENTRIES = 100_000        # Max memories per cartridge
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


def save_manifest(cart_path: str, embeddings: np.ndarray, n_texts: int):
    """Save integrity manifest alongside cartridge."""
    manifest_path = cart_path.rsplit(".", 1)[0] + "_manifest.json"
    manifest = {
        "version": "mcp-v3",
        "count": n_texts,
        "fingerprint": compute_fingerprint(embeddings, n_texts),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Manifest saved: {manifest['fingerprint']} ({n_texts} entries)")
    return manifest


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
        "embeddings": None,       # (N, 768) float32
        "texts": [],              # list[str]
        "binary_corpus": None,    # (N, 768) uint8 — sign_zero encoding for Hamming search
        "signatures": None,       # (N, 4096) float32 or None (legacy, not used in search)
        "lattice": None,          # CUDA wrapper or None
        "gpu_available": False,
        "modified": False,        # True if memory_store was called since last save
        "last_access": time.time(),
        "created_at": time.time(),
        "query_count": 0,
        "mount_count": 0,
        "last_action": None,      # e.g. "search 'earthquake'" or "mount wiki-10k"
        "last_action_time": None,
    }


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
# EMBEDDING VIA SENTENCE-TRANSFORMERS
# ============================================================

_embed_model = None  # lazy-loaded SentenceTransformer

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

def embed_text(text: str, prefix: str = "search_query") -> np.ndarray:
    """Get 768-dim Nomic embedding via SentenceTransformer (matches Studio cartridge embedder)."""
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

    return {
        "embeddings": embeddings,
        "texts": list(passages),
        "version": version,
        "format": "pkl",
    }


def load_npz_cartridge(path: str) -> dict:
    """Load an .npz cartridge (safe — no code execution)."""
    data = np.load(path, allow_pickle=True)

    result = {"format": "npz", "version": "unknown"}

    # Handle various NPZ layouts
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


@mcp.custom_route("/api/store", methods=["POST", "OPTIONS"])
async def rest_store(request: Request) -> JSONResponse:
    """REST wrapper for memory_store — used by Heartbeat browser extension."""
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    try:
        data = await request.json()
        result = memory_store.fn(
            content=data.get("content", ""),
            tags=data.get("tags", ""),
            session_id=data.get("session_id", "")
        )
        return JSONResponse({"status": "ok", "result": result}, headers=_cors_headers())
    except Exception as e:
        log.error(f"REST /api/store error: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500, headers=_cors_headers())


@mcp.custom_route("/api/search", methods=["POST", "OPTIONS"])
async def rest_search(request: Request) -> JSONResponse:
    """REST wrapper for memory_search — used by Heartbeat browser extension."""
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    try:
        data = await request.json()
        result = memory_search.fn(
            query=data.get("query", ""),
            top_k=data.get("top_k", 5),
            session_id=data.get("session_id", "")
        )
        return JSONResponse({"status": "ok", "result": result}, headers=_cors_headers())
    except Exception as e:
        log.error(f"REST /api/search error: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500, headers=_cors_headers())


@mcp.custom_route("/api/status", methods=["GET", "OPTIONS"])
async def rest_status(request: Request) -> JSONResponse:
    """REST health check — returns mounted cartridge + memory count."""
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    try:
        result = get_status.fn()
        return JSONResponse({"status": "ok", "result": result}, headers=_cors_headers())
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500, headers=_cors_headers())


@mcp.custom_route("/api/save", methods=["POST", "OPTIONS"])
async def rest_save(request: Request) -> JSONResponse:
    """REST wrapper for save_cartridge — persist to disk."""
    if request.method == "OPTIONS":
        return JSONResponse({}, headers=_cors_headers())
    try:
        result = save_cartridge.fn()
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


_DEPOT_HTML = """\
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
        state["modified"] = False

        # Compute sign_zero binary corpus for Hamming search (96 bytes per pattern)
        if len(embeddings) > 0:
            state["binary_corpus"] = (embeddings > 0).astype(np.uint8)
        else:
            state["binary_corpus"] = None

        n = len(texts)
        dim = embeddings.shape[1] if len(embeddings) > 0 else 0
        elapsed_ms = (time.time() - t0) * 1000

        log.info(f"Loaded {n} entries, dim={dim}, format={data['format']}, integrity={verify_msg}, sign_zero={n} codes, in {elapsed_ms:.0f}ms")

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

        state["mount_count"] = state.get("mount_count", 0) + 1
        _log_activity(session_id, "mount", cart['name'], elapsed_ms)
        return f"Mounted '{cart['name']}': {n} memories, {dim}-dim, {data['format'].upper()}, integrity={verify_msg}{gpu_msg}. Session: {session_id}"

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

    if state["embeddings"] is None or len(state["embeddings"]) == 0:
        return "Cartridge is empty."

    try:
        t0 = time.time()

        # 1. Embed query
        query_emb = embed_text(query, prefix="search_query")

        # 2. Embedding cosine similarity
        stored = state["embeddings"]
        stored_norms = np.linalg.norm(stored, axis=1, keepdims=True) + 1e-9
        query_norm = np.linalg.norm(query_emb) + 1e-9
        emb_scores = np.dot(stored / stored_norms, query_emb / query_norm)

        # 3. Sign-zero Hamming search: binary XOR distance as secondary signal
        search_mode = "embedding"
        ham_scores = None

        if HAMMING_BLEND > 0 and state["binary_corpus"] is not None:
            try:
                # Sign-zero encode query: bit_i = 1 if query_emb_i > 0
                q_bin = (query_emb > 0).astype(np.uint8)
                corpus_bin = state["binary_corpus"]
                n_bin = min(len(corpus_bin), len(emb_scores))

                # Hamming similarity: 1 - (XOR distance / n_bits)
                n_bits = corpus_bin.shape[1]
                xor = np.bitwise_xor(q_bin, corpus_bin[:n_bin])
                dist = xor.sum(axis=1)
                ham_scores = 1.0 - dist.astype(np.float32) / n_bits

                # Blend: 70% embedding cosine + 30% Hamming
                blended = np.copy(emb_scores)
                blended[:n_bin] = (1.0 - HAMMING_BLEND) * emb_scores[:n_bin] + HAMMING_BLEND * ham_scores
                scores = blended
                search_mode = "hamming+embedding"
                log.info(f"Hamming search: sign_zero blended {1.0 - HAMMING_BLEND:.0%}/{HAMMING_BLEND:.0%}, {n_bin} patterns")
            except Exception as e:
                log.warning(f"Hamming search failed, falling back to embedding: {e}")
                scores = emb_scores
        else:
            scores = emb_scores
            if state["binary_corpus"] is None:
                log.info("No binary corpus — embedding-only search")

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

        # 5. Format results
        results = []
        for rank, (i, final_score, kw_boost) in enumerate(boosted[:top_k], 1):
            if final_score < 0.1:
                continue
            text = state["texts"][i]
            if len(text) > 500:
                text = text[:500] + "..."
            if verbose:
                cos_s = f"{float(emb_scores[i]):.3f}"
                ham_s = f"{float(ham_scores[i]):.3f}" if ham_scores is not None and i < len(ham_scores) else "—"
                kw_s = f"+{kw_boost:.3f}" if kw_boost > 0 else "—"
                results.append(f"#{rank} [{final_score:.3f}] cos={cos_s} ham={ham_s} kw={kw_s}\n{text}")
            else:
                results.append(f"#{rank} [{final_score:.3f}] {text}")

        state["query_count"] = state.get("query_count", 0) + 1
        _log_activity(session_id, "search", f"'{query[:40]}' → {len(boosted[:top_k])} results", elapsed_ms)

        if not results:
            return f"No relevant matches for '{query}' (searched {len(state['texts'])} memories, {elapsed_ms:.0f}ms)"

        kw_label = f"+kw" if keywords else ""
        if search_mode == "hamming+embedding":
            mode_label = f"hamming+embedding 70/30{kw_label}"
        else:
            mode_label = f"embedding-only{kw_label}"
        header = f"Search [{mode_label}]: {len(results)} results from '{state['cartridge_name']}' ({elapsed_ms:.0f}ms)\n"
        return header + "\n\n".join(results)

    except Exception as e:
        log.error(f"Search error: {e}")
        return f"Search error: {e}"


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

    # Reset session state but keep it alive
    state["cartridge_name"] = None
    state["cartridge_path"] = None
    state["embeddings"] = None
    state["texts"] = []
    state["binary_corpus"] = None
    state["signatures"] = None
    state["modified"] = False

    _log_activity(session_id, "unmount", name)
    return f"Unmounted '{name}'.{warning}"


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

    return (
        f"Cartridge: {cart}{modified} | "
        f"Memories: {n}/{MAX_ENTRIES} | "
        f"Embedding dim: {dim} | "
        f"Hamming index: {'Yes' if has_hamming else 'No'} ({ham_bytes}) | "
        f"GPU: {gpu} | "
        f"Sessions: {active_sessions} | "
        f"Session: {session_id} | "
        f"Search: cosine+hamming 70/30+kw | "
        f"Embed: nomic-embed-text-v1.5 via SentenceTransformer"
    )


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
    parser.add_argument("--read-only", action="store_true",
                        help="Disable memory_store and save_cartridge (public server mode)")
    parser.add_argument("--mount", type=str, default=None,
                        help="Auto-mount a cartridge on startup (creates empty cart if not found)")
    args = parser.parse_args()

    _server_config["read_only"] = args.read_only

    log.info("Starting Membot — Brain Cartridge Server...")
    log.info(f"Transport: {args.transport}")
    if args.read_only:
        log.info("READ-ONLY mode: memory_store and save_cartridge disabled")
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
        if args.read_only:
            log.info("Public server mode: write operations blocked")

    mcp.run(transport=args.transport, host=args.host, port=args.port)
