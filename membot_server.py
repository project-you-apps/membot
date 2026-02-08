"""
Membot — Brain Cartridge Server for AI Agents
==============================================
MCP server that gives AI agents swappable, physics-enhanced memory.
Built on the Vector+ Lattice Engine.

Architecture:
  - SentenceTransformer (nomic-ai/nomic-embed-text-v1.5) for 768-dim embeddings
  - Physics-enhanced search: query → lattice settle → L2 signature → blended ranking
  - Blends 70% embedding cosine + 30% L2 physics similarity (when GPU + signatures available)
  - Falls back to embedding-only cosine if no GPU or no signatures
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

Transport: stdio (MCP JSON-RPC)

Usage:
  python membot_server.py

See README.md for agent configuration.

https://github.com/project-you-apps/membot
"""

from mcp.server.fastmcp import FastMCP
import os
import sys
import re
import hashlib
import pickle
import zlib
import time
import json
import logging
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
PHYSICS_ENABLED = True        # 70/30 embedding+physics blend (embedder mismatch fixed)

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
# STATE
# ============================================================

state = {
    "cartridge_name": None,
    "cartridge_path": None,
    "embeddings": None,       # (N, 768) float32
    "texts": [],              # list[str]
    "signatures": None,       # (N, 4096) float32 or None
    "lattice": None,          # CUDA wrapper or None
    "gpu_available": False,
    "modified": False,        # True if memory_store was called since last save
}


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

def init_gpu():
    """Try to load CUDA LatticeRunner. Returns True if successful."""
    if state["lattice"] is not None:
        return state["gpu_available"]
    try:
        sys.path.insert(0, BASE_DIR)
        from multi_lattice_wrapper_v7 import MultiLatticeCUDAv7
        state["lattice"] = MultiLatticeCUDAv7(lattice_size=4096, verbose=0)
        state["gpu_available"] = True
        log.info("CUDA LatticeRunner V7 ready (4096x4096)")
        return True
    except Exception as e:
        log.warning(f"CUDA not available (CPU-only mode): {e}")
        state["gpu_available"] = False
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
def mount_cartridge(name: str) -> str:
    """Mount a brain cartridge by name (or partial name).
    Loads embeddings and text into memory for searching.
    Verifies integrity against manifest if available.

    Args:
        name: Cartridge name (exact or partial match)
    """
    log.info(f"mount_cartridge({name})")

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

        n = len(texts)
        dim = embeddings.shape[1] if len(embeddings) > 0 else 0
        elapsed_ms = (time.time() - t0) * 1000

        log.info(f"Loaded {n} entries, dim={dim}, format={data['format']}, integrity={verify_msg} in {elapsed_ms:.0f}ms")

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
                state["lattice"].load_brain(brain_path)
                gpu_msg = ", brain loaded to GPU"
                log.info("Brain weights loaded to GPU")
            except Exception as e:
                gpu_msg = f", brain load failed: {e}"
                log.warning(f"Brain load failed: {e}")

        return f"Mounted '{cart['name']}': {n} memories, {dim}-dim, {data['format'].upper()}, integrity={verify_msg}{gpu_msg}. Ready."

    except PermissionError as e:
        log.error(f"Security block: {e}")
        return str(e)
    except Exception as e:
        log.error(f"Mount failed: {e}")
        return f"Failed to mount '{clean_name}': {e}"


@mcp.tool()
def memory_search(query: str, top_k: int = 5) -> str:
    """Search the mounted cartridge using lattice physics + embedding similarity.
    Runs the query through the neural lattice (settle → L2 signature) and blends
    physics-based similarity with embedding cosine for ranked results.

    Falls back to embedding-only search if GPU or signatures are unavailable.

    Args:
        query: Natural language search query
        top_k: Number of results to return (default 5)
    """
    if len(query) > MAX_QUERY_LENGTH:
        return f"Query too long ({len(query)} chars). Max is {MAX_QUERY_LENGTH}."

    log.info(f"memory_search('{query[:60]}', top_k={top_k})")

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

        # 3. Physics search: settle query through lattice, compare L2 signatures
        search_mode = "embedding"
        physics_blend = 0.0

        if PHYSICS_ENABLED and state["gpu_available"] and state["lattice"] is not None and state["signatures"] is not None:
            try:
                ml = state["lattice"]
                ml.reset()
                ml.imprint_vector(query_emb)
                ml.settle(frames=2, learn=False)
                query_sig = ml.recall_l2().flatten()

                # Cosine similarity on L2 signatures (4096-dim physics space)
                sigs = state["signatures"]
                # Trim to match if cartridge was extended by memory_store
                n_sigs = min(len(sigs), len(emb_scores))
                sigs = sigs[:n_sigs]

                sig_norms = np.linalg.norm(sigs, axis=1, keepdims=True) + 1e-9
                q_sig_norm = np.linalg.norm(query_sig) + 1e-9
                sig_scores = np.dot(sigs / sig_norms, query_sig / q_sig_norm)

                # Confidence gating: skip physics blend if L2 signatures are degenerate.
                # Gate on sig_std only — low variance means flat attractor (no discrimination).
                # Overlap check removed: at large N, low overlap is expected for cross-domain
                # queries and doesn't indicate bad signatures.
                sig_std = float(np.std(sig_scores))

                gate_physics = sig_std < 0.02

                if gate_physics:
                    scores = emb_scores
                    search_mode = "embedding (physics gated)"
                    physics_blend = 0.0
                    log.info(f"Physics confidence gating: sig_std={sig_std:.4f} — "
                             f"using embedding-only")
                else:
                    # Blend: 70% embedding + 30% physics (matches Studio v0.83 default)
                    blended = np.copy(emb_scores)
                    blended[:n_sigs] = 0.7 * emb_scores[:n_sigs] + 0.3 * sig_scores
                    scores = blended
                    search_mode = "physics+embedding"
                    physics_blend = 0.3
                    log.info(f"Physics search: L2 sig blended 70/30, sig_std={sig_std:.4f}, {n_sigs} patterns")
            except Exception as e:
                log.warning(f"Physics search failed, falling back to embedding: {e}")
                scores = emb_scores
        else:
            scores = emb_scores
            if state["signatures"] is None:
                log.info("No signatures loaded — embedding-only search")
            elif not state["gpu_available"]:
                log.info("No GPU — embedding-only search")

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
            boosted.append((i, base_score + boost))

        boosted.sort(key=lambda x: x[1], reverse=True)

        elapsed_ms = (time.time() - t0) * 1000

        # 5. Format results
        results = []
        for rank, (i, final_score) in enumerate(boosted[:top_k], 1):
            if final_score < 0.1:
                continue
            text = state["texts"][i]
            if len(text) > 500:
                text = text[:500] + "..."
            results.append(f"#{rank} [{final_score:.3f}] {text}")

        if not results:
            return f"No relevant matches for '{query}' (searched {len(state['texts'])} memories, {elapsed_ms:.0f}ms)"

        kw_label = f"+kw" if keywords else ""
        mode_label = f"physics+embedding 70/30{kw_label}" if search_mode == "physics+embedding" else f"embedding-only{kw_label}"
        header = f"Search [{mode_label}]: {len(results)} results from '{state['cartridge_name']}' ({elapsed_ms:.0f}ms)\n"
        return header + "\n\n".join(results)

    except Exception as e:
        log.error(f"Search error: {e}")
        return f"Search error: {e}"


@mcp.tool()
def memory_store(content: str, tags: str = "") -> str:
    """Store new text in the currently mounted cartridge.
    The text is embedded via Nomic and added to the searchable memory.
    If GPU is available, the pattern is also imprinted into the lattice.

    Args:
        content: Text content to memorize (max 10,000 chars)
        tags: Optional metadata tags (prepended to stored text)
    """
    if len(content) > MAX_TEXT_LENGTH:
        return f"Text too long ({len(content)} chars). Max is {MAX_TEXT_LENGTH}."

    if state["cartridge_name"] is None:
        return "No cartridge mounted. Use mount_cartridge first."

    n_current = len(state["texts"]) if state["texts"] else 0
    if n_current >= MAX_ENTRIES:
        return f"Cartridge full ({n_current}/{MAX_ENTRIES}). Save and start a new cartridge."

    log.info(f"memory_store('{content[:60]}...')")

    try:
        t0 = time.time()

        # 1. Embed via Ollama
        emb = embed_text(content, prefix="search_document")

        # 2. Add to embedding matrix
        if state["embeddings"] is None or len(state["embeddings"]) == 0:
            state["embeddings"] = emb.reshape(1, -1)
        else:
            state["embeddings"] = np.vstack([state["embeddings"], emb.reshape(1, -1)])

        # 3. Store text
        stored_text = f"[{tags}] {content}" if tags else content
        state["texts"].append(stored_text)
        state["modified"] = True

        # 4. GPU lattice imprint (if available)
        gpu_msg = ""
        if state["gpu_available"] and state["lattice"] is not None:
            try:
                ml = state["lattice"]
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
def save_cartridge(name: str = "") -> str:
    """Save the current cartridge to disk as a secure .npz file.
    Generates a SHA256 manifest for integrity verification.

    Args:
        name: Optional name for the saved cartridge (default: use mounted name)
    """
    log.info(f"save_cartridge('{name}')")

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
        if state["gpu_available"] and state["lattice"] is not None:
            brain_path = os.path.join(save_dir, f"{save_name}_brain.npy")
            try:
                state["lattice"].save_brain(brain_path)
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
def unmount() -> str:
    """Unmount the current cartridge and free memory."""
    log.info("unmount")

    if state["cartridge_name"] is None:
        return "No cartridge mounted."

    name = state["cartridge_name"]
    warning = ""
    if state["modified"]:
        warning = " WARNING: Unsaved changes were discarded."

    state["cartridge_name"] = None
    state["cartridge_path"] = None
    state["embeddings"] = None
    state["texts"] = []
    state["signatures"] = None
    state["modified"] = False

    return f"Unmounted '{name}'.{warning}"


@mcp.tool()
def get_status() -> str:
    """Get server status: mounted cartridge, memory count, GPU availability."""
    log.info("get_status")

    cart = state["cartridge_name"] or "None"
    n = len(state["texts"]) if state["texts"] else 0
    dim = state["embeddings"].shape[1] if state["embeddings"] is not None and len(state["embeddings"]) > 0 else 0
    has_sigs = state["signatures"] is not None
    gpu = "Ready" if state["gpu_available"] else "Not available"
    modified = " (unsaved changes)" if state["modified"] else ""

    return (
        f"Cartridge: {cart}{modified} | "
        f"Memories: {n}/{MAX_ENTRIES} | "
        f"Embedding dim: {dim} | "
        f"Signatures: {'Yes' if has_sigs else 'No'} | "
        f"GPU: {gpu} | "
        f"Embed: nomic-embed-text-v1.5 via SentenceTransformer"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    log.info("Starting Membot — Brain Cartridge Server...")
    log.info(f"Cartridge dirs: {CARTRIDGE_DIRS}")
    log.info(f"Security: max_entries={MAX_ENTRIES}, max_text={MAX_TEXT_LENGTH}, pkl=trusted-dirs-only")

    # Try GPU at startup (non-fatal)
    init_gpu()

    # Count available cartridges
    carts = find_cartridges()
    log.info(f"Found {len(carts)} cartridges")

    mcp.run(transport="stdio")
