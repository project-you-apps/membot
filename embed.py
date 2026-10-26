"""Embedding backend — Ollama or SentenceTransformer. NO MCP DEPENDENCY, DELIBERATELY.

WHY THIS MODULE EXISTS (2026-08-01)
------------------------------------
This code lived in `membot_server.py`, which begins `from fastmcp import FastMCP`. Because
`multi_cart.py` reached into the server for `embed_text`, **the read-only, no-server search
path transitively required the MCP server package at module load.**

HUB reported the consequence on 2026-07-28, on the box that PUBLISHES the fleet cartridge:

    | interpreter                  | mount            | search                       |
    | membot/.venv/bin/python      | 80 patterns      | 3 hits                       |
    | system python3               | status: "error"  | ModuleNotFoundError: fastmcp |
    | ~/.web4/venv/bin/python      | status: "error"  | ModuleNotFoundError: fastmcp |

Their framing, which is sharper than "a missing requirement" and is the reason this file
exists rather than just a new line in requirements.txt:

    "the dependency is on the *interpreter*, not on a running server... the sharper form is
    'needs the one interpreter the server was installed into', which is a thing no consumer
    can see from the call site."

A consumer that only wants to READ a cartridge should not need the server's dependencies. So
the embedding subsystem lives here, importing nothing from `membot_server`, and both the
server and the read paths import it from here.

**Keep it that way.** If something in this file ever needs an import from `membot_server`,
the coupling is back and the read path silently reacquires the server's dependency tree.

Requires: numpy, plus `sentence_transformers` ONLY if the Ollama backend is unavailable
(imported lazily inside `_get_embed_model`, so an Ollama-backed box never pays for it).
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

# Same logger name the server uses, so output is indistinguishable from before the split.
log = logging.getLogger("membot")

_embed_model = None  # lazy-loaded SentenceTransformer (if not using Ollama)
_embed_backend = os.environ.get("MEMBOT_EMBED_BACKEND", "auto")  # "ollama", "st", or "auto"
_ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_ollama_embed_model = os.environ.get("MEMBOT_OLLAMA_EMBED_MODEL", "nomic-embed-text")

_resolved_backend = None


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


# WHERE THE EMBEDDER RUNS, AND HOW LONG IT STAYS. Both absorbed from dp-web4/membot
# (c31c3c6 2026-09-30, 75cc47c 2026-10-12), where the fleet hit them first: on Legion the
# being's model holds 14.9 GB of a 16 GB card, and an embed without `num_gpu` loads
# nomic-embed-text into the remaining ~1.1 GB beside it — a contention class that already
# cost that machine a swap stall and a watchdog kill. On Sprout, nomic stays resident for
# Ollama's default 5 minutes after every recall, taking room the being could use.
#
# ⭐ CPU IS THE DEFAULT, and dp's reasoning generalizes past his hardware (Andy, 2026-10-25).
# A first pass here kept Ollama's own placement on the grounds that "membot ships to hosts
# whose hardware we do not know." That was wrong twice:
#   - "let Ollama decide" is not neutral. Ollama's auto-placement TAKES the GPU when one
#     exists, so the no-opinion default is really the most aggressive one.
#   - membot embeds ONE text per operation — one per store, one per query; corpus vectors are
#     precomputed in the cart. The accelerator saves tens of ms on a call that already does
#     disk and scoring, and costs a resident ~300 MB-1 GB of VRAM for five minutes after it.
#     On this box that card runs the lattice CUDA work, which is exactly the contention dp
#     describes on Legion.
# ⇒ A background memory layer should not claim the accelerator; the thing doing the THINKING
#   should. Opt in with a layer count on a host where the GPU is genuinely free, or pass
#   "auto" to restore Ollama's own placement.
_num_gpu_raw = (os.getenv("MEMBOT_OLLAMA_EMBED_NUM_GPU") or "").strip().lower()
_ollama_embed_num_gpu = None if _num_gpu_raw == "auto" else int(_num_gpu_raw or "0")
_ollama_embed_keep_alive = (os.getenv("MEMBOT_OLLAMA_EMBED_KEEP_ALIVE") or "").strip() or None


def _embed_via_ollama(text: str) -> np.ndarray:
    """Get embedding from Ollama API (no extra RAM — reuses running Ollama process).

    Embeds on the CPU unless told otherwise. MEMBOT_OLLAMA_EMBED_NUM_GPU takes a layer count
    (0 = CPU, the default) or "auto" to let Ollama place it. MEMBOT_OLLAMA_EMBED_KEEP_ALIVE is
    Ollama's keep_alive ("0" unloads immediately, "30s", "5m", ...); unset, that field is not
    sent and Ollama's own default applies.
    """
    import urllib.request
    body = {"model": _ollama_embed_model, "input": text}
    if _ollama_embed_num_gpu is not None:
        body["options"] = {"num_gpu": _ollama_embed_num_gpu}
    if _ollama_embed_keep_alive is not None:
        body["keep_alive"] = _ollama_embed_keep_alive
    payload = json.dumps(body).encode()
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


def active_backend() -> str:
    """Which backend is in use, resolving it if that has not happened yet.

    Exists so a caller can REPORT the backend without having to embed something first.
    Addresses the "silent Ollama fallback" item from the 2026-07-22 cold-install audit:
    a fallback nobody can see is indistinguishable from a misconfiguration.
    """
    global _resolved_backend
    if _resolved_backend is None:
        _resolved_backend = _resolve_embed_backend()
    return _resolved_backend
