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
# ⚠ WE DO NOT TAKE THEIR num_gpu DEFAULT. dp set it to 0 (force CPU) "because that is the
# fleet's decision" — correct for a box whose GPU belongs to a being, wrong as a product
# default. Membot ships to hosts whose hardware we do not know, and here the 4080 has
# nothing else on it, so forcing CPU would be a silent regression on every store and search.
# ⇒ UNSET MEANS DO NOT SEND THE FIELD, so Ollama keeps deciding and nothing changes for
#   anyone who has not asked. That is exactly the pattern dp's own LATER commit used for
#   keep_alive ("Unset: the field is not sent and Ollama's default applies, so nothing
#   changes for other hosts") — we are applying his second lesson to his first commit.
_ollama_embed_num_gpu = (os.getenv("MEMBOT_OLLAMA_EMBED_NUM_GPU") or "").strip() or None
_ollama_embed_keep_alive = (os.getenv("MEMBOT_OLLAMA_EMBED_KEEP_ALIVE") or "").strip() or None


def _embed_via_ollama(text: str) -> np.ndarray:
    """Get embedding from Ollama API (no extra RAM — reuses running Ollama process).

    MEMBOT_OLLAMA_EMBED_NUM_GPU pins the layer count (0 = CPU); MEMBOT_OLLAMA_EMBED_KEEP_ALIVE
    is Ollama's keep_alive ("0" unloads immediately, "30s", "5m", ...). Unset, neither field
    is sent and Ollama's own defaults apply.
    """
    import urllib.request
    body = {"model": _ollama_embed_model, "input": text}
    if _ollama_embed_num_gpu is not None:
        body["options"] = {"num_gpu": int(_ollama_embed_num_gpu)}
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
