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
from pathlib import Path

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


# ═════════════════════════════════════════════════════════════════════════════
# INTEGRITY FINGERPRINT
#
# ⛔⛔ WHAT THE v1 ALGORITHM ACTUALLY COVERED, measured 2026-11-12:
#
#     h.update(embeddings[0].tobytes())
#     h.update(embeddings[-1].tobytes())
#     h.update(str(n_texts).encode())
#     return h.hexdigest()[:16]
#
#   · the FIRST and LAST embedding only ⇒ every embedding in between could be
#     altered freely and the fingerprint still matched
#   · NO passage text at all ⇒ a passage could be rewritten undetected
#   · truncated to 16 hex chars ⇒ 64 bits
#   · and `verify_manifest` returned (True, "no manifest (legacy cartridge)")
#     when the manifest was ABSENT, so an unmanifested cart VERIFIED.
#
# ⭐⭐ WHY THIS MATTERS MORE THAN IT LOOKS. On 2026-10-18 the boot path was fixed
#   to act on its integrity verdict -- Andy: "if it doesn't see if it's ok what's
#   the point of checking in the first place?" ⇒ So the ENFORCEMENT was correct
#   and the CHECK was nearly meaningless. A gate faithfully enforcing a test that
#   tested almost nothing is worse than a missing gate, because it reads as cover.
#
# ⓘ Found by reading a fleet test TITLE on dp-web4/membot, branch
#   codex/membot-save-race-review:
#     test_manifest_covers_text_and_middle_embedding_not_just_group_metadata
#
# ⭐ A FINGERPRINT IS INTEGRITY, NEVER AUTHENTICITY. It proves the bytes did not
#   change. It says nothing about who produced them. Authenticity needs a
#   signature, which is gate 2 of the gauntlet and is not this function's job.
#   See docs/DESIGN-the-gauntlet-verify-before-materialize-2026-11-12.md
# ═════════════════════════════════════════════════════════════════════════════

# Written into every new manifest and checked on read, so a v1 manifest is a
# NAMED refusal rather than an unexplained mismatch.
FINGERPRINT_ALGO = "sha256-full-v2"
LEGACY_ALGO = "trunc64-ends-v1"


def _hash_embeddings(h: "hashlib._Hash", embeddings) -> int:
    """Hash EVERY embedding, with shape and dtype bound in. Returns rows hashed.

    ⭐ Shape and dtype are hashed because `(2, 768) float32` and `(768, 2) float32`
      hold the same bytes and are not the same cartridge.
    ⓘ Uses a memoryview per row rather than `.tobytes()` on the whole array, so a
      725 MB cart does not get a 725 MB copy made of it to be hashed.
    """
    if embeddings is None:
        h.update(b"embeddings:none")
        return 0
    arr = np.ascontiguousarray(embeddings)
    h.update(f"embeddings:{arr.dtype.str}:{arr.shape}".encode())
    # ⛔ `memoryview().cast()` raises TypeError on ANY shape containing a zero, so
    #   a cart with no rows crashed the hash instead of hashing to a stable value.
    #   Found 2026-11-13 by the first real sweep, on `embed-consistency-test`.
    #   ⚠ This is the same class as the missing length guard in the inline copies
    #     this function replaced -- removed there, reintroduced here in a new form.
    #   ⭐ Nothing is lost by skipping the empty buffer: the shape and dtype line
    #     above already separates (0, 768) from (0, 384) and both from None.
    if arr.size:
        h.update(memoryview(arr).cast("B"))
    # A 0-d array has no len(). Report rows, not a crash.
    return int(arr.shape[0]) if arr.ndim else 0


def _hash_texts(h: "hashlib._Hash", texts) -> int:
    """Hash every passage, length-prefixed. Returns passages hashed.

    ⛔ LENGTH-PREFIXED ON PURPOSE. Hashing concatenated text makes
      ["ab", "c"] and ["a", "bc"] identical, so a passage boundary could be moved
      without changing the digest. The prefix makes the split part of the content.
    """
    if texts is None:
        h.update(b"texts:none")
        return 0
    n = 0
    for t in texts:
        b = (t if isinstance(t, str) else str(t)).encode("utf-8", "surrogatepass")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
        n += 1
    return n


def compute_fingerprint(embeddings: np.ndarray, n_texts: int,
                        texts=None) -> str:
    """Full-coverage integrity fingerprint: every embedding, every passage.

    ⛔ `texts` IS REQUIRED. It is a keyword with no default ONLY so that a caller
      written against the v1 signature fails loudly instead of silently producing
      a weaker hash. ⇒ This is the day's own lesson applied to our own API: a
      function that cannot do its whole job must refuse, not do part of it.
    """
    if texts is None:
        raise TypeError(
            "compute_fingerprint() now requires `texts=`. The v1 algorithm hashed "
            "only the first and last embedding plus a count, covering no passage "
            "text at all; passing no texts would silently reproduce that weakness. "
            "Pass the passage list."
        )
    h = hashlib.sha256()
    h.update(f"{FINGERPRINT_ALGO}\n".encode())
    n_emb = _hash_embeddings(h, embeddings)
    n_txt = _hash_texts(h, texts)
    h.update(f"counts:{n_emb}:{n_txt}:{n_texts}".encode())
    return h.hexdigest()            # ⭐ full 256 bits, not [:16]


def compute_fingerprint_legacy(embeddings: np.ndarray, n_texts: int) -> str:
    """The v1 algorithm. ⛔ NOT FOR WRITING -- kept only to RECOGNIZE old manifests.

    Lets `verify_manifest` tell "this is an old manifest whose content still
    matches under the old weak algorithm, rebuild it" apart from "this is an old
    manifest and the content has changed too". Those need different responses and
    a bare mismatch cannot distinguish them.
    """
    h = hashlib.sha256()
    # ⛔ Was `len(embeddings) > 0`, which raises "len() of unsized object" on a
    #   0-d array -- the same degenerate-shape hole as `_hash_embeddings` had,
    #   in the function kept to RECOGNIZE the old manifests. Found 2026-11-13 on
    #   `x.cart.npz`, one cart after the v2 version of this bug was fixed.
    # ⭐ Identical output for every normal array: for ndim >= 1, `len(a)` IS
    #   `a.shape[0]`. Only the shapes that previously raised behave differently,
    #   and a manifest can never have been written for one of those.
    if (embeddings is not None and getattr(embeddings, "ndim", 0) >= 1
            and embeddings.shape[0] > 0):
        h.update(embeddings[0].tobytes())
        h.update(embeddings[-1].tobytes())
    h.update(str(n_texts).encode())
    return h.hexdigest()[:16]


# Verdicts. ⛔ Only VERIFIED is a pass. Every other value is a refusal with a
# reason, which is the opposite of v1's "no manifest => True".
VERIFIED = "VERIFIED"
MISMATCH = "MISMATCH"
COUNT_MISMATCH = "COUNT_MISMATCH"
NO_MANIFEST = "NO_MANIFEST"
LEGACY_MANIFEST = "LEGACY_MANIFEST"
UNREADABLE = "UNREADABLE"


def verify_manifest_detail(cart_path: str, embeddings: np.ndarray, n_texts: int,
                           texts) -> tuple[str, str]:
    """Return (verdict, message). ⭐ The caller cannot mistake a reason for a pass.

    ⛔⛔ `texts` HAS NO DEFAULT, AND THE DEFAULT IS WHAT BROKE THE BOOT PATH.
      It was `texts=None`, so a caller that omitted it compiled fine and failed
      only when execution reached `compute_fingerprint`. ⚠ On a cart with a v1
      manifest this function returns LEGACY_MANIFEST *before* that line, so the
      three callers that omitted `texts` worked for as long as every manifest was
      stale, and crashed the server the moment the manifests were upgraded.
    ⇒ A parameter needed on only one branch WILL be omitted, and the omission
      hides on the other branch. Required is the only version with no hiding
      place: now every caller fails identically, immediately, on any cart.
    """
    manifest_path = cart_path.rsplit(".", 1)[0] + "_manifest.json"

    if not os.path.exists(manifest_path):
        # ⛔⛔ v1 RETURNED SUCCESS HERE. An unmanifested cart verified, which made
        #   "verified" mean "either checked and clean, or never checkable".
        return NO_MANIFEST, (
            f"no manifest at {os.path.basename(manifest_path)}. An unverifiable "
            f"cartridge is not a verified one. Build a manifest with "
            f"scripts/rebuild_manifests.py."
        )

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except Exception as e:
        return UNREADABLE, f"manifest read error: {e}"

    expected_fp = manifest.get("fingerprint", "")
    algo = manifest.get("fingerprint_algo") or (
        LEGACY_ALGO if len(expected_fp) == 16 else "")

    expected_count = manifest.get("count", -1)
    if expected_count != n_texts:
        return COUNT_MISMATCH, (
            f"COUNT MISMATCH: manifest says {expected_count}, cart has {n_texts}")

    if algo == LEGACY_ALGO:
        # Name it, and say whether the old weak check still passes, because
        # "rebuild me" and "I have been altered" are different emergencies.
        legacy_fp = compute_fingerprint_legacy(embeddings, n_texts)
        still = "and its content still matches under that algorithm" \
            if legacy_fp == expected_fp else \
            "AND THE CONTENT HAS CHANGED under that algorithm too"
        return LEGACY_MANIFEST, (
            f"manifest uses {LEGACY_ALGO}, which covered only the first and last "
            f"embedding and no passage text, {still}. Rebuild with "
            f"scripts/rebuild_manifests.py to get {FINGERPRINT_ALGO}."
        )

    if algo != FINGERPRINT_ALGO:
        return UNREADABLE, (
            f"manifest declares unknown fingerprint_algo {algo!r}; expected "
            f"{FINGERPRINT_ALGO}. Refusing rather than guessing.")

    actual_fp = compute_fingerprint(embeddings, n_texts, texts=texts)
    if expected_fp != actual_fp:
        return MISMATCH, (
            f"FINGERPRINT MISMATCH: expected {expected_fp[:16]}..., "
            f"got {actual_fp[:16]}...")

    return VERIFIED, f"verified ({FINGERPRINT_ALGO}, {actual_fp[:16]}...)"


def verify_manifest(cart_path: str, embeddings: np.ndarray, n_texts: int,
                    texts) -> tuple[bool, str]:
    """(ok, message), where ⭐ ok is True ONLY for VERIFIED.

    The RETURN SHAPE is unchanged from v1 on purpose, so no call site can read a
    truthy reason string as a pass.

    ⛔⛔ CORRECTION, 2026-11-13. This docstring previously said the four existing
      call sites "pick up the corrected behaviour WITHOUT EDITS". That was wrong,
      and writing it down is what stopped me checking:

        - the return shape was compatible, so the calls kept compiling
        - but three of them passed no `texts`, and `texts=None` made that legal
        - and `verify_manifest_detail` returns LEGACY_MANIFEST before it needs
          `texts`, so the gap was invisible while every manifest was still v1

      ⇒ Upgrading the manifests -- the fix -- is what crashed the boot path.
      ⇒ `texts` is now REQUIRED here and in `verify_manifest_detail`. Every call
        site was edited: membot_server.py:5537 and :10774, multi_cart.py:162,
        test_boot_integrity.py.
      ⭐ I verified SIGNATURE compatibility and called it ARGUMENT completeness.
        They are different questions and only one of them was asked.
    """
    verdict, msg = verify_manifest_detail(cart_path, embeddings, n_texts,
                                          texts=texts)
    return verdict == VERIFIED, f"[{verdict}] {msg}"


def unverified_mount_allowed(cart_name: str, allow_all: bool,
                             allow_names=None) -> tuple[bool, str]:
    """May THIS cart mount despite failing verification? Returns (allowed, why).

    ⛔⛔ WHY THE PER-CART FORM EXISTS. `--allow-unverified-mount` is GLOBAL. It does not
      accept a known risk on one cart -- it removes the gate from the process. On a host
      serving several carts, switching it on for the one stale cart silently un-gates
      every cart that was verifying correctly, and the log looks the same.
    ⇒ `--allow-unverified-cart NAME` names the exception, so everything else stays
      fail-closed and the log says which cart was excused and why.
      Andy's ruling, 2026-11-13, on the droplet's split carts.

    ⭐ MATCHING IS EXACT AND CASE-SENSITIVE. A substring rule here would reintroduce
      precisely the ambiguity `b6eac68` removed from `mount_cartridge` -- and it would do
      it on the path that decides whether to skip an integrity check, which is the worst
      possible place for a fuzzy match.
    """
    if allow_names and cart_name in allow_names:
        return True, f"--allow-unverified-cart {cart_name!r}"
    if allow_all:
        return True, "--allow-unverified-mount (GLOBAL: every cart is un-gated)"
    return False, ""


def unused_bypass_names(available_names, allow_names=None) -> list[str]:
    """Names given to --allow-unverified-cart that match no cart on this host.

    ⛔ A typo'd name fails CLOSED, which sounds safe and is the problem: the operator
      believes an exception is in place, the server refuses the cart anyway, and the
      failure is indistinguishable from the flag not working at all.
    ⇒ Worse, the reverse reading is available too -- "the gate is protecting me" -- when
      in fact the intended exception simply never applied to anything.
    ⭐ Same shape as the stale-exemption check in tests/test_no_bare_texts_calls.py: an
      exception that matches nothing must SAY so rather than pass silently.
    """
    if not allow_names:
        return []
    have = {str(n) for n in (available_names or ())}
    return sorted({n for n in allow_names if n not in have})


def is_trusted_directory(path: str) -> bool:
    """True only if `path` lies strictly inside a trusted cartridge directory.

    Comparison is COMPONENT-AWARE, by design. The previous implementation was:

        if real_path.startswith(real_dir + os.sep) or real_path.startswith(real_dir):

    The first clause is correct. ⛔ The SECOND clause made it redundant and defeated
    it: a sibling directory that merely shares the prefix was classified as trusted,
    so `/carts-evil/x.pkl` passed while `/carts` was the trusted directory. And this
    function is the gate the `.pkl` loader relies on, so it was the gate on the one
    path that was actually gated.

    ⭐ `trusted_dir in Path(real_path).parents` is true only when the trusted
    directory is a genuine ANCESTOR, which is the property we meant:
      · `/carts/a.pkl`       -> True   (direct child)
      · `/carts/sub/a.pkl`   -> True   (nested, still inside)
      · `/carts-evil/a.pkl`  -> False  (sibling sharing a prefix)
      · `/carts`             -> False  (the directory is not a file inside itself)
    ⓘ It also avoids two edge cases `startswith` has: a trusted dir that already
    ends in a separator, and a Windows path on a different drive.

    ⚠ `realpath` resolves symlinks first, so a symlink pointing out of a trusted
    directory is judged on its target. That is deliberate.
    ⚠ On a case-insensitive filesystem a differently-cased path fails this check.
    That is a false NEGATIVE, which for a gate is the safe direction.
    """
    real_path = Path(os.path.realpath(path))
    parents = set(real_path.parents)
    for d in CARTRIDGE_DIRS:
        if Path(os.path.realpath(d)) in parents:
            return True
    return False


def load_cartridge_safe(path: str) -> dict:
    """Load a cartridge using the safest available method CURRENTLY IMPLEMENTED.

    ⛔⛔ THIS DOCSTRING USED TO SAY "NPZ files are always safe (no code execution)".
    THAT WAS FALSE, and it outlived every reader who trusted it. An `.npz` opened
    with `allow_pickle=True` unpickles object arrays, and unpickling executes code.
    Two of our six arrays are object dtype (`passages`, `compressed_texts`), so the
    loader has been passing `allow_pickle=True` and an `.npz` is therefore NOT a
    safe thing to open from an untrusted source today.

    ⚠ WHAT IS AND IS NOT TRUE RIGHT NOW:
      · `.pkl` is gated to trusted directories by `is_trusted_directory`.
      · `.npz` is NOT gated at all, and executes code on load. ⇒ The file extension
        is not a safety boundary.
      · The loader also runs BEFORE manifest verification, so nothing has checked
        that the file is what it claims to be at the moment it is materialized.

    ⇒ The repair is the gauntlet: admissible, then authentic, then authorized, in
      that order (docs/DESIGN-the-gauntlet-verify-before-materialize-2026-11-12.md).
      ⭐ Until it lands, mount only carts your own organization produced.
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
