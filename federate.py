"""
federate.py — Federated cart mode for Membot.

Drop-in replacement for Dennis Palatov's JSONL-based federated learning
infrastructure (`shared-context/arc-agi-3/consolidate.py` +
`publish_learning.py`). Same git-sync architecture, same per-machine
append-only model, same daily consolidation timing — but the substrate is
brain carts instead of JSONL, which gives semantic search across the fleet
for free.

Spec: docs/RFC/federated-cart-spec.md
Built on: multi_cart.py (the multi-cart query layer is the foundation)

Public API
==========
    publish_session(session_file, machine_id, fleet_dir)
        Extract learning entries from a SAGE session.json (or already-formatted
        JSONL line list), append them to the machine's federated cart, and save
        the cart back to disk. Used by post-session hooks.

    publish_jsonl_lines(jsonl_lines, machine_id, fleet_dir)
        Lower-level helper: append already-built JSONL entry dicts to a cart.

    consolidate(fleet_dir, output_dir, similarity_threshold=0.85)
        Mount every machine cart in fleet_dir/{machine}/kb.cart, find
        semantically similar patterns across machines, write a consolidated
        cart that contains the cross-machine consensus.

    migrate_jsonl(jsonl_dir, output_dir=None)
        One-time migration: walk a directory of *_learning.jsonl files
        (Dennis's current format), convert each to a brain cart. Non-destructive
        — keeps the JSONL files in place by default.

    load_fleet(fleet_dir)
        Mount every machine cart in fleet_dir as a federated cart in the
        multi-cart pool. Used by solvers at session start to make the fleet's
        learning available for cross-machine search.

Drop-in equivalence to Dennis's scripts
=======================================
    His consolidate.py     →  federate.consolidate(fleet_dir, output_dir)
    His publish_learning.py → federate.publish_session(session_file, machine, fleet_dir)
    JSONL load at solver   →  federate.load_fleet(fleet_dir) + multi_cart.search(...)

The federate module knows nothing about ARC-AGI-3 specifically. The schema
matches Dennis's JSONL format because that's what's needed RIGHT NOW for the
fleet, but any append-only learning log with similar shape (timestamp, source,
event_type, content text) will work.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
import time
from typing import Any, Iterable, Optional

import numpy as np

# Lazy imports inside functions to avoid circular import with membot_server
log = logging.getLogger(__name__)

# Default cart filename inside each machine directory
DEFAULT_MACHINE_CART_NAME = "kb"

# Default name for the consolidated output cart
DEFAULT_CONSOLIDATED_CART_NAME = "kb"

# How similar two patterns must be to count as cross-machine confirmation
DEFAULT_SIMILARITY_THRESHOLD = 0.85

# How similar two patterns must be to count as a contradiction candidate
# (high embedding similarity but low text overlap)
DEFAULT_CONTRADICTION_EMBEDDING_THRESHOLD = 0.75
DEFAULT_CONTRADICTION_TEXT_OVERLAP_MAX = 0.30


# =============================================================================
# JSONL ENTRY → CART TEXT
# =============================================================================

def _entry_to_text(entry: dict) -> str:
    """Convert a JSONL learning entry to the searchable text that will be
    embedded and stored in the cart.

    The text is structured so the LLM consuming search results can read it
    naturally — machine name, event type, and the substantive content all
    visible — and so the embedding captures the semantic content (rule,
    description, insight, meta) rather than just metadata fields.
    """
    machine = entry.get("machine", "?")
    player = entry.get("player", "?")
    game = entry.get("game", "?")
    level = entry.get("level", -1)
    event = entry.get("event", "?")
    timestamp = entry.get("timestamp", "")

    # Header: "[machine/player game L7 event_type @ timestamp]"
    level_str = f"L{level}" if level >= 0 else "L-game"
    header = f"[{machine}/{player} {game} {level_str} {event} @ {timestamp}]"

    # Body varies by event type. We try the most informative fields first.
    body_parts = []

    # level_solved / level_failed
    if "structural_pattern" in entry:
        body_parts.append(f"PATTERN: {entry['structural_pattern']}")
    if "rule" in entry:
        body_parts.append(f"RULE: {entry['rule']}")
    if "actions" in entry and "baseline" in entry:
        body_parts.append(f"ACTIONS: {entry['actions']} (baseline: {entry['baseline']})")
    elif "actions" in entry:
        body_parts.append(f"ACTIONS: {entry['actions']}")

    # game_complete summaries
    if "total_actions" in entry:
        body_parts.append(f"TOTAL: {entry['total_actions']} actions")
        if "total_baseline" in entry:
            body_parts.append(f"BASELINE: {entry['total_baseline']}")
        if "efficiency" in entry:
            body_parts.append(f"EFFICIENCY: {entry['efficiency']}")
    if "levels_solved" in entry and "levels_total" in entry:
        body_parts.append(f"LEVELS: {entry['levels_solved']}/{entry['levels_total']}")

    # game_insight / structural_pattern
    if "insight" in entry:
        body_parts.append(f"INSIGHT: {entry['insight']}")
    if "pattern" in entry:
        body_parts.append(f"PATTERN: {entry['pattern']}")
    if "description" in entry:
        body_parts.append(f"DESCRIPTION: {entry['description']}")

    # Confidence / corroboration
    if "confidence" in entry:
        body_parts.append(f"CONFIDENCE: {entry['confidence']}")
    if "games_confirmed" in entry:
        confirmed = entry["games_confirmed"]
        if isinstance(confirmed, list):
            body_parts.append(f"CONFIRMED IN: {', '.join(confirmed)}")

    # Meta is human commentary, often the most distinctive bit semantically
    if "meta" in entry:
        body_parts.append(f"META: {entry['meta']}")

    if not body_parts:
        # Fallback: dump everything as JSON-ish text
        body_parts.append(json.dumps({k: v for k, v in entry.items()
                                       if k not in {"machine", "player", "game",
                                                    "level", "event", "timestamp"}}))

    return header + "\n" + "\n".join(body_parts)


def _entry_metadata(entry: dict) -> dict:
    """Extract per-pattern metadata fields that should be stored alongside the
    text. These become searchable/filterable fields on the cart pattern record.
    """
    return {
        "machine": entry.get("machine"),
        "player": entry.get("player"),
        "game": entry.get("game"),
        "level": entry.get("level", -1),
        "event": entry.get("event"),
        "timestamp": entry.get("timestamp"),
        "structural_pattern": entry.get("structural_pattern") or entry.get("pattern"),
        "confidence": entry.get("confidence"),
        "actions": entry.get("actions"),
        "baseline": entry.get("baseline"),
    }


def _content_signature(entry: dict) -> str:
    """A stable string signature used to dedup the same entry across multiple
    publishes. We don't want one machine's session to be appended twice if the
    publish hook fires more than once for the same session.
    """
    return "|".join([
        str(entry.get("machine", "")),
        str(entry.get("game", "")),
        str(entry.get("level", -1)),
        str(entry.get("event", "")),
        str(entry.get("timestamp", "")),
        str(entry.get("structural_pattern") or entry.get("pattern") or ""),
    ])


# =============================================================================
# CART READ/WRITE
# =============================================================================

def _machine_cart_path(fleet_dir: str, machine_id: str) -> str:
    """Path to a machine's federated cart inside the fleet directory."""
    machine_dir = os.path.join(fleet_dir, machine_id)
    return os.path.join(machine_dir, f"{DEFAULT_MACHINE_CART_NAME}.cart.npz")


def _load_existing_cart(cart_path: str) -> tuple[list[str], np.ndarray, list[dict], set[str]]:
    """Load an existing federated cart and return (texts, embeddings,
    metadata_list, content_signatures). Empty containers if the cart doesn't
    exist yet (first publish).
    """
    if not os.path.exists(cart_path):
        return [], np.zeros((0, 768), dtype=np.float32), [], set()

    data = np.load(cart_path, allow_pickle=True)
    embeddings = data["embeddings"].astype(np.float32) if "embeddings" in data.files else np.zeros((0, 768), dtype=np.float32)

    # Texts may be stored as 'passages' or 'compressed_texts'
    texts = []
    if "compressed_texts" in data.files:
        import zlib
        for ct in data["compressed_texts"]:
            try:
                texts.append(zlib.decompress(ct.tobytes() if hasattr(ct, "tobytes") else ct).decode("utf-8"))
            except Exception:
                texts.append("")
    elif "passages" in data.files:
        texts = list(data["passages"])

    # Per-pattern metadata, if stored
    metadata_list = []
    if "per_pattern_meta" in data.files:
        try:
            raw = data["per_pattern_meta"]
            if isinstance(raw, np.ndarray) and raw.dtype == object:
                metadata_list = [json.loads(m) if isinstance(m, str) else (m or {}) for m in raw]
            else:
                metadata_list = json.loads(str(raw))
        except Exception as e:
            log.warning(f"[federate] Could not parse per_pattern_meta in {cart_path}: {e}")
            metadata_list = [{} for _ in texts]
    else:
        metadata_list = [{} for _ in texts]

    # Build the dedup signature set from metadata
    sigs = set()
    for meta in metadata_list:
        if isinstance(meta, dict):
            sigs.add("|".join([
                str(meta.get("machine", "")),
                str(meta.get("game", "")),
                str(meta.get("level", -1)),
                str(meta.get("event", "")),
                str(meta.get("timestamp", "")),
                str(meta.get("structural_pattern") or ""),
            ]))

    return texts, embeddings, metadata_list, sigs


def _save_federated_cart(cart_path: str, texts: list[str], embeddings: np.ndarray,
                         metadata_list: list[dict]) -> dict:
    """Save a federated cart with manifest. Uses cartridge_builder.save_cartridge
    if available, otherwise writes the npz directly."""
    output_dir = os.path.dirname(cart_path)
    cart_name = os.path.basename(cart_path).replace(".cart.npz", "")
    os.makedirs(output_dir, exist_ok=True)

    # Per-pattern metadata as a JSON-encoded numpy object array
    meta_json_list = [json.dumps(m, default=str) for m in metadata_list]
    meta_array = np.array(meta_json_list, dtype=object)

    # Compress texts for storage efficiency
    import zlib
    compressed_texts = []
    for t in texts:
        compressed_texts.append(np.void(zlib.compress(t.encode("utf-8"), level=9)))

    # Sign-zero bits for Hamming search
    if len(embeddings) > 0:
        sign_bits = np.packbits((embeddings > 0).astype(np.uint8), axis=1)
    else:
        sign_bits = np.zeros((0, 96), dtype=np.uint8)

    np.savez_compressed(
        cart_path,
        embeddings=embeddings,
        passages=np.array(texts, dtype=object),
        compressed_texts=np.array(compressed_texts, dtype=object),
        sign_bits=sign_bits,
        per_pattern_meta=meta_array,
        version="federate-v1",
    )

    # Write integrity manifest matching membot_server's verify_manifest format
    import hashlib
    h = hashlib.sha256()
    if len(embeddings) > 0:
        h.update(embeddings[0].tobytes())
        h.update(embeddings[-1].tobytes())
    h.update(str(len(texts)).encode())
    fingerprint = h.hexdigest()[:16]

    manifest_path = cart_path.rsplit(".", 1)[0] + "_manifest.json"
    # cart_path ends with .cart.npz so the manifest goes next to it as
    # something_manifest.json — but membot_server.verify_manifest expects
    # the manifest at "<base>_manifest.json" where base is path without .npz
    # not without .cart.npz. Let's match the membot convention exactly.
    manifest_path = cart_path.rsplit(".", 1)[0] + "_manifest.json"
    manifest = {
        "version": "federate-v1",
        "count": len(texts),
        "fingerprint": fingerprint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "format": "federated_cart",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    size_bytes = os.path.getsize(cart_path)
    return {
        "cart_path": cart_path,
        "n_patterns": len(texts),
        "size_bytes": size_bytes,
        "fingerprint": fingerprint,
    }


# =============================================================================
# PUBLISH — Append session learning to a machine's federated cart
# =============================================================================

def publish_jsonl_lines(jsonl_lines: Iterable[dict], machine_id: str,
                        fleet_dir: str) -> dict:
    """Append already-built JSONL entry dicts to the machine's federated cart.

    Args:
        jsonl_lines: Iterable of dicts in Dennis's JSONL format
        machine_id: The machine writing this batch (e.g. "cbp", "sprout")
        fleet_dir: Root of the fleet-learning directory (parent of machine dirs)

    Returns:
        dict with keys: cart_path, added, skipped (dedup), total_in_cart, fingerprint
    """
    from cartridge_builder import embed_texts

    cart_path = _machine_cart_path(fleet_dir, machine_id)

    # Load existing cart (or empty containers if first publish)
    texts, embeddings, metadata_list, existing_sigs = _load_existing_cart(cart_path)

    new_texts = []
    new_metadata = []
    skipped = 0

    for entry in jsonl_lines:
        if not isinstance(entry, dict):
            continue
        sig = _content_signature(entry)
        if sig in existing_sigs:
            skipped += 1
            continue
        existing_sigs.add(sig)
        new_texts.append(_entry_to_text(entry))
        new_metadata.append(_entry_metadata(entry))

    if not new_texts:
        log.info(f"[federate] publish_jsonl_lines({machine_id}): nothing new (skipped {skipped})")
        return {
            "cart_path": cart_path,
            "added": 0,
            "skipped": skipped,
            "total_in_cart": len(texts),
        }

    # Embed only the new texts
    log.info(f"[federate] Embedding {len(new_texts)} new entries for {machine_id}...")
    new_embeddings = embed_texts(new_texts)

    # Concatenate
    if len(embeddings) > 0:
        combined_embeddings = np.vstack([embeddings, new_embeddings]).astype(np.float32)
    else:
        combined_embeddings = new_embeddings.astype(np.float32)
    combined_texts = list(texts) + new_texts
    combined_metadata = list(metadata_list) + new_metadata

    save_result = _save_federated_cart(cart_path, combined_texts, combined_embeddings, combined_metadata)
    save_result["added"] = len(new_texts)
    save_result["skipped"] = skipped
    save_result["total_in_cart"] = len(combined_texts)

    log.info(
        f"[federate] {machine_id}: added {len(new_texts)}, skipped {skipped} dedup, "
        f"total {len(combined_texts)} → {cart_path}"
    )
    return save_result


def publish_session(session_file: str, machine_id: str, fleet_dir: str) -> dict:
    """Extract learning entries from a SAGE session.json (or a JSONL file)
    and append them to the machine's federated cart.

    Supports two input formats:
        1. A .jsonl file (one JSON object per line) — Dennis's existing format
        2. A .json file with a 'learning_entries' or 'entries' key holding a list

    Args:
        session_file: Path to the session JSON or JSONL file
        machine_id: The machine writing this batch
        fleet_dir: Root of the fleet-learning directory

    Returns:
        Same shape as publish_jsonl_lines() return
    """
    if not os.path.exists(session_file):
        raise FileNotFoundError(f"Session file not found: {session_file}")

    entries = []
    if session_file.endswith(".jsonl"):
        with open(session_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning(f"[federate] Skipping malformed JSONL line: {e}")
    else:
        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = data.get("learning_entries") or data.get("entries") or []
        else:
            raise ValueError(f"Unexpected session file shape: {type(data)}")

    return publish_jsonl_lines(entries, machine_id, fleet_dir)


# =============================================================================
# CONSOLIDATE — Cross-machine pattern matching across mounted carts
# =============================================================================

def consolidate(fleet_dir: str, output_dir: Optional[str] = None,
                similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                contradiction_threshold: float = DEFAULT_CONTRADICTION_EMBEDDING_THRESHOLD,
                mode: str = "preserve") -> dict:
    """Mount every machine cart in fleet_dir, find cross-machine pattern
    matches, and write a consolidated cart that captures the cross-machine
    consensus and disagreements.

    Replaces Dennis Palatov's consolidate.py JSONL deduplication script.
    The output is a brain cart that any solver can mount to get the fleet's
    accumulated learning, with corroboration and contradiction signals
    preserved instead of dedup'd away.

    Args:
        fleet_dir: Root of the fleet-learning directory
        output_dir: Where to write the consolidated cart (defaults to
            fleet_dir/../consolidated)
        similarity_threshold: Patterns with cross-machine similarity above
            this count as CONFIRMED_BY (default 0.85)
        contradiction_threshold: Patterns with similarity in [contradiction,
            similarity_threshold) and low text overlap count as
            CONTRADICTED_BY candidates (default 0.75)
        mode: How to handle cross-machine matches. Two options:
            - "preserve" (DEFAULT): Keep ALL variants from every machine in the
              consolidated cart. Cross-cart edges are stored as metadata
              (confirming_machines, contradicting_machines lists). The
              relying party (solver) sees all voices and weighs them itself.
              This aligns with the Web4 trust model — trust is contextual,
              not collapsed by the consolidator. Recommended for federated
              fleets where multiple machines should remain distinct.
            - "collapse": Pick one representative per CONFIRMED_BY connected
              component. Smaller cart, less duplication, but loses individual
              machine voices. Useful when storage is constrained and you
              don't need per-machine attribution.

    Returns:
        dict with keys: output_path, n_machines, total_input_patterns,
        n_consolidated_patterns, n_confirmed_pairs, n_contradicted_pairs,
        mode, elapsed_seconds
    """
    import multi_cart as mc

    t0 = time.time()

    if output_dir is None:
        # Default: ../consolidated relative to fleet_dir
        output_dir = os.path.join(os.path.dirname(fleet_dir.rstrip("/\\")), "consolidated")
    os.makedirs(output_dir, exist_ok=True)

    # Mount every machine cart in fleet_dir
    log.info(f"[federate] Consolidating fleet at {fleet_dir}...")
    mc.unmount_all()  # clean slate

    machine_carts = []
    for machine_dir in sorted(os.listdir(fleet_dir)):
        full_dir = os.path.join(fleet_dir, machine_dir)
        if not os.path.isdir(full_dir):
            continue
        cart_file = os.path.join(full_dir, f"{DEFAULT_MACHINE_CART_NAME}.cart.npz")
        if not os.path.exists(cart_file):
            log.info(f"[federate] No cart at {cart_file}, skipping {machine_dir}")
            continue
        try:
            result = mc.mount(cart_file, cart_id=machine_dir, role="federated",
                              verify_integrity=True)
            machine_carts.append(machine_dir)
            log.info(f"[federate] Mounted {machine_dir}: {result['n_patterns']} patterns")
        except Exception as e:
            log.warning(f"[federate] Failed to mount {machine_dir}: {e}")

    if len(machine_carts) == 0:
        return {
            "output_path": None,
            "n_machines": 0,
            "total_input_patterns": 0,
            "n_consolidated_patterns": 0,
            "n_confirmed_pairs": 0,
            "n_contradicted_pairs": 0,
            "elapsed_seconds": time.time() - t0,
            "error": "no_machine_carts_found",
        }

    # Walk every pattern in every machine cart, search for cross-machine matches
    confirmed_pairs = []  # list of (cart_a, addr_a, cart_b, addr_b, score)
    contradicted_pairs = []
    total_input = 0

    for cart_id in machine_carts:
        state = mc.get_cart(cart_id)
        if not state or not state["has_embeddings"]:
            continue
        n = state["n_patterns"]
        total_input += n
        for local_addr in range(n):
            text = state["texts"][local_addr]
            # Search OTHER machines for similar patterns
            other_carts = [c for c in machine_carts if c != cart_id]
            if not other_carts:
                continue
            results = mc.search(text[:500], top_k=3, scope=other_carts)
            for r in results.get("results", []):
                score = r["score"]
                if score >= similarity_threshold:
                    confirmed_pairs.append({
                        "from": (cart_id, local_addr),
                        "to": (r["cart_id"], r["local_addr"]),
                        "score": score,
                        "type": "CONFIRMED_BY",
                    })
                elif score >= contradiction_threshold:
                    # Could be contradiction — check text overlap
                    other_text = r["text"][:500]
                    overlap = _text_overlap(text[:500], other_text)
                    if overlap < DEFAULT_CONTRADICTION_TEXT_OVERLAP_MAX:
                        contradicted_pairs.append({
                            "from": (cart_id, local_addr),
                            "to": (r["cart_id"], r["local_addr"]),
                            "score": score,
                            "text_overlap": overlap,
                            "type": "CONTRADICTED_BY",
                        })

    # Build the consolidated cart from the original patterns + cross-cart edges.
    # Two modes:
    #   - "preserve" (DEFAULT, per Dennis's Web4 trust framing): keep ALL variants
    #     from every machine, just annotate cross-cart edges as metadata. Trust is
    #     contextual and evaluated by the relying party (the solver), not collapsed
    #     by the consolidator. Larger output cart but no information loss.
    #   - "collapse": pick one representative per CONFIRMED_BY connected component.
    #     Smaller cart but loses individual machine voices.
    if mode == "preserve":
        consolidated_texts, consolidated_meta = _build_preserved_set(
            machine_carts, confirmed_pairs, contradicted_pairs
        )
    elif mode == "collapse":
        consolidated_texts, consolidated_meta, _ = _build_consolidated_set(
            machine_carts, confirmed_pairs, contradicted_pairs
        )
    else:
        raise ValueError(f"Unknown consolidate mode: {mode!r}. Use 'preserve' or 'collapse'.")

    output_cart_path = os.path.join(output_dir, f"{DEFAULT_CONSOLIDATED_CART_NAME}.cart.npz")
    if consolidated_texts:
        from cartridge_builder import embed_texts
        consolidated_embeddings = embed_texts(consolidated_texts).astype(np.float32)
    else:
        consolidated_embeddings = np.zeros((0, 768), dtype=np.float32)

    save_result = _save_federated_cart(
        output_cart_path,
        consolidated_texts,
        consolidated_embeddings,
        consolidated_meta,
    )

    # Also write a stats file alongside (matches Dennis's last_consolidated.json)
    stats = {
        "consolidated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_machines": len(machine_carts),
        "machines": machine_carts,
        "total_input_patterns": total_input,
        "n_consolidated_patterns": len(consolidated_texts),
        "n_confirmed_pairs": len(confirmed_pairs),
        "n_contradicted_pairs": len(contradicted_pairs),
        "similarity_threshold": similarity_threshold,
        "contradiction_threshold": contradiction_threshold,
        "mode": mode,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(os.path.join(output_dir, "last_consolidated.json"), "w") as f:
        json.dump(stats, f, indent=2)

    log.info(
        f"[federate] Consolidated {total_input} input patterns from "
        f"{len(machine_carts)} machines → {len(consolidated_texts)} unique "
        f"({len(confirmed_pairs)} confirmed, {len(contradicted_pairs)} contradicted) "
        f"in {time.time() - t0:.1f}s"
    )

    # Clean up the mounts we created
    mc.unmount_all()

    stats["output_path"] = output_cart_path
    return stats


def _text_overlap(a: str, b: str) -> float:
    """Quick word-set Jaccard similarity for contradiction detection."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _build_preserved_set(machine_carts: list[str],
                          confirmed_pairs: list[dict],
                          contradicted_pairs: list[dict]) -> tuple[list[str], list[dict]]:
    """Build a consolidated cart that PRESERVES all variants from every machine.

    Every original pattern from every machine appears in the output. Cross-cart
    edges (CONFIRMED_BY, CONTRADICTED_BY) are stored as metadata on each pattern
    so the relying party (the solver) can see who confirmed it and who
    disagreed, and weight the trust signal contextually.

    This is the Web4 trust model applied to federation: trust is not collapsed
    by the consolidator. The consolidator's job is to FIND relationships, not to
    decide whose voice wins. The solver decides.

    Returns (texts, metadata_list). One entry per (machine, original pattern).
    """
    import multi_cart as mc

    # Index: (cart_id, addr) → set of confirming (cart_id, addr) keys
    confirmed_by: dict = {}
    for pair in confirmed_pairs:
        a = pair["from"]
        b = pair["to"]
        confirmed_by.setdefault(a, set()).add(b)
        confirmed_by.setdefault(b, set()).add(a)

    contradicted_by: dict = {}
    for pair in contradicted_pairs:
        a = pair["from"]
        b = pair["to"]
        contradicted_by.setdefault(a, set()).add(b)
        contradicted_by.setdefault(b, set()).add(a)

    out_texts: list[str] = []
    out_meta: list[dict] = []

    for cart_id in machine_carts:
        state = mc.get_cart(cart_id)
        if not state:
            continue
        for local_addr in range(state["n_patterns"]):
            key = (cart_id, local_addr)
            text = state["texts"][local_addr] if local_addr < len(state["texts"]) else ""

            # Collect cross-cart edges for this specific pattern
            confirmations = sorted(confirmed_by.get(key, set()))
            contradictions = sorted(contradicted_by.get(key, set()))

            # Confirming machines (deduplicated, alphabetical)
            confirming_machines = sorted(set(c[0] for c in confirmations))
            contradicting_machines = sorted(set(c[0] for c in contradictions))

            # Build a list of (machine, addr) pointers for the solver to follow
            confirmed_by_list = [
                {"machine": c[0], "local_addr": c[1]} for c in confirmations
            ]
            contradicted_by_list = [
                {"machine": c[0], "local_addr": c[1]} for c in contradictions
            ]

            meta = {
                "consolidated": True,
                "preserve_mode": True,
                "source_machine": cart_id,
                "source_local_addr": local_addr,
                "confirming_machines": confirming_machines,
                "contradicting_machines": contradicting_machines,
                "n_confirmations": len(confirmations),
                "n_contradictions": len(contradictions),
                "confirmed_by": confirmed_by_list,
                "contradicted_by": contradicted_by_list,
                # Trust hint for the solver — but the solver decides what to do with it
                "trust_signal": (
                    "high_corroboration" if len(confirming_machines) >= 2 else
                    "single_corroboration" if len(confirming_machines) == 1 else
                    "single_source"
                ),
            }

            out_texts.append(text)
            out_meta.append(meta)

    return out_texts, out_meta


def _build_consolidated_set(machine_carts: list[str],
                             confirmed_pairs: list[dict],
                             contradicted_pairs: list[dict]) -> tuple[list[str], list[dict], int]:
    """Walk every pattern in every machine cart and produce a deduplicated set
    where CONFIRMED_BY pairs collapse to one representative pattern with all
    confirming machines listed in the metadata.

    NOTE: This is the legacy "collapse" mode. New code should prefer
    _build_preserved_set() which keeps all variants per the Web4 trust model
    decision (2026-04-08, requested by dp-web4 fleet).

    Returns (texts, metadata_list, n_unique).
    """
    import multi_cart as mc

    # Build a quick index: (cart_id, addr) → list of confirming (cart_id, addr) pairs
    confirmed_by = {}  # key → set of confirming keys
    for pair in confirmed_pairs:
        a = pair["from"]
        b = pair["to"]
        confirmed_by.setdefault(a, set()).add(b)
        confirmed_by.setdefault(b, set()).add(a)

    # Build a quick index for contradictions
    contradicted_by = {}
    for pair in contradicted_pairs:
        a = pair["from"]
        b = pair["to"]
        contradicted_by.setdefault(a, set()).add(b)
        contradicted_by.setdefault(b, set()).add(a)

    seen = set()  # keys that have already been emitted
    out_texts = []
    out_meta = []

    for cart_id in machine_carts:
        state = mc.get_cart(cart_id)
        if not state:
            continue
        for local_addr in range(state["n_patterns"]):
            key = (cart_id, local_addr)
            if key in seen:
                continue

            # Walk the connected component of CONFIRMED_BY edges
            component = {key}
            stack = [key]
            while stack:
                current = stack.pop()
                for neighbor in confirmed_by.get(current, set()):
                    if neighbor not in component:
                        component.add(neighbor)
                        stack.append(neighbor)

            # Mark every pattern in this component as seen
            seen |= component

            # Pick a representative — the one from the first machine alphabetically
            rep = sorted(component)[0]
            rep_state = mc.get_cart(rep[0])
            rep_text = rep_state["texts"][rep[1]] if rep_state else ""

            # Build metadata listing all confirming machines and any contradictions
            confirming_machines = sorted(set(c[0] for c in component))
            contradictions = []
            for member in component:
                for contra_key in contradicted_by.get(member, set()):
                    contra_state = mc.get_cart(contra_key[0])
                    if contra_state and contra_key[1] < len(contra_state["texts"]):
                        contradictions.append({
                            "machine": contra_key[0],
                            "local_addr": contra_key[1],
                            "text_preview": contra_state["texts"][contra_key[1]][:200],
                        })

            meta = {
                "consolidated": True,
                "confirming_machines": confirming_machines,
                "n_confirmations": len(component),
                "representative": {"machine": rep[0], "local_addr": rep[1]},
                "contradictions": contradictions if contradictions else None,
                "confidence": "high" if len(component) >= 2 else "single_source",
            }

            out_texts.append(rep_text)
            out_meta.append(meta)

    return out_texts, out_meta, len(out_texts)


# =============================================================================
# MIGRATE — One-time JSONL → cart conversion
# =============================================================================

def migrate_jsonl(jsonl_dir: str, output_dir: Optional[str] = None,
                  in_place: bool = False) -> dict:
    """Walk a directory of *_learning.jsonl files and build a brain cart for
    each machine. This is the one-time migration from Dennis's existing JSONL
    format to brain carts.

    Args:
        jsonl_dir: Directory containing per-machine subdirectories with
            *_learning.jsonl files (e.g. fleet-learning/cbp/sb26_learning.jsonl)
        output_dir: Where to write the carts (defaults to in-place in jsonl_dir)
        in_place: If True, write carts alongside the JSONL files in their
            original directories. Default False — non-destructive.

    Returns:
        dict with keys: machines_processed, total_entries, carts_built,
        errors, elapsed_seconds
    """
    t0 = time.time()
    if output_dir is None:
        output_dir = jsonl_dir

    machine_dirs = []
    for entry in sorted(os.listdir(jsonl_dir)):
        full_dir = os.path.join(jsonl_dir, entry)
        if os.path.isdir(full_dir):
            machine_dirs.append(entry)

    if not machine_dirs:
        return {
            "machines_processed": 0,
            "total_entries": 0,
            "carts_built": 0,
            "errors": [],
            "elapsed_seconds": time.time() - t0,
        }

    log.info(f"[federate] Migration: {len(machine_dirs)} machine dirs in {jsonl_dir}")

    total_entries = 0
    carts_built = 0
    errors = []

    for machine_id in machine_dirs:
        machine_dir = os.path.join(jsonl_dir, machine_id)
        jsonl_files = sorted(glob.glob(os.path.join(machine_dir, "*_learning.jsonl")))
        if not jsonl_files:
            log.info(f"[federate] No JSONL files in {machine_id}, skipping")
            continue

        # Aggregate all entries from all per-game JSONL files
        all_entries = []
        for jf in jsonl_files:
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        all_entries.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        errors.append(f"{jf}: {e}")

        if not all_entries:
            continue

        # Determine where to write the cart
        if in_place:
            target_fleet_dir = jsonl_dir
        else:
            target_fleet_dir = output_dir

        try:
            result = publish_jsonl_lines(all_entries, machine_id, target_fleet_dir)
            total_entries += result["added"]
            carts_built += 1
            log.info(
                f"[federate] {machine_id}: migrated {result['added']} entries to "
                f"{result['cart_path']}"
            )
        except Exception as e:
            errors.append(f"{machine_id}: {e}")
            log.warning(f"[federate] Failed to build cart for {machine_id}: {e}")

    elapsed = time.time() - t0
    log.info(
        f"[federate] Migration complete: {carts_built} carts built, "
        f"{total_entries} entries in {elapsed:.1f}s"
    )
    return {
        "machines_processed": len(machine_dirs),
        "total_entries": total_entries,
        "carts_built": carts_built,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 2),
    }


# =============================================================================
# LOAD FLEET — Mount all federated carts for solver use
# =============================================================================

def load_fleet(fleet_dir: str) -> dict:
    """Mount every machine's federated cart into the multi-cart pool with
    role='federated'. This is the call a solver makes at session start to
    make the fleet's accumulated learning available for cross-machine search.

    Returns:
        dict with keys: mounted, total_patterns, machines, errors
    """
    import multi_cart as mc

    if not os.path.isdir(fleet_dir):
        raise ValueError(f"Fleet directory does not exist: {fleet_dir}")

    mounted = []
    errors = []
    total_patterns = 0

    for machine_id in sorted(os.listdir(fleet_dir)):
        machine_dir = os.path.join(fleet_dir, machine_id)
        if not os.path.isdir(machine_dir):
            continue
        cart_path = os.path.join(machine_dir, f"{DEFAULT_MACHINE_CART_NAME}.cart.npz")
        if not os.path.exists(cart_path):
            continue
        try:
            # Use the machine_id as the cart_id so search results are clearly attributed
            result = mc.mount(cart_path, cart_id=machine_id, role="federated",
                              verify_integrity=True)
            mounted.append(result)
            total_patterns += result["n_patterns"]
        except Exception as e:
            errors.append({"machine": machine_id, "error": str(e)})
            log.warning(f"[federate] load_fleet: failed to mount {machine_id}: {e}")

    log.info(
        f"[federate] load_fleet({fleet_dir}): mounted {len(mounted)} machines, "
        f"{total_patterns} total patterns"
    )
    return {
        "mounted": mounted,
        "total_patterns": total_patterns,
        "machines": [m["cart_id"] for m in mounted],
        "errors": errors,
    }


# =============================================================================
# CLI ENTRY (for use as a drop-in script)
# =============================================================================

def _cli():
    """Minimal CLI so this file can be invoked as a drop-in for Dennis's
    consolidate.py and publish_learning.py scripts.

    Usage:
        python federate.py consolidate <fleet_dir> [output_dir]
        python federate.py publish <session_file> <machine_id> <fleet_dir>
        python federate.py migrate <jsonl_dir> [output_dir] [--in-place]
        python federate.py load <fleet_dir>
    """
    if len(sys.argv) < 2:
        print(_cli.__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if cmd == "consolidate":
        if len(sys.argv) < 3:
            print("Usage: python federate.py consolidate <fleet_dir> [output_dir]")
            sys.exit(1)
        fleet_dir = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        result = consolidate(fleet_dir, output_dir)
        print(json.dumps(result, indent=2))

    elif cmd == "publish":
        if len(sys.argv) < 5:
            print("Usage: python federate.py publish <session_file> <machine_id> <fleet_dir>")
            sys.exit(1)
        session_file = sys.argv[2]
        machine_id = sys.argv[3]
        fleet_dir = sys.argv[4]
        result = publish_session(session_file, machine_id, fleet_dir)
        print(json.dumps(result, indent=2))

    elif cmd == "migrate":
        if len(sys.argv) < 3:
            print("Usage: python federate.py migrate <jsonl_dir> [output_dir] [--in-place]")
            sys.exit(1)
        jsonl_dir = sys.argv[2]
        in_place = "--in-place" in sys.argv
        output_dir = None
        for arg in sys.argv[3:]:
            if not arg.startswith("--"):
                output_dir = arg
                break
        result = migrate_jsonl(jsonl_dir, output_dir, in_place=in_place)
        print(json.dumps(result, indent=2))

    elif cmd == "load":
        if len(sys.argv) < 3:
            print("Usage: python federate.py load <fleet_dir>")
            sys.exit(1)
        fleet_dir = sys.argv[2]
        result = load_fleet(fleet_dir)
        print(json.dumps(result, indent=2, default=str))

    else:
        print(f"Unknown command: {cmd}")
        print(_cli.__doc__)
        sys.exit(1)


if __name__ == "__main__":
    _cli()
