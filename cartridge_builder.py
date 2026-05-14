"""
Membot Cartridge Builder
========================
Build brain cartridges from local documents.

Reads .txt, .md, .pdf, and .docx files from a folder (or single file),
embeds them with Nomic, optionally trains the lattice for physics search,
and outputs a complete cartridge package ready for Membot.

Usage:
  # Basic: embed only (fast, no GPU needed)
  python cartridge_builder.py ./my-docs/ --name my-knowledge

  # Full: embed + train brain + capture signatures (GPU required)
  python cartridge_builder.py ./my-docs/ --name my-knowledge --train

  # Single file
  python cartridge_builder.py notes.txt --name my-notes

  # Custom chunk size for long documents
  python cartridge_builder.py ./papers/ --name papers --chunk-size 500

Output:
  cartridges/my-knowledge.cart.npz          # Embeddings + text
  cartridges/my-knowledge_brain.npy         # Hebbian weights (if --train)
  cartridges/my-knowledge_signatures.npz    # L2 signatures (if --train)
  cartridges/my-knowledge_manifest.json     # SHA256 integrity manifest
"""

import os
import sys
import time
import hashlib
import json
import argparse
import zlib
import numpy as np

# Optional PDF/DOCX support
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None


# ============================================================
# DOCUMENT READING
# ============================================================

def read_file(path: str) -> str:
    """Read a single file and return its text content."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    elif ext == ".pdf":
        if PyPDF2 is None:
            print(f"  Skipping {path} (install PyPDF2: pip install PyPDF2)")
            return ""
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = [p.extract_text() or "" for p in reader.pages]
            return "\n".join(pages)

    elif ext == ".docx":
        if docx is None:
            print(f"  Skipping {path} (install python-docx: pip install python-docx)")
            return ""
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)

    elif ext == ".json":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        # Expect list of objects with 'text' or 'content' field
        if isinstance(data, list):
            passages = []
            for item in data:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content", "")
                    if text:
                        passages.append(str(text))
                elif isinstance(item, str):
                    passages.append(item)
            return "\n\n---PASSAGE_BREAK---\n\n".join(passages)
        return str(data)

    elif ext == ".jsonl":
        passages = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("content", str(obj))
                    passages.append(str(text))
                except json.JSONDecodeError:
                    passages.append(line)  # treat as plain text
        return "\n\n---PASSAGE_BREAK---\n\n".join(passages)

    else:
        print(f"  Skipping unsupported file: {path}")
        return ""


def read_folder(folder: str, recursive: bool = False) -> list[tuple[str, str]]:
    """Read all supported documents from a folder.

    Returns list of (filename, text) tuples.
    """
    supported = {".txt", ".md", ".pdf", ".docx", ".json", ".jsonl"}
    results = []

    if recursive:
        for root, dirs, files in os.walk(folder):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in supported:
                    path = os.path.join(root, f)
                    text = read_file(path)
                    if text.strip():
                        rel_path = os.path.relpath(path, folder)
                        results.append((rel_path, text))
    else:
        for f in sorted(os.listdir(folder)):
            if os.path.splitext(f)[1].lower() in supported:
                path = os.path.join(folder, f)
                if os.path.isfile(path):
                    text = read_file(path)
                    if text.strip():
                        results.append((f, text))

    return results


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-based chunks.

    If the text contains ---PASSAGE_BREAK--- sentinels (from JSON/JSONL
    ingestion), split on those instead of by word count. This preserves
    pre-chunked passages from structured data sources.

    Args:
        text: Input text
        chunk_size: Target words per chunk
        overlap: Words of overlap between chunks
    """
    # Pre-chunked input from JSON/JSONL — respect passage boundaries
    if "---PASSAGE_BREAK---" in text:
        passages = [p.strip() for p in text.split("---PASSAGE_BREAK---")]
        return [p for p in passages if p]

    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


# ============================================================
# HIPPOCAMPUS METADATA
# ============================================================

# 64-byte struct matching hippocampus_header.py layout:
# pattern_id(I) + format_version(B) + cartridge_type(B) +
# parent_ptr(I) + child_ptr(I) + sibling_ptr(I) +
# source_hash(I) + sequence_num(H) + timestamp(I) + flags(B) + reserved(35s)
import struct

# H-block (NPZ-stored hippocampus) — one 64-byte struct per pattern in the
# cart NPZ's `hippocampus` array. Distinct from the lattice-encoded H-row
# (a 64-BIT physics-layer header on row 63). Full spec: docs/PATTERN-ANATOMY.md
# §3 "The Hippocampus: H-row and H-block".
#
# CANONICAL FORMAT (12 fields, with perms_byte at offset 29). Mirrored from
# vector-plus-studio-repo/api/cartridge_io.py. Migration from legacy 11-field
# format completed 2026-05-13. Legacy unpacking lives in the one-shot migration
# script tools/migrate_droplet_mempacks_to_supabase.py; new writes always use
# canonical format. format_version (offset 4) discriminates if mixed carts
# are ever encountered.
HIPPO_FORMAT = '<I B B I I I I H I B B 34s'
HIPPO_SIZE = 64  # struct.calcsize(HIPPO_FORMAT)

# format_version values
FORMAT_VERSION_LEGACY    = 1  # legacy 11-field carts, no perms_byte
FORMAT_VERSION_CANONICAL = 2  # canonical 12-field carts with perms_byte

# Flag bits (byte at offset 28)
FLAG_TOMBSTONE  = 0x01
FLAG_PINNED     = 0x02
FLAG_HAS_PARENT = 0x04
FLAG_HAS_CHILD  = 0x08
FLAG_HAS_SIBLING = 0x10

# Perms bits (byte at offset 29 — canonical format only)
PERM_R = 0x01  # readable — included in search results
PERM_W = 0x02  # writable — tombstoneable, restoreable, updateable in place
PERM_X = 0x04  # reserved for future executable / lambda-passage feature
PERM_DEFAULT = PERM_R | PERM_W  # 0x03 — read+write default for new patterns
# Perishability: bits 5-6 encode decay class
# 00 = volatile (algorithmic time decay, pruneable)
# 01 = replaceable (superseded by newer version)
# 10 = archival (permanent, always accessible)
# 11 = reserved
FLAG_PERISH_MASK = 0x60
FLAG_PERISH_VOLATILE    = 0x00  # default — routine entries
FLAG_PERISH_REPLACEABLE = 0x20  # superseded on update
FLAG_PERISH_ARCHIVAL    = 0x40  # permanent (reflections, discoveries, proven mechanics)


def _source_hash(filename: str) -> int:
    """Deterministic 32-bit hash of a source filename."""
    return int(hashlib.md5(filename.encode()).hexdigest()[:8], 16)


def build_metadata(entries: list[str], doc_map: list[tuple[str, int, int]],
                   cart_name: str = "", creator: str = "") -> tuple[list[bytes], bytes]:
    """Build hippocampus metadata for all entries plus a Pattern 0 header.

    Args:
        entries: List of passage texts (the actual cart content)
        doc_map: List of (filename, chunk_index, total_chunks) per entry,
                 tracking which document each entry came from
        cart_name: Cartridge name for Pattern 0
        creator: Creator identifier for Pattern 0

    Returns:
        (metadata_list, pattern0_bytes):
            metadata_list: List of 64-byte packed structs, one per entry
            pattern0_bytes: 4096-byte Pattern 0 header (if available) or None
    """
    n = len(entries)
    now_ts = int(time.time())
    meta = []

    # --- Group entries by source document ---
    doc_groups: dict[str, list[int]] = {}
    for i, (filename, chunk_idx, total_chunks) in enumerate(doc_map):
        doc_groups.setdefault(filename, []).append(i)

    # --- Build per-entry metadata ---
    for i in range(n):
        filename, chunk_idx, total_chunks = doc_map[i]
        src_hash = _source_hash(filename)

        # Find prev/next within same document
        group = doc_groups[filename]
        pos_in_group = group.index(i)
        prev_ptr = group[pos_in_group - 1] + 1 if pos_in_group > 0 else 0  # +1 because Pattern 0 is header
        next_ptr = group[pos_in_group + 1] + 1 if pos_in_group < len(group) - 1 else 0

        flags = 0
        if prev_ptr > 0:
            flags |= FLAG_HAS_PARENT
        if next_ptr > 0:
            flags |= FLAG_HAS_CHILD

        packed = struct.pack(
            HIPPO_FORMAT,
            i + 1,                       # pattern_id (1-based, 0 = header)
            FORMAT_VERSION_CANONICAL,    # format_version
            0,                           # cartridge_type = knowledge
            prev_ptr,                    # parent_ptr (PREV)
            next_ptr,                    # child_ptr (NEXT)
            0,                           # sibling_ptr
            src_hash,                    # source_hash
            chunk_idx,                   # sequence_num (0-based chunk position)
            now_ts,                      # timestamp
            flags,                       # flags
            PERM_DEFAULT,                # perms_byte (R+W default)
            b'\x00' * 34,                # reserved
        )
        meta.append(packed)

    # --- Pattern 0 header (simplified — just pack into metadata format) ---
    # Full CartridgeHeader uses 4096 bytes across all rows.
    # Here we store a lightweight 64-byte version in the metadata array
    # so the hippocampus struct is consistent. The full Pattern 0 header
    # is a separate concern for GPU-trained cartridges.
    pattern0 = struct.pack(
        HIPPO_FORMAT,
        0,                           # pattern_id = 0 (header)
        FORMAT_VERSION_CANONICAL,    # format_version
        0,                           # cartridge_type = knowledge
        0,                           # parent_ptr (none)
        1 if n > 0 else 0,           # child_ptr → first real pattern
        0,                           # sibling_ptr
        0,                           # source_hash (N/A for header)
        0,                           # sequence_num
        now_ts,                      # timestamp
        FLAG_PINNED,                 # flags: pinned (never evict)
        PERM_R,                      # perms_byte (R only — header is system-managed)
        b'\x00' * 34,                # reserved
    )

    return meta, pattern0


# ============================================================
# EMBEDDING
# ============================================================

_embed_model = None

def get_embedder():
    """Lazy-load SentenceTransformer (same model as Membot and Studio)."""
    global _embed_model
    if _embed_model is None:
        print("Loading Nomic embedder (first run downloads ~270 MB)...")
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
        print("Embedder ready.")
    return _embed_model


def embed_texts(texts: list[str], batch_size: int = 32,
                cooldown_every: int = 500, cooldown_secs: float = 3.0) -> np.ndarray:
    """Embed a list of texts using Nomic with GPU cooldown pauses.

    Truncates passages to ~8000 chars (~2000 tokens) to stay within
    Nomic's 2048-token context and avoid GPU OOM on long poetry passages.

    Processes in macro-batches of `cooldown_every` entries, pausing
    `cooldown_secs` between each to let the GPU cool down.

    Args:
        texts: List of text passages to embed
        batch_size: SentenceTransformer internal batch size (default: 32)
        cooldown_every: Entries between cooldown pauses (default: 500)
        cooldown_secs: Seconds to pause for GPU cooling (default: 3.0)
    """
    model = get_embedder()
    prefixed = [f"search_document: {t[:8000]}" for t in texts]

    # Small enough to do in one shot — no cooldown needed
    if len(prefixed) <= cooldown_every:
        embeddings = model.encode(prefixed, batch_size=batch_size,
                                  show_progress_bar=True, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    # Large corpus: process in macro-batches with cooldown pauses.
    # CHUNK-FLUSH: write each batch to a temp file and clear RAM to prevent
    # the deceleration bug (163/sec → 6/sec from growing numpy arrays).
    import tempfile, glob

    n = len(prefixed)
    t0 = time.time()
    flush_dir = tempfile.mkdtemp(prefix="cartridge_embed_")
    shard_paths = []

    for start in range(0, n, cooldown_every):
        end = min(start + cooldown_every, n)
        batch = prefixed[start:end]
        embs = model.encode(batch, batch_size=batch_size,
                            show_progress_bar=False, convert_to_numpy=True)

        # Flush to disk immediately — don't accumulate in RAM
        shard_path = os.path.join(flush_dir, f"shard_{start:08d}.npy")
        np.save(shard_path, embs.astype(np.float32))
        shard_paths.append(shard_path)
        del embs  # free CPU memory
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # force PyTorch to release GPU memory blocks
        import gc
        gc.collect()  # force garbage collection

        elapsed = time.time() - t0
        rate = end / elapsed if elapsed > 0 else 0
        eta = (n - end) / rate if rate > 0 else 0
        print(f"  Embedded {end}/{n} ({rate:.0f}/sec, ETA {eta:.0f}s)")

        # Cooldown pause (skip after last batch)
        if end < n:
            print(f"  Cooling down {cooldown_secs}s...")
            time.sleep(cooldown_secs)

    # Reassemble from shards
    print(f"  Reassembling {len(shard_paths)} shards ...")
    all_embeddings = [np.load(p) for p in shard_paths]
    result = np.vstack(all_embeddings).astype(np.float32)

    # Cleanup temp files
    for p in shard_paths:
        os.remove(p)
    os.rmdir(flush_dir)

    return result


# ============================================================
# CARTRIDGE SAVING
# ============================================================

def save_cartridge(output_dir: str, name: str, embeddings: np.ndarray, texts: list[str],
                   metadata: list[bytes] = None, pattern0: bytes = None):
    """Save cartridge as secure NPZ with integrity manifest and hippocampus metadata."""
    os.makedirs(output_dir, exist_ok=True)
    cart_path = os.path.join(output_dir, f"{name}.cart.npz")

    # Compress texts
    compressed_texts = []
    for t in texts:
        compressed_texts.append(np.void(zlib.compress(t.encode("utf-8"), level=9)))

    save_kwargs = dict(
        embeddings=embeddings,
        passages=np.array(texts, dtype=object),
        compressed_texts=np.array(compressed_texts, dtype=object),
        version="mcp-v4",
    )

    # Add hippocampus metadata if provided
    if metadata is not None:
        # Pack as a 2D byte array: (n_passages, 64)
        meta_array = np.frombuffer(b''.join(metadata), dtype=np.uint8).reshape(-1, HIPPO_SIZE)
        save_kwargs["hippocampus"] = meta_array
    if pattern0 is not None:
        save_kwargs["pattern0"] = np.frombuffer(pattern0, dtype=np.uint8)

    np.savez_compressed(cart_path, **save_kwargs)

    # Integrity manifest
    h = hashlib.sha256()
    if len(embeddings) > 0:
        h.update(embeddings[0].tobytes())
        h.update(embeddings[-1].tobytes())
    h.update(str(len(texts)).encode())
    fingerprint = h.hexdigest()[:16]

    manifest = {
        "version": "mcp-v4",
        "count": len(texts),
        "has_hippocampus": metadata is not None,
        "fingerprint": fingerprint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = os.path.join(output_dir, f"{name}.cart_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    size_mb = os.path.getsize(cart_path) / (1024 * 1024)
    return cart_path, size_mb, fingerprint


# ============================================================
# HIPPOCAMPUS READING
# ============================================================

def read_metadata(cart_data: dict) -> list[dict]:
    """Unpack hippocampus metadata from a loaded cartridge NPZ.

    Args:
        cart_data: Dict-like from np.load() of a .cart.npz file

    Returns:
        List of metadata dicts (one per passage), or empty list if no metadata.
        Each dict has: pattern_id, format_version, cartridge_type, prev, next,
        sibling, source_hash, sequence_num, timestamp, flags
    """
    if "hippocampus" not in cart_data:
        return []

    raw = cart_data["hippocampus"]  # shape: (n, 64) uint8
    result = []
    for row in raw:
        vals = struct.unpack(HIPPO_FORMAT, row.tobytes())
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


# ============================================================
# LATTICE TRAINING (OPTIONAL — REQUIRES GPU)
# ============================================================

def train_and_sign(output_dir: str, name: str, embeddings: np.ndarray, texts: list[str],
                   train_frames: int = 5, settle_frames: int = 2):
    """Train lattice on embeddings and capture L2 signatures.

    Requires CUDA GPU and lattice_cuda_v7.dll/.so.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)

    from multi_lattice_wrapper_v7 import MultiLatticeCUDAv7

    brain_path = os.path.join(output_dir, f"{name}_brain.npy")
    sig_path = os.path.join(output_dir, f"{name}_signatures.npz")

    n = len(embeddings)
    ml = MultiLatticeCUDAv7(lattice_size=4096, verbose=0)

    # Phase 1: Train
    print(f"\nTraining lattice on {n} patterns ({train_frames} frames)...")
    t0 = time.time()
    for i, emb in enumerate(embeddings):
        ml.reset()
        ml.imprint_vector(emb.astype(np.float32))
        ml.settle(frames=train_frames, learn=True)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"  Trained {i+1}/{n} ({rate:.1f}/sec, ETA {eta:.0f}s)")

    print(f"  Training done: {time.time()-t0:.1f}s")
    ml.save_brain(brain_path)
    brain_mb = os.path.getsize(brain_path) / (1024 * 1024)
    print(f"  Brain saved: {brain_mb:.1f} MB")

    # Phase 2: Capture L2 signatures
    print(f"\nCapturing L2 signatures ({settle_frames} frames)...")
    signatures = np.zeros((n, 4096), dtype=np.float32)
    t0 = time.time()
    for i, emb in enumerate(embeddings):
        ml.reset()
        ml.imprint_vector(emb.astype(np.float32))
        ml.settle(frames=settle_frames, learn=False)
        signatures[i] = ml.recall_l2().flatten()
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"  Captured {i+1}/{n} ({rate:.1f}/sec, ETA {eta:.0f}s)")

    print(f"  Capture done: {time.time()-t0:.1f}s")

    # Save signatures
    titles = [t[:50] for t in texts]
    np.savez_compressed(
        sig_path,
        pattern_ids=np.arange(n, dtype=np.int32),
        signatures=signatures,
        titles=np.array(titles, dtype=object),
        n_patterns=n,
        signature_dim=4096,
        signature_method="l2",
        settle_frames=settle_frames,
    )
    sig_mb = os.path.getsize(sig_path) / (1024 * 1024)
    print(f"  Signatures saved: {sig_mb:.1f} MB")

    return brain_path, sig_path


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build brain cartridges from local documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cartridge_builder.py ./my-docs/ --name my-knowledge
  python cartridge_builder.py ./my-docs/ --name my-knowledge --train
  python cartridge_builder.py notes.txt --name my-notes
  python cartridge_builder.py ./papers/ --name papers --chunk-size 500 --recursive
        """
    )
    parser.add_argument("source", help="File or folder to read")
    parser.add_argument("--name", required=True, help="Cartridge name")
    parser.add_argument("--output-dir", default="cartridges", help="Output directory (default: cartridges/)")
    parser.add_argument("--chunk-size", type=int, default=300, help="Words per chunk for long documents (default: 300)")
    parser.add_argument("--overlap", type=int, default=50, help="Word overlap between chunks (default: 50)")
    parser.add_argument("--no-chunk", action="store_true", help="Don't chunk — one entry per file")
    parser.add_argument("--no-prefix", action="store_true", help="Don't prepend filename to passages (cleaner embeddings for conversational data)")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--train", action="store_true", help="Train lattice + capture signatures (requires GPU)")
    parser.add_argument("--train-frames", type=int, default=5, help="Training settle frames (default: 5)")
    parser.add_argument("--settle-frames", type=int, default=2, help="Signature capture settle frames (default: 2)")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size (default: 64)")
    parser.add_argument("--cooldown-every", type=int, default=500, help="Entries between GPU cooldown pauses (default: 500)")
    parser.add_argument("--cooldown-secs", type=float, default=3.0, help="Seconds to pause for GPU cooling (default: 3.0)")

    args = parser.parse_args()

    # 1. Read documents
    print(f"\n{'='*60}")
    print(f"Membot Cartridge Builder")
    print(f"{'='*60}\n")

    source = os.path.abspath(args.source)

    if os.path.isfile(source):
        print(f"Reading file: {source}")
        text = read_file(source)
        if not text.strip():
            print("Error: File is empty or unsupported.")
            return
        docs = [(os.path.basename(source), text)]
    elif os.path.isdir(source):
        print(f"Reading folder: {source}")
        docs = read_folder(source, recursive=args.recursive)
        if not docs:
            print("Error: No supported files found (.txt, .md, .pdf, .docx)")
            return
    else:
        print(f"Error: {source} not found")
        return

    print(f"  Found {len(docs)} documents")

    # 2. Chunk (tracking document origins for hippocampus linking)
    entries = []
    doc_map = []  # (filename, chunk_index, total_chunks) per entry
    for filename, text in docs:
        if args.no_chunk:
            if args.no_prefix:
                entries.append(text)
            else:
                entries.append(f"{filename}\n{text}")
            doc_map.append((filename, 0, 1))
        else:
            chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
            for i, chunk in enumerate(chunks):
                if args.no_prefix:
                    entries.append(chunk)
                elif len(chunks) > 1:
                    entries.append(f"{filename} (part {i+1}/{len(chunks)})\n{chunk}")
                else:
                    entries.append(f"{filename}\n{chunk}")
                doc_map.append((filename, i, len(chunks)))

    print(f"  Chunked into {len(entries)} entries ({args.chunk_size} words/chunk)")

    # 3. Embed (with GPU cooldown pauses for large corpora)
    print(f"\nEmbedding {len(entries)} entries (cooldown: {args.cooldown_secs}s every {args.cooldown_every})...")
    t0 = time.time()
    embeddings = embed_texts(entries, batch_size=args.batch_size,
                             cooldown_every=args.cooldown_every,
                             cooldown_secs=args.cooldown_secs)
    embed_time = time.time() - t0
    print(f"  Embedded in {embed_time:.1f}s ({len(entries)/embed_time:.1f} entries/sec)")

    # 4. Build hippocampus metadata
    print(f"\nBuilding hippocampus metadata...")
    metadata, pattern0 = build_metadata(entries, doc_map, cart_name=args.name)
    n_docs = len(set(fn for fn, _, _ in doc_map))
    n_linked = sum(1 for m in metadata if struct.unpack_from('<I', m, 6)[0] > 0 or struct.unpack_from('<I', m, 10)[0] > 0)
    print(f"  {len(metadata)} entries from {n_docs} documents, {n_linked} with PREV/NEXT links")

    # 5. Save cartridge
    print(f"\nSaving cartridge...")
    cart_path, size_mb, fingerprint = save_cartridge(
        args.output_dir, args.name, embeddings, entries,
        metadata=metadata, pattern0=pattern0,
    )
    print(f"  {cart_path} ({size_mb:.1f} MB, {fingerprint})")

    # 6. Train (optional)
    if args.train:
        try:
            brain_path, sig_path = train_and_sign(
                args.output_dir, args.name, embeddings, entries,
                train_frames=args.train_frames,
                settle_frames=args.settle_frames,
            )
        except ImportError:
            print("\nError: GPU training requires lattice_cuda_v7.dll/.so and multi_lattice_wrapper_v7.py")
            print("Cartridge saved without brain/signatures (embedding-only search will work).")
        except Exception as e:
            print(f"\nTraining failed: {e}")
            print("Cartridge saved without brain/signatures (embedding-only search will work).")

    # 7. Summary
    print(f"\n{'='*60}")
    print(f"CARTRIDGE READY")
    print(f"  Name:       {args.name}")
    print(f"  Entries:    {len(entries)}")
    print(f"  Dimensions: {embeddings.shape[1]}")
    print(f"  Documents:  {n_docs}")
    print(f"  Linked:     {n_linked} entries with PREV/NEXT navigation")
    print(f"  Cartridge:  {cart_path}")
    if args.train:
        print(f"  Brain:      {args.output_dir}/{args.name}_brain.npy")
        print(f"  Signatures: {args.output_dir}/{args.name}_signatures.npz")
    print(f"  Manifest:   {args.output_dir}/{args.name}.cart_manifest.json")
    print(f"\nDrop into membot's cartridges/ folder and mount it!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
