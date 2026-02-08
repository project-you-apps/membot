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

    else:
        print(f"  Skipping unsupported file: {path}")
        return ""


def read_folder(folder: str, recursive: bool = False) -> list[tuple[str, str]]:
    """Read all supported documents from a folder.

    Returns list of (filename, text) tuples.
    """
    supported = {".txt", ".md", ".pdf", ".docx"}
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

    Args:
        text: Input text
        chunk_size: Target words per chunk
        overlap: Words of overlap between chunks
    """
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


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of texts using Nomic."""
    model = get_embedder()
    prefixed = [f"search_document: {t}" for t in texts]
    embeddings = model.encode(prefixed, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype(np.float32)


# ============================================================
# CARTRIDGE SAVING
# ============================================================

def save_cartridge(output_dir: str, name: str, embeddings: np.ndarray, texts: list[str]):
    """Save cartridge as secure NPZ with integrity manifest."""
    os.makedirs(output_dir, exist_ok=True)
    cart_path = os.path.join(output_dir, f"{name}.cart.npz")

    # Compress texts
    compressed_texts = []
    for t in texts:
        compressed_texts.append(np.void(zlib.compress(t.encode("utf-8"), level=9)))

    np.savez_compressed(
        cart_path,
        embeddings=embeddings,
        passages=np.array(texts, dtype=object),
        compressed_texts=np.array(compressed_texts, dtype=object),
        version="mcp-v3",
    )

    # Integrity manifest
    h = hashlib.sha256()
    if len(embeddings) > 0:
        h.update(embeddings[0].tobytes())
        h.update(embeddings[-1].tobytes())
    h.update(str(len(texts)).encode())
    fingerprint = h.hexdigest()[:16]

    manifest = {
        "version": "mcp-v3",
        "count": len(texts),
        "fingerprint": fingerprint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_path = os.path.join(output_dir, f"{name}.cart_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    size_mb = os.path.getsize(cart_path) / (1024 * 1024)
    return cart_path, size_mb, fingerprint


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
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--train", action="store_true", help="Train lattice + capture signatures (requires GPU)")
    parser.add_argument("--train-frames", type=int, default=5, help="Training settle frames (default: 5)")
    parser.add_argument("--settle-frames", type=int, default=2, help="Signature capture settle frames (default: 2)")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size (default: 64)")

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

    # 2. Chunk
    entries = []
    for filename, text in docs:
        if args.no_chunk:
            entries.append(f"{filename}\n{text}")
        else:
            chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
            for i, chunk in enumerate(chunks):
                if len(chunks) > 1:
                    entries.append(f"{filename} (part {i+1}/{len(chunks)})\n{chunk}")
                else:
                    entries.append(f"{filename}\n{chunk}")

    print(f"  Chunked into {len(entries)} entries ({args.chunk_size} words/chunk)")

    # 3. Embed
    print(f"\nEmbedding {len(entries)} entries...")
    t0 = time.time()
    embeddings = embed_texts(entries, batch_size=args.batch_size)
    embed_time = time.time() - t0
    print(f"  Embedded in {embed_time:.1f}s ({len(entries)/embed_time:.1f} entries/sec)")

    # 4. Save cartridge
    print(f"\nSaving cartridge...")
    cart_path, size_mb, fingerprint = save_cartridge(args.output_dir, args.name, embeddings, entries)
    print(f"  {cart_path} ({size_mb:.1f} MB, {fingerprint})")

    # 5. Train (optional)
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

    # 6. Summary
    print(f"\n{'='*60}")
    print(f"CARTRIDGE READY")
    print(f"  Name:       {args.name}")
    print(f"  Entries:    {len(entries)}")
    print(f"  Dimensions: {embeddings.shape[1]}")
    print(f"  Cartridge:  {cart_path}")
    if args.train:
        print(f"  Brain:      {args.output_dir}/{args.name}_brain.npy")
        print(f"  Signatures: {args.output_dir}/{args.name}_signatures.npz")
    print(f"  Manifest:   {args.output_dir}/{args.name}.cart_manifest.json")
    print(f"\nDrop into membot's cartridges/ folder and mount it!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
