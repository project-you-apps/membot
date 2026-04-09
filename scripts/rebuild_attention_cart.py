"""
rebuild_attention_cart.py — One-shot rebuild of attention-is-all-you-need.cart.npz
with all fields preserved: hippocampus, per_pattern_meta, sign_bits, pattern0.

Background: At some point the cart on disk got rebuilt without preserving
the hippocampus row, while the older committed version had hippocampus but
lacked per_pattern_meta. Neither was complete. This script does a fresh
rebuild from the source PDF using cartridge_builder (which gives us
hippocampus + pattern0), then patches the cart to add the curated
per_pattern_meta from the existing on-disk version, then re-saves with
sign_bits computed from the embeddings.

The result is a cart that has EVERYTHING:
  - embeddings (float32, computed fresh from the PDF)
  - passages (uncompressed text)
  - compressed_texts (zlib for storage efficiency)
  - hippocampus (episodic linking row)
  - pattern0 (cart-level metadata)
  - per_pattern_meta (curated per-passage metadata: tags, owner, description...)
  - sign_bits (precomputed sign-zero binary corpus for fast Hamming search)

Run from membot/:
    python scripts/rebuild_attention_cart.py
    python scripts/rebuild_attention_cart.py --dry-run    # show plan, don't write
"""

import argparse
import hashlib
import json
import os
import sys
import time
import zlib

import numpy as np

# scripts/ is one level deep — add membot/ to sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_MEMBOT_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _MEMBOT_DIR)

from cartridge_builder import (
    read_file,
    chunk_text,
    embed_texts,
    build_metadata,
    save_cartridge,
)

# Hardcoded paths for this one-shot
PROJECT_ROOT = os.path.abspath(os.path.join(_MEMBOT_DIR, ".."))
PDF_SOURCE = os.path.join(
    PROJECT_ROOT,
    "docs",
    "Books, Papers, and Theories",
    "Industry Papers, etc",
    "Attention-Is-All-You-Need.pdf",
)
CART_NAME = "attention-is-all-you-need"
CART_PATH = os.path.join(_MEMBOT_DIR, "cartridges", f"{CART_NAME}.cart.npz")
OUTPUT_DIR = os.path.join(_MEMBOT_DIR, "cartridges")

# Chunking parameters that produced the original 24 chunks (cart-builder defaults)
CHUNK_SIZE = 300
OVERLAP = 50


def extract_existing_per_pattern_meta(cart_path: str):
    """Read the existing on-disk cart and extract its per_pattern_meta JSON.
    Returns the parsed list of dicts, or None if absent.
    """
    if not os.path.exists(cart_path):
        return None
    try:
        data = np.load(cart_path, allow_pickle=True)
        if "per_pattern_meta" not in data.files:
            return None
        meta = data["per_pattern_meta"]
        # Stored as 0-d ndarray holding a single string (a JSON list of dicts)
        s = meta.item() if meta.ndim == 0 else str(meta)
        parsed = json.loads(s)
        if not isinstance(parsed, list):
            return None
        return parsed
    except Exception as e:
        print(f"  WARN: could not extract per_pattern_meta from {cart_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without writing files")
    args = parser.parse_args()

    print("=" * 60)
    print("ATTENTION CART REBUILD (one-shot, complete)")
    print("=" * 60)
    print()

    # 1. Sanity checks
    if not os.path.exists(PDF_SOURCE):
        print(f"FATAL: PDF not found at {PDF_SOURCE}")
        sys.exit(1)
    print(f"PDF source:  {PDF_SOURCE}")
    print(f"Output cart: {CART_PATH}")
    print()

    # 2. Extract existing per_pattern_meta from current on-disk cart
    print("[1/5] Extracting existing per_pattern_meta...")
    existing_meta = extract_existing_per_pattern_meta(CART_PATH)
    if existing_meta is None:
        print("      No existing per_pattern_meta found — will build cart without it")
    else:
        print(f"      Found {len(existing_meta)} per-pattern metadata entries")
        if existing_meta and isinstance(existing_meta[0], dict):
            sample = existing_meta[0]
            print(f"      Sample fields: {list(sample.keys())}")
            if "tags" in sample:
                print(f"      Tags from first entry: {sample['tags']}")
    print()

    # 3. Read and chunk the PDF
    print("[2/5] Reading and chunking PDF...")
    text = read_file(PDF_SOURCE)
    if not text.strip():
        print("FATAL: PDF read returned empty text. Check that PyPDF2 is installed.")
        sys.exit(1)
    print(f"      Read {len(text)} characters")

    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    print(f"      Chunked into {len(chunks)} pieces (chunk_size={CHUNK_SIZE}, overlap={OVERLAP})")
    print()

    # Match cartridge_builder.py format: prefix each chunk with filename + part number
    filename = os.path.basename(PDF_SOURCE)
    entries = []
    doc_map = []
    if len(chunks) > 1:
        for i, chunk in enumerate(chunks):
            entries.append(f"{filename} (part {i+1}/{len(chunks)})\n{chunk}")
            doc_map.append((filename, i, len(chunks)))
    else:
        entries.append(f"{filename}\n{chunks[0]}")
        doc_map.append((filename, 0, 1))

    # 4. Sanity check chunk count vs existing per_pattern_meta length
    if existing_meta is not None and len(existing_meta) != len(entries):
        print(f"      WARN: chunk count mismatch! existing per_pattern_meta has "
              f"{len(existing_meta)} entries, new chunking produced {len(entries)}.")
        print(f"      The per_pattern_meta will be padded/truncated to fit the new chunking.")
        # Pad with copies of the first entry, or truncate
        if len(existing_meta) < len(entries):
            template = existing_meta[0] if existing_meta else {}
            existing_meta = existing_meta + [dict(template) for _ in range(len(entries) - len(existing_meta))]
            # Update chunk indices on the padded entries
            for i in range(len(entries)):
                if i < len(existing_meta) and isinstance(existing_meta[i], dict):
                    existing_meta[i]["chunk"] = i
                    existing_meta[i]["chunks"] = len(entries)
        else:
            existing_meta = existing_meta[:len(entries)]
    elif existing_meta is not None:
        # Counts match — just sync the chunk/chunks fields in case they're stale
        for i in range(len(entries)):
            if isinstance(existing_meta[i], dict):
                existing_meta[i]["chunk"] = i
                existing_meta[i]["chunks"] = len(entries)

    if args.dry_run:
        print("[3/5] DRY RUN — would embed", len(entries), "entries")
        print("[4/5] DRY RUN — would build hippocampus + pattern0 metadata")
        print("[5/5] DRY RUN — would save cart with embeddings + hippocampus + pattern0 + per_pattern_meta + sign_bits")
        print()
        print("Re-run without --dry-run to actually rebuild.")
        return

    # 5. Embed
    print("[3/5] Embedding (this loads Nomic — first run downloads ~270 MB)...")
    t0 = time.time()
    embeddings = embed_texts(entries, batch_size=32)
    embed_time = time.time() - t0
    print(f"      Embedded {len(entries)} entries in {embed_time:.1f}s "
          f"({len(entries)/embed_time:.1f} entries/sec)")
    print(f"      Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    print()

    # 6. Build hippocampus + pattern0
    print("[4/5] Building hippocampus metadata...")
    metadata, pattern0 = build_metadata(entries, doc_map, cart_name=CART_NAME)
    print(f"      Built {len(metadata)} hippocampus entries + pattern0")
    print()

    # 7. Save the cart with hippocampus + pattern0 (the cart_builder default)
    print("[5/5] Saving cart with all fields...")
    cart_path, size_mb, fingerprint = save_cartridge(
        OUTPUT_DIR, CART_NAME, embeddings, entries,
        metadata=metadata, pattern0=pattern0,
    )
    print(f"      Base cart saved: {cart_path} ({size_mb:.2f} MB, fingerprint {fingerprint})")

    # 8. Patch in per_pattern_meta + sign_bits (cart_builder.save_cartridge doesn't write these)
    print("      Patching in per_pattern_meta and sign_bits...")
    data = np.load(cart_path, allow_pickle=True)
    save_kwargs = {k: data[k] for k in data.files}

    # Add per_pattern_meta (as a 0-d ndarray holding the JSON string, matching the existing format)
    if existing_meta is not None:
        meta_json = json.dumps(existing_meta)
        save_kwargs["per_pattern_meta"] = np.array(meta_json, dtype=object)
        print(f"        per_pattern_meta: {len(existing_meta)} entries, {len(meta_json)} chars")
    else:
        print("        per_pattern_meta: skipped (no source data)")

    # Add sign_bits (precomputed sign-zero binary corpus for Hamming search)
    sign_bits = np.packbits((embeddings > 0).astype(np.uint8), axis=1)
    save_kwargs["sign_bits"] = sign_bits
    print(f"        sign_bits: {sign_bits.shape}")

    # Re-save
    np.savez_compressed(cart_path, **save_kwargs)

    # 9. Rebuild manifest with the FINAL fingerprint (which is the same as save_cartridge gave us
    # because we didn't change embeddings or passage count)
    final_size = os.path.getsize(cart_path)
    print(f"      Final cart size: {final_size / 1024:.1f} KB")
    print()

    # Verify the final cart by loading it back and checking all expected keys are present
    print("Verifying...")
    final = np.load(cart_path, allow_pickle=True)
    expected_keys = {"embeddings", "passages", "compressed_texts", "hippocampus",
                     "pattern0", "per_pattern_meta", "sign_bits"}
    actual_keys = set(final.files)
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    print(f"  Keys present: {sorted(actual_keys)}")
    if missing:
        print(f"  WARN: missing keys: {sorted(missing)}")
    if extra:
        print(f"  Extra keys (probably fine): {sorted(extra)}")
    if not missing:
        print("  ALL EXPECTED FIELDS PRESENT")

    # Re-run the rebuild_manifests script logic to update the manifest fingerprint
    print()
    print("Updating manifest fingerprint to match final cart...")
    h = hashlib.sha256()
    h.update(embeddings[0].tobytes())
    h.update(embeddings[-1].tobytes())
    h.update(str(len(entries)).encode())
    final_fingerprint = h.hexdigest()[:16]
    manifest_path = cart_path.rsplit(".", 1)[0] + "_manifest.json"
    manifest = {
        "version": "mcp-v4",
        "count": len(entries),
        "has_hippocampus": True,
        "fingerprint": final_fingerprint,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rebuilt_by": "scripts/rebuild_attention_cart.py",
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest written: {manifest_path}")
    print(f"  Fingerprint: {final_fingerprint}")
    print()
    print("=" * 60)
    print("REBUILD COMPLETE")
    print("=" * 60)
    print(f"Cart:     {cart_path}")
    print(f"Manifest: {manifest_path}")
    print()
    print("Next: re-run tests/test_multi_cart.py to verify the cart mounts cleanly")
    print("with verify_integrity=True (and the FINGERPRINT MISMATCH warning is gone).")


if __name__ == "__main__":
    main()
