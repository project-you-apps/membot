"""
Inject hippocampus metadata into an existing .cart.npz
======================================================
Reads the cart, figures out document groupings from passage tags,
builds PREV/NEXT links, and re-saves with hippocampus data.

No re-embedding needed — just adds the missing metadata.

Usage:
  python inject_hippocampus.py cartridges/gutenberg-poetry.cart.npz
"""

import os
import sys
import re
import struct
import hashlib
import time
import zlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cartridge_builder import HIPPO_FORMAT, HIPPO_SIZE, build_metadata, save_cartridge


def extract_source(text: str) -> str:
    """Extract source identifier from a passage's tag prefix.

    Poetry passages look like:
      [Poem: "Title" from Collection by Author]
    or:
      [Collection by Author]

    Knowledge/wiki passages might have other prefixes.
    Falls back to first 80 chars as source key.
    """
    # Poetry tag: [Poem: "title" from Collection by Author]
    m = re.match(r'^\[Poem:\s*(?:"[^"]*"\s*from\s*)?(.+?\s+by\s+.+?)\]', text)
    if m:
        return m.group(1).strip()

    # Generic tag: [Collection by Author]
    m = re.match(r'^\[([^\]]+)\]', text)
    if m:
        return m.group(1).strip()

    # No tag — use first line as source
    first_line = text.split('\n')[0][:80].strip()
    return first_line or "unknown"


def main():
    if len(sys.argv) < 2:
        print("Usage: python inject_hippocampus.py <cart.npz>")
        sys.exit(1)

    cart_path = sys.argv[1]
    print(f"Loading: {cart_path}")

    data = np.load(cart_path, allow_pickle=True)
    keys = list(data.keys())
    print(f"Keys: {keys}")

    if "hippocampus" in data:
        print("Cart already has hippocampus! Nothing to do.")
        return

    # Get texts
    if "compressed_texts" in data:
        texts = []
        for ct in data["compressed_texts"]:
            try:
                texts.append(zlib.decompress(bytes(ct)).decode("utf-8"))
            except Exception:
                texts.append("[decompress error]")
    elif "passages" in data:
        texts = [str(p) for p in data["passages"]]
    else:
        print("ERROR: No texts found in cart!")
        sys.exit(1)

    print(f"Passages: {len(texts)}")

    # Build doc_map by grouping consecutive passages with same source
    doc_map = []
    groups = {}  # source -> list of indices
    for i, text in enumerate(texts):
        source = extract_source(text)
        if source not in groups:
            groups[source] = []
        groups[source].append(i)

    # Build doc_map: (source, chunk_index_within_source, total_chunks_in_source)
    source_counters = {}
    for i, text in enumerate(texts):
        source = extract_source(text)
        if source not in source_counters:
            source_counters[source] = 0
        chunk_idx = source_counters[source]
        total = len(groups[source])
        doc_map.append((source, chunk_idx, total))
        source_counters[source] += 1

    print(f"Sources: {len(groups)}")
    # Show a few examples
    for src, indices in list(groups.items())[:5]:
        print(f"  {src}: {len(indices)} passages (idx {indices[0]}-{indices[-1]})")

    # Build hippocampus
    print(f"\nBuilding hippocampus metadata...")
    metadata, pattern0 = build_metadata(texts, doc_map, cart_name=os.path.basename(cart_path))

    n_linked = sum(1 for m in metadata if struct.unpack('<I', m[6:10])[0] > 0 or struct.unpack('<I', m[10:14])[0] > 0)
    print(f"  {len(metadata)} entries, {n_linked} with PREV/NEXT links")

    # Re-save: copy all existing arrays + add hippocampus
    print(f"\nRe-saving with hippocampus...")
    meta_array = np.frombuffer(b''.join(metadata), dtype=np.uint8).reshape(-1, HIPPO_SIZE)

    save_kwargs = {}
    for key in data.keys():
        save_kwargs[key] = data[key]
    save_kwargs["hippocampus"] = meta_array
    if pattern0 is not None:
        save_kwargs["pattern0"] = np.frombuffer(pattern0, dtype=np.uint8)

    # Save to same path (overwrite)
    np.savez_compressed(cart_path, **save_kwargs)
    size_mb = os.path.getsize(cart_path) / (1024 * 1024)
    print(f"  Saved: {cart_path} ({size_mb:.1f} MB)")

    # Verify
    print(f"\nVerifying...")
    check = np.load(cart_path, allow_pickle=True)
    if "hippocampus" in check:
        h = check["hippocampus"]
        print(f"  hippocampus: shape={h.shape}, dtype={h.dtype}")
        fmt = '<I B B I I I I H I B 35s'
        for i in range(min(3, len(h))):
            vals = struct.unpack(fmt, h[i].tobytes())
            prev = vals[3] if vals[3] > 0 else None
            nxt = vals[4] if vals[4] > 0 else None
            print(f"  [{i}] pattern_id={vals[0]} prev={prev} next={nxt}")
        print("  OK!")
    else:
        print("  FAILED — hippocampus not found after save!")


if __name__ == "__main__":
    main()
