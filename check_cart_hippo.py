"""Quick check: does a .cart.npz have hippocampus data?"""
import sys, os, struct
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "cartridges/gutenberg-poetry.cart.npz"
print(f"Checking: {path}")

data = np.load(path, allow_pickle=True)
print(f"Keys: {list(data.keys())}")

if "hippocampus" in data:
    h = data["hippocampus"]
    print(f"hippocampus: shape={h.shape}, dtype={h.dtype}")
    # Unpack first few entries
    fmt = '<I B B I I I I H I B 35s'
    for i in range(min(5, len(h))):
        vals = struct.unpack(fmt, h[i].tobytes())
        prev = vals[3] if vals[3] > 0 else None
        nxt = vals[4] if vals[4] > 0 else None
        print(f"  [{i}] pattern_id={vals[0]} prev={prev} next={nxt} seq={vals[7]} flags={vals[9]:02x}")
    # Count linked entries
    n_linked = 0
    for row in h:
        vals = struct.unpack(fmt, row.tobytes())
        if vals[3] > 0 or vals[4] > 0:
            n_linked += 1
    print(f"Linked entries: {n_linked}/{len(h)}")
else:
    print("NO hippocampus key found!")

if "pattern0" in data:
    print(f"pattern0: {len(data['pattern0'])} bytes")
else:
    print("No pattern0 key.")
