#!/usr/bin/env python3
"""walk_associate_core(cart_id=...) — HUB's seam, asked 2026-06-17, shipped 2026-10-26.

THE SEAM. The Web4 hub's membox sidecar mounts a cart by PATH through
`multi_cart.mount(...)`, because `mount_cartridge` resolves a registry NAME against
membot's depot and cannot open an arbitrary `.cart.npz`. But `walk_associate_core`
read its corpus out of the SESSION subsystem, so a cart the sidecar had mounted was
unreachable from walk. HUB asked for one of three fixes; we answered option 2.

§5 is the one that matters to them and is the claim we made in writing on 2026-10-24:
"Walk's returned `idx` indexes the same passage order as `multi_cart.search`'s
`local_addr`." That was asserted from reading the code. This measures it.

No network, no GPU. Builds a tiny real cart on disk, mounts it both ways.

Run from membot/:  python test_walk_cart_id.py
"""
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_MEMBOT_DIR = _HERE if os.path.exists(os.path.join(_HERE, "membot_server.py")) \
    else os.path.dirname(_HERE)
sys.path.insert(0, _MEMBOT_DIR)

import numpy as np                                      # noqa: E402

failures = []


def check(label, ok, detail=""):
    if not ok:
        failures.append(label)
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if detail and not ok else ""))


# Deterministic, dependency-free embeddings: a passage's vector is decided by which
# keyword it contains, so "nearest" is predictable and no model is loaded.
TOPICS = ["alpha", "beta", "gamma", "delta"]


def fake_embed(text, *a, **kw):
    v = np.zeros(768, dtype=np.float32)
    low = (text or "").lower()
    for i, t in enumerate(TOPICS):
        if t in low:
            v[i * 10:(i * 10) + 10] = 1.0
    if not v.any():
        v[700] = 1.0
    return v / (np.linalg.norm(v) + 1e-9)


import membot_server as m                               # noqa: E402
import multi_cart as mc                                 # noqa: E402
import embed as _embed_mod                              # noqa: E402

m.embed_text = fake_embed
_embed_mod.embed_text = fake_embed
try:
    mc.embed_text = fake_embed
except Exception:
    pass

TEXTS = [
    "alpha is the first letter and alpha appears often",
    "beta follows alpha in the ordering",
    "gamma is the third and relates to beta",
    "delta closes the set after gamma",
    "alpha again, a second mention of alpha entirely",
]

tmp = tempfile.mkdtemp(prefix="walkcart-")
cart_path = os.path.join(tmp, "seamtest.cart.npz")
try:
    embs = np.stack([fake_embed(t) for t in TEXTS]).astype(np.float32)
    np.savez_compressed(
        cart_path,
        embeddings=embs,
        passages=np.array(TEXTS, dtype=object),
        compressed_texts=np.array(TEXTS, dtype=object),
        version="mcp-v3",
    )

    print("\n1 - the seam opens: a multi_cart mount is now walkable")
    mounted = mc.mount(cart_path, cart_id="seamtest", verify_integrity=False)
    check("cart mounted through multi_cart", mc.get_cart("seamtest") is not None, mounted)
    r = m.walk_associate_core("alpha", top_k=3, cart_id="seamtest")
    check("walk returns no error with cart_id", r["error"] is None, r["error"])
    check("walk found primaries", len(r["primary"]) > 0, r)

    print("\n2 - the two-key shim: cart_name is reported, not a KeyError")
    # multi_cart state has `cart_name`; walk used to read `cartridge_name`.
    check("result['cart'] is populated", bool(r["cart"]), r["cart"])
    check("header names the cart, not 'None'",
          "None" not in r["header"].split("from")[-1], r["header"])

    print("\n3 - a pooled cart is NOT mutated by a read path")
    st = mc.get_cart("seamtest")
    check("no query_count written into the shared cart state",
          "query_count" not in st, sorted(st.keys()))

    print("\n4 - a bad cart_id fails loudly and names what IS mounted")
    bad = m.walk_associate_core("alpha", cart_id="no-such-cart")
    check("bad cart_id is an error, not a silent session fallback",
          bad["error"] is not None, bad)
    check("error names the real mounted cart so a typo is obvious",
          "seamtest" in (bad["error"] or ""), bad["error"])
    check("error does not leak results", bad["primary"] == [] and bad["missed"] == [], bad)

    print("\n5 - *** THE CLAIM WE MADE TO HUB: walk idx == multi_cart local_addr ***")
    hits = mc.search("alpha", top_k=5, scope="seamtest")
    rows = hits.get("results", hits) if isinstance(hits, dict) else hits
    addr_by_text = {}
    for h in rows:
        if isinstance(h, dict) and "local_addr" in h:
            addr_by_text[(h.get("text") or "")[:40]] = h["local_addr"]
    check("multi_cart.search returned addressable rows", bool(addr_by_text), rows)
    if addr_by_text:
        agree = mismatch = 0
        for item in r["primary"]:
            key = (item["text"] or "")[:40]
            if key in addr_by_text:
                if addr_by_text[key] == item["idx"]:
                    agree += 1
                else:
                    mismatch += 1
        check("every shared passage has idx == local_addr",
              mismatch == 0 and agree > 0, f"agree={agree} mismatch={mismatch}")
        check("walk idx indexes the same texts list",
              all(TEXTS[i["idx"]] == i["text"] for i in r["primary"] if i["idx"] < len(TEXTS)),
              [(i["idx"], i["text"][:30]) for i in r["primary"]])

    print("\n6 - the default path is untouched (no cart_id = session, as before)")
    sid = "walk-seam-session"
    m._sessions.pop(sid, None)
    s = m._get_session(sid)
    check("a session with no cartridge still errors the old way",
          m.walk_associate_core("alpha", session_id=sid)["error"] ==
          "No cartridge mounted. Use mount_cartridge first.",
          m.walk_associate_core("alpha", session_id=sid)["error"])
    s["cartridge_name"] = "session-cart"
    s["texts"] = list(TEXTS)
    s["embeddings"] = embs
    s["has_embeddings"] = True
    sr = m.walk_associate_core("alpha", top_k=3, session_id=sid)
    check("session walk still works", sr["error"] is None and len(sr["primary"]) > 0, sr)
    check("session walk reports cartridge_name", sr["cart"] == "session-cart", sr["cart"])
    check("session DOES get its query_count incremented",
          s.get("query_count", 0) >= 1, s.get("query_count"))

    print("\n7 - cart_id wins over session_id when both are given")
    both = m.walk_associate_core("alpha", top_k=3, session_id=sid, cart_id="seamtest")
    check("cart_id takes precedence", both["cart"] != "session-cart", both["cart"])

finally:
    try:
        mc.unmount("seamtest")
    except Exception:
        pass
    shutil.rmtree(tmp, ignore_errors=True)

print("\n" + "=" * 62)
if failures:
    print(f"VERDICT: {len(failures)} FAILED")
    for f in failures:
        print(f"  - {f}")
else:
    print("VERDICT: all checks passed")
print("=" * 62)
sys.exit(1 if failures else 0)
