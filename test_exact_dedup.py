#!/usr/bin/env python3
"""Exact-dedup refuses byte-identical re-stores, and the index stays correct and incremental.

The FEATURE is dp's (bfce6ab, 2026-06-27): clients that re-emit recurring patterns every
session bloat the cartridge and pollute recall, so a byte-identical re-store is skipped.

The IMPLEMENTATION is ours and that is why this file exists. dp's was
`stored_text in state["texts"]` -- an O(n) linear scan on every store, correct at fleet
cart sizes and wrong at 2.4M passages. Ours builds a set once per session and keeps it
incrementally, which introduces a failure mode his version could not have: an index that
drifts out of step with the list it summarizes. §3 and §4 exist for exactly that.

No network, no GPU, no disk. Drives memory_store against a hand-made session dict.

Run from membot/:  python test_exact_dedup.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MEMBOT_DIR = _HERE if os.path.exists(os.path.join(_HERE, "membot_server.py")) \
    else os.path.dirname(_HERE)
sys.path.insert(0, _MEMBOT_DIR)

import numpy as np                                      # noqa: E402
import membot_server as m                               # noqa: E402

failures = []


def check(label, ok, detail=""):
    if not ok:
        failures.append(label)
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if detail and not ok else ""))


def fresh_session():
    """A minimal writable session, bypassing mount/embed entirely."""
    sid = "dedup-test"
    m._sessions.pop(sid, None)
    st = m._get_session(sid)
    st["cartridge_name"] = "dedup-test"
    st["texts"] = []
    st["embeddings"] = np.zeros((0, 768), dtype=np.float32)
    st["binary_corpus"] = None
    st["mount_read_only"] = False
    return sid, st


# Embedding is not what is under test: make it deterministic and free.
m.embed_text = lambda text, *a, **kw: np.ones(768, dtype=np.float32)
try:
    import embed as _e
    _e.embed_text = m.embed_text
except Exception:
    pass
_store = getattr(m.memory_store, "fn", m.memory_store)

print("\n1 - a byte-identical re-store is refused")
sid, st = fresh_session()
r1 = _store("the kettle is on", session_id=sid)
r2 = _store("the kettle is on", session_id=sid)
check("first store is accepted", "Stored memory" in r1, r1)
check("second, identical store is refused", "Duplicate" in r2, r2)
check("only ONE passage landed", len(st["texts"]) == 1, len(st["texts"]))
check("embeddings row count matches texts",
      st["embeddings"].shape[0] == len(st["texts"]),
      (st["embeddings"].shape, len(st["texts"])))

print("\n2 - tags are part of the identity, not decoration")
sid, st = fresh_session()
_store("same body", session_id=sid)
r = _store("same body", tags="urgent", session_id=sid)
check("same text under a DIFFERENT tag is a different memory",
      "Stored memory" in r, r)
check("both landed", len(st["texts"]) == 2, st["texts"])
r = _store("same body", tags="urgent", session_id=sid)
check("re-storing that tagged form IS a duplicate", "Duplicate" in r, r)

print("\n3 - the incremental index does not drift")
sid, st = fresh_session()
for i in range(5):
    _store(f"entry {i}", session_id=sid)
check("five distinct stores all landed", len(st["texts"]) == 5, len(st["texts"]))
check("index size tracks the list", len(st.get("_dedup_set", ())) == 5, st.get("_dedup_set"))
check("counter tracks the list", st.get("_dedup_n") == 5, st.get("_dedup_n"))
r = _store("entry 3", session_id=sid)
check("an earlier entry is still caught after later writes", "Duplicate" in r, r)

print("\n4 - an append made behind the index's back is detected, not missed")
# The whole risk of caching: another path mutates texts and the set goes stale. The
# length guard must notice and rebuild rather than silently admitting a duplicate.
st["texts"].append("snuck in directly")
r = _store("snuck in directly", session_id=sid)
check("stale index is rebuilt and the duplicate is caught", "Duplicate" in r, r)
check("rebuild did not lose earlier entries",
      "entry 0" in st["_dedup_set"] and "entry 3" in st["_dedup_set"],
      sorted(st["_dedup_set"]))

print("\n5 - sessions do not share a dedup index")
sid_a, st_a = fresh_session()
_store("shared wording", session_id=sid_a)
sid_b = "dedup-test-b"
m._sessions.pop(sid_b, None)
st_b = m._get_session(sid_b)
st_b["cartridge_name"] = "dedup-test-b"
st_b["texts"] = []
st_b["embeddings"] = np.zeros((0, 768), dtype=np.float32)
st_b["binary_corpus"] = None
st_b["mount_read_only"] = False
r = _store("shared wording", session_id=sid_b)
check("the same text in ANOTHER session is not a duplicate", "Stored memory" in r, r)

print("\n" + "=" * 62)
if failures:
    print(f"VERDICT: {len(failures)} FAILED")
    for f in failures:
        print(f"  - {f}")
else:
    print("VERDICT: all checks passed")
print("=" * 62)
sys.exit(1 if failures else 0)
