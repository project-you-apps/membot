"""Shared cart payload pool — correctness and liveness. 2026-10-18.

Step 2 of the demo-session work. Step 1 (commit f496684) gave every browser tab its
own session; on its own that multiplies RAM, because mount_cartridge loaded the cart
from disk per session. This pool loads once per PATH and hands sessions references.

⛔ THE TEST THAT MATTERS IS NOT "IS IT CORRECT", IT IS "IS IT SHARED". A pool that
quietly loads twice passes every behavioural check — same results, same search, same
everything — and delivers nothing. So §2 counts disk loads, and §3 asserts object
IDENTITY rather than equality. Both would fail if the pool were bypassed.

⛔ And sharing is only safe if nothing writes in place, so §5 runs the actual hazard:
memory_store appends to texts in place while rebinding embeddings. Without
copy-on-write that grows one session's text list inside every other session.

No GPU, no network, no model load — embed_text is stubbed. Output is ~30 lines.
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import membot_server as M          # noqa: E402

save_as_npz = M.save_as_npz
save_manifest = M.save_manifest

RESULTS = []


def raw(name):
    """The plain function behind an @mcp.tool decorator.

    FastMCP replaces the module attribute with a FunctionTool object, so calling
    mount_cartridge(...) directly raises TypeError. The tool keeps the original
    under .fn; fall back to the attribute itself for anything undecorated.
    """
    obj = getattr(M, name)
    return getattr(obj, "fn", obj)


mount_cartridge = raw("mount_cartridge")
memory_store = raw("memory_store")
unmount = raw("unmount")

# ⭐ TWO NEGATIVE CONTROLS, and they catch DIFFERENT checks. Neither alone is enough.
#
#   --no-pool  disables sharing outright. Catches §1-§4 and §6: identity, load counts,
#              refcounts, eviction.
#              ⛔ It does NOT exercise §5. With every session on its own copy there is
#              nothing to corrupt, so the copy-on-write checks pass with CoW removed —
#              a check identical in both states.
#
#   --no-cow   leaves sharing ON and removes ONLY copy-on-write. This is the arm that
#              makes §5 mean something: the writer's in-place texts.append() lands in
#              the shared list and every other holder sees it.
#
# The suite MUST FAIL in each mode. A green run is the failure that looks like success.
NO_POOL = "--no-pool" in sys.argv
NO_COW = "--no-cow" in sys.argv
CONTROL = NO_POOL or NO_COW

if NO_POOL:
    def _unshared_load(cart_path):
        return M.load_cartridge_safe(cart_path), False
    M._shared_load = _unshared_load
    M._shared_register = lambda state, sid, path: None
    M._shared_release = lambda state, sid: None
    M._detach_for_write = lambda state, sid: None
    print("\n*** CONTROL A: sharing disabled — §1-§4, §6 EXPECTED TO FAIL ***")
elif NO_COW:
    M._detach_for_write = lambda state, sid: None
    print("\n*** CONTROL B: sharing ON, copy-on-write removed — §5 EXPECTED TO FAIL ***")


def check(label, condition, detail=""):
    RESULTS.append((label, bool(condition), detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


# ---------------------------------------------------------------- fixtures
TMP = tempfile.mkdtemp(prefix="membot_pool_")
RNG = np.random.default_rng(7)


def make_cart(name, n=40, dim=768):
    path = os.path.join(TMP, f"{name}.cart.npz")
    emb = RNG.standard_normal((n, dim)).astype(np.float32)
    texts = [f"{name} passage {i} about lattices and memory" for i in range(n)]
    save_as_npz(path, emb, texts)
    save_manifest(path, emb, n)
    return {"name": name, "path": path}


CART_A = make_cart("pool-test-alpha")
CART_B = make_cart("pool-test-beta")

M.find_cartridges = lambda: [CART_A, CART_B]
M.embed_text = lambda text, prefix="": RNG.standard_normal(768).astype(np.float32)
M._server_config["read_only"] = False

# Count real disk loads. The pool is only doing something if this stops growing.
_real_load = M.load_cartridge_safe
LOADS = {"n": 0}


def counting_load(path):
    LOADS["n"] += 1
    return _real_load(path)


M.load_cartridge_safe = counting_load


def refs(cart):
    """Holders of a path. Tolerant of absence so the --no-pool control reports clean
    FAILs instead of a KeyError — a crash proves something changed, not that each
    individual check discriminates."""
    return M._shared_refs.get(cart["path"], set())


def payload(cart):
    return M._shared_cart_data.get(cart["path"], {})


def reset():
    M._sessions.clear()
    M._shared_cart_data.clear()
    M._shared_refs.clear()
    LOADS["n"] = 0


# ---------------------------------------------------------------- §1 baseline
print("\n§1 · one session mounts — the pool is populated")
reset()
mount_cartridge("pool-test-alpha", session_id="tab1")
s1 = M._sessions["tab1"]
check("cart mounted", s1["cartridge_name"] == "pool-test-alpha")
check("payload loaded from disk exactly once", LOADS["n"] == 1, f"loads={LOADS['n']}")
check("pool holds 1 path", len(M._shared_cart_data) == 1)
check("session is a registered holder", refs(CART_A) == {"tab1"})

# ---------------------------------------------------------------- §2 LIVENESS
print("\n§2 · LIVENESS — 10 more tabs must add ZERO disk loads")
for i in range(2, 12):
    mount_cartridge("pool-test-alpha", session_id=f"tab{i}")
check("11 sessions, still 1 disk load", LOADS["n"] == 1, f"loads={LOADS['n']}")
check("11 holders registered", len(refs(CART_A)) == 11)
check("still 1 payload in the pool", len(M._shared_cart_data) == 1)

# ---------------------------------------------------------------- §3 IDENTITY
print("\n§3 · IDENTITY — equal arrays would prove nothing; these must be the SAME object")
s2 = M._sessions["tab2"]
check("embeddings are one object", s1["embeddings"] is s2["embeddings"])
check("texts are one object", s1["texts"] is s2["texts"])
check("binary_corpus is one object", s1["binary_corpus"] is s2["binary_corpus"])
check("sqlite_conn NOT shared (unmount closes it)",
      "sqlite_conn" not in payload(CART_A))

# ---------------------------------------------------------------- §4 isolation
print("\n§4 · a different cart is a different payload")
mount_cartridge("pool-test-beta", session_id="tab2")
check("second cart loaded from disk", LOADS["n"] == 2, f"loads={LOADS['n']}")
check("pool holds 2 paths", len(M._shared_cart_data) == 2)
check("tab2 released alpha on remount", "tab2" not in refs(CART_A))
check("alpha down to 10 holders", len(refs(CART_A)) == 10)
check("tab1 unaffected", M._sessions["tab1"]["cartridge_name"] == "pool-test-alpha")

# ---------------------------------------------------------------- §5 THE HAZARD
print("\n§5 · COPY-ON-WRITE — the in-place append that would corrupt every reader")
reset()
mount_cartridge("pool-test-alpha", session_id="writer")
mount_cartridge("pool-test-alpha", session_id="reader")
w, r = M._sessions["writer"], M._sessions["reader"]
n_before = len(r["texts"])
shared_before = w["texts"] is r["texts"]
out = memory_store("a new memory written by exactly one session", session_id="writer")
check("memory_store succeeded", "Stored" in out or "stored" in out.lower(), out[:60])
check("writer grew by 1", len(w["texts"]) == n_before + 1, f"{n_before}->{len(w['texts'])}")
check("READER DID NOT GROW", len(r["texts"]) == n_before, f"reader={len(r['texts'])}")
check("they were shared before the write", shared_before)
check("they are separate objects after", w["texts"] is not r["texts"])
check("writer detached from the pool", "writer" not in M._shared_refs.get(CART_A["path"], set()))
check("reader still holds the shared payload", r["texts"] is payload(CART_A).get("texts"))
check("writer's embeddings match its texts",
      len(w["embeddings"]) == len(w["texts"]), f"{len(w['embeddings'])} vs {len(w['texts'])}")
check("reader's embeddings match its texts",
      len(r["embeddings"]) == len(r["texts"]), f"{len(r['embeddings'])} vs {len(r['texts'])}")

# ---------------------------------------------------------------- §6 release
print("\n§6 · release — unmount, expiry and eviction must all free the payload")
reset()
mount_cartridge("pool-test-alpha", session_id="u1")
mount_cartridge("pool-test-alpha", session_id="u2")
unmount(session_id="u1")
check("unmount drops one holder", refs(CART_A) == {"u2"})
check("payload still resident for u2", CART_A["path"] in M._shared_cart_data)
unmount(session_id="u2")
check("last unmount evicts the payload", CART_A["path"] not in M._shared_cart_data)
check("ref set cleaned up", CART_A["path"] not in M._shared_refs)

reset()
mount_cartridge("pool-test-alpha", session_id="old")
M._sessions["old"]["last_access"] = time.time() - (M.SESSION_TIMEOUT_SEC + 60)
M._get_session("fresh")          # triggers the expiry sweep
check("expired session released its payload", CART_A["path"] not in M._shared_cart_data,
      f"pool={list(M._shared_cart_data)}")

# ---------------------------------------------------------------- §7 poisoning
print("\n§7 · a cart that FAILS integrity must not stay cached as pre-verified")
reset()
bad = make_cart("pool-test-bad", n=12)
M.find_cartridges = lambda: [CART_A, CART_B, bad]
_good_verify = M.verify_manifest
M.verify_manifest = lambda *a, **k: (False, "forced failure for test")
out1 = mount_cartridge("pool-test-bad", session_id="v1")
check("first mount refused", "SECURITY" in out1, out1[:50])
check("bad payload not left in the pool", bad["path"] not in M._shared_cart_data)
out2 = mount_cartridge("pool-test-bad", session_id="v2")
check("SECOND mount also refused (no cache-hit bypass)", "SECURITY" in out2, out2[:50])
M.verify_manifest = _good_verify

# ---------------------------------------------------------------- §8 budget
print("\n§8 · RAM BUDGET — the pool shares duplicates but must also bound DISTINCT carts")
reset()
M.find_cartridges = lambda: [CART_A, CART_B]
M._server_config["max_pool_bytes"] = None

est_a = M._cart_ram_estimate(CART_A["path"])
check("estimate is non-zero and read without loading", est_a > 0, f"{est_a/1024:.0f} KB")

# A budget that fits exactly one cart. Not a round number pulled from the air: it is
# derived from the measured estimate, so the test still discriminates if cart sizes
# change.
M._server_config["max_pool_bytes"] = int(est_a * 1.5)
mount_cartridge("pool-test-alpha", session_id="b1")
check("first cart admitted", M._sessions["b1"]["cartridge_name"] == "pool-test-alpha")
out = mount_cartridge("pool-test-beta", session_id="b2")
check("SECOND, DIFFERENT cart refused", "Not enough cartridge memory" in out, out[:52])
check("refusal names the resident cart", "alpha" in out)
check("refused cart did not enter the pool", len(M._shared_cart_data) == 1)
check("refused session has no cartridge", M._sessions["b2"]["cartridge_name"] is None)

# ⭐ The case the whole design turns on: a cart ALREADY resident must still admit,
# however tight the budget, because another holder costs nothing.
M._server_config["max_pool_bytes"] = 1
out2 = mount_cartridge("pool-test-alpha", session_id="b3")
check("resident cart admits even at a 1-byte budget",
      M._sessions["b3"]["cartridge_name"] == "pool-test-alpha", out2[:40])
check("still one payload for three holders", len(M._shared_cart_data) == 1)

# Control: with the budget lifted, the same second mount must SUCCEED. Without this,
# "refused" could mean the mount was broken rather than bounded.
M._server_config["max_pool_bytes"] = est_a * 50
out3 = mount_cartridge("pool-test-beta", session_id="b4")
check("CONTROL — same mount succeeds once the budget allows it",
      M._sessions["b4"]["cartridge_name"] == "pool-test-beta", out3[:52])
M._server_config["max_pool_bytes"] = None

# ---------------------------------------------------------------- verdict
passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print("\n" + "=" * 62)
if CONTROL:
    # Inverted: with the mechanism off, FAILURES are the proof the suite measures it.
    # And a control is only meaningful if it catches the section it targets, so each
    # arm names the checks it must break rather than settling for "something failed".
    caught = {label for label, ok, _ in RESULTS if not ok}
    if NO_POOL:
        arm, must_catch = "A (sharing off)", {
            "11 sessions, still 1 disk load",
            "embeddings are one object",
            "texts are one object",
        }
    else:
        arm, must_catch = "B (copy-on-write off)", {
            "READER DID NOT GROW",
            "they are separate objects after",
        }
    missed = must_catch - caught
    ok_control = not missed
    print(f"CONTROL {arm}: {total - passed}/{total} checks failed")
    if ok_control:
        print("VERDICT: CONTROL PASSED — the suite detects this mechanism's absence")
    else:
        # ASCII only: this is the FAILURE branch, so it has never run, and a
        # UnicodeEncodeError here would replace the diagnosis with a stack trace at
        # exactly the moment the diagnosis mattered.
        print("VERDICT: CONTROL FAILED - these still PASSED with the mechanism "
              "removed, so they are not measuring it:")
        for label in sorted(missed):
            print(f"   not discriminating: {label}")
    for label in sorted(caught):
        print(f"   caught: {label}")
    print("=" * 62)
    sys.exit(0 if ok_control else 1)

print(f"VERDICT: {passed}/{total} checks passed" + ("" if passed == total else "  <<< FAILURES"))
for label, ok, detail in RESULTS:
    if not ok:
        print(f"   FAILED: {label}   {detail}")
print("=" * 62)
sys.exit(0 if passed == total else 1)
