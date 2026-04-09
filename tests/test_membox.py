"""
test_membox.py — Standalone test for Membox Phase 1 (locking + tagging).

Run from the membot directory:
    python tests/test_membox.py

Exercises every Phase 1 capability:
  1. Mount a cart in Membox mode
  2. Single-agent imprint and read
  3. Two-agent concurrent imprint with lock serialization
  4. Lock contention — agent B blocks until agent A releases
  5. Lock timeout — acquire returns False after the timeout window
  6. Lock holder query — correct agent_id mid-write, None when free
  7. Read never blocks — search succeeds while another agent holds the lock
  8. Lock auto-release on lease expiration (crash recovery)
  9. agent_id stamping — every pattern has agent_id in per_pattern_meta
  10. Invalid lock release — wrong agent can't release another agent's lock
  11. Multiple patterns from multiple agents end up correctly attributed

Uses a temp copy of an existing cart so we don't pollute the real one.
Cleans up on exit.
"""

import os
import shutil
import sys
import tempfile
import threading
import time

# Tests live in membot/tests/ — add the parent (membot/) to sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_MEMBOT_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _MEMBOT_DIR)

import membox as mb
import multi_cart as mc

CARTRIDGES_DIR = os.path.join(_MEMBOT_DIR, "cartridges")


def find_test_cart():
    """Find a small cart we can copy and write to in the test."""
    candidates = [
        "attention-is-all-you-need.cart.npz",
        "membot_devlog.cart.npz",
        "tui-research.cart.npz",
    ]
    for name in candidates:
        path = os.path.join(CARTRIDGES_DIR, name)
        if os.path.exists(path):
            return path
    return None


def main():
    print("=" * 70)
    print("MEMBOX PHASE 1 TEST")
    print("=" * 70)
    print()

    src_cart = find_test_cart()
    if not src_cart:
        print(f"FATAL: no test cart found in {CARTRIDGES_DIR}")
        sys.exit(1)

    print(f"Source cart: {src_cart}")
    print()

    # Copy to a temp dir so we don't mutate the real cart
    tmpdir = tempfile.mkdtemp(prefix="membox_test_")
    test_cart = os.path.join(tmpdir, "test_membox.cart.npz")
    shutil.copy(src_cart, test_cart)
    # Also copy the manifest if it exists alongside
    src_manifest = src_cart.rsplit(".", 1)[0] + "_manifest.json"
    if os.path.exists(src_manifest):
        shutil.copy(src_manifest, os.path.join(tmpdir, "test_membox_manifest.json"))
    print(f"Test cart copied to: {test_cart}")
    print()

    try:
        mb.unmount_all()
        mc.unmount_all()

        # =========================================================
        # Test 1: mount in Membox mode
        # =========================================================
        print("--- Test 1: mount in Membox mode ---")
        result = mb.mount(test_cart, cart_id="test", role="working",
                          verify_integrity=False)
        print(f"  mounted: cart_id={result['cart_id']}, "
              f"n_patterns={result['n_patterns']}, "
              f"membox={result['membox']}, "
              f"lease={result['lease_seconds']}s")
        assert result["membox"] is True
        assert result["cart_id"] == "test"
        baseline_n = result["n_patterns"]
        print(f"  PASS (baseline: {baseline_n} patterns)")
        print()

        # =========================================================
        # Test 2: list_mounts shows the Membox cart
        # =========================================================
        print("--- Test 2: list_mounts ---")
        mounts = mb.list_mounts()
        print(f"  found {len(mounts)} Membox carts")
        assert len(mounts) == 1
        assert mounts[0]["cart_id"] == "test"
        assert mounts[0]["lock"]["holder"] is None
        print(f"  lock holder when idle: {mounts[0]['lock']['holder']}")
        print("  PASS")
        print()

        # =========================================================
        # Test 3: single-agent imprint
        # =========================================================
        print("--- Test 3: single-agent imprint ---")
        r = mb.imprint(
            "test",
            text="Membox Phase 1 test pattern from alice",
            agent_id="alice",
            tags="test,membox-phase1",
            reasoning="Verifying single-agent write works",
        )
        print(f"  imprint result: ok={r['ok']}, local_addr={r.get('local_addr')}, "
              f"agent_id={r.get('agent_id')}")
        assert r["ok"] is True
        assert r["agent_id"] == "alice"
        assert r["local_addr"] == baseline_n  # appended at the end
        print(f"  PASS (appended at index {r['local_addr']})")
        print()

        # =========================================================
        # Test 4: lock holder query mid-write
        # =========================================================
        print("--- Test 4: lock holder query when idle and held ---")
        idle_holder = mb.lock_holder("test")
        assert idle_holder is None
        print(f"  idle holder: {idle_holder}")

        # Acquire manually so we can query mid-hold
        ok = mb.acquire_lock("test", agent_id="alice", timeout_ms=1000)
        assert ok is True
        held_holder = mb.lock_holder("test")
        assert held_holder == "alice"
        print(f"  held holder: {held_holder}")
        mb.release_lock("test", agent_id="alice")
        print("  released")
        post_holder = mb.lock_holder("test")
        assert post_holder is None
        print(f"  post-release holder: {post_holder}")
        print("  PASS")
        print()

        # =========================================================
        # Test 5: lock timeout — agent B can't get lock if A holds
        # =========================================================
        print("--- Test 5: lock timeout when contended ---")
        ok = mb.acquire_lock("test", agent_id="alice", timeout_ms=1000)
        assert ok is True
        t0 = time.time()
        ok_b = mb.acquire_lock("test", agent_id="bob", timeout_ms=200)
        elapsed = time.time() - t0
        print(f"  bob's acquire returned {ok_b} after {elapsed*1000:.0f}ms")
        assert ok_b is False
        assert 0.18 < elapsed < 0.50  # ~200ms with some slack
        mb.release_lock("test", agent_id="alice")
        print("  PASS")
        print()

        # =========================================================
        # Test 6: invalid release (wrong agent) raises PermissionError
        # =========================================================
        print("--- Test 6: invalid release raises PermissionError ---")
        ok = mb.acquire_lock("test", agent_id="alice", timeout_ms=1000)
        assert ok is True
        try:
            mb.release_lock("test", agent_id="bob")
            print("  FAIL: should have raised PermissionError")
            return
        except PermissionError as e:
            print(f"  raised PermissionError: {e}")
        mb.release_lock("test", agent_id="alice")
        print("  PASS")
        print()

        # =========================================================
        # Test 7: read never blocks while another agent holds the lock
        # =========================================================
        print("--- Test 7: read never blocks ---")
        ok = mb.acquire_lock("test", agent_id="alice", timeout_ms=1000)
        assert ok is True
        t0 = time.time()
        result = mb.search("test", "Membox phase 1 test pattern", top_k=3)
        elapsed_ms = (time.time() - t0) * 1000
        n_results = len(result.get("results", []))
        print(f"  search returned {n_results} results in {elapsed_ms:.0f}ms while alice holds the lock")
        # Should return quickly, not block on the lock
        assert elapsed_ms < 5000  # generous; should be much faster
        assert n_results > 0  # should find the alice imprint from test 3
        mb.release_lock("test", agent_id="alice")
        print("  PASS")
        print()

        # =========================================================
        # Test 8: two-agent concurrent imprint serializes correctly
        # =========================================================
        print("--- Test 8: two-agent concurrent imprint ---")
        results: dict = {"alice": None, "bob": None}
        timings: dict = {"alice": None, "bob": None}

        def worker(agent: str, text: str):
            t0 = time.time()
            r = mb.imprint("test", text=text, agent_id=agent,
                           reasoning=f"concurrent test from {agent}")
            results[agent] = r
            timings[agent] = time.time() - t0

        t_alice = threading.Thread(
            target=worker, args=("alice", "Concurrent write from Alice")
        )
        t_bob = threading.Thread(
            target=worker, args=("bob", "Concurrent write from Bob")
        )

        # Start both at the same time
        t_alice.start()
        t_bob.start()
        t_alice.join(timeout=10)
        t_bob.join(timeout=10)

        print(f"  alice timing: {timings['alice']*1000:.0f}ms, ok={results['alice']['ok']}")
        print(f"  bob timing:   {timings['bob']*1000:.0f}ms, ok={results['bob']['ok']}")
        assert results["alice"]["ok"] is True
        assert results["bob"]["ok"] is True
        # Both writes should have unique local_addrs (proves they didn't overwrite)
        assert results["alice"]["local_addr"] != results["bob"]["local_addr"]
        print(f"  alice local_addr={results['alice']['local_addr']}, bob local_addr={results['bob']['local_addr']}")
        print("  PASS (both writes attributed and serialized)")
        print()

        # =========================================================
        # Test 9: agent_id stamping survives in per_pattern_meta
        # =========================================================
        print("--- Test 9: agent_id stamping ---")
        # Search for both writes and verify their per_pattern_meta contains the right agent_id
        result = mb.search("test", "concurrent write", top_k=10)
        found_alice = False
        found_bob = False
        for r in result.get("results", []):
            membox_meta = r.get("membox_meta", {})
            agent = membox_meta.get("agent_id") if isinstance(membox_meta, dict) else None
            text_preview = r.get("text", "")[:80]
            if agent == "alice" and "Alice" in text_preview:
                found_alice = True
            if agent == "bob" and "Bob" in text_preview:
                found_bob = True
            if agent in ("alice", "bob"):
                print(f"    [{r['cart_id']}#{r['local_addr']}] agent={agent} {text_preview!r}")
        assert found_alice, "alice's pattern should be attributed to alice"
        assert found_bob, "bob's pattern should be attributed to bob"
        print("  PASS (both writes correctly attributed)")
        print()

        # =========================================================
        # Test 10: status reports stats correctly
        # =========================================================
        print("--- Test 10: status reports ---")
        s = mb.status("test")
        print(f"  n_patterns: {s['n_patterns']}")
        print(f"  writes_by_agent: {s['writes_by_agent']}")
        print(f"  recent_writes count: {len(s['recent_writes'])}")
        assert s["writes_by_agent"].get("alice", 0) >= 2  # test 3 + test 8
        assert s["writes_by_agent"].get("bob", 0) >= 1   # test 8
        print("  PASS")
        print()

        # =========================================================
        # Test 11: lease-based crash recovery
        # =========================================================
        print("--- Test 11: lease-based crash recovery ---")
        # Re-mount with a very short lease so the test runs fast
        mb.unmount_all()
        mc.unmount_all()
        result = mb.mount(test_cart, cart_id="test", role="working",
                          verify_integrity=False, lease_seconds=1)
        # Acquire and "crash" (just don't release)
        ok = mb.acquire_lock("test", agent_id="alice", timeout_ms=500)
        assert ok is True
        print(f"  alice acquired (lease=1s)")
        # Wait for lease to expire
        time.sleep(1.2)
        # Bob should now be able to acquire — the expired lease auto-releases
        t0 = time.time()
        ok_b = mb.acquire_lock("test", agent_id="bob", timeout_ms=500)
        elapsed = time.time() - t0
        print(f"  bob acquired={ok_b} after lease expiry, in {elapsed*1000:.0f}ms")
        assert ok_b is True
        mb.release_lock("test", agent_id="bob")
        print("  PASS (crashed lock auto-released)")
        print()

        # =========================================================
        # Cleanup
        # =========================================================
        print("--- Cleanup: unmount_all ---")
        n = mb.unmount_all()
        print(f"  unmounted {n} Membox carts")
        print()

        print("=" * 70)
        print("ALL MEMBOX PHASE 1 TESTS PASSED")
        print("=" * 70)

    finally:
        # Always clean up the temp dir
        try:
            mb.unmount_all()
            mc.unmount_all()
        except Exception:
            pass
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
            print(f"Cleaned up {tmpdir}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
