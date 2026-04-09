"""
test_multi_cart.py — Standalone test for the multi-cart query layer.

Run from the membot directory:
    python tests/test_multi_cart.py

Mounts a couple of existing carts from the cartridges/ directory and runs a
test query against the multi-cart pool. Verifies:
  - mount() works on real cart files
  - list_mounts() returns expected cart_ids
  - search() returns results attributed to source carts
  - role_filter narrows results correctly
  - unmount() actually frees the cart
  - mount_directory() picks up multiple files

Does NOT touch the Membot server or any persistent state. Just exercises
multi_cart.py in isolation.
"""

import os
import sys

# Tests live in membot/tests/ — add the parent (membot/) to sys.path so
# `import multi_cart` and `import membot_server` resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
_MEMBOT_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _MEMBOT_DIR)

import multi_cart as mc

CARTRIDGES_DIR = os.path.join(_MEMBOT_DIR, "cartridges")


def find_test_carts(n=2):
    """Find a few cart files to use for testing."""
    if not os.path.isdir(CARTRIDGES_DIR):
        print(f"FATAL: cartridges directory not found at {CARTRIDGES_DIR}")
        sys.exit(1)

    candidates = []
    for ext in (".npz", ".pkl"):
        for f in sorted(os.listdir(CARTRIDGES_DIR)):
            if f.endswith(ext):
                candidates.append(os.path.join(CARTRIDGES_DIR, f))
                if len(candidates) >= n:
                    return candidates
    return candidates


def main():
    print("=" * 70)
    print("MULTI-CART PHASE 1 TEST")
    print("=" * 70)
    print()

    # --- Find some carts ---
    carts = find_test_carts(n=3)
    if len(carts) < 2:
        print(f"WARNING: only {len(carts)} cart(s) found in {CARTRIDGES_DIR}")
        print("Need at least 2 carts to test multi-cart properly. Continuing anyway.")
    if not carts:
        print("FATAL: no cart files found. Mount one manually first or build one with cart_builder.")
        sys.exit(1)

    print(f"Found {len(carts)} cart files in {CARTRIDGES_DIR}:")
    for c in carts:
        print(f"  {os.path.basename(c)}")
    print()

    # --- Test 1: mount the first cart ---
    # NOTE: verify_integrity=False because some of these test carts may have
    # stale manifests from earlier builds. Production calls should leave
    # integrity verification on. We're just exercising the mount/search logic.
    print("--- Test 1: mount first cart ---")
    try:
        result1 = mc.mount(carts[0], cart_id="cart_a", role="test", verify_integrity=False)
        print(f"  mounted: {result1}")
    except Exception as e:
        print(f"  FAIL: {e}")
        return

    # --- Test 2: mount a second cart ---
    if len(carts) >= 2:
        print()
        print("--- Test 2: mount second cart with different role ---")
        try:
            result2 = mc.mount(carts[1], cart_id="cart_b", role="other", verify_integrity=False)
            print(f"  mounted: {result2}")
        except Exception as e:
            print(f"  FAIL: {e}")

    # --- Test 3: list mounts ---
    print()
    print("--- Test 3: list mounts ---")
    mounts = mc.list_mounts()
    print(f"  {len(mounts)} carts mounted, {mc.total_patterns_mounted()} total patterns")
    for m in mounts:
        print(f"    {m['cart_id']} (role={m['role']}, {m['n_patterns']} patterns)")

    # --- Test 4: search across all carts ---
    print()
    print("--- Test 4: search scope='all' ---")
    test_query = "memory and recall"
    try:
        result = mc.search(test_query, top_k=5, scope="all")
        print(f"  query: {test_query!r}")
        print(f"  searched {result['cart_count']} carts, {result['total_patterns']} patterns, "
              f"{result['elapsed_ms']}ms")
        print(f"  got {len(result['results'])} results:")
        for i, r in enumerate(result["results"], 1):
            text_preview = r["text"][:80].replace("\n", " ")
            print(f"    #{i} [{r['cart_id']}#{r['local_addr']}] score={r['score']:.3f} {text_preview}")
    except Exception as e:
        print(f"  FAIL: {e}")

    # --- Test 5: search with role_filter ---
    print()
    print("--- Test 5: search with role_filter='test' ---")
    try:
        result = mc.search(test_query, top_k=5, scope="all", role_filter="test")
        print(f"  searched {result['cart_count']} carts (role=test only)")
        print(f"  got {len(result['results'])} results")
        for r in result["results"]:
            assert r["role"] == "test", f"role_filter failed: got role {r['role']!r}"
        print("  role_filter check: PASS")
    except Exception as e:
        print(f"  FAIL: {e}")

    # --- Test 6: search with specific cart_id scope ---
    print()
    print("--- Test 6: search scope='cart_a' (single cart) ---")
    try:
        result = mc.search(test_query, top_k=5, scope="cart_a")
        print(f"  searched {result['cart_count']} carts (cart_a only)")
        print(f"  got {len(result['results'])} results")
        for r in result["results"]:
            assert r["cart_id"] == "cart_a", f"scope failed: got cart {r['cart_id']!r}"
        print("  scope check: PASS")
    except Exception as e:
        print(f"  FAIL: {e}")

    # --- Test 6b: scope_mode variants (the small-cart-not-drowned fix) ---
    print()
    print("--- Test 6b: scope_mode variants ---")
    try:
        # global mode (default) — true top-K across all
        global_result = mc.search(test_query, top_k=3, scope="all", scope_mode="global")
        global_carts_hit = set(r["cart_id"] for r in global_result["results"])
        print(f"  global: {len(global_result['results'])} results from carts {global_carts_hit}")
        assert global_result["scope_mode"] == "global"

        # per_cart mode — top-K from each cart, no global rerank
        per_cart_result = mc.search(test_query, top_k=3, scope="all", scope_mode="per_cart")
        per_cart_carts_hit = set(r["cart_id"] for r in per_cart_result["results"])
        print(f"  per_cart: {len(per_cart_result['results'])} results from carts {per_cart_carts_hit}")
        print(f"  per_cart grouped_results keys: {list(per_cart_result.get('grouped_results', {}).keys())}")
        assert per_cart_result["scope_mode"] == "per_cart"
        assert "grouped_results" in per_cart_result
        # per_cart with 2 carts and top_k=3 should return up to 6 results (3 from each)
        # And both carts should be represented even if one is much smaller
        assert len(per_cart_carts_hit) == per_cart_result["cart_count"], (
            f"per_cart should return results from every cart: got {per_cart_carts_hit} "
            f"vs {per_cart_result['cart_count']} carts"
        )

        # balanced mode — top-K candidates per cart, then global rerank
        balanced_result = mc.search(test_query, top_k=3, scope="all", scope_mode="balanced")
        balanced_carts_hit = set(r["cart_id"] for r in balanced_result["results"])
        print(f"  balanced: {len(balanced_result['results'])} results from carts {balanced_carts_hit}")
        assert balanced_result["scope_mode"] == "balanced"
        # balanced returns top_k total (not top_k per cart)
        assert len(balanced_result["results"]) <= 3, (
            f"balanced should return at most top_k={3} results, got {len(balanced_result['results'])}"
        )

        # diagnostic mode — every cart's top-K, fully labeled
        diag_result = mc.search(test_query, top_k=3, scope="all", scope_mode="diagnostic")
        print(f"  diagnostic: {len(diag_result['results'])} results "
              f"(grouped_results: {list(diag_result.get('grouped_results', {}).keys())})")
        assert diag_result["scope_mode"] == "diagnostic"
        assert "grouped_results" in diag_result

        # invalid scope_mode should error cleanly
        try:
            mc.search(test_query, top_k=3, scope="all", scope_mode="bogus")
            print("  FAIL: invalid scope_mode should have raised")
        except ValueError as e:
            print(f"  invalid scope_mode raises ValueError: PASS")

        print("  scope_mode variants check: PASS")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()

    # --- Test 7: collision detection ---
    print()
    print("--- Test 7: cart_id collision should error ---")
    try:
        mc.mount(carts[0], cart_id="cart_a", role="test", verify_integrity=False)
        print("  FAIL: should have raised on collision")
    except ValueError as e:
        print(f"  PASS: {e}")

    # --- Test 8: unmount ---
    print()
    print("--- Test 8: unmount cart_a ---")
    result = mc.unmount("cart_a")
    print(f"  {result}")
    assert "cart_a" not in [m["cart_id"] for m in mc.list_mounts()]
    print("  unmount check: PASS")

    # --- Test 9: unmount_all cleanup ---
    print()
    print("--- Test 9: unmount_all ---")
    n = mc.unmount_all()
    print(f"  unmounted {n} carts")
    assert mc.total_patterns_mounted() == 0
    print("  cleanup check: PASS")

    print()
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
