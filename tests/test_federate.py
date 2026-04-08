"""
test_federate.py — Test the federated cart layer against Dennis's real data.

Run from the membot directory:
    python tests/test_federate.py

Uses Dennis's actual sb26_learning.jsonl from shared-context/arc-agi-3 to:
  1. Migrate JSONL → brain cart for cbp machine
  2. Mount the cart and verify it's queryable
  3. Run a search against it
  4. Build a fake "sprout" machine cart from the same data (simulated)
  5. Run consolidate() to find cross-machine matches
  6. Verify the consolidated cart has expected structure

Does NOT touch any production data — writes to a temp directory.

Note: This test triggers a real Nomic embedding load, which is slow on first
run (~270 MB download if not cached). Subsequent runs are fast.
"""

import json
import os
import shutil
import sys
import tempfile
import time

# Tests live in membot/tests/ — add the parent (membot/) to sys.path so
# `import federate` and `import multi_cart` resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
_MEMBOT_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _MEMBOT_DIR)

import federate
import multi_cart as mc

# project-you root is two levels up from membot/tests/
PROJECT_ROOT = os.path.abspath(os.path.join(_MEMBOT_DIR, ".."))
REAL_JSONL = os.path.join(
    PROJECT_ROOT, "shared-context", "arc-agi-3", "fleet-learning", "cbp", "sb26_learning.jsonl"
)


def main():
    print("=" * 70)
    print("FEDERATE PHASE 1 TEST")
    print("=" * 70)
    print()

    if not os.path.exists(REAL_JSONL):
        print(f"FATAL: real JSONL not found at {REAL_JSONL}")
        print("Make sure shared-context is cloned at the project root.")
        sys.exit(1)

    print(f"Using real data from: {REAL_JSONL}")
    with open(REAL_JSONL, "r", encoding="utf-8") as f:
        line_count = sum(1 for line in f if line.strip())
    print(f"  ({line_count} JSONL lines)")
    print()

    # Use a temp directory so we don't pollute the real shared-context
    tmpdir = tempfile.mkdtemp(prefix="federate_test_")
    print(f"Working in temp dir: {tmpdir}")
    print()

    try:
        # --- Setup: copy the real JSONL into a fake fleet structure ---
        fleet_dir = os.path.join(tmpdir, "fleet-learning")
        cbp_dir = os.path.join(fleet_dir, "cbp")
        sprout_dir = os.path.join(fleet_dir, "sprout")
        os.makedirs(cbp_dir)
        os.makedirs(sprout_dir)
        shutil.copy(REAL_JSONL, os.path.join(cbp_dir, "sb26_learning.jsonl"))
        # Make a "sprout" copy with slightly different content so consolidate
        # has something to match across machines
        with open(REAL_JSONL, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        # Sprout sees the same patterns but writes them as "sprout"
        sprout_lines = []
        for entry in lines:
            sprout_entry = dict(entry)
            sprout_entry["machine"] = "sprout"
            sprout_entry["timestamp"] = entry.get("timestamp", "").replace("T10:", "T11:")
            sprout_lines.append(sprout_entry)
        with open(os.path.join(sprout_dir, "sb26_learning.jsonl"), "w") as f:
            for entry in sprout_lines:
                f.write(json.dumps(entry) + "\n")

        print("--- Setup: fleet structure created ---")
        print(f"  cbp/sb26_learning.jsonl ({len(lines)} entries)")
        print(f"  sprout/sb26_learning.jsonl ({len(sprout_lines)} entries, simulated)")
        print()

        # --- Test 1: migrate JSONL → carts ---
        print("--- Test 1: migrate_jsonl ---")
        t0 = time.time()
        result = federate.migrate_jsonl(fleet_dir, in_place=True)
        elapsed = time.time() - t0
        print(f"  carts_built: {result['carts_built']}")
        print(f"  total_entries: {result['total_entries']}")
        print(f"  errors: {len(result['errors'])}")
        print(f"  elapsed: {elapsed:.1f}s")
        for err in result["errors"]:
            print(f"    ! {err}")

        if result["carts_built"] == 0:
            print("  FAIL: no carts built")
            return

        cbp_cart = os.path.join(cbp_dir, "kb.cart.npz")
        sprout_cart = os.path.join(sprout_dir, "kb.cart.npz")
        assert os.path.exists(cbp_cart), f"cbp cart not at {cbp_cart}"
        assert os.path.exists(sprout_cart), f"sprout cart not at {sprout_cart}"
        print(f"  cbp cart: {os.path.getsize(cbp_cart)} bytes")
        print(f"  sprout cart: {os.path.getsize(sprout_cart)} bytes")
        print()

        # --- Test 2: load_fleet (mount all machine carts at once) ---
        print("--- Test 2: load_fleet ---")
        mc.unmount_all()
        load_result = federate.load_fleet(fleet_dir)
        print(f"  mounted: {len(load_result['mounted'])} machines")
        print(f"  total_patterns: {load_result['total_patterns']}")
        print(f"  machines: {load_result['machines']}")
        print(f"  errors: {len(load_result['errors'])}")
        for err in load_result["errors"]:
            print(f"    ! {err}")
        assert len(load_result["mounted"]) == 2, f"expected 2 machines mounted, got {len(load_result['mounted'])}"
        print()

        # --- Test 3: cross-fleet search ---
        print("--- Test 3: cross-fleet search ---")
        query = "border color identifies parent slot"
        search_result = mc.search(query, top_k=5, role_filter="federated")
        print(f"  query: {query!r}")
        print(f"  searched {search_result['cart_count']} carts, "
              f"{search_result['total_patterns']} patterns, "
              f"{search_result['elapsed_ms']}ms")
        print(f"  got {len(search_result['results'])} results:")
        for i, r in enumerate(search_result["results"], 1):
            text_preview = r["text"][:100].replace("\n", " ")
            print(f"    #{i} [{r['cart_id']}#{r['local_addr']}] score={r['score']:.3f}")
            print(f"        {text_preview}")
        assert len(search_result["results"]) > 0, "search returned no results"
        # Should hit both cbp and sprout because they have same content
        cart_ids_hit = set(r["cart_id"] for r in search_result["results"])
        print(f"  cart_ids hit: {cart_ids_hit}")
        assert "cbp" in cart_ids_hit or "sprout" in cart_ids_hit, "neither machine appeared in results"
        print("  cross-fleet search: PASS")
        print()

        # --- Test 4: consolidate ---
        print("--- Test 4: consolidate ---")
        # Unmount first because consolidate will mount everything fresh
        mc.unmount_all()
        consolidated_dir = os.path.join(tmpdir, "consolidated")
        cons_result = federate.consolidate(
            fleet_dir,
            output_dir=consolidated_dir,
            similarity_threshold=0.85,
        )
        print(f"  n_machines: {cons_result['n_machines']}")
        print(f"  total_input_patterns: {cons_result['total_input_patterns']}")
        print(f"  n_consolidated_patterns: {cons_result['n_consolidated_patterns']}")
        print(f"  n_confirmed_pairs: {cons_result['n_confirmed_pairs']}")
        print(f"  n_contradicted_pairs: {cons_result['n_contradicted_pairs']}")
        print(f"  elapsed: {cons_result['elapsed_seconds']}s")
        print(f"  output: {cons_result.get('output_path')}")
        assert cons_result["n_machines"] == 2
        assert cons_result["total_input_patterns"] >= line_count * 2  # both machines
        # Since cbp and sprout have identical text content, almost everything
        # should be confirmed across machines
        assert cons_result["n_confirmed_pairs"] > 0, "expected some cross-machine confirmations"
        print()

        # --- Test 5: load consolidated cart and search ---
        print("--- Test 5: search consolidated cart ---")
        consolidated_path = cons_result.get("output_path")
        if consolidated_path and os.path.exists(consolidated_path):
            mc.unmount_all()
            mount_result = mc.mount(consolidated_path, cart_id="consolidated", role="consolidated")
            print(f"  mounted consolidated: {mount_result['n_patterns']} patterns")
            search_result = mc.search("hierarchy parent identity", top_k=3, scope="consolidated")
            print(f"  search returned {len(search_result['results'])} results")
            for r in search_result["results"]:
                text_preview = r["text"][:100].replace("\n", " ")
                print(f"    [{r['cart_id']}#{r['local_addr']}] score={r['score']:.3f} {text_preview}")
        else:
            print("  SKIP: no consolidated cart written")
        print()

        # --- Test 6: publish_session adds new entries to existing cart ---
        print("--- Test 6: publish_session appends to existing cart ---")
        mc.unmount_all()
        new_entries = [
            {
                "timestamp": "2026-04-08T01:00:00",
                "machine": "cbp",
                "player": "claude",
                "game": "test_game",
                "level": 0,
                "event": "level_solved",
                "actions": 5,
                "baseline": 10,
                "structural_pattern": "test_pattern",
                "rule": "this is a test rule",
                "meta": "added by test_federate.py",
            }
        ]
        new_session_file = os.path.join(tmpdir, "new_session.jsonl")
        with open(new_session_file, "w") as f:
            for e in new_entries:
                f.write(json.dumps(e) + "\n")

        before = federate._load_existing_cart(cbp_cart)
        before_count = len(before[0])
        result = federate.publish_session(new_session_file, "cbp", fleet_dir)
        print(f"  before: {before_count} patterns")
        print(f"  after: {result['total_in_cart']} patterns")
        print(f"  added: {result['added']}, skipped: {result['skipped']}")
        assert result["added"] == 1, f"expected 1 added, got {result['added']}"
        assert result["total_in_cart"] == before_count + 1
        print("  publish_session: PASS")
        print()

        # --- Test 7: re-publishing the same entries should dedup ---
        print("--- Test 7: re-publish same entries should dedup ---")
        result = federate.publish_session(new_session_file, "cbp", fleet_dir)
        print(f"  added: {result['added']}, skipped: {result['skipped']}")
        assert result["added"] == 0, f"expected 0 added on re-publish, got {result['added']}"
        assert result["skipped"] == 1, f"expected 1 skipped, got {result['skipped']}"
        print("  dedup: PASS")
        print()

        print("=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)

    finally:
        print()
        print(f"Cleaning up {tmpdir}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        mc.unmount_all()


if __name__ == "__main__":
    main()
