#!/usr/bin/env python
"""Smoke test for walk_associate MCP tool.

Mounts a cart and runs walk_associate with a query set appropriate to the
cart's content. Verifies the tool returns BOTH primary matches AND "you may
have missed" results, demonstrating the walk-hop algorithm is firing correctly.

Usage:
    python tools/smoke_walk_associate.py
    python tools/smoke_walk_associate.py --cart wiki_nomic_100k
    python tools/smoke_walk_associate.py --cart gutenberg-poetry --query "death"
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Force utf-8 on stdout/stderr so non-ASCII characters in wiki content
# (e.g. "Yōko Ōno", "Tōkyō") don't trip Windows cp1252 default encoding.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Make membot_server importable from the membot dir
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import membot_server  # noqa: E402


def _call(tool):
    """FastMCP 2.x stores the original function on .fn; 3.x stores on .__wrapped__."""
    return getattr(tool, 'fn', getattr(tool, '__wrapped__', tool))


# Query sets tuned per cart-content-type. Each entry: (query, top_k, min_hits, temp, label).
QUERY_SETS = {
    "gutenberg-poetry": [
        ("solitude", 5, 1, 0.0, "tight-cluster query, min_hits=1 (sanity check walk is firing)"),
        ("love and loss", 10, 2, 0.0, "broad theme, default min_hits=2"),
        ("death", 10, 2, 0.0, "broad theme, different vocabulary"),
        ("love and loss", 10, 2, 0.4, "broad theme, temperature=0.4 (basin escape)"),
    ],
    "wiki_nomic_100k": [
        # Entity-rich queries -- walk should surface related entities the user
        # didn't directly query for. These are the John/Yoko -> Cynthia Lennon
        # and Caesar/Brutus -> Cassius demo queries from the 6/1-6/2 ship.
        ("Caesar and Brutus", 10, 2, 0.0, "Roman history entity walk; expect Cassius, Senate, etc."),
        ("John Lennon and Yoko Ono", 10, 2, 0.0, "Beatles personal history; expect Cynthia Lennon"),
        ("Beatles breakup", 10, 2, 0.0, "Pop history; expect related band/member articles"),
        ("Caesar and Brutus", 10, 2, 0.4, "Same Caesar query, temperature=0.4 -- adjacent-empire surfacing"),
    ],
    # Default fallback for unknown carts -- generic queries
    "_default": [
        ("knowledge", 10, 2, 0.0, "generic semantic walk"),
        ("history", 10, 2, 0.0, "generic semantic walk"),
        ("knowledge", 10, 2, 0.4, "generic semantic walk with temperature"),
    ],
}


def main():
    parser = argparse.ArgumentParser(description="walk_associate smoke test")
    parser.add_argument("--cart", default="gutenberg-poetry",
                        help="Cart name to mount (default: gutenberg-poetry)")
    parser.add_argument("--query", default=None,
                        help="Single query override (skips the per-cart query set)")
    args = parser.parse_args()

    print("=" * 80)
    print(f"walk_associate smoke test -- cart={args.cart}")
    print("=" * 80)

    # 1. List available carts
    list_cartridges = _call(membot_server.list_cartridges)
    print("\n[1] list_cartridges() -- (first 700 chars)")
    print(list_cartridges()[:700])

    # 2. Mount the target cart
    mount_cartridge = _call(membot_server.mount_cartridge)
    cart_name = args.cart
    print(f"\n[2] mount_cartridge('{cart_name}')")
    mount_result = mount_cartridge(cart_name)
    print(mount_result[:600])

    # 3. Run walk_associate with the cart-appropriate query set
    walk_associate = _call(membot_server.walk_associate)

    if args.query:
        # Single ad-hoc query override
        test_queries = [(args.query, 10, 2, 0.0, "user-supplied query"),
                        (args.query, 10, 2, 0.4, "user-supplied query, temperature=0.4")]
    else:
        test_queries = QUERY_SETS.get(args.cart, QUERY_SETS["_default"])

    for query, k, min_hits, temperature, label in test_queries:
        print(f"\n[3] walk_associate(query='{query}', top_k={k}, walk_min_hits={min_hits}, temperature={temperature})")
        print(f"    case: {label}")
        print("-" * 80)
        result = walk_associate(
            query=query,
            top_k=k,
            walk_min_hits=min_hits,
            walk_max_show=5,
            temperature=temperature,
        )
        # Print just the header + missed section (skip the long primary content to keep output readable)
        lines = result.split("\n")
        header = lines[0] if lines else ""
        # Find "You may have missed" section
        missed_idx = next((i for i, l in enumerate(lines) if "may have missed" in l.lower()), -1)
        print(header)
        if missed_idx > 0:
            print("\n".join(lines[missed_idx:missed_idx+30]))
        else:
            print("(no missed section in output)")
        print()

    # 4. Compare against memory_search baseline using the first query
    memory_search = _call(membot_server.memory_search)
    baseline_query = test_queries[0][0]
    print(f"\n[4] memory_search('{baseline_query}', top_k=5) for comparison")
    print("-" * 80)
    print(memory_search(query=baseline_query, top_k=5))

    print("\n" + "=" * 80)
    print("Smoke test complete. Walk should have surfaced 'may have missed' results")
    print("that don't appear in memory_search's top-5 -- those are walk's value-add.")
    print("=" * 80)


if __name__ == "__main__":
    main()
