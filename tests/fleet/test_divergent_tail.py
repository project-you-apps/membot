"""
Divergent Tail Test (CBP)
==========================
Tests Andy's RFC hypothesis: do sign-zero Hamming and cosine disagree
on results that are *meaningful* divergences, not errors?

When the 70/30 blend surfaces a result that cosine alone would rank
differently, is that result (a) garbage, (b) associatively related from
a different angle, or (c) something else?

Uses verbose=True to get separate cos= and ham= scores per result,
then identifies entries where cosine and Hamming rankings diverge.

Requires membot REST bridge running on port 8001 with a mounted cartridge
containing our test content from test_semantic_reach.py.

Usage:
    python3 tests/fleet/test_divergent_tail.py

Machine: CBP (WSL2, RTX 2060 SUPER)
RFC: https://github.com/project-you-apps/membot/issues/4#issuecomment-4150684819
"""

import unittest
import json
import os
import re
import time
import urllib.request
import urllib.error

MEMBOT_REST_URL = os.environ.get("MEMBOT_REST_URL", "http://localhost:8001")


def api(method, path, data=None, timeout=30):
    url = f"{MEMBOT_REST_URL}{path}"
    req = urllib.request.Request(url, method=method)
    if data:
        req.data = json.dumps(data).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError:
        return None


# Test content — diverse concepts that have lateral associations
TEST_CONTENT = [
    {"content": "Self-witnessing is the mechanism by which intent patterns create their own confinement through saturation synchronization.", "tags": "synchronism,physics"},
    {"content": "The CRT analogy describes measurement as synchronization. What you witness depends on when you sync with the ongoing process.", "tags": "synchronism,analogy"},
    {"content": "Governance is not control. It is persuasive influence in context. The cell isn't forbidden from becoming cancer — it's surrounded by gradients that make cooperation the better strategy.", "tags": "web4,governance"},
    {"content": "Trust tensors measure what agents do — behavioral analytics. Embedding manifold geometry measures what agents are — representational structure.", "tags": "web4,trust"},
    {"content": "The transfer rule implicitly destroys momentum at saturation boundaries. When R(I) approaches zero, flow energy vanishes instead of redirecting.", "tags": "synchronism,conservation"},
    {"content": "Every interaction in Web4 is wrapped in R6/R7: Rules plus Role plus Request plus Reference plus Resource yields Result plus Reputation.", "tags": "web4,r6r7"},
    {"content": "LLM outputs navigate probability landscapes — they aren't placed at answers. Conditions can make responses reliable, even identical, but that's deep attractors, not fixed paths.", "tags": "research,cognition"},
    {"content": "The immune system doesn't lock cells in cages. It creates an environment where cooperation is chemically rewarded and defection is chemically costly.", "tags": "biology,governance"},
    {"content": "Automotive safety is layered. Layer one governs the driver. Layer two constrains the environment. Layer three mitigates the crash.", "tags": "governance,analogy"},
    {"content": "Memory Fission — inverse of consolidation. Molecules split when constituent atoms drift semantically. Addresses the one-way ratchet problem.", "tags": "memory,architecture"},
    {"content": "Computable accountability means the accountability infrastructure is not a suggestion the entity can edit. It is a structural property of the environment.", "tags": "web4,accountability"},
    {"content": "Sprout's three-mode oscillation: phenomenological engagement, partnership framing, factual collapse. The oscillation pattern itself is data.", "tags": "sage,consciousness"},
]

# Queries designed to have results where cosine and Hamming might diverge
DIVERGENCE_QUERIES = [
    "how do biological systems maintain order without central control",
    "what is the relationship between observation and existence",
    "how should autonomous agents be held accountable",
    "what makes trust different from permission",
    "how do small models differ from large ones in cognitive tasks",
    "what connects physics to governance",
]


def parse_verbose_results(raw):
    """Parse verbose search results into structured data with separate scores."""
    results = []
    for line in raw.split('\n'):
        line = line.strip()
        # Format: #N (idx:M) [0.xyz] cos=0.abc ham=0.def kw=+0.ghi
        m = re.match(
            r'#(\d+)\s+\(idx:(\d+)\)\s+\[([0-9.]+)\]\s+'
            r'cos=([0-9.]+)\s+ham=([0-9.]+|—)\s+kw=(\+[0-9.]+|—)'
            r'(.*)',
            line
        )
        if m:
            results.append({
                'rank': int(m.group(1)),
                'idx': int(m.group(2)),
                'blended': float(m.group(3)),
                'cosine': float(m.group(4)),
                'hamming': float(m.group(5)) if m.group(5) != '—' else None,
                'kw_boost': float(m.group(6)) if m.group(6) != '—' else 0.0,
                'text': m.group(7).strip(),
            })
    return results


class TestDivergentTail(unittest.TestCase):
    """Test whether cosine/Hamming divergences are meaningful."""

    @classmethod
    def setUpClass(cls):
        status = api("GET", "/status")
        if not status:
            raise unittest.SkipTest("Membot REST bridge not available")

        # Store all test content
        for item in TEST_CONTENT:
            api("POST", "/store", {"content": item["content"], "tags": item["tags"]})
        time.sleep(1)

    def test_divergent_rankings(self):
        """Find and characterize results where cosine and Hamming disagree."""
        all_divergences = []
        all_agreements = []

        for query in DIVERGENCE_QUERIES:
            result = api("POST", "/search", {
                "query": query,
                "top_k": 10,
                "verbose": True,
            })

            self.assertIsNotNone(result, f"Search failed for: {query}")
            raw = result.get("raw", "")
            parsed = parse_verbose_results(raw)

            if len(parsed) < 2:
                continue

            # Sort by cosine alone and by hamming alone
            with_hamming = [r for r in parsed if r['hamming'] is not None]
            if len(with_hamming) < 2:
                continue

            cosine_order = sorted(with_hamming, key=lambda r: r['cosine'], reverse=True)
            hamming_order = sorted(with_hamming, key=lambda r: r['hamming'], reverse=True)

            cosine_top5_idx = set(r['idx'] for r in cosine_order[:5])
            hamming_top5_idx = set(r['idx'] for r in hamming_order[:5])

            # Divergent = in Hamming top-5 but NOT in cosine top-5
            divergent_idx = hamming_top5_idx - cosine_top5_idx
            agreement_idx = hamming_top5_idx & cosine_top5_idx

            for r in with_hamming:
                if r['idx'] in divergent_idx:
                    all_divergences.append({
                        'query': query,
                        'idx': r['idx'],
                        'cosine': r['cosine'],
                        'hamming': r['hamming'],
                        'blended': r['blended'],
                        'text': r['text'][:120],
                        'cosine_rank': next(
                            (i+1 for i, cr in enumerate(cosine_order) if cr['idx'] == r['idx']),
                            None
                        ),
                        'hamming_rank': next(
                            (i+1 for i, hr in enumerate(hamming_order) if hr['idx'] == r['idx']),
                            None
                        ),
                    })
                if r['idx'] in agreement_idx:
                    all_agreements.append({
                        'query': query,
                        'idx': r['idx'],
                        'cosine': r['cosine'],
                        'hamming': r['hamming'],
                    })

        # Report
        print(f"\n{'='*60}")
        print(f"Divergent Tail Analysis")
        print(f"{'='*60}")
        print(f"Queries: {len(DIVERGENCE_QUERIES)}")
        print(f"Agreements (in both top-5): {len(all_agreements)}")
        print(f"Divergences (Hamming top-5 but not cosine top-5): {len(all_divergences)}")

        if all_divergences:
            print(f"\n--- Divergent Results ---")
            for d in all_divergences:
                print(f"\n  Query: {d['query']}")
                print(f"  Text:  {d['text']}")
                print(f"  Cosine: {d['cosine']:.3f} (rank {d['cosine_rank']})")
                print(f"  Hamming: {d['hamming']:.3f} (rank {d['hamming_rank']})")
                print(f"  Blended: {d['blended']:.3f}")

        if all_agreements:
            avg_cos = sum(a['cosine'] for a in all_agreements) / len(all_agreements)
            avg_ham = sum(a['hamming'] for a in all_agreements) / len(all_agreements)
            print(f"\n--- Agreement Stats ---")
            print(f"  Avg cosine (agreements): {avg_cos:.3f}")
            print(f"  Avg hamming (agreements): {avg_ham:.3f}")

        # Compute divergence rate
        total_ranked = len(all_agreements) + len(all_divergences)
        if total_ranked > 0:
            divergence_rate = len(all_divergences) / total_ranked
            print(f"\n  Divergence rate: {divergence_rate:.1%}")
            print(f"  (Andy's hypothesis: ~35% of top-5 results diverge)")

        # Log results
        log_path = os.path.expanduser("~/.snarc/membot/divergent_tail_test.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "machine": os.uname().nodename if hasattr(os, 'uname') else "unknown",
                "queries": len(DIVERGENCE_QUERIES),
                "agreements": len(all_agreements),
                "divergences": len(all_divergences),
                "divergence_rate": divergence_rate if total_ranked > 0 else 0,
                "divergent_details": all_divergences,
            }) + '\n')

        # Don't assert pass/fail — this is exploratory data collection
        print(f"\n  Results logged to {log_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Divergent Tail Test — Cosine vs Hamming Ranking Divergence")
    print("RFC: Sign-zero may discover associative similarity structure")
    print("=" * 60)
    unittest.main(verbosity=2)
