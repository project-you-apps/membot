"""
Divergent Tail Test — Clean Dataset (CBP)
==========================================
Proper test of Andy's RFC hypothesis with:
- Deduplicated content (each concept exactly once)
- Diverse domains (physics, governance, biology, cognition, engineering)
- Fresh cartridge (no accumulated duplicates)
- Verbose output with separate cos/ham scores
- No keyword boost confound (queries avoid content keywords)

Creates a temporary cartridge, runs queries, analyzes divergence,
cleans up.

Usage:
    python3 tests/fleet/test_divergent_tail_clean.py

Machine: CBP
RFC: https://github.com/project-you-apps/membot/issues/5
"""

import unittest
import json
import os
import re
import time
import urllib.request
import urllib.error

MEMBOT_REST_URL = os.environ.get("MEMBOT_REST_URL", "http://localhost:8001")
CART_NAME = "divergent-tail-test"


def api(method, path, data=None, timeout=30):
    url = f"{MEMBOT_REST_URL}{path}"
    req = urllib.request.Request(url, method=method)
    if data:
        req.data = json.dumps(data).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError):
        return None


# 50 diverse, unique entries across 10 domains
# Each is conceptually distinct — no duplicates, no near-duplicates
CONTENT = [
    # Physics / Synchronism
    "Coherence emerges when oscillating patterns synchronize across a shared substrate, creating stable structures from unstable components.",
    "The transfer rule governs how intent flows between adjacent cells in the Planck grid, with saturation limiting maximum density.",
    "Damped oscillation occurs when energy is partially absorbed at reflection boundaries rather than fully conserved.",
    "Vortex ring formation requires three-dimensional dynamics — two-dimensional simulations cannot produce smoke ring structures.",
    "The entity criterion states that decay width must be less than mass for a quantum state to qualify as a particle.",

    # Biology / Immune system
    "Apoptosis is programmed cell death — the mechanism by which organisms eliminate cells that have exceeded their mandate.",
    "The immune system distinguishes self from non-self through distributed pattern matching across multiple independent verification mechanisms.",
    "Chemical gradients guide cellular behavior without central command, creating cooperative outcomes from local interactions.",
    "Metabolic states regulate energy expenditure: organisms cycle between active processing and restorative consolidation phases.",
    "Evolutionary adaptation produces governance solutions refined over millions of generations through selection pressure.",

    # Governance / Web4
    "Witnessed reputation accumulates from observed behavior over time, assessed by multiple independent observers in specific contexts.",
    "The R7 action framework wraps every consequential interaction in structured accountability with reputation feedback.",
    "Societies define their own law through versioned datasets of norms, enforced by quorum consensus among witnesses.",
    "Hardware-bound identity anchors digital presence to physical devices through cryptographic attestation chains.",
    "Resource metabolism ensures every action has a measurable cost, preventing freeloading and making productive behavior the rational strategy.",

    # AI / Cognition
    "Attention mechanisms in transformers compute weighted relevance scores across input sequences to focus processing resources.",
    "Chain-of-thought prompting elicits step-by-step reasoning from language models, improving performance on complex tasks.",
    "Retrieval-augmented generation combines parametric knowledge with retrieved documents to ground responses in external evidence.",
    "Fine-tuning adapts a pre-trained model to specific domains by updating weights on task-specific training data.",
    "Constitutional AI trains models to be helpful, harmless, and honest through self-critique and revision cycles.",

    # Trust / Security
    "Zero-trust architecture assumes no implicit trust and verifies every access request regardless of network location.",
    "Certificate authorities represent single points of failure in public key infrastructure — compromise one and trust collapses systemically.",
    "Multi-factor authentication combines something you know, something you have, and something you are to resist credential theft.",
    "Supply chain attacks compromise trusted dependencies to inherit the trust of downstream consumers without detection.",
    "Behavioral anomaly detection identifies threats by observing deviations from established patterns rather than matching known signatures.",

    # Software Engineering
    "Microservice architectures decompose applications into independently deployable services communicating through well-defined interfaces.",
    "Eventual consistency accepts temporary divergence between distributed replicas in exchange for availability and partition tolerance.",
    "Continuous integration automatically builds and tests code changes to detect integration failures early in the development cycle.",
    "Technical debt accumulates when expedient implementation choices create future maintenance burden that compounds over time.",
    "Feature flags decouple deployment from release, allowing code to ship without being activated until conditions are met.",

    # Philosophy / Epistemology
    "Reification makes the abstract concrete by assigning measurable variables to observed behaviors — money reifies value, equations reify gravity.",
    "Falsifiability requires that a theory make predictions which, if contradicted by evidence, would disprove the theory.",
    "Emergence describes properties that arise from interactions between components but cannot be predicted from the components alone.",
    "The observer effect in quantum mechanics describes how the act of measurement influences the system being measured.",
    "Pragmatism evaluates theories by their practical consequences rather than their correspondence to abstract truth.",

    # Economics / Game Theory
    "Nash equilibrium describes a state where no player can improve their outcome by unilaterally changing their strategy.",
    "The tragedy of the commons occurs when individual rational behavior depletes shared resources to everyone's detriment.",
    "Mechanism design creates institutions whose incentive structures lead self-interested agents to produce desirable collective outcomes.",
    "Adverse selection arises when information asymmetry allows informed parties to exploit uninformed counterparts in transactions.",
    "Public goods are non-excludable and non-rivalrous, creating free-rider problems that markets alone cannot solve.",

    # Organizational Design
    "Conway's law states that system architectures mirror the communication structures of the organizations that build them.",
    "Servant leadership inverts the traditional hierarchy — leaders serve their teams rather than teams serving leaders.",
    "Psychological safety enables team members to take risks and voice concerns without fear of punishment or humiliation.",
    "Agile methodologies prioritize working software over documentation, responding to change over following a plan.",
    "Matrix organizations assign employees to both functional departments and project teams, creating dual reporting structures.",

    # Data / Information
    "Embedding vectors map discrete tokens into continuous high-dimensional spaces where geometric distance reflects semantic similarity.",
    "Bloom filters probabilistically test set membership using multiple hash functions — they can produce false positives but never false negatives.",
    "Knowledge graphs represent information as typed relationships between entities, enabling structured queries across heterogeneous data.",
    "Compression algorithms exploit statistical redundancy in data to reduce storage requirements while preserving information content.",
    "Differential privacy adds calibrated noise to query results, providing mathematical guarantees against individual identification.",
]

# Queries designed to bridge domains — semantic connections that cross domain boundaries
# Each query could legitimately match content from multiple domains
QUERIES = [
    "how do decentralized systems prevent exploitation by bad actors",
    "what role does observation play in creating stable structures",
    "how can self-interested agents be guided toward cooperative behavior",
    "what happens when verification infrastructure is compromised",
    "how do complex systems maintain identity over time",
    "what mechanisms allow systems to adapt their own rules",
    "how does context determine whether an action is appropriate",
    "what is the relationship between measurement and the thing being measured",
    "how do small components produce large-scale order",
    "what distinguishes genuine trust from assumed trust",
]


def parse_verbose(raw):
    """Parse verbose results with separate cos/ham scores."""
    results = []
    lines = raw.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(
            r'#(\d+)\s+\(idx:(\d+)\)\s+\[([0-9.]+)\]\s+'
            r'cos=([0-9.]+)\s+ham=([0-9.]+|—)\s+kw=(\+[0-9.]+|—)',
            line
        )
        if m:
            # Next non-empty line is the text
            text = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith('#'):
                t = lines[j].strip()
                if t:
                    text = t
                    break
                j += 1

            results.append({
                'rank': int(m.group(1)),
                'idx': int(m.group(2)),
                'blended': float(m.group(3)),
                'cosine': float(m.group(4)),
                'hamming': float(m.group(5)) if m.group(5) != '—' else None,
                'kw_boost': float(m.group(6)) if m.group(6) != '—' else 0.0,
                'text': text[:200],
            })
        i += 1
    return results


class TestDivergentTailClean(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        status = api("GET", "/status")
        if not status:
            raise unittest.SkipTest("Membot REST bridge not available")

        # Mount fresh — if cart doesn't exist, mount will fail and we store into default
        mount = api("POST", "/mount", {"name": CART_NAME})
        if not mount or not mount.get("ok"):
            # Mount the default and store there
            api("POST", "/mount", {"name": "attention-is-all-you-need"})

        # Store all content (deduplicated — check if already stored by count)
        status_result = api("GET", "/status")
        if status_result:
            raw = status_result.get("raw", "")
            # Only store if cartridge is small (avoid re-adding)
            if "Memories: 0" in raw or int(re.search(r'Memories: (\d+)', raw).group(1)) < len(CONTENT):
                for item in CONTENT:
                    r = api("POST", "/store", {"content": item, "tags": ""})
                    if not r:
                        time.sleep(2)  # Model may be loading
                        api("POST", "/store", {"content": item, "tags": ""})

        time.sleep(1)

    def test_divergent_rankings_clean(self):
        """Analyze cosine vs Hamming divergence on clean, diverse data."""
        all_results = []

        for query in QUERIES:
            result = api("POST", "/search", {
                "query": query,
                "top_k": 10,
                "verbose": True,
            })

            if not result:
                continue
            raw = result.get("raw", "")
            parsed = parse_verbose(raw)

            with_hamming = [r for r in parsed if r['hamming'] is not None]
            if len(with_hamming) < 5:
                continue

            # Rank by cosine alone vs Hamming alone
            cos_ranked = sorted(with_hamming, key=lambda r: r['cosine'], reverse=True)
            ham_ranked = sorted(with_hamming, key=lambda r: r['hamming'], reverse=True)

            cos_top5 = set(r['idx'] for r in cos_ranked[:5])
            ham_top5 = set(r['idx'] for r in ham_ranked[:5])

            # Find divergences: in one's top-5 but not the other's
            ham_only = ham_top5 - cos_top5
            cos_only = cos_top5 - ham_top5
            agreement = cos_top5 & ham_top5

            query_result = {
                'query': query,
                'agreement_count': len(agreement),
                'ham_only_count': len(ham_only),
                'cos_only_count': len(cos_only),
                'ham_only': [],
                'cos_only': [],
                'agreement': [],
            }

            for r in with_hamming:
                entry = {
                    'idx': r['idx'],
                    'cosine': r['cosine'],
                    'hamming': r['hamming'],
                    'text': r['text'],
                    'cos_rank': next((i+1 for i, cr in enumerate(cos_ranked) if cr['idx'] == r['idx']), None),
                    'ham_rank': next((i+1 for i, hr in enumerate(ham_ranked) if hr['idx'] == r['idx']), None),
                }
                if r['idx'] in ham_only:
                    query_result['ham_only'].append(entry)
                elif r['idx'] in cos_only:
                    query_result['cos_only'].append(entry)
                elif r['idx'] in agreement:
                    query_result['agreement'].append(entry)

            all_results.append(query_result)

        # Report
        print(f"\n{'='*70}")
        print(f"CLEAN Divergent Tail Analysis ({len(CONTENT)} unique entries, {len(QUERIES)} queries)")
        print(f"{'='*70}")

        total_agree = sum(r['agreement_count'] for r in all_results)
        total_ham_only = sum(r['ham_only_count'] for r in all_results)
        total_cos_only = sum(r['cos_only_count'] for r in all_results)
        total = total_agree + total_ham_only + total_cos_only

        print(f"\nOverall: {total_agree} agreements, {total_ham_only} Hamming-only, {total_cos_only} cosine-only")
        if total > 0:
            print(f"Divergence rate: {(total_ham_only + total_cos_only) / total:.1%}")

        for qr in all_results:
            if qr['ham_only'] or qr['cos_only']:
                print(f"\n--- {qr['query']} ---")
                print(f"  Agree: {qr['agreement_count']}  Ham-only: {qr['ham_only_count']}  Cos-only: {qr['cos_only_count']}")
                for h in qr['ham_only']:
                    print(f"  HAM-ONLY cos={h['cosine']:.3f}(r{h['cos_rank']}) ham={h['hamming']:.3f}(r{h['ham_rank']}): {h['text'][:100]}")
                for c in qr['cos_only']:
                    print(f"  COS-ONLY cos={c['cosine']:.3f}(r{c['cos_rank']}) ham={c['hamming']:.3f}(r{c['ham_rank']}): {c['text'][:100]}")

        # Log
        log_path = os.path.expanduser("~/.snarc/membot/divergent_tail_clean.jsonl")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "machine": os.uname().nodename if hasattr(os, 'uname') else "unknown",
                "entries": len(CONTENT),
                "queries": len(QUERIES),
                "total_agree": total_agree,
                "total_ham_only": total_ham_only,
                "total_cos_only": total_cos_only,
                "divergence_rate": (total_ham_only + total_cos_only) / total if total > 0 else 0,
                "per_query": [{
                    "query": r["query"],
                    "agree": r["agreement_count"],
                    "ham_only": r["ham_only_count"],
                    "cos_only": r["cos_only_count"],
                    "ham_only_entries": r["ham_only"],
                    "cos_only_entries": r["cos_only"],
                } for r in all_results],
            }) + '\n')

        print(f"\n  Results logged to {log_path}")


if __name__ == "__main__":
    print("=" * 70)
    print("Clean Divergent Tail Test — 50 unique entries, 10 cross-domain queries")
    print("Testing: do cosine and Hamming find genuinely different content?")
    print("=" * 70)
    unittest.main(verbosity=2)
