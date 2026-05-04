#!/usr/bin/env python3
"""BEIR benchmark driver for Membot.

Runs a BEIR dataset against Membot's sign-zero hamming + 70/30 cosine blend
+ keyword reranking pipeline. Computes Recall@K and MRR vs the published qrels.

This is the empirical rebuttal to NodeMind's asserted (but not measured)
claim that standard sign-bit binarization "breaks down on out-of-distribution
queries." See:
  memory/concept-clusters/CC_competitive-landscape-nodemind_2026-05.md

USAGE
-----

Two-step flow because cart-build is expensive (embed once, query many):

    # 1. Build the BEIR cart (downloads dataset via HuggingFace, embeds, saves)
    python tools/benchmarks/run_beir.py --dataset scifact --build-only

    # 2. Mount the cart on a running membot and run queries
    #    (assumes you have a membot HTTP server running locally on :8000)
    python -m membot --transport http --port 8000 --writable &
    curl -X POST http://localhost:8000/api/mount \
         -H 'Content-Type: application/json' \
         -d '{"name":"beir-scifact"}'
    python tools/benchmarks/run_beir.py --dataset scifact --skip-build

    # Or do both in one go (script will tell you to mount the cart manually
    # in between, since it can't restart the server itself):
    python tools/benchmarks/run_beir.py --dataset scifact

DATASETS (small ones recommended for first runs)
-----------------------------------------------

    scifact     ~5K docs, ~300 queries        (smoke test)
    nfcorpus    ~3.6K docs, ~323 queries      (smallest)
    fiqa        ~57K docs, ~648 queries       (financial QA — real OOD)
    trec-covid  ~171K docs, ~50 queries       (post-cutoff OOD)

Larger datasets exist (msmarco, hotpotqa, etc.) but those are multi-day
affairs at our scale; not needed for the rebuttal.

OUTPUT
------

Per-dataset table:

    Recall@1  Recall@5  Recall@10   MRR@10
    ---       ---       ---         ---
    0.xxxx    0.xxxx    0.xxxx      0.xxxx

Compare to:
- BGE-M3 / cosine baselines published in the BEIR paper appendix
- HNSW-tuned recall (typically 0.95-0.99 on most BEIR datasets at N=10)
- Headline target: "sign-zero binary holds @ Recall@N within Y pp of float32"

DEPENDENCIES
------------

    pip install datasets         # HuggingFace datasets, for BEIR loading

Plus whatever cartridge_builder.py already needs (sentence-transformers,
numpy). Membot's own deps cover the search side.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add membot/ to sys.path so we can import cartridge_builder directly.
SCRIPT_DIR = Path(__file__).resolve().parent
MEMBOT_DIR = SCRIPT_DIR.parent.parent  # membot/tools/benchmarks/run_beir.py -> membot/
sys.path.insert(0, str(MEMBOT_DIR))


# ---------------------------------------------------------------------------
# BEIR loader (via HuggingFace datasets — no `beir` library needed)
# ---------------------------------------------------------------------------

def load_beir_dataset(name: str, cache_dir: str | None = None):
    """Load corpus + queries + qrels for a BEIR dataset via HF datasets.

    Returns:
        corpus: dict[doc_id] = {"title": str, "text": str}
        queries: dict[query_id] = str (the query text itself)
        qrels: dict[query_id] = dict[doc_id] = relevance_score (int > 0)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print(f"[load] BeIR/{name} corpus + queries...")
    corpus_ds = load_dataset(f"BeIR/{name}", "corpus", cache_dir=cache_dir, split="corpus")
    queries_ds = load_dataset(f"BeIR/{name}", "queries", cache_dir=cache_dir, split="queries")

    print(f"[load] BeIR/{name}-qrels test split...")
    qrels_ds = load_dataset(f"BeIR/{name}-qrels", cache_dir=cache_dir, split="test")

    corpus = {}
    for row in corpus_ds:
        corpus[str(row["_id"])] = {
            "title": row.get("title", "") or "",
            "text": row.get("text", "") or "",
        }

    queries = {str(row["_id"]): row["text"] for row in queries_ds}

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        q_id = str(row["query-id"])
        d_id = str(row["corpus-id"])
        score = int(row["score"])
        if score > 0:
            qrels.setdefault(q_id, {})[d_id] = score

    # Only keep queries that have at least one relevant doc in the qrels
    queries = {q: queries[q] for q in queries if q in qrels}

    print(f"[load] {len(corpus):,} docs, {len(queries):,} queries with qrels")
    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# Cart build (uses cartridge_builder's existing API)
# ---------------------------------------------------------------------------

def build_cart_from_corpus(corpus: dict, cart_name: str, output_dir: str):
    """Embed BEIR corpus into a Membot cart.

    Each doc becomes one passage prefixed with [BEIR,doc=<id>] so that search
    results' tag fields can be parsed back to BEIR doc_ids for scoring.

    Title and body are concatenated as 'Title: <t>\\n\\n<text>' (or just text
    if no title) — this is the standard BEIR convention.
    """
    import cartridge_builder  # late import — sys.path is patched at module top

    # Materialize corpus in deterministic order so we can map index -> doc_id
    doc_ids: list[str] = list(corpus.keys())
    passages: list[str] = []
    for doc_id in doc_ids:
        d = corpus[doc_id]
        title = d.get("title", "").strip()
        text = d.get("text", "").strip()
        body = f"Title: {title}\n\n{text}" if title else text
        # Tag prefix: parsed by membot at search time and returned in
        # result.tags so we can recover the doc_id from each hit.
        passages.append(f"[BEIR,doc={doc_id}]{body}")

    print(f"[build] Embedding {len(passages):,} passages...")
    t0 = time.time()
    embeddings = cartridge_builder.embed_texts(passages)
    print(f"[build] Embed done in {time.time() - t0:.1f}s "
          f"({embeddings.shape}, dtype={embeddings.dtype})")

    print(f"[build] Saving cart to {output_dir}/{cart_name}.cart.npz ...")
    cartridge_builder.save_cartridge(output_dir, cart_name, embeddings, passages)
    print(f"[build] Cart saved.")


# ---------------------------------------------------------------------------
# Query loop (HTTP — assumes membot server with cart mounted)
# ---------------------------------------------------------------------------

def search_query(server_url: str, query: str, top_k: int):
    """POST /api/search and return list of (doc_id, score) for the top_k hits.

    Doc_id is extracted from the result's tags string by parsing 'doc=<id>'.
    Results without a recoverable doc_id (e.g. cart not built by this driver)
    are silently skipped.
    """
    import urllib.error
    import urllib.request

    body = json.dumps({"query": query, "top_k": top_k}).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/api/search",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    results: list[tuple[str, float]] = []
    for r in data.get("results", []):
        tags = r.get("tags", "") or ""
        doc_id = None
        for tag in tags.split(","):
            tag = tag.strip()
            if tag.startswith("doc="):
                doc_id = tag[4:].strip()
                break
        if doc_id is None:
            continue
        results.append((doc_id, float(r.get("score", 0.0))))
    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def recall_at_k(results_per_query, qrels, k):
    """Macro-averaged Recall@k across queries."""
    recalls = []
    for q_id, results in results_per_query.items():
        relevant = set(qrels.get(q_id, {}).keys())
        if not relevant:
            continue
        retrieved = [d for d, _ in results[:k]]
        hits = sum(1 for d in retrieved if d in relevant)
        recalls.append(hits / len(relevant))
    return sum(recalls) / len(recalls) if recalls else 0.0


def mrr_at_k(results_per_query, qrels, k):
    """Mean Reciprocal Rank @ k."""
    rrs = []
    for q_id, results in results_per_query.items():
        relevant = set(qrels.get(q_id, {}).keys())
        if not relevant:
            continue
        rr = 0.0
        for rank, (d, _) in enumerate(results[:k], start=1):
            if d in relevant:
                rr = 1.0 / rank
                break
        rrs.append(rr)
    return sum(rrs) / len(rrs) if rrs else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="BEIR dataset name (e.g. scifact, nfcorpus, fiqa, trec-covid). "
             "Any HF BeIR/<name> + BeIR/<name>-qrels pair works.",
    )
    p.add_argument("--top-k", type=int, default=10, help="Top-k retrieved per query (default 10)")
    p.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="Running membot HTTP server with the BEIR cart mounted",
    )
    p.add_argument(
        "--cart-dir",
        default=str(MEMBOT_DIR / "cartridges"),
        help="Where to write the BEIR cart (default: <membot>/cartridges/)",
    )
    p.add_argument("--cache-dir", default=None, help="HuggingFace datasets cache dir")
    p.add_argument(
        "--build-only",
        action="store_true",
        help="Stop after building the cart (don't query). Use this first, mount the cart, "
             "then re-run with --skip-build to score.",
    )
    p.add_argument(
        "--skip-build",
        action="store_true",
        help="Assume the cart already exists and is mounted on the server.",
    )
    p.add_argument(
        "--log-jsonl",
        default=None,
        help="Optional path to write per-query results (q_id, retrieved doc_ids, scores) "
             "as JSONL for offline analysis.",
    )
    args = p.parse_args()

    cart_name = f"beir-{args.dataset}"

    # 1. Load BEIR
    corpus, queries, qrels = load_beir_dataset(args.dataset, cache_dir=args.cache_dir)

    # 2. Build cart (or skip)
    if not args.skip_build:
        Path(args.cart_dir).mkdir(parents=True, exist_ok=True)
        build_cart_from_corpus(corpus, cart_name, args.cart_dir)

    if args.build_only:
        print(f"\n[done] Cart at {args.cart_dir}/{cart_name}.cart.npz")
        print(f"[next] Mount it on the membot server, then re-run this script with --skip-build:")
        print(f"  curl -X POST {args.server_url}/api/mount \\")
        print(f"       -H 'Content-Type: application/json' \\")
        print(f"       -d '{{\"name\":\"{cart_name}\"}}'")
        return

    # 3. Run queries
    print(f"[query] Running {len(queries):,} queries against {args.server_url}...")
    results_per_query: dict[str, list[tuple[str, float]]] = {}
    log_f = open(args.log_jsonl, "w", encoding="utf-8") if args.log_jsonl else None
    t0 = time.time()
    failures = 0
    for i, (q_id, q_text) in enumerate(queries.items()):
        if i and i % 100 == 0:
            elapsed = time.time() - t0
            print(f"[query]   {i:,}/{len(queries):,} ({elapsed:.1f}s, "
                  f"{elapsed / i * 1000:.1f}ms/query)")
        try:
            results_per_query[q_id] = search_query(args.server_url, q_text, args.top_k)
        except Exception as e:
            failures += 1
            if failures <= 5:
                print(f"[query] WARN: query {q_id} failed: {e}")
            results_per_query[q_id] = []
        if log_f:
            log_f.write(json.dumps({
                "query_id": q_id,
                "query": q_text,
                "results": results_per_query[q_id],
            }) + "\n")
    if log_f:
        log_f.close()

    elapsed = time.time() - t0
    print(f"[query] Done in {elapsed:.1f}s "
          f"({elapsed / max(len(queries), 1) * 1000:.1f}ms/query, {failures} failures)")

    # 4. Score
    print(f"\n=== {args.dataset} : sign-zero hamming + 70/30 cosine blend + kw rerank ===")
    print(f"  Queries scored: {len(results_per_query):,}  "
          f"Top-k: {args.top_k}  Failures: {failures}")
    print()
    print(f"  {'Metric':<14}{'Value':>10}")
    print(f"  {'-' * 14}{'-' * 10:>10}")
    for k in (1, 5, 10, args.top_k):
        if k <= 0 or k > args.top_k:
            continue
        if k in (1, 5, 10) or k == args.top_k:
            r = recall_at_k(results_per_query, qrels, k)
            print(f"  Recall@{k:<7}{r:>10.4f}")
    for k in (10, args.top_k):
        if k <= 0 or k > args.top_k:
            continue
        if k == 10 or k == args.top_k:
            m = mrr_at_k(results_per_query, qrels, k)
            print(f"  MRR@{k:<10}{m:>10.4f}")


if __name__ == "__main__":
    main()
