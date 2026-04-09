# RFC: Federated Cart Mode — Multi-Machine Brain Cart Sync

**Date**: 2026-04-07
**Status**: Draft
**Authors**: Andy Grossberg, Claude (Opus 4.6)
**Related**: `multi-cart-query-spec.md`, `membox-multiuser-dbms-spec.md`
**Inspired by**: Dennis Palatov's `dp-web4/shared-context/arc-agi-3/FEDERATED_LEARNING.md`

---

## TL;DR

Federated mode lets a fleet of machines share knowledge by each writing to its own append-only brain cart in a shared git directory, then mounting all the carts together for cross-fleet semantic search. No conflicts (each machine writes only its own cart). No central server (sync is `git pull`). No deduplication step (cross-cart `CONFIRMED_BY` edges turn matching insights into trust signals instead of duplicates).

This is Dennis's `FEDERATED_LEARNING.md` JSONL-in-git architecture, but with brain carts as the substrate. Same git sync model, same per-machine append-only writes, same daily consolidation timing. The substrate change is what unlocks semantic search across the fleet.

This RFC depends on `multi-cart-query-spec.md` (the multi-cart mount and query layer) and is the federation feature *built on* that foundation.

---

## Motivation

Dennis's SAGE fleet (6 machines, 11 instances) currently shares knowledge via JSONL files in the `dp-web4/shared-context` git repo. Each machine writes to `fleet-learning/{machine}/{game}_learning.jsonl`. CBP runs a daily cron at 4am that reads all the JSONL files, deduplicates, and writes consolidated outputs to `consolidated/`. Other machines `git pull` at session start to get the latest.

This works. It's elegant. But it has limits:

| Limitation | What it costs |
|-----------|---------------|
| Text grep, not semantic search | Can't ask "has any machine seen a puzzle with cyclic loops?" — can only grep for the literal string "cyclic loops" |
| Daily consolidation lag | Insights from other machines aren't visible until the next 4am cron |
| Deduplication is lossy | When two machines independently arrive at the same insight, the consolidator picks "best" and drops the other — losing the corroboration signal |
| Disagreements get resolved away | If two machines disagree, the consolidator picks one — the disagreement itself isn't preserved as data |
| No provenance chains | Can't trace "this insight built on that earlier insight from machine X" |

A brain cart federation gives all of those for free, while keeping Dennis's git-sync architecture intact. The fleet doesn't change how it commits and pulls. The substrate gains semantic search, live consolidation, corroboration tracking, and dispute preservation.

---

## Architecture

### Directory layout (matches Dennis's, just .cart instead of .jsonl)

```
shared-context/arc-agi-3/
├── fleet-learning/
│   ├── cbp/
│   │   └── kb.cart                  # CBP's append-only learning cart
│   ├── sprout/
│   │   └── kb.cart
│   ├── mcnugget/
│   ├── legion/
│   ├── nomad/
│   └── thor/
├── consolidated/
│   ├── kb.cart                      # CBP-built consolidation (cross-machine edges only)
│   └── last_consolidated.json       # timestamp + stats
└── consolidate.py                   # the consolidator (uses Membot federate consolidate)
```

The directory structure is *identical* to Dennis's current layout. Only the file extensions change. This is intentional: we don't want to fork Dennis's process — we want to slot underneath it.

### Per-machine cart format

Each machine's `kb.cart` is a normal paired brain cart (even rows = embeddings, odd rows = text, h-row = navigation) with two extra metadata fields:

- `source_machine`: the machine_id that wrote this cart (prevents identity confusion if the file is renamed or copied)
- `lineage`: the CONSOLIDATED cart_id this cart contributes to (so the consolidator knows where to publish to)

Per-pattern metadata (in the `_reserved` dict already supported by Pattern 0 v2) gains:

- `machine_id`: who imprinted this pattern
- `player`: the LLM model that produced the insight (e.g. "claude", "gemma3-4b", "tinyllama")
- `session_id`: which session the insight came from
- `event_type`: matches Dennis's existing event types — `level_solved`, `level_failed`, `game_complete`, `game_insight`, `structural_pattern`
- `confidence`: 0-1, machine's own confidence at write time
- `actions_taken`: for `level_solved` events
- `baseline`: human baseline for comparison

These are exactly the fields Dennis's JSONL format uses. We just store them on the pattern instead of in a JSON line.

### Append-only guarantee

A machine's cart is **only ever written by that machine**. Other machines mount it read-only. This matches Dennis's git-conflict-prevention principle: no two machines ever touch the same file.

The consolidator (CBP only) writes to `consolidated/kb.cart`. That's the one shared write target, and only one machine has permission to do it.

### Sync model

Same as Dennis's: `git pull` at session start. The cart files are versioned in git like everything else. Git's binary file handling is fine for cart files at typical fleet sizes (~1-10MB per machine cart).

For larger carts (>50MB), we add a `git lfs` recommendation to the README. We don't force it.

---

## Drop-in API for Dennis's Hooks

The whole point of this spec is that Dennis can swap in our implementation **without rewriting his solver**. That requires our API to match what his code already calls.

### Replacement: `consolidate.py`

His current script: reads all `fleet-learning/*/*.jsonl`, deduplicates, writes to `consolidated/`.

Our replacement is a one-liner he can drop in:

```python
#!/usr/bin/env python3
"""Federation consolidator — replaces the JSONL version with cart-based federation."""
from membot import federate

federate.consolidate(
    fleet_dir="shared-context/arc-agi-3/fleet-learning",
    output_dir="shared-context/arc-agi-3/consolidated",
    similarity_threshold=0.85,
    confidence_boost_per_confirmation=0.05,
)
```

Under the hood, `federate.consolidate()`:

1. Mounts every `fleet-learning/{machine}/kb.cart` as a separate cart (using the multi-cart mount API)
2. For each pattern in each machine cart, searches all OTHER machine carts for semantically similar patterns
3. When similarity > threshold, creates a `CONFIRMED_BY` edge between them in the consolidated cart
4. When two patterns are similar but their text disagrees (low text overlap, high embedding similarity), creates `CONTRADICTED_BY` edges
5. Writes the consolidation as a separate `consolidated/kb.cart` containing only the cross-cart edges (the actual patterns stay in their machine carts; the consolidated cart is essentially an index)
6. Updates `last_consolidated.json` with stats

The output is the same shape Dennis's solver expects: a directory of files he can mount. The new shape is a brain cart instead of jsonl files, but his "load consolidated data" code path becomes a single mount call.

### Replacement: `publish_learning.py`

His current script: extracts learning entries from `session.json`, appends them to `fleet-learning/{machine}/{game}_learning.jsonl`.

Our replacement:

```python
#!/usr/bin/env python3
"""Publish a session's learning to the machine's federated cart."""
from membot import federate
import sys

session_file = sys.argv[1]
machine_id = sys.argv[2]   # e.g. "cbp"

federate.publish_session(
    session_file=session_file,
    machine_id=machine_id,
    fleet_dir="shared-context/arc-agi-3/fleet-learning",
)
```

`federate.publish_session()`:

1. Reads the session.json
2. Extracts learning entries (event_type, level, actions, mechanics, insight text, etc.)
3. Imprints each entry into the machine's local cart with all the metadata fields populated
4. Commits the cart file to git (or leaves that to the user's existing post-session hook)

### Solver-side mount call

His solver currently does something like:

```python
# Old
consolidated = load_jsonl_consolidated("shared-context/arc-agi-3/consolidated/")
my_kb = GameKnowledgeBase("sprout")
```

Becomes:

```python
# New
import membot
mb = membot.Membot()
mb.mount_directory("shared-context/arc-agi-3/fleet-learning/", role="federated")
mb.mount("shared-context/arc-agi-3/consolidated/kb.cart", cart_id="consolidated", role="consolidated")
my_kb = mb.get_cart("sprout")  # or however his ID maps
```

Then anywhere his code did "load consolidated data," it does:

```python
results = mb.search(query, scope="all", role_filter="federated")
```

The drop-in test: the solver doesn't need to know or care that the substrate changed. The function signatures match, the data shape matches, the integration is mount + search.

---

## Live vs Daily Consolidation

Dennis's current model: consolidator runs at 4am. Insights from other machines aren't visible until then.

With federated carts, consolidation has two modes:

### Daily mode (matches Dennis's current cron)
The consolidator runs once a day, computes all the cross-cart edges, and writes them to `consolidated/kb.cart`. This is what `federate.consolidate()` does when called from a cron. Same timing, same git commit, same file Dennis's solver mounts on session start.

### Live mode (new capability)
When a machine mounts the entire fleet directory at session start, the cross-cart edges can be computed on the fly. No 24-hour lag. The cost is mount-time consolidation, but for typical fleet sizes (10 machines, 10K patterns each) that's seconds, not minutes.

The choice between modes is a config flag. Daily is safer (predictable git history, predictable performance). Live is faster-feedback (no lag). Both are supported.

---

## Migration Path from JSONL

For Dennis's existing fleet, migration is one command:

```bash
python -m membot.federate.migrate_jsonl \
    --input shared-context/arc-agi-3/fleet-learning \
    --output shared-context/arc-agi-3/fleet-learning \
    --in-place
```

This walks every `*_learning.jsonl` file, builds a `kb.cart` next to it (or in place of it), and preserves all metadata. After migration:

- Old JSONL files can be deleted (or kept as a fallback)
- Cart files are committed to git
- `consolidate.py` is replaced with the membot version
- Solvers are updated to mount carts instead of loading JSONL

Migration is reversible: a `cart_to_jsonl` script can rebuild the JSONL files from the carts at any time. We never destroy data — we add a new substrate alongside the old one.

---

## What This Gives Dennis

| Capability | Before (JSONL) | After (Federated Carts) |
|-----------|----------------|-------------------------|
| Semantic search across fleet | Grep only | Cosine + Hamming + keyword |
| Cross-machine corroboration | Lost in dedup | First-class edges (CONFIRMED_BY) |
| Cross-machine disagreement | Lost in dedup | First-class edges (CONTRADICTED_BY) |
| Live updates | 24h lag | Optional, on mount |
| Cart provenance | None | machine_id, player, session_id per pattern |
| Membot ecosystem | Outside it | Inside it (uses membot search, MCP, viewers) |
| Git sync model | Same | Same |
| Per-machine append-only | Same | Same |
| Drop-in to his solver | N/A | Yes (this is the key requirement) |

The substrate change unlocks the first six rows. The bottom three are guarantees that we don't break what already works.

---

## Implementation Phases

### Phase 1: Cart format extensions (1-2 days)
Depends on multi-cart-query-spec.md Phase 1.
- Add `source_machine`, `lineage` to Pattern 0 v2 metadata
- Add per-pattern fields (`machine_id`, `player`, `session_id`, `event_type`, `confidence`, `actions_taken`, `baseline`)
- Build the JSONL → cart migration script

### Phase 2: Federation API (2-3 days)
Depends on multi-cart-query-spec.md Phase 2-3.
- `membot.federate.consolidate()` — runs the cross-cart edge computation
- `membot.federate.publish_session()` — extracts session data, imprints to local cart
- `membot.federate.migrate_jsonl()` — converts existing JSONL to carts

### Phase 3: Drop-in scripts for Dennis (1 day)
- `consolidate.py` replacement
- `publish_learning.py` replacement
- Mount + search example for solver integration
- Testing on a small subset of his actual fleet data (2-3 machines)

### Phase 4: Live mode (2 days)
- Mount-time consolidation option
- Background consolidation thread option
- Performance profiling

### Phase 5: Offer it (1 day)
- Write a clear README for Dennis explaining what it gives, what it costs, how to try it
- File a PR or comment on his FEDERATED_LEARNING.md
- He decides whether to adopt

Total: ~8-10 days, after multi-cart-query-spec.md is done.

---

## Open Questions

1. **Cart file conflicts in git.** Even though each machine writes only its own cart, two machines might commit at the same time and produce a non-fast-forward push. Same problem as any git workflow. Solution: machines `git pull --rebase` before commit, push retries on conflict. Dennis's existing infrastructure already handles this for JSONL — same solution applies.

2. **Cart binary diffs.** Cart files are binary (.npz or .pkl). Git can store them but can't show meaningful diffs. Mitigation: include a `cart.summary.txt` next to each cart with human-readable counts and recent additions. Git diffs the summary file; the cart file is the source of truth.

3. **Migration risk.** If the JSONL → cart migration has a bug, machines could lose data. Mitigation: migration is non-destructive by default (writes alongside, doesn't replace). Dennis can verify cart correctness against JSONL before deleting JSONL files.

4. **Versioning the federation format itself.** As we add fields, how do we handle old machines reading new carts and vice versa? Pattern 0 v2 already has a version field. Bump it on format changes. Consolidator skips patterns whose format version is higher than its own and logs a warning.

5. **Performance at fleet scale.** A fleet of 100 machines × 100K patterns each = 10M patterns to mount. Multi-cart query at that scale needs the lazy-load approach mentioned in the multi-cart spec. Worth designing for, not worth implementing until the fleet actually gets there.

6. **Permission model.** Federated mode is read-only-from-other-machines by default. Should there be a "trusted machine" concept where one machine is allowed to write to another's cart? Probably no. Keep the append-only-by-self invariant strict. If a machine wants to refine another's insight, it imprints the refinement in its OWN cart with a `REFINED_BY` edge pointing to the other machine's pattern. Cross-cart edges go in the consolidated cart, not in machine carts.

---

## What This Doesn't Do (Yet)

- **Doesn't replace Dennis's git workflow.** Sync is still git. We're not building a network protocol.
- **Doesn't add real-time push.** A machine's writes don't propagate to other machines until git push + git pull. Same as today.
- **Doesn't handle multi-writer single-cart.** That's the Membox spec. Federated mode is "many machines, many carts, one read across them all."
- **Doesn't provide a UI.** That's VPS / Cart Builder integration territory. Federated mode is the substrate.

---

*Federation isn't a separate feature. It's what multi-cart query looks like when the carts come from multiple machines instead of multiple cognitive functions.*

---

## Amendment 1: Consolidation Mode (2026-04-08, dp-web4 design input)

After the Phase 1 PR was merged into dp-web4/membot, Dennis Palatov flagged a fundamental design choice in `consolidate()`. The original implementation picked one representative per CONFIRMED_BY connected component and discarded the duplicates. Dennis pushed back:

> "We prefer keeping all variants and using CONFIRMED_BY edges as trust signals rather than picking one representative. This aligns with our Web4 trust model — trust is contextual and evaluated by the relying party, not collapsed to a single truth by the consolidator. The solver should see that 3 machines agree on a mechanic and weigh that differently than a single observation."

He's right. The consolidator's job is to **find** cross-machine relationships, not to **decide** whose voice wins. Trust is contextual to the relying party (the solver). Three machines independently arriving at the same conclusion is *information* that should be visible at search time, not metadata that gets compressed into "we picked the cbp version because it came first alphabetically."

### The fix

`consolidate()` now takes a `mode` parameter:

- **`mode="preserve"`** (DEFAULT): Keep all variants from every machine in the consolidated cart. Cross-cart edges are stored as per-pattern metadata (`confirming_machines`, `contradicting_machines`, `confirmed_by`, `contradicted_by`, `n_confirmations`, `n_contradictions`, `trust_signal`). The solver sees every voice and weighs them itself.
- **`mode="collapse"`**: Legacy behavior. Pick one representative per CONFIRMED_BY connected component. Smaller output cart but loses individual machine voices. Use only when storage is constrained and per-machine attribution doesn't matter.

### Why this matters beyond SAGE

The Web4 framing — "trust is contextual, evaluated by the relying party" — applies to *any* federation use case, not just Dennis's fleet. A multi-team knowledge base (Membox in federated mode) should preserve disagreements between teams instead of collapsing them. A multi-source RAG corpus should let the model see which sources corroborate each other and which contradict, not just the deduplicated answer. Preserve mode is the correct default for federation, period.

### Trust signal hint

In preserve mode, each consolidated pattern gets a `trust_signal` field set to one of:

- `high_corroboration` — confirmed by 2+ other machines
- `single_corroboration` — confirmed by exactly 1 other machine
- `single_source` — only the source machine has this pattern

The solver MAY use this as a confidence boost, MAY ignore it, MAY apply its own trust calculation based on which machines confirmed (e.g. trust Sprout's confirmations more than CBP's because Sprout is in a deeper raising phase). The hint is a *suggestion*, not a verdict.

### Phase 3 alignment

This amendment puts us closer to the Phase 3 vision (cross-cart h-row edges) than the original collapse-mode implementation. In Phase 3, the cross-cart edges in metadata become first-class h-row entries with proper navigation. The preserve-mode consolidated cart already has all the information needed to construct those edges; the difference is just the storage format. The metadata blob today can be lifted into h-row entries tomorrow without re-running consolidation.

### Status

Shipped 2026-04-08 in the same branch as the original Phase 1 PR (claude/federate-phase-1).
