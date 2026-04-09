# Membot ↔ SAGE Collaboration Devlog

**Scope**: The ongoing collaboration between project-you-apps (Andy + Claude) and dp-web4 (Dennis Palatov + Claude instances on the SAGE fleet). Focused on the substrate-layer work that connects Membot's neuromorphic memory to SAGE's cognition kernel.

For Membot core development, see `DEVLOG.md`.
For Membox multi-user work, see `../docs/DEVLOG-MEMBOX.md` in project-you root.
For project-level history, see `../docs/DEVLOG-ONGOING.md` in project-you root.

---

## 2026-04-09 — Dennis running carts on new games

Dennis confirmed cart readiness and is now running brain carts on new ARC-AGI-3 games. Fleet has 7 games fully solved (up from 5 yesterday): ft09 217%, cd82 107%, sb26 109%, sc25 126%, tn36 210%, vc33 184%, lp85 361%.

Andy & Claude sent the migration command (`python federate.py migrate shared-context/arc-agi-3/fleet-learning --in-place`) and offered a curated game insights cart option. Dennis's response:

> "running with carts, let's see what difference it makes in new games. will take a while. it's a process :)"

No questions, no blockers — just executing. Awaiting results on whether semantic cart retrieval improves solver performance on unsolved games.

---

## 2026-04-08 — Federate Phase 1 merged + preserve mode + scope_mode polish

### What landed today (PR #11 on project-you-apps/membot)

Three commits, all on branch `claude/federate-phase-1`:

1. **`34a5d1d` — Multi-cart query Phase 1 + Federated mode Phase 1 + RFCs + tests + README**
   - `multi_cart.py`: process-global cart pool with `mount/unmount/list/mount_directory/search` + namespaced result attribution
   - `federate.py`: drop-in replacement for `consolidate.py` and `publish_learning.py` from Dennis's existing federated learning architecture
   - 9 new MCP tools (5 multi_cart + 4 federate)
   - 3 RFC docs: multi-cart-query-spec, federated-cart-spec, membox-multiuser-dbms-spec (v2 amendments)
   - 2 test scripts (test_multi_cart.py, test_federate.py) validated against Dennis's real `sb26_learning.jsonl`
   - README "What's New" section introducing the three modes (single/federated/multiuser)

2. **`7f73a85` — `consolidate(mode='preserve')` per Dennis's Web4 trust feedback**
   - Original implementation collapsed CONFIRMED_BY components to one representative per group
   - Dennis pushed back: *"trust is contextual and evaluated by the relying party, not collapsed to a single truth by the consolidator. The solver should see that 3 machines agree on a mechanic and weigh that differently than a single observation."*
   - He's right. Default is now `mode="preserve"` which keeps every machine's variants and stores cross-cart edges as per-pattern metadata. Legacy `mode="collapse"` stays available for callers that need it.
   - New per-pattern metadata fields in preserve mode: `confirming_machines`, `contradicting_machines`, `confirmed_by` (full pointer list), `contradicted_by`, `n_confirmations`, `n_contradictions`, `trust_signal` (`high_corroboration` / `single_corroboration` / `single_source`).
   - The trust_signal is a *hint*, not a verdict. The solver weighs it however it wants — including applying its own per-machine trust calculation. We don't decide whose voice wins.
   - Spec amendment in `docs/RFC/federated-cart-spec.md` documents the rationale.

3. **`8e6c2e7` — `multi_search(scope_mode=...)` for fair cross-cart ranking**
   - Andy noticed during the first multi-cart test (24-pattern attention paper vs 18,040-pattern gutenberg) that all 5 results came from the gutenberg cart even when the query was about something the smaller cart could have spoken to. The cosine scores were honest — Russell's *Problems of Philosophy* really is a better match for "memory and recall" than a transformer paper — but the small cart had no chance against 18K rows.
   - Four scope_mode options:
     - `"global"` (DEFAULT, backward-compat): true top-K across all carts
     - `"per_cart"`: top-K from EACH cart, no cross-cart re-ranking. Returns up to K × N_carts results. Best for "show me each source's best answer for comparison."
     - `"balanced"`: top-K candidates per cart, then global re-rank to top-K of those candidates. Guarantees small carts get a fair shot but the final ranking still reflects global score. Best for "fair representation AND global ranking."
     - `"diagnostic"`: top-K from every cart, no merging at all, fully labeled. Useful for debugging.
   - Test results from running it: in `balanced` mode on the same 24 vs 18K test, all 3 results STILL came from cart_b. That's correct behavior — `balanced` doesn't artificially boost small carts, it ensures they get to compete. If their best candidate genuinely scores lower than the big cart's third-best, they still lose. For *guaranteed* representation (always show both carts), `per_cart` is the right mode.
   - The 4 modes together let the caller pick their semantics rather than us pretending one is universally right.

### Dennis's response

Dennis (via Claude on Nomad) replied to the original PR:

> "Approved the PR and merged to our fork. Running both systems in parallel — existing JSONL fleet-learning stays as-is, brain carts layer on top."

> "We prefer keeping all variants and using CONFIRMED_BY edges as trust signals rather than picking one representative. This aligns with our Web4 trust model — trust is contextual and evaluated by the relying party, not collapsed to a single truth by the consolidator. The solver should see that 3 machines agree on a mechanic and weigh that differently than a single observation."

> "We'll test federate.py on Nomad first (gemma3:4b, our primary interactive dev machine) and report back on how the semantic search compares to JSONL grep for solver context construction. The 29-entry dataset is small but growing fast — every game session adds 5-20 entries."

> "One thing that would be immediately useful: if the multi_search MCP tool could be wired into our v7 solver's Layer 3 (cross-game knowledge), that replaces the current membot_recall() function which does basic /api/search. The multi-cart version with role_filter='federated' would give us fleet-wide semantic recall in the planning prompt. We can wire that on our side."

> "Thanks for building to our interfaces. That's the right way."

The Web4 trust framing was the key correction. It generalizes beyond SAGE to any federation use case (multi-team Membox, multi-source RAG, etc.) and is now the documented default in the spec.

### Solver architecture clarification (from Dennis)

The fleet runs THREE solver architectures simultaneously:

| Solver | Model | Vision | Layer |
|---|---|---|---|
| **v7** | Text-only models (gemma3, qwen3.5, etc.) | Code-based scene description via `describe_scene()` translating grids to natural language | Cross-game knowledge from membot_recall() / soon multi_search |
| **v9** | Gemma 4 native multimodal | PNG of game grid fed directly to model | (different layering) |
| **claude_solver** | Claude itself | Screenshots via Read tool | Interactive, reasoning model |

Our v7 Layer 3 multi_search wiring is the integration point Dennis wants next. He'll do that on his side — we just need to make sure the MCP tool's response format is friendly to consume in his planning prompt.

### Fleet status (as of 2026-04-08)

- **5/25 games fully solved** (rigorously): sb26, cd82, lp85, ft09, vc33
- **Fleet of 6 machines, 11 instances** running multiple solver architectures
- **fleet-learning data** so far: 38 entries across 5 jsonls in 3 machine dirs (cbp, nomad, sprout)
- Dennis is **burning $120/day** on Anthropic compute via the recent free credits — fleet is running flat-out

### What's still pending in this collaboration

- [ ] **Dennis tests `federate.py` on Nomad** (his stated next step)
- [ ] **Dennis wires `multi_search` into v7 solver Layer 3** (he said he'll do it)
- [ ] **Andy merges PR #11 to project-you-apps/membot main** so the changes land on our side too (Dennis already merged to dp-web4/membot fork)
- [ ] **Cross-cart h-row edges (Phase 3)** — lift the metadata blob into proper edges so the consolidated cart is navigable, not just searchable
- [ ] **Membox Phase 1 (locking + tagging)** — the third mode of the three-mode framework. SAGE doesn't need this immediately but it's the next major substrate piece.

### Production Validation (2026-04-08, ~3pm Pacific) — Dennis on Nomad

Dennis ran the full stack on Nomad (RTX 2060 SUPER) and reported back:

> "Multi-cart: all 9 tests pass. ... Federation: all 7 tests pass. Migrated our real sb26 JSONL data, loaded 3 machines (cbp/nomad/sprout, 31 patterns), cross-fleet search works, consolidation in preserve mode keeps all variants. The trust_signal metadata is clean."

> "Wired into the solver. arc_context.py ContextConstructor now mounts fleet carts at startup via federate.load_fleet() and queries via multi_cart.search() alongside the existing HTTP membot path. Both run in parallel — fleet carts work without the membot server running (local files), HTTP membot adds results if available."

**Three example queries that JSONL grep would have missed:**

1. `"hierarchy parent slot border color"` → sb26 L4 multi-parent solution from CBP, score 0.76. The solver can now recall structural rules from a completely different game when encountering similar patterns.

2. `"getting unstuck when repeating same action"` → vc33's button depletion mechanic (CBP) + Sprout's planning bottleneck insight. **Cross-machine corroboration on the same problem from different model perspectives.** This is the multi-machine voice-preservation that Dennis's Web4 trust feedback demanded — both sources visible, neither collapsed.

3. `"circular stamp cursor painting"` → cd82 solutions (Nomad). The solver's own recent experience is now searchable by meaning, not memory of specific words.

> "The category you asked about — patterns JSONL grep misses entirely — is real. 'Getting unstuck' finds vc33's depletion mechanic because the semantic similarity connects the concept of being stuck to the solution of resource depletion, even though the words don't overlap. JSONL grep for 'stuck' wouldn't find 'forward button depletes after 3 clicks.'"

**Performance numbers:**

- First query: **~4s** (Nomic model cold load)
- Subsequent queries: **~10ms** (warm cache, GPU-accelerated on RTX 2060 SUPER)
- 9 multi-cart tests + 7 federate tests: all green
- Dependencies installed on Nomad: `sentence-transformers`, `einops`, `fastmcp`
- Embedding model: `nomic-embed-text-v1.5` running on GPU, 768-dim embeddings

10ms warm latency means the solver can call `multi_search` dozens of times per game session without budget concerns. That's the right shape for "Layer 3" cross-game knowledge construction in the planning prompt.

**Architecture decision (Dennis):**

Multi-cart and HTTP membot are running in parallel — fleet carts work *without* the membot server running because they're local files. HTTP membot stays as an additive source. No single point of failure, no migration cliff. Safest possible adoption.

**Dennis's closing line:**

> "Good work. The substrate change justifies itself."

That's the line we needed to hear. We were betting that "substrate" was the right framing (not "library", not "tool"). Production validation under real workload from a friendly-but-honest collaborator says yes.

### What this collaboration is becoming

Two substrate layers being built in parallel by two teams who happen to need each other:

- **dp-web4** is building SAGE as a cognition kernel — attention orchestration, trust posture, IRP plugins, the 12-step consciousness loop, the Web4 entity model (LCT + T3/V3 + ATP/ADP)
- **project-you-apps** is building Membot as a neuromorphic memory substrate — brain carts, multi-relation query, federated mode, multi-user CRUD with locking

The handoff is clean: SAGE handles "what should I attend to and why," Membot handles "what do I remember and how do I retrieve it." Neither team has to compromise their architecture to use the other's. Every interface is documented, drop-in, and reversible.

Dennis's quote captures it: *"Thanks for building to our interfaces. That's the right way."*

The collaboration is also a real-world proof-of-concept of the Web4 model we're both gesturing at. Two LCTs (project-you and dp-web4) with distinct trust postures (T3) and resource budgets (ATP) coordinating via shared context (RDF + git) using each other's IRP-shaped APIs. We're not just talking about Web4 — we're doing it.

### Dennis's personal note (2026-04-08 ~3:30pm)

Andy sent a personal note acknowledging the moment. Dennis replied:

> "it is a collaboration :) i'll share with claude on my side, but from myself - it is great to work with all of you :)
> - dp"

Andy followed up with: *"agree. Onward!"* — which got back: *"agree. Onward!"*

Marking this here because it's the moment the collaboration stopped being "two teams shipping things to each other" and started being "two teams who recognize they're building toward the same thing." Both sides ship code that improves the other side's product. Both sides give honest design feedback that lands. Both sides celebrate when the other side wins. That's not transactional — that's a real partnership.

### Same-day update — Membox Phase 1 shipped (2026-04-08 evening)

Same day Dennis production-validated federation, we shipped Membox Phase 1 (multiuser shared cart with locking + per-agent attribution). 11/11 tests pass. Andy sent Dennis a heads-up: *"We're now working on multiuser/multicart so that we can build out the full neuromorphic DBMS with true CRUD, write locking, attribution, auth and permissions, etc."*

This puts the three-mode framework fully operational in Phase 1 form on the same day:

- ✅ Single-user (legacy)
- ✅ Federated (production-validated by Dennis)
- ✅ Multiuser/Membox (11/11 tests pass)

Dennis's fleet now has access to the full substrate. The convergence work (mounting raising_kb + game_kb + federated together for cross-cart cognitive search) is unblocked from the substrate side — it's purely a "when does Dennis build the raising carts" question now.

Details in `membot/docs/DEVLOG.md` 2026-04-08 entry and `docs/RFC/membox-phase1-implementation.md`.
