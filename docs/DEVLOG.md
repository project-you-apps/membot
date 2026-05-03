# Membot Devlog

## 2026-05-03 — Provenance feature shipped on the public arxiv demo (RAG+)

A morning-into-afternoon polish session on the public Membot demo at https://www.project-you.app/membot/app turned into a substantial product upgrade. Headline result: Membot now demonstrates **three-tier transparent provenance** as a first-class feature. Pitch line: *"Vector Plus is RAG+."*

### What landed (in deploy order)

1. **Demo header tagline + intro** — "Self-contained semantic search without the LLM. Click a cart to mount it, then search with plain English — no generation, no hallucinations." Plus haystack-size reframe in the result-meta strip ("8 of 2,400,000 results in 1497ms"). Commit `6e2c0ce`, drop-max-width follow-up `9f9a3b1`.
2. **Hamming-popcount recovery** — discovered during the deploy that the chunked-popcount + lookup-table optimization was hot-patched on the droplet on 2026-05-02 and never committed back. Pulled it from the droplet backup, spliced into local, committed cleanly. Commit `63053c5`. New rule filed (see Discipline below).
3. **Provenance wiring v1** — REST `/api/search` taught to fetch full passages from the SQLite sidecar via `_sqlite_fetch_passages` for split carts; new `_soft_truncate` helper for sentence-boundary card text. Returns `paper_id` + `source_db` per result. Commit `775020e`.
4. **`numpy.int64` SQLite binding fix** — sqlite3 silently no-matches on `np.int64` indices. `[int(i) for i, _ in boosted[:top_k]]` cast at both REST and MCP search sites. The provenance wiring landed but returned empty until this fix. Commit `c543e08`.
5. **Source-DB label fix** — `state.get("cartridge","")` was returning empty (real key is `cartridge_name`); now uses `os.path.basename(state["sqlite_db_path"])` so the label is the actual filename (e.g. `arxiv_2400k_text.db`). Commit `caa69f9`.
6. **Two-step UX pivot** — Andy's reframe mid-session: rather than auto-fetch the source DB on search (hides the cart vs DB distinction), make every DB consultation a **labeled user click**. `/api/search` now returns the in-cart preview only + a `source_db` hint; `/api/passage` does the SQLite fetch on demand and returns the full passage + `paper_id`. Modal opens instantly with the preview, shows a CTA `📂 Load full passage from <db> →`, click → swap to full content + source line at bottom. Self-documenting architecture. Commit `279cc45`.
7. **Copy fix + BUGS.md** — CTA reads "Load full **passage**" (was "document") to generalize across cart types. New `BUGS.md` in repo root with severity classes (A/B/C) and the first two open bugs filed (Gutenberg PREV/NEXT discontinuity, cart-mounting UX feedback). Commit `44adc8a`.

### Architectural reframe (the real win)

The split-cart pattern (NPZ index in RAM + SQLite on disk + external publisher URL) was originally a RAM-pressure workaround for the 2 GB droplet. Andy reframed it mid-session as a positioning win: **three honest tiers of provenance, all visible, all clickable.** Most vector-RAG demos show a chunk with no source visible — the demo now shows the chunk, the labeled DB it came from, the paper ID, and the publisher URL. The two-step UX pivot makes it self-documenting: every DB hit is a user action, not a hidden side effect.

CC filed: `memory/concept-clusters/CC_provenance-as-feature_2026-05.md`. This is the version that goes in the next pitch deck.

### Discipline rule filed

Hot-patching the droplet without committing back to the local repo was the root cause of the Hamming regression scare. New feedback memory: `feedback_no_droplet_hot_patches.md`. Edit local → test → commit → push → scp → restart. If `git diff` doesn't show it, it doesn't exist. Andy: *"I need to get more tidy about this so we don't waste work in the future!"*

### Known issues (filed in BUGS.md)

- **B-high** — Gutenberg-classics PREV/NEXT chunks read as discontinuous, suspect cart-build chunker truncates tail of each chunk.
- **C** — Cart-mounting UX: highlight should move on click (not on mount-complete), search input should disable during mount.

### Pipeline metric for the day

Seven membot commits, all deployed, all verified. The discipline pipeline (edit local → test → commit → push → scp → restart) ran cleanly on every push after the first hot-patch-recovery confusion. The follow-up "drop max-width" round took ~90 seconds end-to-end, validating that the discipline isn't slow once you're in the rhythm.

---

## 2026-04-08 (afternoon) — Membox Phase 1 SHIPPED — All Three Modes Live

The third mode of the three-mode framework is now operational. Membot is the first **neuromorphic database** with single-user, federated multi-machine, AND multiuser-shared modes all running on the same substrate.

### What landed

- **`membot/membox.py`** — `CartLock` (write mutex with lease-based crash recovery), `imprint(cart_id, text, agent_id, ...)` convenience API, `search(cart_id, query)` (delegates to multi_cart, enriches results with `membox_meta`), `mount/unmount/list_mounts/status`. Reads never block on writes — classic many-readers-one-writer.
- **`multi_cart.imprint_with_meta(cart_id, text, per_pattern_meta)`** — the general-purpose runtime imprint API used by Membox (and any future write path). Embeds new text, appends to in-memory state, persists the cart preserving hippocampus + pattern0 + per_pattern_meta + sign_bits + version. `_persist_cart()` is the on-disk write helper.
- **9 new MCP tools**: `membox_mount`, `membox_unmount`, `membox_list`, `membox_imprint`, `membox_search`, `membox_acquire_lock`, `membox_release_lock`, `membox_lock_holder`, `membox_status`.
- **`tests/test_membox.py`** — 11 tests, all pass first run. Mount, single-agent imprint, lock holder query, lock timeout under contention (~205ms), invalid release raises PermissionError, read never blocks (28ms while lock held), two-agent concurrent imprint via Python threads (both serialize correctly with unique local_addrs and proper attribution), agent_id stamping survives in per_pattern_meta, status reports per-agent write counts, lease-based crash recovery.
- **Spec docs**: new `docs/RFC/membox-phase1-implementation.md` (concrete implementation spec), Amendment 2 added to `docs/RFC/membox-multiuser-dbms-spec.md` (Phase 1 SHIPPED marker + test results).
- **README** "What's New" section updated to mark Membox Phase 1 SHIPPED with code example.

### Performance numbers (RTX 4080 Super local)

- First imprint: 2,131ms (Nomic cold load)
- Subsequent imprints: ~19ms (warm cache + append + persist)
- Read while write lock held: 28ms (zero blocking)
- ~50 writes/sec per agent under contention

### Why this matters

Multi-cart query (foundation) + Federation (multi-machine) were powerful but invisible to a non-technical observer. Membox is the **product unlock** because it has a 90-second visible demo: two textareas, two agents, both writing to the same cart with serialized attribution, lock indicator showing who's holding the write lock right now. That's the demo an angel investor can grasp in 30 seconds because they've used Google Docs.

The pitch becomes "Google Docs for AI memory." Phase 2 (version chains + dispute detection) adds the conflict-resolution UI moment. Phase 3 (permissions) makes it multi-tenant. Phase 4 (admin agent) handles HITL workflows. Phase 5 (VPS integration) makes it the visible product.

Andy's framing: *"This multi-user/multi-cart feature that gives us true CRUD suddenly puts project-you over the top. And it potentially moves VPS into being an actual product I can pitch to angels that they might understand BY SHOWING IT TO THEM DIRECTLY."*

### Status of all three modes

- ✅ **Single-user** (legacy Membot, untouched)
- ✅ **Federated** (Phase 1 shipped 2026-04-08 morning, production-validated by Dennis on Nomad — see `DEVLOG-SAGE-COLLAB.md`)
- ✅ **Multiuser/Membox** (Phase 1 shipped 2026-04-08 afternoon, 11/11 tests pass)

PR #11 on `project-you-apps/membot`, branch `claude/federate-phase-1`, will get a new commit containing all of this Membox work.

---

## 2026-04-08 — Multi-Cart Phase 1 + Federate Phase 1 + scope_mode polish

The substrate change that turns Membot from a single-relation library into a multi-relation neuromorphic database. Three commits on PR #11 (project-you-apps/membot, branch `claude/federate-phase-1`):

### multi_cart.py (the foundation)

Process-global cart pool with namespaced query and result attribution. Existing single-cart per-session API stays unchanged.

```python
multi_mount("./identity.cart", cart_id="me", role="identity")
multi_mount("./gutenberg.cart", cart_id="lib", role="reference")
multi_mount("./fleet/cbp.cart", cart_id="cbp", role="federated")

multi_search("uncertainty tolerance", scope="all")
# → [me#42], [lib#9131], [cbp#3]  ← attributed to source carts
```

API: `mount`, `unmount`, `list_mounts`, `mount_directory`, `search`, `get_cart`, `total_patterns_mounted`, `unmount_all`.

Search supports `scope` (`"all"` / `"local"` / cart_id / list) and `role_filter` (one role string or list). 5 new MCP tools expose the API.

Validated by `tests/test_multi_cart.py` (10 tests, all pass against real cartridges).

### federate.py (drop-in for Dennis Palatov's SAGE federated learning)

Built on multi_cart. Same git-sync model and per-machine append-only writes as Dennis's existing JSONL architecture, brain carts as the substrate.

- `publish_session(session_file, machine_id, fleet_dir)` — replaces his `publish_learning.py`
- `consolidate(fleet_dir, output_dir, mode='preserve')` — replaces his `consolidate.py`
- `migrate_jsonl(jsonl_dir, in_place=True)` — one-time JSONL → cart migration (non-destructive)
- `load_fleet(fleet_dir)` — solver entry point: mount every machine's cart with `role='federated'`

`consolidate()` defaults to `mode='preserve'` (per Dennis's Web4 trust feedback): keeps every machine's variants and stores cross-cart edges as per-pattern metadata. The legacy `mode='collapse'` is still available for callers that need a smaller cart and don't care about per-machine attribution.

Per-pattern metadata in preserve mode includes `confirming_machines`, `contradicting_machines`, `confirmed_by`, `contradicted_by`, `n_confirmations`, `n_contradictions`, `trust_signal` (`high_corroboration` / `single_corroboration` / `single_source`). Trust signals are *hints*, not verdicts — the solver weighs them however it wants.

4 new MCP tools: `federate_publish`, `federate_consolidate`, `federate_migrate_jsonl`, `federate_load`.

Validated by `tests/test_federate.py` (8 tests, all pass against Dennis's real `sb26_learning.jsonl`).

### scope_mode polish

`multi_search` now takes a `scope_mode` parameter for fair cross-cart ranking when carts are different sizes:

| scope_mode | Behavior | Use when |
|---|---|---|
| `"global"` (default) | True top-K across all carts. Backward-compatible. | "What's the single best answer regardless of source?" |
| `"per_cart"` | Top-K from EACH cart, no cross-cart re-rank. Up to K × N_carts results. | "Show me each source's best answer for comparison." |
| `"balanced"` | Top-K candidates per cart, then global re-rank to top-K. | "Fair representation AND global ranking." |
| `"diagnostic"` | Top-K from every cart, no merging, fully labeled. | Debugging cross-cart search behavior. |

`per_cart` and `diagnostic` also return a `grouped_results` dict in the response keyed by cart_id.

Real-world test: 24-pattern attention paper cart vs 18,040-pattern gutenberg cart, query "memory and recall":
- `global` mode: all 5 results came from gutenberg (Bertrand Russell on memory dominated)
- `per_cart` mode: 6 results, 3 from each cart (both represented)
- `balanced` mode: 3 results, all from gutenberg (the attention paper's best candidate genuinely scored lower than gutenberg's third-best — balanced doesn't artificially boost, it ensures fair *opportunity*)
- `diagnostic` mode: 6 results fully grouped by source

The four modes give the caller their choice of semantics. Spec amendment in `docs/RFC/multi-cart-query-spec.md`.

### What this unblocks

- **Federated mode for SAGE**: Dennis already merged the PR into dp-web4/membot fork and is testing on Nomad
- **Cart-per-cognitive-function**: a Membot instance can hold raising_kb + game_kb + working_memory + identity carts simultaneously, queried as one substrate
- **Membox Phase 1** (next): the third mode of the three-mode framework — multiuser CRUD with locking + tagging
- **The "neuromorphic DBMS" launch positioning**: with multi-cart + federation real, the framing is no longer aspirational

### Files added or changed

```
multi_cart.py                          (new, ~450 lines)
federate.py                            (new, ~700 lines)
membot_server.py                       (+9 MCP tools, ~400 lines)
tests/test_multi_cart.py               (new, ~200 lines)
tests/test_federate.py                 (new, ~200 lines)
tests/README.md                        (new, conventions)
.gitignore                             (test artifact patterns)
docs/RFC/multi-cart-query-spec.md      (new, foundational spec + Amendment 1)
docs/RFC/federated-cart-spec.md        (new, federation spec + Amendment 1)
docs/RFC/membox-multiuser-dbms-spec.md (new, v2 amendments folding modes)
docs/DEVLOG-SAGE-COLLAB.md             (new, sustained collab narrative)
README.md                              ("What's New" section)
```

PR: https://github.com/project-you-apps/membot/pull/11

Detailed collaboration narrative: `docs/DEVLOG-SAGE-COLLAB.md`

---

## 2026-04-01 — Membox RFC: Multi-User Neuromorphic DBMS

Concept crystallized: extend Membot from single-user carts to multi-user DBMS with CRUD, locking, versioning, access control, and conflict resolution. Append-only version chains (Git for memories), DISPUTED flags for semantic conflicts, membox.txt for default permissions, admin agent template for automated governance. VPS = human frontend, MCP = agent backend, same cart.

Full RFC: `docs/RFC/membox-multiuser-dbms-spec.md`
Dedicated devlog: `docs/DEVLOG-MEMBOX.md`

---

## 2026-03-24 — Wiki Brain Cart Deployed, Writable Agent Workspace, TUI Session Persistence Gap

### Wiki 2.4M Brain Cart — DEPLOYED TO DROPLET
- Built split cart from physics-trained wiki brain: `wiki_2400k_brain_index.npz` (380 MB) + `wiki_2400k_brain_text.db` (1.3 GB)
- Added Wikipedia URLs to SQLite: `https://en.wikipedia.org/wiki/{Title}` auto-generated from first line of each passage
- Uploaded both files + updated `membot_server.py` to droplet
- Both arXiv and wiki carts live — **4.8 million entries** on a $12/mo droplet
- Constraint: only one cart mounted at a time on 2 GB RAM (both indexes = ~760 MB > available)

### Droplet Port Map (finalized)
| Port | Service | Auth | Status |
|------|---------|------|--------|
| 8000 | Membot read-only | None | Running |
| 8040 | Membot writable | API key | Stopped (needs 4 GB upgrade) |
| 8100 | Session memory | Bearer token | Running |

The 804x convention: base port + 40 = writable/agent variant. Future: 8140 for session memory agent variant.

### TUI Session Persistence — GAP IDENTIFIED
OpenClaw TUI stores session transcripts as JSONL at `~/.openclaw/sessions/<session-id>/transcript.jsonl`. Same format family as Claude Code sessions. But no scanner wired up yet — TUI agent research is ephemeral.

**Plan:** Build independent TUI scanner (separate from Claude Code scanner — streams must never cross). Points to own cart `tui_sessions.pkl`. Same embed + append pipeline, different source directory.

### Competitive Analysis: Letta/Claude Subconscious
- Letta (the MemGPT team) shipping a Claude Code plugin with background agent, 8 structured memory blocks, async transcript processing via lifecycle hooks
- "Whisper" mode injects suggestions as observations, never blocks Claude
- No physics, no portable artifacts, Claude Code only
- Sixth competitor surveyed (Supermemory, virtual-context, claude-mem, Interloom, Letta)

### Interloom — $16.5M Seed for Enterprise Process Mining
- Munich startup building "Context Graph" from enterprise docs (emails, tickets, transcripts)
- Captures tacit knowledge (70% of operational decisions never documented)
- Commerzbank: reduced doc/reality gap from 50% to 5%
- We *could* do this with our ingestion pipeline + physics association for a fraction of the cost. But enterprise sales is a different game. Filed under "arrows in quiver."

### Neuro-Symbolic Energy Paper — Ammunition for Pitch
Duggan et al. (arXiv:2602.19260): neuro-symbolic methods outperform VLAs at 100x less energy, 95% vs 34% success, 34 min vs 1.5 day training.

Direct parallel to our architecture:
- Their PDDL symbolic planning = our attractor dynamics + SimHash structure
- Their learned diffusion policies = our Hebbian-learned associations
- Their 80x energy savings = our LLM-free pipeline

**Key quote:** "Explicit symbolic structure improves reliability, data efficiency, and energy consumption."

Our pipeline is LLM-free end to end. Training energy for 2.4M brain cart: ~8.8 MJ estimated (5.75 hrs × 320W GPU). Competitor equivalent would require millions of LLM calls for fact extraction, relationship building, and compaction.

### Auto-Append Status Discrepancy — RESOLVED
MCP `session_status` reported `Auto-appends: 0` but the actual server console showed active appending:
```
[AutoAppend] grew: 2026-02-26 | b6a4d6a9... | +1 passages (was 2490)
[AutoAppend] Done: +1 passages (111ms embed, 4.9s total). Cartridge now has 15531 passages.
```
The counter in the status endpoint may reset on server restart or not track correctly. The pipeline IS working — exchanges flow into the cart via the JSONL scanner. False alarm escalated and corrected.

### Session Startup Protocol — NEW
Added `feedback_session_startup.md` to memory: at each session start, check the hot stack and read the most recent journal entry. Goal: continuity of *thought*, not just continuity of *facts*. Evolves toward REMINDS_OF triggering journal associations organically.

---

## 2026-03-23 — Split Cart Format, 2.4M ArXiv on Droplet, First Agent Research Session

### Split Cart Format — NEW FEATURE
Invented a two-file cartridge format to solve the RAM problem for large-scale deployment.

**The problem:** 2.4M passages = ~2.8 GB of Python strings in RAM. A $12/mo droplet (2 GB) can't hold it.

**The solution:** Separate the search index from the text.

| File | Format | Size (2.4M) | Lives in | Purpose |
|------|--------|-------------|----------|---------|
| `_index.npz` | NumPy compressed | ~377 MB | RAM | Sign bits + 200-char snippets + metadata |
| `_text.db` | SQLite | ~1-4 GB | Disk | Full passages, paged on demand per query hit |

Hamming search + keyword reranking run entirely in RAM against the index. Only the top-K display results trigger SQLite lookups (~1ms each). 88% RAM reduction vs loading everything.

SQLite needs no server — Python's built-in `sqlite3` module reads the `.db` file directly.

**Tools built:**
- `build_sqlite_cart.py` — general-purpose split cart builder (any .pkl → index + SQLite)
- `compress_cart.py` — general-purpose zlib cart compressor (any .pkl)

### Server Changes
- `load_npz_cartridge`: detects `has_sqlite=True`, loads snippets, opens SQLite sidecar
- Mount: opens SQLite connection, stores in session state
- Search: snippets for keyword reranking (RAM), full text from SQLite (disk) for display only
- Unmount: closes SQLite connection cleanly
- Search mode label: `hamming-only+kw+sqlite`
- Fixed OOM on packed Hamming search: replaced `np.unpackbits` (expanded 2.4M×96 → 2.4M×768 = 1.72 GiB) with byte-level popcount lookup table. O(N×96) instead of O(N×768).

### ArXiv 2.4M Cart — LIVE ON DROPLET
- `arxiv_2400k_index.npz` (360 MB) + `arxiv_2400k_text.db` (3.5 GB)
- 2,400,000 arXiv paper abstracts with titles, categories, and arXiv URLs
- Searchable on the $12/mo droplet with 377 MB RAM usage
- Search: ~9-18s latency (brute-force Hamming, fixable with FAISS binary index)

### Wiki 2.4M Brain Cart — LIVE ON DROPLET
- `wiki_2400k_brain_index.npz` (380 MB) + `wiki_2400k_brain_text.db` (1.3 GB)
- 2,400,000 Wikipedia articles with auto-generated Wikipedia URLs
- Physics-trained (3 passes V7.5 Hebbian learning) — sign bits encode attractor topology
- Note: only one cart at a time on 2 GB droplet; both indexes exceed available RAM

### Writable Agent Workspace
- New systemd service `membot-writable` on port 8040
- API key protected (`waving-cat-agents-2026`)
- Shares `cartridges/` with read-only instance — agents can search big carts AND create their own
- Currently stopped (RAM constraint) — needs 4 GB droplet upgrade

### First Live Agent Research Session — MILESTONE
OpenClaw TUI (Sonnet) autonomously:
1. Mounted `arxiv_2400k_index` on the live droplet
2. Searched "attractor networks" — found Hopfield network papers
3. Fetched and summarized "Vector Symbolic FSMs in Attractor Neural Networks"
4. Discovered our sign-zero encoding is **SimHash / Hyperplane LSH** (Charikar 2002) — established prior art
5. Identified our novelty boundary: SimHash on raw embeddings = known; SimHash on physics-shaped attractor states after Hebbian training = novel
6. Wrote a full benchmark methodology document (`physics-lsh-benchmark-methodology.md`) with 15+ citations from the arXiv cart
7. Read and analyzed "The Price Is Not Right" (Duggan et al., arXiv:2602.19260) — neuro-symbolic methods outperform VLAs at 100x less energy
8. Connected the paper's findings to our architecture

**Key research finding:** Our sign-zero encoding is SimHash (Charikar 2002). Cannot patent the encoding itself. Our novelty is combining it with Hebbian settle dynamics — "SimHash on attractor states" — which nobody else does.

### Competitive Analysis

**Supermemory, virtual-context, claude-mem, Letta/Subconscious** — all surveyed. All require LLM calls for memory operations (fact extraction, relationship building, compaction). We don't.

**The insight:** Our entire memory substrate is LLM-free. Build, store, search, recall — no tokens consumed. The only LLM is the agent on the other end deciding what to search for.

> "The only AI memory system that doesn't need AI to run."

**Comparison at 2.4M scale:**

| | Pinecone | FAISS (IVF+PQ) | Supermemory | Our split cart |
|---|---|---|---|---|
| Index size | ~7 GB (cloud) | ~300-600 MB | Cloud | 377 MB |
| Search latency | ~50ms | ~5-20ms | ~100ms | ~13s (fixable) |
| LLM in pipeline | No | No | Yes (every op) | No |
| Association | No | No | No | Yes (physics) |
| Monthly cost | $70+ | Self-hosted | Usage-based | $12/mo |

### Troubleshooting Docs Added
- "Session not found" (-32600) error after server restart/crash: documented cause and fix for OpenClaw, Claude Code, Claude Desktop
- Large cart OOM crash: documented prevention via split carts

### README Updates
- Split cart format section with tables, build instructions
- Updated brain cartridge file table to include split format
- Troubleshooting section

### Three Commits Pushed
1. `1e4f82b` — Split cart format: SQLite sidecar for low-RAM deployment
2. `eb89095` — Fix OOM on packed Hamming search: popcount lookup table
3. `955fac6` — Add troubleshooting section: session errors, OOM on large carts

---

## 2026-02-08 — Initial Release + OpenClaw Integration

### What Shipped
- Membot repo created at `github.com/project-you-apps/membot`
- 7 MCP tools: list_cartridges, mount_cartridge, memory_search, memory_store, save_cartridge, unmount, get_status
- SentenceTransformer nomic-embed-text-v1.5 (matches Studio cartridge embedder exactly)
- 70/30 physics+embedding blend with keyword reranking (GPU), embedding-only fallback (CPU)
- Security: NPZ-first saves, PKL sandboxing, SHA256 manifests, input sanitization, resource caps
- Cartridge builder: `cartridge_builder.py` for building from .txt/.md/.pdf/.docx
- Sample cartridge: "Attention Is All You Need" (24 chunks, pre-built embeddings)
- Dual license: MIT for Python, proprietary for CUDA binary

### OpenClaw Integration (WSL Debian on PlateauofLeng)
- stdio transport working perfectly with OpenClaw TUI
- All 7 tools register as `membot_*` prefix
- Garfield physics proof: physics swapped rank #5 from generic "Barack Obama" to contextual "Assassination of James A. Garfield"
- 100k demo: 217ms embedding-only, ~9.6s physics blend

### OpenClaw Agent Dispatch — BROKEN
- TUI sessions get MCP adapter tools; headless `openclaw agent --agent X --message "..."` does NOT
- `tools.alsoAllow` in agent config read as `tools.allow` — "unknown entries" warning
- `tools` key NOT valid at `agents.defaults` level (only per-agent)
- Tested 7+ rounds across multiple config approaches — confirmed limitation of openclaw-mcp-adapter plugin (210 lines TS by androidStern)
- Workaround: use TUI for demos

### Embedding Mismatch Bug (FIXED)
- Root cause: Originally used Ollama for query embeddings, but cartridges built with SentenceTransformer
- Same model weights != same vectors (different inference paths produce subtle drift)
- Ambiguous queries (like "Garfield") broke; strong queries masked the problem
- Fix: Switched server to SentenceTransformer to match Studio exactly
- Lesson: NEVER mix embedding models between build and query

### Key Decisions
- Physics confidence gating: sig_std only (removed overlap check — zero overlap at 100k is expected for cross-domain queries)
- `PHYSICS_ENABLED = True` (re-enabled after embedder fix)
- Versioned PKL format (v7.0-8.3) loading with nested `data["data"]` unwrapping

---

## 2026-02-09 — HTTP Transport + DO Deployment

### HTTP Transport Added
- Switched import: `from mcp.server.fastmcp` → `from fastmcp` (standalone FastMCP 2.14.5)
- New CLI args: `--transport stdio|http|sse`, `--host`, `--port`
- Default: stdio (backward compatible with OpenClaw, Claude Desktop)
- HTTP mode: `python membot_server.py --transport http --port 8000`
- Endpoint: `http://{host}:{port}/mcp` (Streamable HTTP protocol)

### Middleware (HTTP mode only)
- **Rate limiting**: 60 requests/minute per client IP, sliding window
- **API key auth**: `MEMBOT_API_KEY` env var → requires `Authorization: Bearer <key>` header
- No env var = no auth (for local testing)

### Local Testing (WSL)
- FastMCP 2.14.5 installed in `~/vplus/venv/`
- Server starts clean: CUDA V7 ready, 5 cartridges found, Uvicorn on 0.0.0.0:8000
- curl handshake: MCP initialize returns `"name":"Membot"` with all capabilities
- FastMCP Client round-trip: `list_cartridges` returns all 5 carts over HTTP

### DO Droplet Deployment (DONE)
- Droplet: SHACKERS-1, 137.184.227.79, Debian 10, 2 GB RAM, 50 GB disk
- Hosts andygrossberg.com and project-you.app (can't rebuild)
- Added 2 GB swap (SentenceTransformer needs ~1.5 GB)
- Debian 10 repos EOL'd — switched to archive.debian.org
- Python 3.7.3 too old — built Python 3.11.8 from source (`make altinstall`)
- Venv at `/opt/membot/venv/` with fastmcp, sentence-transformers, numpy
- Strategy: CPU-only (embedding search), PlateauofLeng keeps GPU for physics demos
- Demo cartridge deployed: attention-is-all-you-need (24 memories, 0.1 MB)
- systemd service: `membot.service`, auto-start on boot, auto-restart on crash
- API key auth via `MEMBOT_API_KEY` env var in systemd unit

### Remote Testing (DONE)
- FastMCP Client from WSL → droplet: list_cartridges, mount, search all working
- Claude Code connected as remote MCP client via user config (`~/.claude.json`)
- Full round-trip: list → mount → search → store → search again → save
- First search: ~13s (model cold load on 2 GB RAM), subsequent: ~131ms
- Membot tools appear natively in Claude Code sessions

### Read-Only Mode
- `--read-only` flag disables `memory_store` and `save_cartridge`
- For public-facing server: users can browse and search, not write
- Architecture: public server is read-only dispensary, local Membot is full CRUD
- Users build cartridges locally, upload finished ones to public server

### Architecture Decision: No Gateway Needed
- For remote hosting, Membot IS the server — runs Streamable HTTP directly via Uvicorn
- OpenClaw gateway only relevant for local TUI use
- Any MCP client (Claude Code, Claude Desktop, custom agents) connects directly to HTTP endpoint
- This is the "agents come get carts" vision

### Ideas Filed (Future)
- **Hippocampus as graph nodes**: Reserve columns for parent/child/sibling pointers, turning each pattern into a linked list node. Physics finds implicit associations; linked list encodes explicit relationships. Use case: session memory trees, taxonomies, conversation threads.
- **LatticeRunner as LLM session memory**: Claude Code journals key moments into a cartridge during work, queries it after compaction. Associative recall via physics, not just keyword grep on flat text. Challenge: remembering to journal (needs hook or MEMORY.md instruction).
- **Fork openclaw-mcp-adapter**: Only 210 lines TS. Could fix agent dispatch tool provisioning. Low priority vs HTTP transport.

---

## 2026-02-09 (Evening) — OpenClaw Adapter Deep Dive + Strategy

### Forked openclaw-mcp-adapter
- Fork at `github.com/project-you-apps/openclaw-mcp-adapter`
- Original: `github.com/androidStern-personal/openclaw-mcp-adapter` (MIT, 10 stars, 2 forks)
- Three files, ~210 lines total: `index.ts` (60), `mcp-client.ts` (108), `config.ts` (51)
- Supports both stdio and HTTP (StreamableHTTP) transports
- Uses `@modelcontextprotocol/sdk` ^1.0.0

### Root Cause Analysis — Agent Dispatch Tool Visibility
- **Confirmed**: `api.registerTool()` registers at **gateway runtime level only**
- **TUI sessions** share the gateway runtime → see plugin-registered tools
- **Agent dispatch** (`openclaw agent --agent X --message "..."`) builds tool set from **core tool registry only** — doesn't query plugins
- **ACP layer explicitly disables MCP** — `mcpCapabilities` sets HTTP and SSE to `false`, MCP servers passed during session creation are silently ignored
- **`tools.alsoAllow`** doesn't work for plugin tools — gets read as `tools.allow` and fails with "unknown entries"
- **Native MCP support** (Issue #4834) was **closed as "not planned"** on 2026-02-01
- This is an OpenClaw core limitation, not an adapter bug

### mcporter Investigation
- **What it is**: Standalone CLI + OpenClaw built-in skill by steipete (Peter Steinberger)
- **Repo**: `github.com/steipete/mcporter`
- **How it works**: OpenClaw shells out to `mcporter` binary per tool call — no persistent connection
- **Key insight**: mcporter works because it's registered as a **skill**, not via `api.registerTool()`. Skills ARE visible to agent dispatch; tools are not.
- **Config**: Supports HTTP transport — `"baseUrl": "http://137.184.227.79:8000/mcp"` in mcporter.json
- **Overhead**: ~2.4s cold-start per call (spawn, connect, JSON-RPC init, call, exit)
- **Verdict**: Works but slow. Every tool call is a fresh subprocess + connection.

### The Real Fix — Skills vs Tools
- **Skills** are visible to agent dispatch. **Tools** (via `api.registerTool()`) are not.
- If the adapter registered MCP tools as **skills** instead of tools, agent dispatch would see them
- This keeps the adapter's persistent connection advantage (no cold-start per call)
- Potential fix: find `api.registerSkill()` or equivalent in OpenClaw plugin-sdk
- The plugin-sdk is bundled/minified at `dist/plugin-sdk/index.js` (~18K lines) — undocumented

### Strategic Decision
- **For Membot's use case**: HTTP transport to Claude Code already works (proven today). OpenClaw is a visibility/hype play, not a technical necessity.
- **For the community**: The skills-vs-tools insight is the real contribution. A PR that registers MCP tools as skills would fix the adapter for everyone.
- **Next steps**: Dig into plugin-sdk for skill registration API, or file the issue with the analysis and let androidStern/OpenClaw core team respond.

### Picks and Shovels Positioning
- Every agent framework has the same unsolved problem: persistent, searchable, associative memory across sessions
- Brain cartridges are framework-agnostic — MCP over HTTP means any client can connect
- Don't care who wins the agent wars — Membot serves them all
- Physics search differentiates on ambiguous/cross-domain queries, not routine RAG
- Target users: researchers, analysts, investigators, creative teams — not db admins optimizing ticket routing

### Vector Plus Studio 1.0 Feature Spec
- React front end with full CRUD (CREATE via file ingestor, READ via mount+search, UPDATE with source-mismatch warning, DELETE with tombstones, RESTORE from tombstones)
- File picker for cart loading and document ingestion (txt, md, rtf, docx, pdf, xlsx)
- Hybrid cosine/physics search slider
- Document observer with expand preview, full page view, open-in-native-app
- Text+embedding verified association (legal/medical provenance)
- CPU fallback (no GPU required)
- Spec file: `vector-benchmark-demo/cuda/Vector_Plus_Studio_React_Front_End_Features_List.md`

### Plugin-SDK Deep Dive — No `api.registerSkill()` Exists

- **Skills are file-based only** — SKILL.md files in `~/.openclaw/skills/<name>/` or workspace `skills/`
- OpenClaw discovers them at startup, injects into system prompt as XML for ALL agent runs (including dispatch)
- Agent reads SKILL.md, learns what commands to run, shells out via `exec` or `mcporter call`
- **Every MCP skill in the OpenClaw ecosystem** works this way (clickup-mcp, guru-mcp, etc.)
- No programmatic `api.registerSkill()` — the plugin-sdk only has `api.registerTool()` (gateway-scoped)
- SKILL.md frontmatter supports: name, description, homepage, user-invocable, command-dispatch, metadata (requires.bins, requires.env)

### The Actual Fix — Create a SKILL.md (Not a Code Change)

- The adapter fork is NOT the right fix — the problem isn't in the adapter, it's in OpenClaw's architecture
- **Solution**: Create `~/.openclaw/skills/membot/SKILL.md` that teaches agents to call Membot via `mcporter call`
- mcporter supports HTTP transport: `mcporter call http://137.184.227.79:8000/mcp.memory_search query="..."`
- ~2.4s cold-start per call (subprocess spawn), but agent dispatch CAN see and invoke it
- This matches the blessed pattern used by every MCP skill in the ecosystem
- No adapter modification, no plugin-sdk hacking, no OpenClaw core PR needed

### What to Tell androidStern

- The adapter works perfectly for TUI — that's its intended scope
- Agent dispatch uses a separate tool provisioning path that doesn't include plugin-registered tools
- The OpenClaw-blessed workaround is SKILL.md + mcporter (file-based, not programmatic)
- Native MCP support (Issue #4834) was closed as "not planned" on 2026-02-01
- Suggestion: adapter README should document this limitation and point to SKILL.md pattern as alternative
- **Filed as Issue #6**: `github.com/androidStern-personal/openclaw-mcp-adapter/issues/6`

### mcporter Round-Trip Proven

- Installed mcporter via `sudo npm install -g mcporter` (118 packages)
- Configured server: `mcporter config add membot --url http://137.184.227.79:8000/mcp --header 'Authorization=Bearer ...' --transport http --scope home`
- Config saved to `~/.mcporter/mcporter.json`
- `--allow-http` only needed for ad-hoc URL calls, not configured servers
- **Full round-trip**: `mcporter call membot.list_cartridges` → 1 cartridge found
- **Mount**: `mcporter call membot.mount_cartridge name=attention` → 25 memories, verified
- **Search**: `mcporter call membot.memory_search query="multi-head attention mechanism"` → 5 results, 0.844 top score, 11.9s (cold model load)
- Bash `!` in API key causes history expansion — use single quotes or change to `1`

### Agent Dispatch — STILL BROKEN (Different Reason)

- SKILL.md created at `~/.openclaw/skills/membot/SKILL.md` — agent didn't see it
- SOUL.md created at `~/.openclaw/agents/research-bot/agent/SOUL.md` with explicit mcporter bash instructions
- CLAUDE.md also written to agent workspace — no effect
- **Agent keeps looking for native `membot_*` tools** instead of using Bash to run mcporter
- Possible causes: agent has cached conversation context from yesterday's failures, or agent dispatch doesn't inject SOUL.md from agentDir, or Bash tool not available in dispatch mode
- **mcporter itself works perfectly** — the problem is getting OpenClaw's agent dispatch to read the instructions and use Bash
- **Next step**: Fresh debug session — check if dispatch reads SOUL.md at all, check available tools in dispatch, try `--no-history` if it exists

### gh CLI Installed on Windows

- `winget install GitHub.cli` — GitHub CLI 2.86.0
- Located at `C:\Program Files\GitHub CLI\gh.exe`
- Authenticated via `gh auth login`
- Used to file Issue #6 on androidStern's repo

---

## 2026-02-10 — OpenClaw Agent Round-Trip Proven

### The SOUL.md Mystery

Found **three** SOUL.md files for research-bot on WSL:

| File | Contents | Problem |
| ---- | -------- | ------- |
| `~/.openclaw/agents/research-bot/SOUL.md` | Old instructions: call native `membot_*` tools | **Agent was reading this one** |
| `~/.openclaw/agents/research-bot/agent/SOUL.md` | Our new mcporter-based instructions | Correct but not taking priority |
| `~/.openclaw/workspace/SOUL.md` | Global OpenClaw personality ("You're not a chatbot...") | Clean, no membot refs |

**Root cause**: Agent dispatch reads the SOUL.md at the agent root level, not the `agent/` subdirectory. The old file told it to call `membot_list_cartridges` etc. as native tools — which don't exist in dispatch mode.

### The Fix: Merged SOUL.md

Combined behavioral guidance from the old SOUL.md with mechanical instructions from the new one:

- **From old**: "Use cartridge results first before your own knowledge", "cite your sources", "if the cartridge doesn't have what you need, say so", web search fallback, memory_store
- **From new**: Explicit Bash + mcporter commands, "Do NOT look for native membot tools" warning

Deployed merged version to both `research-bot/SOUL.md` and `research-bot/agent/SOUL.md`. Local copy at `membot/SOUL-research-bot-merged.md`.

### Clean First-Dispatch

```bash
openclaw agent --agent research-bot --message "Search the attention cartridge for information about positional encoding"
```

Agent immediately:

1. Mounted attention cartridge via `mcporter call membot.mount_cartridge name="attention"`
2. Searched via `mcporter call membot.memory_search query="positional encoding"`
3. Returned synthesized answer with sources cited (0.782 top score)

**No nudge needed.** First dispatch worked clean.

### Full Chain Proven

```text
OpenClaw agent dispatch
  → reads SOUL.md (mcporter instructions)
  → Bash tool: mcporter call membot.mount_cartridge
  → mcporter subprocess → HTTP POST to 137.184.227.79:8000/mcp
  → Membot mounts attention cartridge (25 memories)
  → Bash tool: mcporter call membot.memory_search
  → Membot embeds query → cosine search → returns 5 results
  → Agent synthesizes answer with citations
```

### Status

- **OpenClaw TUI**: Works via mcp-adapter plugin (native tools)
- **OpenClaw agent dispatch**: Works via SOUL.md + mcporter (Bash tool)
- **Claude Code**: Works via native MCP client (HTTP transport)
- **Remaining**: Change API key (`!` → `1`) on droplet and mcporter config
