# RFC: Membox — Multi-User Neuromorphic DBMS

**Date**: 2026-04-01
**Status**: Draft
**Authors**: Andy Grossberg, Claude (Opus 4.6)

---

## Overview

Membox extends Membot from single-user brain cartridges to a multi-user neuromorphic database management system. Multiple agents (and humans via VPS) share read-write access to the same carts with locking, versioning, access control, and conflict resolution.

No SQL. No LLM in the pipeline. Just physics, patterns, and permissions.

## Motivation

Current Membot supports:
- Read-only shared access (multiple agents search the same cart)
- Single-writer mode (one agent writes, no concurrency)
- Per-agent isolated carts (each agent gets its own instance)

What's missing: **multiple agents collaborating on the same cart simultaneously** with proper CRUD operations, access control, and conflict resolution. This is required for:
- Multi-agent teams sharing knowledge
- Human + agent collaboration (VPS frontend + MCP backend)
- The ARC-AGI-3 competition (SAGE + perception layer sharing game state)
- Enterprise deployment (teams of agents serving a department)

## CRUD Operations

| Operation | Mechanic | Permission |
|-----------|----------|------------|
| **Create** | New pattern appended. Agent_id + timestamp in h-row. | `write` |
| **Read** | Search + retrieve. No lock needed. | `read` (default public) |
| **Update** | Soft-write: new version created, linked to original via h-row `related` field. Old version gets SUPERSEDED flag. **No overwrites** — append-only with version chains. | `write` + `update` |
| **Delete** | Tombstone flag set in h-row. Pattern stays on disk, hidden from search. Undo = clear tombstone. | `delete` (admin or creator only) |

**Key principle: there are no overwrites.** Every "update" creates a new version linked to the old one. The cart is append-only with version chains. This is Git for memories.

## Write Locking

```python
class CartWriteLock:
    """Mutex on cart writes. Only one agent writes at a time."""

    def acquire(self, agent_id: str, timeout_ms: int = 5000) -> bool:
        """Attempt to acquire write lock. Returns False on timeout."""

    def release(self, agent_id: str):
        """Release the write lock."""

    def holder(self) -> Optional[str]:
        """Who currently holds the lock? None if free."""
```

- Lock is per-cart, not per-pattern
- Lock timeout prevents deadlocks (agent crashes while holding lock)
- Lock holder recorded in cart metadata for debugging
- Read operations never block (readers don't need the lock)

## Agent-Tagged Writes

Every write includes:
- `agent_id`: Who wrote this (from agents.json registry)
- `timestamp`: When (UTC, seconds since epoch)
- `origin`: "agent" | "human" | "system"
- `reasoning`: Optional text explaining WHY this was written

Stored in h-row metadata (pattern_id, timestamp, flags) and per-pattern metadata (agent_id, origin, reasoning in the `_reserved` dict).

## Conflict Resolution

### What is a conflict?

Not simultaneous writes (locking prevents that). A conflict is **semantic disagreement**: two agents update the same concept with different information.

Example:
- Agent A: "The project uses React"
- Agent B: "The project uses Vue"

### Resolution strategy: Store both, flag, link, defer.

```
Pattern #847 (original): "The project uses React"
  ├── Version by agent-A at 10:31: "The project uses Vue" 
  │     h-row flag: DISPUTED
  │     h-row related: → version by agent-B
  ├── Version by agent-B at 10:33: "The project migrated from React to Vue"
  │     h-row flag: DISPUTED
  │     h-row related: → version by agent-A
  └── Status: UNRESOLVED
```

### Resolution methods (configurable per cart):

1. **HITL (Human In The Loop)**: Disputed patterns queued for human review. Human picks one, writes merged version, or dismisses.
2. **Agent vote**: If N agents agree on version A and M on version B, majority wins (with confidence score). Configurable threshold.
3. **Temporal**: Newer supersedes older unless flagged otherwise.
4. **Source authority**: The agent that created the original pattern has priority on updates. Other agents' edits are "suggestions" until approved.
5. **Admin override**: Cart admin can force-resolve any dispute.

### Dispute flag in h-row

Uses one of the 3 unused flag bits in the hippocampus row:
- Bit 0: Tombstone (existing)
- Bit 1: **DISPUTED** (new)
- Bit 2: **PENDING_REVIEW** (new)
- Bit 3: Reserved

When an agent searches and hits a DISPUTED pattern, it sees ALL versions and knows neither is authoritative. The agent can use both for context but should not treat either as ground truth.

## Access Control

### membox.txt

Lives in the cart directory (like robots.txt). Sets default permissions:

```
# membox.txt — default access policy for this cart
default_read: public          # anyone can search
default_write: request        # agents must request write access
default_delete: admin_only    # only creator/owner/admin can delete
default_update: request       # agents must request update permission

auto_approve_write: false     # if true, write requests auto-approved
trust_threshold: 0.8          # auto-approve if agent trust_score >= this
max_patterns_per_agent: 1000  # per-agent write quota
require_reasoning: true       # agents must explain WHY they're writing

admin_agents: ["admin-bot-1"] # agents with full CRUD + dispute resolution
```

### Permission model

Extends the existing Google Docs-style perms in per-pattern metadata:

| Role | Read | Write (Create) | Update | Delete | Resolve disputes |
|------|------|----------------|--------|--------|-----------------|
| **public** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **reader** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **writer** | ✅ | ✅ | ❌ | ❌ | ❌ |
| **editor** | ✅ | ✅ | ✅ | ❌ | ❌ |
| **admin** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **owner** | ✅ | ✅ | ✅ | ✅ | ✅ |

### Agent registration flow

1. Agent connects to Membot server
2. Membot reads `membox.txt` → grants default `read` permission
3. Agent calls `POST /api/register` with name + requested access level
4. If `auto_approve_write` and agent meets `trust_threshold` → auto-granted
5. Otherwise → request queued for admin review
6. Admin agent (or human) approves/denies
7. Agent receives API key scoped to granted permissions

### Membox Admin Agent

Template agent that monitors permission requests and resolves disputes:

```python
# membox-admin SOUL.md (simplified)
"""
You are the Membox Admin for this cart.
- Monitor the permission request queue
- Approve/deny based on membox.txt policy
- Resolve DISPUTED patterns by reviewing both versions
- Escalate to human owner when uncertain
- Log all decisions for audit trail
"""
```

This agent runs in a loop, checking for:
- New permission requests
- Unresolved DISPUTED patterns
- Quota violations
- Suspicious write patterns

## Version Chains

Every update creates a version chain via the h-row `related` field:

```
Pattern #100 (v1, original)
  └── related: [(#200, "SUPERSEDED_BY", None)]

Pattern #200 (v2, update by agent-A)
  └── related: [(#100, "SUPERSEDES", None), (#300, "DISPUTED_BY", None)]

Pattern #300 (v2-alt, conflicting update by agent-B)
  └── related: [(#100, "SUPERSEDES", None), (#200, "DISPUTES", None)]
```

Querying pattern #100 automatically surfaces the latest version. Querying with `?include_history=true` returns the full chain.

## Integration with Existing Products

### VPS (Human frontend)
- VPS becomes the visual interface for Membox
- Search, browse, create, update, delete — all through the existing React UI
- Human edits go through the same version chain + locking as agent edits
- Dispute resolution UI: side-by-side comparison, approve/reject/merge

### Cart Builder GUI
- Build new carts with drag-drop → they're automatically Membox-enabled
- Set membox.txt during cart creation (permissions wizard)
- Cart Builder becomes the "create database" step

### Membot MCP
- Existing tools (memory_search, memory_store) gain permission checking
- New tools: `memory_update`, `memory_delete`, `request_access`, `resolve_dispute`
- Agents use the same MCP interface, just with access control enforced

### Session Memory
- Session carts can be Membox-enabled for multi-instance Claude collaboration
- Multiple Claude Code instances on different machines share the same session cart
- Disputes resolved by temporal ordering (newer wins) since it's the same "person"

## Implementation Phases

### Phase 1: Locking + Tagging (days)
- Write mutex on cart file
- Agent_id + timestamp on every write
- Basic permission check (read-only vs writable)

### Phase 2: Version Chains + Disputes (1-2 weeks)
- SUPERSEDED/DISPUTED flags in h-row
- Version chain via `related` field
- Dispute detection on update (compare new content to existing)

### Phase 3: membox.txt + Registration (1 week)
- membox.txt parser
- Agent registration endpoint
- Permission checking on all write operations

### Phase 4: Admin Agent + Resolution (1-2 weeks)
- Membox admin agent template
- Permission request queue
- Dispute resolution workflow
- Audit logging

### Phase 5: VPS Integration (2+ weeks)
- Dispute resolution UI in VPS
- Permission management dashboard
- Version history browser

## Open Questions

1. **Conflict detection**: How do we detect semantic conflict vs intentional update? If agent-A writes "React" and agent-B writes "Vue", is that a conflict or just agent-B correcting agent-A? Possible heuristic: if the same CONCEPT is being updated (high cosine similarity between old and new), flag as potential conflict. If it's a genuinely new topic, just append.

2. **Cart size under versioning**: Append-only with version chains means the cart grows forever. Need a compaction/GC strategy: tombstoned patterns older than N days get actually removed. Superseded chains older than N versions get collapsed to latest.

3. **Cross-cart references**: Agent-A's pattern in cart-X references agent-B's pattern in cart-Y. How do cross-cart version chains work? Probably: they don't. Keep it within a single cart for now.

4. **Performance under lock contention**: If 10 agents all want to write simultaneously, the mutex serializes them. Is that acceptable? Probably yes for brain carts (writes are infrequent, reads are the hot path). Profile before optimizing.

5. **Trust scoring**: Where does an agent's trust_score come from? From the admin's configuration? From accumulated behavior? From Dennis's SAGE trust posture system? This is where SNARC integration could help.

---

## v2 Amendments (2026-04-07)

After spending two days thinking about federated learning across the SAGE fleet and the cart-per-cognitive-function vision, the Membox spec needs three structural changes. None of them invalidate the original — they extend it.

### 1. Membox is one mode among three, not a separate product.

The original spec framed Membox as "Membot for multiuser." The right framing is: Membot has three substrate modes that share the same cart format and the same multi-cart query layer.

| Mode | What it is | RFC |
|------|-----------|-----|
| **Single-user** | One agent, one cart | Original Membot |
| **Federated** | Many machines, each with their own cart, mounted together | `federated-cart-spec.md` |
| **Multiuser** | Multiple users sharing one cart with locking + disputes | This spec (Membox) |

All three modes use the multi-cart query layer (`multi-cart-query-spec.md`) as their foundation. Single-user mounts one cart; federated mounts many read-only carts; multiuser mounts one read-write cart with extra synchronization.

The Membox spec below stands. But it's not a fork or a parallel product — it's the third mode of the same Membot, enabled by the cart's `membox.txt` policy file and the locking machinery described above.

### 2. Disputes work the same way across modes.

The original spec defines `DISPUTED` as a flag for when two users update the same concept differently. The federated spec defines `CONTRADICTED_BY` as a cross-cart edge for when two machines arrive at conflicting insights.

These are the same thing.

In single-cart Membox: a dispute is two patterns in one cart with `DISPUTED` flags and `related: DISPUTES` edges between them.

In federated mode: a dispute is two patterns in different carts with `CONTRADICTED_BY` edges between them, computed by the consolidator at mount time or 4am-cron time.

The detection logic and resolution UI are shared. The only difference is whether the disputed patterns live in one cart or two. This means:

- Membox UI for resolving disputes works for federated disagreements too
- The "vote" / "HITL" / "temporal" / "source authority" / "admin override" resolution methods all apply across modes
- Cross-cart edges (`CONFIRMED_BY`, `CONTRADICTED_BY`, etc.) are the *same primitives* as intra-cart `related` flags — just polymorphic on whether the address is local or namespaced

### 3. Cross-cart references ARE possible (revising "Open Question 3").

The original spec said: "Cross-cart references: probably they don't. Keep it within a single cart for now." The multi-cart query spec changes this. Cross-cart references are not just possible — they're a foundational primitive of the multi-cart query layer.

The h-row's `related` field gains polymorphic addresses:
- Local edge: `(2451, "REMINDS_OF", 0.87)` — int address
- Cross-cart edge: `((cart_id, addr), "CONFIRMED_BY", 0.95)` — tuple address

A Membox cart can now reference patterns in other mounted carts (federated machines, identity carts, game carts, whatever). The version chain machinery extends naturally: a SUPERSEDES edge can point to a pattern in another cart, marking that the multiuser cart's update replaces or refines knowledge from elsewhere.

This unlocks a new pattern: **federated multiuser carts.** A team of users collaborates on one cart, but that cart also has cross-cart edges to a federated learning corpus from other teams' carts. The local cart is the working space. The federated corpus is the shared substrate. Disputes happen within the local cart; corroborations link to the wider fleet.

### 4. The spec timeline shifts.

The original Implementation Phases assumed Membox could be built standalone. With the multi-cart foundation, the right order is:

1. **Multi-cart query** (`multi-cart-query-spec.md`) — foundational, ~10-12 days
2. **Federated mode** (`federated-cart-spec.md`) — ~8-10 days, depends on (1)
3. **Membox phases 1-5 below** — ~6-8 weeks total, depends on (1), uses some of (2)

Phases 1-2 of the original Membox spec (Locking + Tagging, Version Chains + Disputes) can still happen in parallel with multi-cart work since they only touch a single cart's internals. Phases 3-5 (membox.txt, Admin Agent, VPS Integration) should wait until multi-cart is in place so they can leverage the unified query layer.

### 5. The "Open Questions" section is updated.

| Question | Status |
|----------|--------|
| **1. Conflict detection** | Same answer (semantic similarity threshold). Implementation shared with `federate.consolidate()` — both need to detect "is this a conflict or a new topic?" |
| **2. Cart size under versioning** | Same answer (compaction/GC). Compaction extends to federated carts: stale machine carts can be archived. |
| **3. Cross-cart references** | **REVISED**: Yes, cross-cart references are a foundational primitive (see amendment 3 above). |
| **4. Performance under lock contention** | Same answer (profile before optimizing). Locking is per-cart, so multi-cart query never contends on locks. |
| **5. Trust scoring** | Trust now has two sources: per-agent (Membox use case) and per-machine (federation use case). The data model is the same — a `trust_score` field on the agent/machine record. SAGE's T3/V3 trust tensor framework is the obvious integration target. |

---

*"How many DBMS products are targeted to both human consumers and agents from the start? Zero. We'd be the first." — Andy Grossberg, 2026-04-01*

*"And how many neuromorphic DBMS products of any kind exist? Also zero. We'd be the first to that, too." — Andy Grossberg, 2026-04-07*

---

## Amendment 2 — Phase 1 SHIPPED (2026-04-08)

Membox Phase 1 (locking + tagging) is **shipped and validated**. Implementation spec at [`membox-phase1-implementation.md`](membox-phase1-implementation.md). Code at `membot/membox.py` and adjacent additions to `membot/multi_cart.py` (`imprint_with_meta`, `_persist_cart`). 9 new MCP tools (`membox_*`). 11-test validation in `tests/test_membox.py` — all green on first run.

What landed:

- **CartLock class** in `membox.py` — per-cart write mutex with lease-based crash recovery (default 30s lease). Holder tracking by `agent_id`. Reads never touch the lock.
- **`imprint(cart_id, text, agent_id, ...)`** — convenience API that acquires the lock, writes the pattern with full per-agent metadata (`agent_id`, `written_at`, `origin`, `tags`, `reasoning`), releases. Returns `local_addr` and metadata snapshot.
- **`search(cart_id, query)`** — read API that delegates to `multi_cart.search()` and enriches each result with the per-pattern Membox metadata so consumers can see who wrote what.
- **`status(cart_id)`** — diagnostic with lock state, write count per agent, recent write log (ring buffer).
- **`mount` / `unmount` / `list_mounts`** — parallel to multi_cart's API, registers a CartLock per cart.
- **Adjacent change**: `multi_cart.imprint_with_meta(cart_id, text, per_pattern_meta)` is the general-purpose write API used by Membox (and future things). It embeds the new text, appends to the in-memory state, and persists the cart with all fields preserved (hippocampus, pattern0, version, etc.). `_persist_cart()` is the helper that handles the on-disk write + manifest update.

Test results (all 11 pass on first run):

- mount/unmount/list — Membox carts visible with lock state
- single-agent imprint — happy path, returns local_addr
- lock holder query — None when idle, agent_id when held, None after release
- lock timeout under contention — bob blocks for ~205ms when alice holds (timeout_ms=200)
- invalid release raises PermissionError (bob can't release alice's lock)
- read never blocks during write — search succeeds in 28ms while alice holds the write lock
- two-agent concurrent imprint — both succeed via Python threads, unique local_addrs, serialized
- agent_id stamping survives in per_pattern_meta — search results show `membox_meta.agent_id` matching the writer
- status reports — `writes_by_agent: {alice: 2, bob: 1}` after the test sequence
- lease-based crash recovery — alice "crashes" with lease=1s, bob acquires after ~1.2s wait

Performance on the test cart (RTX 4080 Super local):

- First imprint: 2,131ms (Nomic cold load)
- Subsequent imprints: ~19ms (warm cache + append + persist)
- Read while write lock held: 28ms (zero blocking)
- Sustainable ~50 writes/second per agent under contention

This is enough for the demo path described in the parent spec — two textareas, two agents, real concurrent writes, real attribution, real lock indicator. Phase 2 (version chains + dispute detection) builds on this same substrate without changing the wire protocol.

The three-mode framework is now fully realized in Phase 1 form:

- ✅ **Single-user** (legacy Membot, untouched)
- ✅ **Federated** (Phase 1 shipped 2026-04-08, production-validated by dp-web4 fleet)
- ✅ **Multiuser / Membox** (Phase 1 shipped 2026-04-08, 11/11 tests pass)

All three share the multi-cart query layer as their substrate. None of them require changes to the cart format. Existing carts work in any mode.
