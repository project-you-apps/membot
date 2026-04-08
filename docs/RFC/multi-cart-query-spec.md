# RFC: Multi-Cart Query — The Foundation of Membot as a DBMS

**Date**: 2026-04-07
**Status**: Draft
**Authors**: Andy Grossberg, Claude (Opus 4.6)
**Related**: `membox-multiuser-dbms-spec.md`, `federated-cart-spec.md`

---

## TL;DR

Membot today owns one mounted cart per session. To become a database (instead of a library), Membot must own *many mounted carts* and let queries span them transparently. This RFC defines:

1. The mount model (attaching multiple carts to one Membot instance)
2. The namespacing of cart contents (so addresses don't collide)
3. The query API (search scope, result attribution)
4. Cross-cart h-row edges (the navigation layer that spans cart boundaries)

This is the foundational change that unblocks federated mode (`federated-cart-spec.md`) and multiuser CRUD (`membox-multiuser-dbms-spec.md`). Without it, both are bolt-ons. With it, they're natural extensions of the same architecture.

---

## Motivation

The cart-per-cognitive-function vision (raising_kb / game_kb / hot_stack / context_window / identity) requires a Membot instance that holds multiple carts simultaneously and lets the LLM reason across them. The federated learning vision (per-machine carts, fleet-wide search) requires the same thing at a different scale.

Currently:
- One Membot instance = one mounted cart
- Search returns results from "the cart" — there's no other cart to compare to
- The lattice substrate has no notion of cart boundaries
- The hippocampus row only addresses patterns inside its own cart

Target:
- One Membot instance = many mounted carts
- Search returns results across all mounted carts, attributed to their source
- Query scope is configurable (single cart, group of carts, all carts)
- The hippocampus row can address patterns in *other* mounted carts via namespaced IDs
- Cross-cart relationships are first-class (CONFIRMED_BY, CONTRADICTED_BY, REFINED_BY, SUPERSEDED_BY)

This is the difference between "a library that holds one collection" and "a database that holds many relations." It's also the difference between Membot-as-tool and Membot-as-product.

---

## Mount Model

### Mount API

```python
membot.mount(cart_path: str, cart_id: str = None, role: str = None)
```

- `cart_path`: filesystem path or URL to the cart file (.npz, .pkl, or split format)
- `cart_id`: short stable identifier (defaults to filename without extension)
- `role`: optional semantic tag — `identity`, `episodic`, `semantic`, `working`, `federated`, `consolidated`, etc.

The role is metadata for the agent to understand *why* a cart was mounted, not for the lattice to dispatch differently. All carts use the same physics; the role is human/LLM-facing.

Multiple carts can share the same role (e.g. a federated mount might have a dozen carts all with `role="machine_kb"`).

### Mount Lifecycle

```python
membot.mount("./carts/raising_sprout.cart", cart_id="raising", role="identity")
membot.mount("./carts/game_sb26.cart", cart_id="game_sb26", role="game")
membot.mount("./fleet/cbp.cart", cart_id="cbp", role="federated")
membot.mount("./fleet/sprout.cart", cart_id="sprout", role="federated")

membot.list_mounts()
# → [
#     {cart_id: "raising", role: "identity", patterns: 2451, status: "loaded"},
#     {cart_id: "game_sb26", role: "game", patterns: 487, status: "loaded"},
#     {cart_id: "cbp", role: "federated", patterns: 13422, status: "loaded"},
#     {cart_id: "sprout", role: "federated", patterns: 8201, status: "loaded"},
#   ]

membot.unmount(cart_id="game_sb26")
```

### Mount Discovery

For federated mode, mounting an entire directory should be a one-call operation:

```python
membot.mount_directory("./fleet/", role="federated", pattern="*.cart")
# → mounts cbp.cart, sprout.cart, mcnugget.cart, etc. as separate cart_ids
```

This is how Dennis's solver would mount the fleet's federated learning data on session start: one call, all machines mounted.

---

## Namespacing

Every pattern in every mounted cart gets a globally-unique address: `(cart_id, local_address)`. The cart_id is the namespace prefix.

### Why namespacing matters

Without it, two carts can both have a pattern at local address `42` and the lattice can't distinguish them. With namespacing, `(raising, 42)` and `(game_sb26, 42)` are unambiguous.

### Hippocampus row uses namespaced addresses

Cross-cart edges in h-row reference `(cart_id, addr)` tuples instead of bare integers. When a pattern in `game_sb26` says "this insight was confirmed by `(sprout, 1842)`", the h-row entry stores the full tuple. The mount layer resolves it transparently when you query.

### Local h-row entries stay local

Edges that point within the same cart still use bare integers. Only cross-cart edges carry the cart_id. This keeps existing single-cart behavior unchanged.

```python
# Local edge (existing format)
{"related": [(2451, "REMINDS_OF", 0.87)]}

# Cross-cart edge (new format)
{"related": [(2451, "REMINDS_OF", 0.87), (("sprout", 1842), "CONFIRMED_BY", 0.95)]}
```

The first entry of each tuple is either an int (local) or a `(cart_id, addr)` tuple (cross-cart). Same shape, just polymorphic on the address.

---

## Query API

### Search scope

```python
membot.search(
    query: str,
    top_k: int = 10,
    scope: str | list[str] = "all",   # "all" | "local" | cart_id | [cart_id, ...]
    role_filter: str | list[str] = None,  # only search carts with these roles
    include_source: bool = True,
)
```

Scope options:
- `"all"` (default) — search every mounted cart
- `"local"` — search only the primary/first-mounted cart (backward compatible with single-cart code)
- `"raising"` — search only the cart with cart_id="raising"
- `["raising", "game_sb26"]` — search just these two
- `role_filter="federated"` — search every cart with role="federated"

Combinations work: `scope="all", role_filter="federated"` searches every federated cart only.

### Result attribution

Every result carries its source cart_id:

```python
results = membot.search("loop topology with parking strategy", top_k=5)
# → [
#     {
#       cart_id: "sprout",
#       local_addr: 1842,
#       score: 0.93,
#       text: "Discovered side track parking on loop puzzles...",
#       role: "federated",
#       confidence: 0.95,
#       source_machine: "sprout",   # if federated, the originating machine
#     },
#     {
#       cart_id: "game_sb26",
#       local_addr: 312,
#       score: 0.89,
#       ...
#     },
#   ]
```

The LLM consuming these results can now distinguish "this insight came from my own raising history" from "this insight came from machine cbp's last week's session." That distinction matters for trust and for narrative voice when the LLM responds.

### Cross-cart ranking

When searching multiple carts, results are ranked globally — not per-cart with merging. The cosine + Hamming + keyword scores are computed against the query in each cart's own embedding space (assumed identical, which it is when all carts use Nomic v1.5), then sorted into one ranked list.

If carts use different embedding models, the mount layer rejects the mount with an explicit error. We don't try to translate between embedding spaces.

### Query patterns the multi-cart model enables

```python
# Ask: "what does my fleet collectively know about cyclic loops in puzzles?"
membot.search("cyclic loop puzzle pattern", role_filter="federated")

# Ask: "what does my own past say about uncertainty?"
membot.search("uncertainty tolerance", scope="raising")

# Ask: "is this game state similar to anything in any of my mounted memory?"
membot.search(grid_description, scope="all")

# Ask: "what's the current consensus on this concept across the fleet,
# and where do machines disagree?"
results = membot.search("the goal of this puzzle", scope="all", include_source=True)
groups = group_by_concept(results)  # see CONFIRMED_BY / CONTRADICTED_BY edges
```

The last one is where the cross-cart h-row edges become powerful — see below.

---

## Cross-Cart H-Row Edges

The hippocampus row is already the typed-edge layer in our brain cart. Today its edges stay local. For multi-cart, edges can cross cart boundaries.

### New edge types

These are first-class h-row edge types for cross-cart relationships:

| Edge | Meaning | Use case |
|------|---------|----------|
| `CONFIRMED_BY` | Same concept independently arrived at by another cart | Federated trust signal |
| `CONTRADICTED_BY` | Same input, different conclusion in another cart | Dispute / disagreement |
| `REFINED_BY` | Another cart added detail to this concept | Knowledge enrichment |
| `SUPERSEDED_BY` | A newer cart's pattern replaces this one | Versioning across carts |
| `SOURCED_FROM` | This pattern originally came from another cart | Provenance |
| `MERGED_FROM` | This pattern is the consolidation of patterns in N carts | Consolidation result |

These coexist with existing local edge types (`REMINDS_OF`, `RELATED_TO`, `PRECEDES`, etc.).

### How cross-cart edges get created

Three mechanisms:

1. **Explicit imprint with cross-cart relation**
   ```python
   membot.imprint(
       text="...", cart_id="raising",
       relations=[(("game_sb26", 312), "REFINED_BY", 0.92)]
   )
   ```

2. **Automatic via federation consolidation** (see federated-cart-spec.md)
   When the consolidator runs (or when carts are mounted), patterns above a similarity threshold across carts get auto-linked with `CONFIRMED_BY`. This is how "two machines independently noticed the same thing" becomes a trust signal instead of a deduplication problem.

3. **Via dispute detection** (see membox-multiuser-dbms-spec.md)
   When a multiuser cart detects two writes that semantically conflict, it auto-creates `CONTRADICTED_BY` edges between them. This is the federation case applied to a single shared cart.

### Why this matters

Today's "search and merge" approach to federated knowledge is:
- Run a query on each cart
- Take the top N from each
- Sort by score
- Hope nothing important got dropped

That's lossy. Patterns that occur in two carts get returned twice with no signal that they corroborate each other.

Cross-cart edges fix this. When a pattern in cart A has `CONFIRMED_BY` edges to similar patterns in carts B and C, the search can:
- Return the cart A pattern *once* with a confidence boost based on the number of confirmations
- Surface the supporting evidence from B and C only on request
- Show contradictions explicitly when they exist

This is the difference between **merging documents** and **modeling agreement**. The first is what RAG does. The second is what a database does.

---

## Implementation Notes

### Backward compatibility

Single-cart code keeps working unchanged. If you mount only one cart and never specify scope, search behaves exactly as it does today.

```python
# Old code, still works:
membot.load_cart("./my.cart")
membot.search("hello world", top_k=5)
```

Internally, `load_cart()` becomes `mount(cart_path, cart_id="default")` and `search()` defaults to `scope="all"`. Since there's only one cart, scope makes no difference.

### Memory cost

Each mounted cart loads its embeddings + lattice + h-row into RAM. Mounting 10 carts of 10K patterns each = ~10× the memory of one cart. For the federated case (a fleet of 6-12 machines, ~10K patterns each), that's ~150 MB total. Manageable.

For larger deployments, lazy loading: keep h-row + metadata in RAM, page embeddings from disk on demand. Phase 2.

### Query cost

Multi-cart search is O(N_carts × cart_search_cost). For 10 carts of 10K patterns each, that's 10× a single cart's search time. With our current 13ms search speed, that's 130ms — still fast enough for interactive use. For very large federations, the same lazy-load approach can fan out searches in parallel.

### Identity and locking

Multi-cart introduces a new question: when multiple carts are mounted and one has multiuser locking enabled (Membox), which cart owns the lock? Answer: locking is per-cart, always. A multi-cart query that touches a write-locked cart still gets to read it (reads never block). Writes go to one specific cart at a time and acquire that cart's lock specifically.

---

## What This Unblocks

1. **Federated mode** — fleet-wide search becomes one query against many mounted federated carts
2. **Multiuser CRUD** — disputes across machine carts work the same way as disputes within a shared cart
3. **Cart-per-cognitive-function** — raising/game/hot_stack/identity all become mounted carts in one Membot instance
4. **VPS as a "database browser"** — VPS gains a "mount this cart" button and a "search across mounted carts" search bar
5. **Cross-cart evidence aggregation** — the LLM can ask "what does ALL my memory say about X" instead of querying carts one at a time

---

## Open Questions

1. **Cart-id collision policy.** If a user mounts two carts with the same default cart_id (e.g. two different `sprout.cart` files), what happens? Probably: error, force them to specify a unique cart_id. Don't auto-rename — silent renames are a debugging nightmare.

2. **Mount persistence.** Are mounts session-scoped (lost on restart) or persisted? Probably both — a `mounts.json` config file lists default mounts, plus a runtime mount API for ad-hoc additions.

3. **Cart unloading on memory pressure.** Should mounted-but-unused carts get evicted from RAM after N minutes idle? Probably yes for the cloud Membot tier where memory matters.

4. **Embedding model heterogeneity.** Today we assume all carts use Nomic v1.5. If a user mounts a cart built with a different embedder, the search results won't be comparable. Two options: reject the mount (safe), or rescore the foreign cart's results separately and merge with a flag (complex). Start with reject.

5. **Federation consolidation at mount time vs background.** When multi-cart mode mounts a directory of federated carts, should it immediately scan for `CONFIRMED_BY` edges or do it lazily? Probably background — don't make mount slow. But the first cross-cart search after mount may be less informative until consolidation runs.

---

## Implementation Phases

### Phase 1: Mount API + namespacing (1-2 days)
- `mount()` / `unmount()` / `list_mounts()` / `mount_directory()`
- Cart_id namespacing in internal data structures
- Backward-compat aliases (`load_cart` → `mount`)

### Phase 2: Multi-cart search (2-3 days)
- `scope` parameter on search
- Result attribution with cart_id and role
- Global ranking across carts

### Phase 3: Cross-cart h-row edges (2-3 days)
- New edge types in h-row format
- Polymorphic address resolution (int vs tuple)
- Edge creation API for explicit links

### Phase 4: Mount discovery + persistence (1-2 days)
- `mounts.json` config
- `mount_directory()` glob support

### Phase 5: Stress test + perf (2 days)
- Mount 10+ carts simultaneously
- Profile search latency under load
- Memory profiling

Total: ~10-12 days of focused work for the foundational change.

---

*This is the work that turns Membot from a library into a database. Federated mode and multiuser CRUD are both downstream of it. Build this first.*

---

## Amendment 1: Search Ranking Modes (2026-04-07, post-Phase-1)

After Phase 1 shipped and was tested with two carts of vastly different sizes
(24 patterns vs 18,040 patterns), Andy noticed that the global ranking strategy
can let one large cart dominate the results, making smaller carts effectively
invisible even when they hold relevant content.

The fix: expose `scope_mode` as a search parameter so callers can choose the
ranking strategy that fits their use case.

### Modes

```python
multi_search(query, top_k=5, scope="all", scope_mode="global")
```

| `scope_mode` | Behavior | When to use |
|---|---|---|
| `"global"` (default) | True top-K across all mounted carts. Current Phase 1 behavior. | "What's the best answer overall, regardless of source?" |
| `"per_cart"` | Top-K from each cart, returned grouped by cart_id, no cross-cart re-ranking. Returns up to `K × N_carts` results. | "Show me the best of each source so I can compare them." |
| `"balanced"` | Top-K from each cart as candidates, then global re-ranking of those candidates to top-K. | "Don't let one big cart dominate, but give me the global best of what each cart offered." |
| `"diagnostic"` | Top-K_per_cart from EVERY cart with no merging at all, just labeled by source. | Debugging — "show me what each cart thinks the answer is." |

### Default behavior

`global` stays the default because it matches the most natural mental model
("show me the best results"). `balanced` is the most useful new mode for
federated learning use cases where you want every machine to be heard.

### Implementation cost

~30 lines in `multi_cart.search()`. The `_search_one_cart()` helper already
returns a per-cart result list, so the merge step is just a different sort/limit
strategy depending on mode. No changes to mount/unmount or to the underlying
data model.

### Relationship to Phase 3 (cross-cart edges)

The deeper fix is Phase 3 of this spec: cross-cart `CONFIRMED_BY` edges
precomputed at consolidation time. Once those exist, you don't need round-robin
strategies at search time — you query the consolidated cart and get one result
per consensus group with the corroborating evidence baked in. The `scope_mode`
parameter is the short-term polish that makes Phase 1 production-ready while
Phase 3 is being designed.

### Status

Filed for implementation tomorrow. ~30 lines + tests.
