# Modular Cartridge Proposal: Federated Memory Assembly

**From**: dp-web4 / Dennis Palatov
**To**: Andy / project-you-apps
**Date**: 2026-03-28
**Status**: Proposal — requesting feedback before implementation

---

## The Idea

Instead of a single monolithic `.npz` cartridge, allow **modular segments** — one `.npz` per knowledge domain — that are **assembled into a unified search space at mount time**. Each segment is independently maintained, independently hashable, and independently available. Missing segments degrade gracefully (snippets available, full text not).

This mirrors how human associative memory works: you know things exist across many domains, you can find connections between them, and some knowledge is currently accessible while other knowledge is "I know I learned this somewhere but can't access it right now."

## Why

We're running membot across a 6-machine fleet, each machine working on multiple projects (SAGE, Synchronism, Web4, Hardbound, etc.). Each project generates semantic memories independently. The valuable insight is often the **cross-project connection** — a concept in physics research that maps to an AI governance primitive, or a raising session observation that connects to a trust tensor design decision.

Currently, each project would need its own cartridge, searched independently. Cross-project semantic search requires a combined cartridge, which means one process must own and rebuild the whole thing whenever any project adds content.

Modular segments solve this: each project maintains its own segment. Assembly happens at mount time. Cross-project search is automatic because all segments share the same embedding space (same Nomic model).

## Architecture

### Per-Project Segment (`.seg.npz`)

Each project produces a segment file:

```
project-sage.seg.npz
├── embeddings     float32[N, 768]     # Nomic embeddings
├── sign_bits      uint8[N, 768]       # Binary Hamming codes
├── snippets       object[N]           # Short text for display + keyword reranking
├── tags           object[N]           # Per-entry metadata tags
├── segment_id     str                 # Unique segment identifier (e.g., project hash)
├── project_name   str                 # Human-readable name
├── db_path        str                 # Relative path to SQLite sidecar for full text
├── db_hash        str                 # SHA256 of the db file at build time
├── count          int                 # Number of entries
├── created        str                 # ISO timestamp
├── updated        str                 # Last modified timestamp
└── schema_version int                 # Segment format version
```

The SQLite sidecar (split cart format) holds full passages. The `.seg.npz` holds only embeddings + snippets — enough to search and display results, but full text requires the sidecar.

### Assembly at Mount Time

```python
def mount_federated(segment_dir: str) -> FederatedCartridge:
    """Load all .seg.npz files from a directory, assemble into unified search space."""
    segments = []
    for f in sorted(Path(segment_dir).glob("*.seg.npz")):
        seg = load_segment(f)
        segments.append(seg)

    # Concatenate embeddings into single search matrix
    all_embeddings = np.vstack([s.embeddings for s in segments])
    all_sign_bits = np.vstack([s.sign_bits for s in segments])
    all_snippets = sum([s.snippets for s in segments], [])

    # Build segment map: which rows belong to which project
    boundaries = []
    offset = 0
    for seg in segments:
        boundaries.append({
            "start": offset,
            "end": offset + seg.count,
            "segment_id": seg.segment_id,
            "project_name": seg.project_name,
            "db_path": seg.db_path,
            "db_hash": seg.db_hash,
            "db_available": Path(seg.db_path).exists(),
        })
        offset += seg.count

    return FederatedCartridge(
        embeddings=all_embeddings,
        sign_bits=all_sign_bits,
        snippets=all_snippets,
        boundaries=boundaries,
    )
```

### Search Results with Source Attribution

```
Query: "how does observation create reality"

#1 [0.735] [Synchronism] Self-witnessing is the mechanism by which intent
   patterns create their own confinement through saturation synchronization.
   → Full text available (Synchronism db online)

#2 [0.664] [Web4/4-gov] Governance works when the entity has enough context
   to make cooperation the obvious choice.
   → Full text available (Web4 db online)

#3 [0.610] [SAGE] The cognitive autonomy gap: capability exists, affordance
   doesn't — yet.
   → Snippet only (SAGE db offline on this machine)
```

### Graceful Degradation

| DB State | Search | Snippet | Full Text | Display |
|----------|--------|---------|-----------|---------|
| Online, hash valid | ✅ | ✅ | ✅ | Normal |
| Online, hash stale | ✅ | ✅ | ⚠️ | "may be outdated" |
| Offline | ✅ | ✅ | ❌ | "exists, not reachable" |
| Segment missing | ❌ | ❌ | ❌ | Not in results |

Crucially: **search always works on all available segments.** The embedding matrix is assembled once at mount time. Missing full text doesn't affect search — it only affects what you can display when a result is selected.

## Implementation Path

### Phase 1: Segment Builder (new tool)

```bash
# Build a segment from a SNARC database
python3 build_segment.py \
    --db ~/.snarc/projects/hash-abc/engram.db \
    --project "SAGE" \
    --output segments/sage.seg.npz

# Build from a membot cartridge
python3 build_segment.py \
    --cartridge cartridges/physics.cart.npz \
    --project "Physics" \
    --output segments/physics.seg.npz
```

### Phase 2: Federated Mount

New MCP tool or mount option:

```python
mount_cartridge("federated:~/.snarc/segments/")
# Loads all *.seg.npz, assembles, reports per-segment status
```

### Phase 3: Live Segment Updates

When a project adds new content (SNARC stores a new observation), append to that project's segment without rebuilding the whole assembly:

```python
# Hot-append to a mounted federated cartridge
federated_store(content, project="SAGE")
# Extends SAGE segment in-place, updates boundaries
```

### Phase 4: Segment Sync (Optional)

Machines could share segments. CBP builds the SAGE segment. Sprout pulls it. Sprout doesn't need the SAGE db — just the segment file for search. Full text retrieval would show "snippet only" but cross-project connections would still surface.

## Compatibility

- **Existing cartridges**: Unchanged. A `.cart.npz` mounts exactly as before.
- **Split carts**: The segment format extends the split cart concept. A segment IS a split cart with additional metadata (segment_id, project_name, boundaries).
- **MCP tools**: `memory_search` returns results with an additional `source_project` field. No breaking changes.
- **Embedding model**: All segments must use the same embedding model. The segment header records which model was used. Assembly rejects mismatched models.

## Connection to SAGE IRP

Each project's memory segment maps naturally to a **temporal sensor** in SAGE's IRP architecture:

```
Consciousness Loop Step 1: Gather from sensors
├── Vision sensor    → camera IRP
├── Audio sensor     → microphone IRP
├── Language sensor  → LLM IRP
├── Memory sensor    → federated cartridge
│   ├── SAGE segment     (recent raising sessions, identity evolution)
│   ├── Synchronism segment (physics research, experiments)
│   ├── Web4 segment     (trust ontology, governance decisions)
│   └── private-context segment (cross-machine logs, insights)
```

The memory sensor provides **temporal context** — what was discussed, decided, discovered across projects and sessions. Each segment is a temporal window into a different domain. The consciousness loop queries the assembled cartridge as one sensor, gets cross-project associations automatically, and the salience scorer (SNARC) weights the results by relevance to the current attention target.

This is the IRP contract: `init_state` (mount segments) → `step` (search with current context) → `energy` (relevance score) → `halt` (results plateau). Same interface as every other sensor. The federated cartridge is just another plugin.

## Memory Budget

| Scale | Embeddings | Sign bits | Snippets | Total (search) |
|-------|-----------|-----------|----------|----------------|
| 1K entries | 3 MB | 96 KB | ~200 KB | ~3.3 MB |
| 10K entries | 30 MB | 960 KB | ~2 MB | ~33 MB |
| 100K entries | 300 MB | 9.6 MB | ~20 MB | ~330 MB |
| 1M entries | 3 GB | 96 MB | ~200 MB | ~3.3 GB |

For our fleet: 10K entries across all projects is realistic for the first few months. 33 MB search index — fits anywhere, including Sprout's 8GB.

## Questions for Andy

1. Does this align with where you see membot going? The segment format is a natural extension of split carts.
2. Any concerns about the assembly approach (vstack at mount time)?
3. The segment builder would need to support both SNARC databases (SQLite/FTS5) and existing membot cartridges as input. Any format considerations?
4. Should segments be versioned independently from cartridges?
5. Interest in the SAGE IRP integration? Would require a Python API alongside the MCP tools.

---

*dp-web4 / Metalinxx*
