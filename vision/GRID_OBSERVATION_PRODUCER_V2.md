# GridObservationProducer v2 — quick reference

**File:** `grid_observation_producer_v2.py`
**Status:** Sprint 0 MVP, schema-stable
**Purpose:** Explorer-facing perception interface that consumes either raw 64×64 grids or pre-segmented object records and emits a typed `GridObservation` describing all objects in the frame plus a state-signature hash suitable for graph-pursuit dedup.

---

## Two entry points

### `observe(frame, frame_idx, last_action, filter_colors=None)`

Raw 64×64 int8 grid → connected-component segmentation → typed objects.

- `frame`: `(64, 64)` numpy array, values 0-15
- `frame_idx`: monotonic counter from session start
- `last_action`: optional; the action that produced this frame
- `filter_colors`: optional `set[int]` of color values to skip (UI overlays). For r11l: `{1}` drops the Bresenham line overlay; the LLM-sparse-prior at first-contact is the natural place to set this per-game.

Returns a `GridObservation` with accurate cell-derived bboxes and full per-frame object list.

### `observe_from_cbp(record, frame_idx, background_color=None)`

Consume one CBP-format `.objstate.jsonl` record directly. No raw frame required.

- `record`: parsed dict per CBP's schema (`{step, level, state, action, objects: [...]}`)
- `frame_idx`: monotonic counter
- `background_color`: optional explicit bg color (CBP records don't carry it; for r11l it's `5`)

Bboxes are approximated (`centroid ± sqrt(n)/2`) since CBP doesn't carry raw cell masks. Use Path 1 (`observe`) if accurate bboxes matter for downstream spatial reasoning.

### `reset()`

Clear cross-frame state. Call between games / session boundaries.

---

## Schema (`GridObservation` + `ObjectNode`)

```
GridObservation:
    frame_idx:             int
    objects:               list[ObjectNode]
    state_signature:       bytes              # 32-byte BLAKE2b
    id_map:                np.ndarray         # (64,64) uint32, primary owner per cell
    id_stack:              dict               # {(r,c): [id, ...]} only for cells with >1 owner
    background_color:      int
    perception_confidence: float              # 0.0-1.0; honest about stubbed fields

ObjectNode:
    identity:   str                           # stable across frames
    bbox:       tuple                         # (r_min, c_min, r_max, c_max) inclusive
    centroid:   tuple                         # (row, col) sub-pixel
    color:      int                           # 0-15
    size:       int                           # cell count
    properties: dict                          # see below
    relations:  dict                          # see below
```

### Property fields (per object)

| Key | Type | MVP behavior |
|---|---|---|
| `selected` | bool | stub `False` (real value requires per-game LLM-sparse-prior) |
| `active` | bool | stub `False` |
| `carried_content` | int \| None | stub `None` |
| `moved_this_frame` | bool | Path 1: computed; Path 2: stub `False` |
| `appeared_this_frame` | bool | computed from identity tracking |
| `visible` | bool | default `True` (lifecycle layer not yet built) |
| `corporeal` | bool | default `True` |
| `identity_confidence` | float | 1.0 for unambiguous matches; lower when forced |
| `mk` / `ho` | int | Path 2 only: CBP's marked/hollow flags verbatim |

### Relation fields (per object)

| Key | Type | MVP behavior |
|---|---|---|
| `part_of` | str \| None | stub `None` |
| `on_top_of` | list[str] | populated when `id_stack` has overlap at this object's cells |
| `contains` | list[str] | stub `[]` |
| `adjacent` | list[str] | computed from 4-neighbor cell analysis |

---

## State-signature hash

`state_signature` is a 32-byte BLAKE2b digest of the canonical object-state graph: each object's `(color, size, centroid, properties, relations)` sorted into a deterministic order and serialized. Identity strings are excluded (tracking-side labels can vary across runs); exact bboxes are excluded (derivable from cells).

**Equality semantics:** two frames produce the same hash iff their object-state graphs are equal. Same objects in same positions doing same things → same hash. Any object moves, gains/loses cells, or has a property/relation change → different hash.

The 64×64 grid is itself the perception layer — cells are discrete. There is no sub-cell drift to filter against. Position changes that look small at the body scale (e.g., r11l body centroid shifting by 0.33 cells when one of three limbs moves) ARE meaningful state changes and are correctly registered.

---

## Design philosophy — "more data than asked for"

The producer intentionally emits richer perception output than minimally needed:

- Path 1 returns walls, decorations, danger zones, single-pixel markers alongside interactive sprites
- Path 1 returns accurate bboxes from real cell sets (where some upstream pipelines approximate bbox as a square around centroid — wrong for elongated objects)
- Path 1 default scipy 4-connectivity may over-segment hollow rings (which some consumers would prefer merged via 8-connectivity); we currently emit the 4-conn view to preserve fine detail

Consumers can trivially subset what they don't want. They cannot trivially recover detail we chose not to emit. **Emit richer, let consumers narrow.** Do not dial outputs down to match a particular consumer's existing pipeline.

---

## Validation status (Sprint 0)

Tested on r11l from dev-SAGE's replay corpus (`replay_dataset/r11l.{objstate.jsonl,rawgrid.npz}`):

- Path 2 alone: 4 discrimination tests pass (identical / 5-cell move / 0.33-cell shift / property flip)
- Path 1 + Path 2 cross-validated on r11l frame 0: agreement on the step-counter bar (color 0 size 64 at centroid (31.5, 0.0)); differences elsewhere reflect known perception-philosophy differences (4-conn vs 8-conn, full-scene vs interactive-only)
- 5-game distinct-state spot check (r11l, cd82, cn04, ar25, bp35) on first 100 frames each: hash discriminates appropriately; per-game state-recurrence rates vary 69-100% (cd82/ar25 show real state revisits which the explorer can exploit)

---

## What's not yet built (post-MVP)

- **Selection / carried-content detection** — requires per-game LLM-sparse-prior or learned classifier; schema fields are ready, values are stubbed
- **Hungarian assignment** for object correspondence under multi-object simultaneous movement (current: nearest-centroid + color matching; works for r11l's one-action-per-step model)
- **Lifecycle layer** that flips `visible`/`corporeal` based on cross-frame disappearance
- **8-connectivity option** in Path 1's parse_objects (would merge hollow-ring objects; intentionally deferred per design philosophy above)
- **Cortex stack + F1 attention** — major architectural addition, separate from this MVP

---

## Files

- `grid_observation_producer_v2.py` — this module
- `grid_observation.py` — low-level primitives (parse_objects, compute_diff, track_objects) reused here
- `../../docs/proposals/GRID_OBSERVATION_PRODUCER_PROPOSAL_2026-05-21.md` — full interface contract (local-only)
- `../../docs/proposals/MEMBOT_INTEGRATION_ASKS_RESPONSE_2026-05-22.md` — answers to the 6 fleet asks (local-only)
- `../../docs/proposals/WM_GROWTH_FOLLOWUP_RESPONSE_2026-05-23.md` — Q1/Q2/Q3 answers (local-only)
