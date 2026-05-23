"""GridObservationProducer v2 — explorer-facing perception interface.

Sprint 0 MVP per:
- docs/proposals/GRID_OBSERVATION_PRODUCER_PROPOSAL_2026-05-21.md (contract)
- docs/proposals/MEMBOT_INTEGRATION_ASKS_RESPONSE_2026-05-22.md (Ask 1/2/3 answers)
- docs/proposals/WM_GROWTH_FOLLOWUP_RESPONSE_2026-05-23.md (Q1/Q2/Q3 answers)
- shared-context/arc-agi-3/game-mechanics/r11l.md (target game for validation)

Two entry points:
  - observe(frame, frame_idx, last_action, filter_colors=None):
        Raw 64x64 grid → ObjectNode list via parse_objects (full connected-
        component analysis). Returns EVERYTHING in the frame: walls,
        decorations, UI markers, single-pixel features, etc. Accurate cell
        sets + accurate bboxes. Consumer is free to subset.
  - observe_from_cbp(record, frame_idx, background_color=None):
        Consume CBP-format .objstate.jsonl records (already-segmented by
        their pipeline). Lighter-weight; no raw frame needed; uses CBP's
        centroid+size with approximated bbox (centroid ± sqrt(n)/2).

Both produce the same GridObservation schema. Path 1 is the richer view;
Path 2 is the faster path when CBP's pre-segmented output is sufficient.

DESIGN PHILOSOPHY — "more data than asked for":
The producer intentionally provides richer perception output than what
upstream consumers might minimally need. Examples: Path 1 returns walls
and decorations alongside interactive sprites; Path 1 returns accurate
bboxes derivable from the actual cell sets where CBP's pipeline
approximates bbox as a square around the centroid (which is wrong for
elongated objects like step-counter bars or wall strips). Consumers can
trivially subset what they don't want (filter_colors arg in Path 1,
property/relation field omission downstream); they cannot trivially
RECOVER detail we chose not to emit. Therefore: emit richer, let
consumers narrow. Do NOT dial outputs down to match a particular
consumer's existing pipeline.

Schema is stable from MVP onward. Field semantics never change; only the
quality of the values evolves as cortices come online in later sprints.
Real values today: identity, bbox, centroid, color, size, adjacent,
moved_this_frame, appeared_this_frame, visible/corporeal, id_map,
state_signature. Stub-defaulted today: selected (False), active (False),
carried_content (None), part_of (None), on_top_of ([]), contains ([]).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Reuse existing low-level primitives — these stay the workhorses.
from grid_observation import (
    parse_objects as _parse_objects_raw,
    compute_diff as _compute_diff_raw,
    track_objects as _track_objects_raw,
)


# ─── Schema dataclasses (the explorer's contract) ──────────────────────

@dataclass
class ObjectNode:
    """One discriminated object in the current frame.

    Identity is stable across frames when the object IS the same object
    (per the producer's tracking heuristics). Properties and relations
    are extracted per frame and may vary frame-to-frame even when identity
    is stable.
    """
    identity:   str
    bbox:       tuple                  # (r_min, c_min, r_max, c_max) inclusive
    centroid:   tuple                  # (row, col) sub-pixel
    color:      int                    # 0-15
    size:       int                    # cell count
    properties: dict = field(default_factory=dict)
    relations:  dict = field(default_factory=dict)


@dataclass
class GridObservation:
    """Producer output for one frame. Consumed by the explorer's read_state()."""
    frame_idx:             int
    objects:               list                       # list[ObjectNode]
    state_signature:       bytes                      # 32-byte BLAKE2b-256
    id_map:                np.ndarray                 # (64,64) int32, primary owner per cell
    id_stack:              dict = field(default_factory=dict)   # {(r,c): [id, ...]} only when stacked
    background_color:      int = 0
    perception_confidence: float = 1.0


# ─── Canonical state-graph hash ────────────────────────────────────────

def canonical_state_hash(objects: list, background_color: int) -> bytes:
    """Compute a 32-byte BLAKE2b hash of the canonical object-state graph.

    Two frames produce the same hash iff their object-state graphs are
    equal under the equality rules in STATE_REPRESENTATION_SPEC §3:
      - Same objects in same positions doing same things → same hash
      - Any object moves to a different cell layout → different hash
      - Any property/relation change → different hash

    Hash inputs (per object): color + size + centroid + properties + relations.
    Identity strings are intentionally EXCLUDED — identity is a tracking-side
    label that can vary across runs. Exact bbox is also excluded since it's
    derivable from the cell set (and CBP's adapter approximates it anyway);
    centroid is the authoritative position signal.

    Note on position: the 64×64 grid IS the perception layer — cells are
    discrete; centroid is exactly mean(covered_cells); there is no
    sub-cell drift to filter against. The "cosmetic flicker" the explorer
    spec discusses exists in the upstream raw display image and is filtered
    by CBP's segmenter before .objstate.jsonl is produced. We see post-
    filter data; no further dedup quantization is appropriate (and would
    risk merging real state changes — e.g., r11l body centroid shifts
    by ~0.33 cells when one of three limbs moves, which IS a state change
    even though it's sub-cell at the body level).

    Centroid is rounded to 2 decimals only to avoid float-representation
    artifacts from different code paths producing the same value.
    """
    canonical = {
        "bg": int(background_color),
        "objs": sorted(
            [
                {
                    "c":     int(obj.color),
                    "s":     int(obj.size),
                    "cr":    round(float(obj.centroid[0]), 2),  # row
                    "cc":    round(float(obj.centroid[1]), 2),  # col
                    "props": json.dumps(obj.properties, sort_keys=True),
                    "rels":  json.dumps(obj.relations, sort_keys=True),
                }
                for obj in objects
            ],
            key=lambda d: (d["c"], d["s"], d["cr"], d["cc"]),
        ),
    }
    canonical_bytes = json.dumps(canonical, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(canonical_bytes, digest_size=32).digest()


# ─── Tile-map identity (Ask 2 answer: per-cell ID array) ───────────────

def build_id_map(objects: list, raw_dicts: list,
                 shape: tuple = (64, 64)) -> tuple:
    """Build the (64,64) per-cell owner-ID array + the stack-dict for overlaps.

    Cell-IDs ARE the identity carrier (Ask 2 answer). When a cell is owned by
    exactly one object, id_map holds that object's stable identity. When
    multiple objects claim a cell (rare for r11l basics; comes up with
    overlapping sprites like body+limb at centroid), id_map holds the
    top-of-stack and id_stack records the full z-order.

    Args:
        objects:   List[ObjectNode] (the contract output)
        raw_dicts: matching list[dict] from parse_objects_raw, kept aligned by index
                   (raw_dicts[i] has "cells" we need; objects[i] has the identity str)
        shape:     grid shape, default (64, 64) for ARC-AGI-3

    Returns:
        (id_map, id_stack) — np.ndarray of int32 (0 = background, otherwise
        per-object identity-hash) and dict {(r,c): [id_str, ...]} for stacked
        cells only.
    """
    # We store identity as uint32 hash for the array (BLAKE2b digest_size=4
    # produces values up to 2^32-1, which exactly fits uint32); the stack
    # dict carries the original identity strings. Hashing is just to fit the
    # dtype; the canonical-state-hash uses the string form so this is decorative.
    id_map = np.zeros(shape, dtype=np.uint32)
    id_stack: dict = {}

    for obj, raw in zip(objects, raw_dicts):
        # Identity hash for the array (collision risk negligible at K=20 objects)
        id_hash = int(hashlib.blake2b(obj.identity.encode(), digest_size=4).hexdigest(), 16)
        if id_hash == 0:
            id_hash = 1  # reserve 0 for background

        for r, c in raw["cells"]:
            if id_map[r, c] != 0:
                # Stacked cell — record both in the stack dict
                prev_owner = id_map[r, c]
                # Find the prev owner's identity string from objects
                prev_identity = next(
                    (o.identity for o, rd in zip(objects, raw_dicts)
                     if int(hashlib.blake2b(o.identity.encode(), digest_size=4).hexdigest(), 16) == prev_owner),
                    f"unknown_{prev_owner}",
                )
                id_stack.setdefault((r, c), [prev_identity]).append(obj.identity)
            id_map[r, c] = id_hash

    return id_map, id_stack


# ─── Adjacency relations from cell-neighbor analysis ───────────────────

def compute_adjacency(objects: list, raw_dicts: list,
                      shape: tuple = (64, 64)) -> dict:
    """For each object, find which other objects share a 4-neighbor cell border.

    Returns: dict {identity: [adjacent_identity, ...]} for all objects with
    at least one adjacent neighbor.
    """
    # Build a quick lookup: cell -> identity
    cell_owner: dict = {}
    for obj, raw in zip(objects, raw_dicts):
        for r, c in raw["cells"]:
            cell_owner[(r, c)] = obj.identity

    adjacency: dict = {obj.identity: set() for obj in objects}

    for obj, raw in zip(objects, raw_dicts):
        for r, c in raw["cells"]:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                    neighbor = cell_owner.get((nr, nc))
                    if neighbor is not None and neighbor != obj.identity:
                        adjacency[obj.identity].add(neighbor)

    return {k: sorted(v) for k, v in adjacency.items() if v}


# ─── The producer class ────────────────────────────────────────────────

class GridObservationProducerV2:
    """The explorer's perception entry point. Stateful across frames within
    a session; reset between sessions or between game changes.

    Schema-stable from MVP onward. Internals will evolve as cortices come
    online; the contract above is what the explorer reads.
    """

    def __init__(self, *, level_id: Optional[str] = None) -> None:
        self.level_id = level_id
        self._prev_objects_raw: list = []     # last frame's raw dicts (with cells)
        self._prev_identities:  dict = {}     # color → list of stable identity strings
        self._next_id_counter: int = 0
        self._seen_identities: set = set()    # for appeared_this_frame detection

    def reset(self) -> None:
        """Clear cross-frame state. Call between games or session boundaries."""
        self._prev_objects_raw = []
        self._prev_identities = {}
        self._next_id_counter = 0
        self._seen_identities = set()

    def _assign_identity(self, curr_raw: dict, prev_match: Optional[dict]) -> str:
        """Assign a stable identity string. If matched to a prior object, reuse
        its identity; otherwise mint a new one."""
        if prev_match is not None and "identity" in prev_match:
            return prev_match["identity"]
        new_id = f"obj_{self._next_id_counter:04d}"
        self._next_id_counter += 1
        return new_id

    def observe(self,
                frame: np.ndarray,
                frame_idx: int,
                last_action: Optional[int] = None,
                filter_colors: Optional[set] = None,
                ) -> GridObservation:
        """Process one frame. Returns the structured observation + state signature.

        Args:
            frame: (64, 64) int8 grid, values 0-15
            frame_idx: monotonic counter from session start
            last_action: action that produced this frame (None for first frame)
            filter_colors: set of color values to treat as UI-overlay noise (skip
                during object extraction). For r11l: {1} drops the Bresenham line
                overlay that connects limbs to body. Per-game value; ideally set
                by LLM-sparse-prior at first-contact.
        """
        # Step 1: raw object extraction via existing primitive
        raw_objects = _parse_objects_raw(frame)

        # Filter UI-overlay colors (per-game configurable)
        if filter_colors:
            raw_objects = [obj for obj in raw_objects if obj["color"] not in filter_colors]

        bg_color = int(np.bincount(frame.ravel().astype(np.int32), minlength=16).argmax())

        # Step 2: cross-frame identity correspondence
        # Use the existing track_objects for "what moved" + nearest-centroid match.
        # We attach a stable identity string to each curr-raw dict.
        moved_records = _track_objects_raw(self._prev_objects_raw, raw_objects)

        # Build a quick lookup: which prev object did each curr object match?
        # track_objects gives us moves keyed by the prev's int id, not by our string identity.
        # We assign identity by: if the curr object's bbox-overlap-fraction with any
        # prev object exceeds a threshold, inherit that prev's identity; else mint new.
        # For MVP: use color + nearest-centroid as the cheap proxy.
        prev_by_color: dict = {}
        for prev in self._prev_objects_raw:
            prev_by_color.setdefault(prev["color"], []).append(prev)

        for curr in raw_objects:
            curr_cent = np.array(curr["centroid"])
            color = curr["color"]
            best_match = None
            best_dist = 10.0
            for prev in prev_by_color.get(color, []):
                d = float(np.linalg.norm(curr_cent - np.array(prev["centroid"])))
                if d < best_dist:
                    best_dist = d
                    best_match = prev
            curr["identity"] = self._assign_identity(curr, best_match)
            curr["_matched_prev"] = best_match  # used below for property derivation

        # Step 3: build ObjectNodes with properties + relations
        adjacency = compute_adjacency(
            [ObjectNode(identity=c["identity"], bbox=tuple(c["bbox"]),
                        centroid=tuple(c["centroid"]), color=c["color"],
                        size=c["size"]) for c in raw_objects],
            raw_objects,
        )

        objects: list = []
        for curr in raw_objects:
            prev = curr.get("_matched_prev")
            identity = curr["identity"]
            appeared = identity not in self._seen_identities
            moved = prev is not None and (
                abs(curr["centroid"][0] - prev["centroid"][0]) > 0.01
                or abs(curr["centroid"][1] - prev["centroid"][1]) > 0.01
            )

            obj = ObjectNode(
                identity=identity,
                bbox=tuple(curr["bbox"]),
                centroid=tuple(curr["centroid"]),
                color=int(curr["color"]),
                size=int(curr["size"]),
                properties={
                    "selected":         False,         # stub: per-game LLM-prior populates
                    "active":           False,         # stub
                    "carried_content":  None,          # stub
                    "moved_this_frame": bool(moved),
                    "appeared_this_frame": bool(appeared),
                    "visible":          True,          # default; lifecycle layer will manage
                    "corporeal":        True,          # default; lifecycle layer will manage
                    "identity_confidence": 1.0 if prev is not None or appeared else 0.5,
                },
                relations={
                    "part_of":   None,                 # stub
                    "on_top_of": [],                   # stub (filled when z-stack populated)
                    "contains":  [],                   # stub
                    "adjacent":  adjacency.get(identity, []),
                },
            )
            objects.append(obj)
            self._seen_identities.add(identity)

        # Step 4: tile-map identity (Ask 2 answer)
        id_map, id_stack = build_id_map(objects, raw_objects)

        # Step 5: canonical state-signature hash
        state_sig = canonical_state_hash(objects, bg_color)

        # Step 6: stash this frame's raw objects (with identities attached) for next frame
        # Strip the _matched_prev key before stashing; we don't need it next time
        for curr in raw_objects:
            curr.pop("_matched_prev", None)
        self._prev_objects_raw = raw_objects

        return GridObservation(
            frame_idx=frame_idx,
            objects=objects,
            state_signature=state_sig,
            id_map=id_map,
            id_stack=id_stack,
            background_color=bg_color,
            perception_confidence=0.85,  # MVP: honest about stubbed fields
        )

    # ─── Path 2: CBP-format adapter ────────────────────────────────────
    #
    # Consumes dev-SAGE's pre-extracted .objstate.jsonl records directly,
    # skipping our parse_objects step. Lets us validate the schema against
    # all 14,798 replay transitions today without needing raw 64×64 frames.
    # Trade-off: we accept CBP's segmentation as ground truth (so we can't
    # cross-check ours against theirs from this code path; that's what raw
    # grids would enable).

    def observe_from_cbp(self,
                         record: dict,
                         frame_idx: int,
                         shape: tuple = (64, 64),
                         background_color: Optional[int] = None,
                         ) -> GridObservation:
        """Convert one CBP-format objstate record to our GridObservation contract.

        Record shape (per CBP_REPLY_TO_MEMBOT_RESPONSE §A):
            {
                "step": int, "level": int, "state": str,
                "action": {"id": str, "x": int?, "y": int?},
                "objects": [{"id": int, "c": int, "x": float, "y": float,
                             "n": int, "mk": int, "ho": int}, ...]
            }

        Field mapping CBP → our schema:
            id  → identity (prefixed for namespace clarity: "cbp_{game_or_session}_{id}")
            c   → color
            x,y → centroid (NOTE: CBP uses (x=col, y=row) per their schema)
            n   → size
            mk  → properties.mk (verbatim — game-specific semantics)
            ho  → properties.ho (verbatim — outline vs filled)

        We DON'T have raw bbox from CBP; we approximate as a square of side √n
        centered at (x,y). This is exactly what CBP uses internally (per his
        reply: "currently use approx bbox = centroid ± √n / 2").

        Identity tracking: CBP's id is stable across most frames (their
        segmenter holds the same label), but per Sprout's m0r0 analysis ~11%
        of frames show referent shifts. For MVP we accept CBP's id as the
        identity; future revisions can layer Hungarian re-assignment on top.
        """
        # Background color resolution:
        #   1. If caller passed it explicitly, use that
        #   2. Otherwise default to 0
        # CBP's .objstate.jsonl doesn't carry background explicitly — bg is
        # the "everything not in an object" implicit. Per-game caller-supplied
        # value (or LLM-sparse-prior config) is the right answer. For r11l
        # specifically, background is color 5 (grey) per the mechanics doc.
        bg_color = background_color if background_color is not None else 0

        cbp_objects = record.get("objects", []) or []
        objects: list = []
        raw_for_idmap: list = []

        for cbp_obj in cbp_objects:
            identity = f"cbp_{self.level_id or 'session'}_{cbp_obj['id']}"
            x = float(cbp_obj.get("x", 0))   # CBP uses (x=col, y=row)
            y = float(cbp_obj.get("y", 0))
            color = int(cbp_obj.get("c", 0))
            size = int(cbp_obj.get("n", 0))
            mk = int(cbp_obj.get("mk", 0))
            ho = int(cbp_obj.get("ho", 0))

            # Approximated bbox: square of side √n centered at (x,y) per CBP convention.
            # Note CBP's (x,y) = (col, row); our bbox is (r_min, c_min, r_max, c_max).
            half = max(1, int(np.sqrt(size) / 2))
            r_center, c_center = int(round(y)), int(round(x))
            r_min = max(0, r_center - half)
            r_max = min(shape[0] - 1, r_center + half)
            c_min = max(0, c_center - half)
            c_max = min(shape[1] - 1, c_center + half)
            bbox = (r_min, c_min, r_max, c_max)

            # Approximated cells: rectangle filled by the bbox
            approx_cells = [
                (r, c)
                for r in range(r_min, r_max + 1)
                for c in range(c_min, c_max + 1)
            ]

            appeared = identity not in self._seen_identities
            # We don't know if it moved without comparing to prev — defer for now;
            # the schema field is honest about uncertainty via identity_confidence
            obj = ObjectNode(
                identity=identity,
                bbox=bbox,
                centroid=(round(y, 2), round(x, 2)),     # convert CBP (x,y) → our (row,col)
                color=color,
                size=size,
                properties={
                    "selected":            False,        # stub — per-game prior populates
                    "active":              False,        # stub
                    "carried_content":     None,         # stub
                    "moved_this_frame":    False,        # would need prev-frame compare
                    "appeared_this_frame": bool(appeared),
                    "visible":             True,
                    "corporeal":           True,
                    "identity_confidence": 0.89,         # CBP's segmenter has ~89% smooth (~11% teleport per Sprout's m0r0 finding)
                    "mk":                  mk,           # CBP verbatim
                    "ho":                  ho,           # CBP verbatim
                },
                relations={
                    "part_of":   None,
                    "on_top_of": [],
                    "contains":  [],
                    "adjacent":  [],  # filled below from adjacency pass
                },
            )
            objects.append(obj)
            # Synthesize a raw-style dict for build_id_map (which expects "cells")
            raw_for_idmap.append({"cells": approx_cells, "color": color})
            self._seen_identities.add(identity)

        # Adjacency pass — same as the raw-path version, just from approximated cells
        adjacency = compute_adjacency(objects, raw_for_idmap, shape=shape)
        for obj in objects:
            obj.relations["adjacent"] = adjacency.get(obj.identity, [])

        # Tile-map identity (Ask 2 answer)
        id_map, id_stack = build_id_map(objects, raw_for_idmap, shape=shape)

        # Canonical state-signature
        state_sig = canonical_state_hash(objects, bg_color)

        return GridObservation(
            frame_idx=frame_idx,
            objects=objects,
            state_signature=state_sig,
            id_map=id_map,
            id_stack=id_stack,
            background_color=bg_color,
            perception_confidence=0.80,  # honest: stubbed properties + approximated cells
        )
