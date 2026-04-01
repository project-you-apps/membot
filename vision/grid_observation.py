"""
grid_observation.py — GridObservation producer for ARC-AGI-3

Receives a 64x64 int8 game grid, parses it into objects, computes frame diffs,
embeds with CLIP, and packages as a GridObservation for SAGE to consume.

Usage:
    from grid_observation import GridObservationProducer

    producer = GridObservationProducer()  # loads CLIP once
    obs = producer.observe(frame, step_number=0, action_taken=0, level_id="g50t-1")
    # obs is a GridObservation dataclass ready for SAGE

NOTE: Andy runs all tests. Do not run test scripts from Claude Code.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from scipy import ndimage
from PIL import Image

# ─── Color map for rendering grids as RGB images ───────────────────────
COLOR_MAP = {
    0: [40, 40, 40],    1: [255, 0, 0],      2: [0, 200, 0],     3: [255, 255, 0],
    4: [0, 0, 255],     5: [255, 0, 255],     6: [0, 255, 255],   7: [255, 128, 0],
    8: [128, 128, 255],  9: [255, 255, 255],  10: [128, 0, 0],    11: [0, 128, 0],
    12: [128, 128, 0],  13: [0, 0, 128],     14: [128, 0, 128],  15: [0, 128, 128],
}


# ─── GridObservation dataclass ─────────────────────────────────────────
@dataclass
class GridObservation:
    """What the perception layer hands to SAGE each frame."""

    # Raw grid
    frame_raw: np.ndarray              # (64, 64) int8, values 0-15

    # Structured perception (connected component analysis)
    objects: List[Dict]                # [{"id": int, "color": int, "cells": [[r,c],...], "bbox": [r1,c1,r2,c2], "centroid": [r,c], "size": int}]

    # Frame diff (what changed since last frame)
    changes: List[Dict]                # [{"cell": [r,c], "was": int, "now": int}]
    moved: List[Dict]                  # [{"id": int, "color": int, "from_bbox": [...], "to_bbox": [...], "delta": [dr,dc]}]

    # Embedding (for cartridge search)
    embedding: np.ndarray              # CLIP or custom encoding vector

    # Context
    step_number: int
    action_taken: int                  # 0 = no action yet (first frame), 1-5 = last action
    level_id: str

    # Stats
    num_objects: int = 0
    num_changes: int = 0
    num_colors: int = 0
    background_color: int = 0          # most frequent color (assumed background)

    # Optional: perception notes
    perception_notes: Optional[str] = None


# ─── Connected component analysis ─────────────────────────────────────
def parse_objects(frame: np.ndarray, background: int = None) -> List[Dict]:
    """Extract connected components from a 64x64 discrete color grid.

    Args:
        frame: (64, 64) int8 grid, values 0-15
        background: color index to treat as background (skip). Auto-detected if None.

    Returns:
        List of object dicts with id, color, cells, bbox, centroid, size.
    """
    if background is None:
        # Most frequent color is background
        counts = np.bincount(frame.ravel().astype(np.int32), minlength=16)
        background = int(np.argmax(counts))

    objects = []
    obj_id = 0

    for color in range(16):
        if color == background:
            continue

        mask = (frame == color)
        if not mask.any():
            continue

        # Label connected components for this color
        labeled, n_components = ndimage.label(mask)

        for comp in range(1, n_components + 1):
            cells = np.argwhere(labeled == comp)
            if len(cells) == 0:
                continue

            r_min, c_min = cells.min(axis=0)
            r_max, c_max = cells.max(axis=0)
            centroid = cells.mean(axis=0)

            objects.append({
                "id": obj_id,
                "color": int(color),
                "cells": cells.tolist(),
                "bbox": [int(r_min), int(c_min), int(r_max), int(c_max)],
                "centroid": [round(float(centroid[0]), 2), round(float(centroid[1]), 2)],
                "size": len(cells),
            })
            obj_id += 1

    return objects


# ─── Frame diff ────────────────────────────────────────────────────────
def compute_diff(prev_frame: np.ndarray, curr_frame: np.ndarray) -> List[Dict]:
    """Compute cell-level changes between two frames.

    Returns list of {"cell": [r,c], "was": int, "now": int}.
    """
    if prev_frame is None:
        return []

    diff_mask = prev_frame != curr_frame
    changed = np.argwhere(diff_mask)

    changes = []
    for cell in changed:
        r, c = int(cell[0]), int(cell[1])
        changes.append({
            "cell": [r, c],
            "was": int(prev_frame[r, c]),
            "now": int(curr_frame[r, c]),
        })

    return changes


# ─── Object tracking ──────────────────────────────────────────────────
def track_objects(prev_objects: List[Dict], curr_objects: List[Dict],
                  max_distance: float = 10.0) -> List[Dict]:
    """Match objects between frames by color + nearest centroid.

    Returns list of {"id": int, "color": int, "from_bbox": [...], "to_bbox": [...], "delta": [dr, dc]}.
    Only includes objects that actually moved.
    """
    if not prev_objects or not curr_objects:
        return []

    moved = []

    # Group by color for matching
    prev_by_color = {}
    for obj in prev_objects:
        prev_by_color.setdefault(obj["color"], []).append(obj)

    curr_by_color = {}
    for obj in curr_objects:
        curr_by_color.setdefault(obj["color"], []).append(obj)

    # Match within each color group
    for color in prev_by_color:
        if color not in curr_by_color:
            continue

        prev_group = prev_by_color[color]
        curr_group = curr_by_color[color]

        # Simple nearest-centroid matching
        used = set()
        for p_obj in prev_group:
            p_cent = np.array(p_obj["centroid"])
            best_dist = max_distance
            best_match = None

            for i, c_obj in enumerate(curr_group):
                if i in used:
                    continue
                c_cent = np.array(c_obj["centroid"])
                dist = np.linalg.norm(p_cent - c_cent)
                if dist < best_dist and dist > 0.01:  # moved at all?
                    best_dist = dist
                    best_match = i

            if best_match is not None:
                used.add(best_match)
                c_obj = curr_group[best_match]
                delta = [
                    round(c_obj["centroid"][0] - p_obj["centroid"][0], 2),
                    round(c_obj["centroid"][1] - p_obj["centroid"][1], 2),
                ]
                moved.append({
                    "id": p_obj["id"],
                    "color": color,
                    "from_bbox": p_obj["bbox"],
                    "to_bbox": c_obj["bbox"],
                    "delta": delta,
                })

    return moved


# ─── Grid rendering ───────────────────────────────────────────────────
def render_grid(frame: np.ndarray, scale: int = 4) -> Image.Image:
    """Render a 64x64 grid to an RGB PIL Image (scaled up for CLIP)."""
    rgb = np.zeros((*frame.shape, 3), dtype=np.uint8)
    for val, color in COLOR_MAP.items():
        mask = frame == val
        rgb[mask] = color
    img = Image.fromarray(rgb)
    return img.resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)


# ─── GridObservation Producer ─────────────────────────────────────────
class GridObservationProducer:
    """Produces GridObservation from raw game frames.

    Handles CLIP model loading, frame history, and object tracking state.
    """

    def __init__(self, use_clip: bool = True, clip_model: str = "ViT-B-32",
                 clip_pretrained: str = "laion2b_s34b_b79k", render_scale: int = 4):
        """Initialize the producer.

        Args:
            use_clip: If True, loads CLIP for embedding. If False, uses zero vector.
            clip_model: CLIP model name for open_clip.
            clip_pretrained: Pretrained weights name.
            render_scale: Scale factor for rendering grids before CLIP encoding.
        """
        self.use_clip = use_clip
        self.render_scale = render_scale
        self.clip_model = None
        self.clip_preprocess = None
        self.embedding_dim = 512  # ViT-B-32 default

        if use_clip:
            self._load_clip(clip_model, clip_pretrained)

        # State for tracking across frames
        self.prev_frame = None
        self.prev_objects = None
        self.current_level = None

    def _load_clip(self, model_name: str, pretrained: str):
        """Load CLIP model (one-time, ~400MB)."""
        import open_clip
        import torch

        print(f"[GridVision] Loading CLIP {model_name}...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.clip_model.eval()
        self.embedding_dim = 512  # ViT-B-32
        self._torch = torch
        print(f"[GridVision] CLIP ready ({self.embedding_dim}-dim)")

    def _embed_frame(self, frame: np.ndarray) -> np.ndarray:
        """Embed a game frame using CLIP.

        Renders the grid as an RGB image, preprocesses for CLIP, encodes.
        Returns a normalized embedding vector.
        """
        if self.clip_model is None:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        img = render_grid(frame, scale=self.render_scale)
        img_tensor = self.clip_preprocess(img).unsqueeze(0)

        with self._torch.no_grad():
            emb = self.clip_model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb[0].numpy().astype(np.float32)

    def observe(self, frame: np.ndarray, step_number: int, action_taken: int,
                level_id: str, perception_notes: str = None) -> GridObservation:
        """Produce a GridObservation from a raw game frame.

        Args:
            frame: (64, 64) int8/uint8 grid, values 0-15
            step_number: Current step in the game
            action_taken: Action that produced this frame (0 = initial)
            level_id: Level/environment identifier
            perception_notes: Optional text annotation from our layer

        Returns:
            GridObservation ready for SAGE consumption
        """
        frame = frame.astype(np.int8)

        # Reset tracking state on level change
        if level_id != self.current_level:
            self.prev_frame = None
            self.prev_objects = None
            self.current_level = level_id

        # Background detection
        counts = np.bincount(frame.ravel().astype(np.int32), minlength=16)
        background = int(np.argmax(counts))

        # Connected components
        objects = parse_objects(frame, background=background)

        # Frame diff
        changes = compute_diff(self.prev_frame, frame)

        # Object tracking
        moved = track_objects(self.prev_objects, objects) if self.prev_objects else []

        # Embedding
        embedding = self._embed_frame(frame)

        # Stats
        unique_colors = len(np.unique(frame))

        # Auto-generate perception notes if not provided
        if perception_notes is None and (changes or moved):
            parts = []
            if objects:
                parts.append(f"{len(objects)} objects detected")
            if changes:
                parts.append(f"{len(changes)} cells changed")
            if moved:
                parts.append(f"{len(moved)} objects moved")
            perception_notes = ". ".join(parts) + "." if parts else None

        # Build observation
        obs = GridObservation(
            frame_raw=frame,
            objects=objects,
            changes=changes,
            moved=moved,
            embedding=embedding,
            step_number=step_number,
            action_taken=action_taken,
            level_id=level_id,
            num_objects=len(objects),
            num_changes=len(changes),
            num_colors=unique_colors,
            background_color=background,
            perception_notes=perception_notes,
        )

        # Update state for next frame
        self.prev_frame = frame.copy()
        self.prev_objects = objects

        return obs

    def reset(self):
        """Reset tracking state (call between levels/games)."""
        self.prev_frame = None
        self.prev_objects = None
        self.current_level = None


# ─── Swappable encoder interface ──────────────────────────────────────
class GridEncoder:
    """Base class for swappable grid encoders."""
    def encode(self, frame: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


class CLIPGridEncoder(GridEncoder):
    """CLIP-based encoder (~400MB, ~50-100ms). General purpose."""
    def __init__(self, model="ViT-B-32", pretrained="laion2b_s34b_b79k", scale=4):
        import open_clip, torch
        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.model.eval()
        self.scale = scale
        self._dim = 512

    def encode(self, frame: np.ndarray) -> np.ndarray:
        img = render_grid(frame, scale=self.scale)
        t = self.preprocess(img).unsqueeze(0)
        with self._torch.no_grad():
            emb = self.model.encode_image(t)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].numpy().astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim


class HandcraftedGridEncoder(GridEncoder):
    """Hand-crafted features. Zero dependencies, ~0.1ms. For constrained environments."""
    def __init__(self, dim=128):
        self._dim = dim

    def encode(self, frame: np.ndarray) -> np.ndarray:
        features = []

        # Color histogram (16 bins, normalized)
        counts = np.bincount(frame.ravel().astype(np.int32), minlength=16).astype(np.float32)
        counts /= counts.sum() + 1e-8
        features.extend(counts.tolist())  # 16

        # Row/col activity (how many non-background cells per row/col)
        bg = int(np.argmax(counts))
        active = (frame != bg).astype(np.float32)
        row_activity = active.mean(axis=1)  # 64
        col_activity = active.mean(axis=0)  # 64
        # Subsample to 16 each
        features.extend(row_activity[::4].tolist())  # 16
        features.extend(col_activity[::4].tolist())  # 16

        # Quadrant density (4 quadrants x 16 colors = 64)
        h, w = frame.shape
        for qr in [slice(0, h//2), slice(h//2, h)]:
            for qc in [slice(0, w//2), slice(w//2, w)]:
                q = frame[qr, qc].ravel().astype(np.int32)
                qcounts = np.bincount(q, minlength=16).astype(np.float32)
                qcounts /= qcounts.sum() + 1e-8
                features.extend(qcounts.tolist())  # 16

        # Symmetry scores (horizontal, vertical)
        h_sym = np.mean(frame == frame[:, ::-1])
        v_sym = np.mean(frame == frame[::-1, :])
        features.extend([float(h_sym), float(v_sym)])  # 2

        # Connected component count per color (16)
        for c in range(16):
            if c == bg:
                features.append(0.0)
                continue
            mask = (frame == c)
            if not mask.any():
                features.append(0.0)
                continue
            _, n = ndimage.label(mask)
            features.append(float(n))

        # Pad or truncate to target dim
        vec = np.array(features[:self._dim], dtype=np.float32)
        if len(vec) < self._dim:
            vec = np.pad(vec, (0, self._dim - len(vec)))

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec

    @property
    def dim(self) -> int:
        return self._dim


# ─── Visual utilities ─────────────────────────────────────────────────
def render_clean(frame: np.ndarray, scale: int = 8, gridlines: bool = False) -> Image.Image:
    """Render a grid as a clean image — what the game looks like to play.

    Args:
        frame: (64, 64) int8 grid
        scale: Pixel scale factor
        gridlines: If True, draw faint gridlines between cells

    Returns:
        PIL Image, clean game view
    """
    from PIL import ImageDraw

    img = render_grid(frame, scale=scale)

    if gridlines:
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for r in range(0, h, scale):
            draw.line([(0, r), (w, r)], fill=(60, 60, 60), width=1)
        for c in range(0, w, scale):
            draw.line([(c, 0), (c, h)], fill=(60, 60, 60), width=1)

    return img


def render_comparison(obs: GridObservation, scale: int = 8) -> Image.Image:
    """Side-by-side: clean game view (left) + annotated analysis view (right).

    This is for Andy to compare "what the game looks like" vs "what we're storing."
    """
    from PIL import ImageDraw

    clean = render_clean(obs.frame_raw, scale=scale, gridlines=True)
    annotated = render_observation(obs, scale=scale)

    # Side by side with labels
    gap = 20
    total_w = clean.width + annotated.width + gap
    max_h = max(clean.height, annotated.height)
    combined = Image.new("RGB", (total_w, max_h + 30), (30, 30, 30))
    combined.paste(clean, (0, 0))
    combined.paste(annotated, (clean.width + gap, 0))

    draw = ImageDraw.Draw(combined)
    draw.text((10, max_h + 5), "Game View (what the agent sees)", fill=(150, 200, 150))
    draw.text((clean.width + gap + 10, max_h + 5), "Analysis View (what we store)", fill=(200, 200, 150))

    return combined


def render_observation(obs: GridObservation, scale: int = 8, show_objects: bool = True,
                       show_diff: bool = True) -> Image.Image:
    """Render a GridObservation as a detailed PNG with annotations.

    Args:
        obs: GridObservation to render
        scale: Pixel scale factor (8 = 512x512 output)
        show_objects: Draw bounding boxes around detected objects
        show_diff: Highlight changed cells in red overlay

    Returns:
        PIL Image with grid + annotations
    """
    from PIL import ImageDraw, ImageFont

    # Base grid render
    img = render_grid(obs.frame_raw, scale=scale)
    draw = ImageDraw.Draw(img)

    # Draw object bounding boxes
    if show_objects:
        OBJ_COLORS = [(255, 255, 0), (0, 255, 255), (255, 128, 0), (128, 255, 0),
                       (255, 0, 128), (0, 128, 255), (255, 255, 128), (128, 255, 255)]
        for obj in obs.objects:
            r1, c1, r2, c2 = obj["bbox"]
            box_color = OBJ_COLORS[obj["id"] % len(OBJ_COLORS)]
            # Scale coordinates
            x1, y1 = c1 * scale, r1 * scale
            x2, y2 = (c2 + 1) * scale - 1, (r2 + 1) * scale - 1
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
            # Label with id and color
            label = f"#{obj['id']} c{obj['color']}"
            draw.text((x1 + 2, y1 + 2), label, fill=box_color)

    # Highlight changed cells
    if show_diff and obs.changes:
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        for change in obs.changes:
            r, c = change["cell"]
            x1, y1 = c * scale, r * scale
            x2, y2 = (c + 1) * scale, (r + 1) * scale
            overlay_draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 80))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # Add info bar at bottom
    info_height = 60
    info_img = Image.new("RGB", (img.width, img.height + info_height), (30, 30, 30))
    info_img.paste(img, (0, 0))
    info_draw = ImageDraw.Draw(info_img)

    info_text = (f"Step {obs.step_number} | Action {obs.action_taken} | "
                 f"{obs.num_objects} objects | {obs.num_changes} changes | "
                 f"{obs.num_colors} colors | bg={obs.background_color}")
    info_draw.text((10, img.height + 5), info_text, fill=(200, 200, 200))

    if obs.perception_notes:
        info_draw.text((10, img.height + 25), obs.perception_notes[:80], fill=(150, 200, 150))

    if obs.moved:
        move_text = " | ".join([f"obj c{m['color']} Δ{m['delta']}" for m in obs.moved[:3]])
        info_draw.text((10, img.height + 42), f"Moved: {move_text}", fill=(200, 200, 150))

    return info_img


def render_diff(prev_frame: np.ndarray, curr_frame: np.ndarray, scale: int = 8) -> Image.Image:
    """Render a side-by-side diff of two frames.

    Left = previous, Center = current, Right = diff (changed cells highlighted).
    """
    img_prev = render_grid(prev_frame, scale=scale)
    img_curr = render_grid(curr_frame, scale=scale)

    # Diff image: show current with changed cells highlighted
    diff_mask = prev_frame != curr_frame
    diff_rgb = np.zeros((*curr_frame.shape, 3), dtype=np.uint8)
    for val, color in COLOR_MAP.items():
        mask = curr_frame == val
        diff_rgb[mask] = color
    # Highlight changes in bright red
    diff_rgb[diff_mask] = [255, 50, 50]
    img_diff = Image.fromarray(diff_rgb).resize(
        (curr_frame.shape[1] * scale, curr_frame.shape[0] * scale), Image.NEAREST)

    # Compose side by side
    total_w = img_prev.width * 3 + 20  # 10px gap between each
    combined = Image.new("RGB", (total_w, img_prev.height + 30), (30, 30, 30))
    combined.paste(img_prev, (0, 0))
    combined.paste(img_curr, (img_prev.width + 10, 0))
    combined.paste(img_diff, (img_prev.width * 2 + 20, 0))

    draw = ImageDraw.Draw(combined)
    draw.text((10, img_prev.height + 5), "Previous", fill=(150, 150, 150))
    draw.text((img_prev.width + 20, img_prev.height + 5), "Current", fill=(150, 150, 150))
    draw.text((img_prev.width * 2 + 30, img_prev.height + 5), "Diff (red=changed)", fill=(255, 100, 100))

    return combined


def save_observation(obs: GridObservation, path: str, scale: int = 8):
    """Save a GridObservation as an annotated PNG."""
    img = render_observation(obs, scale=scale)
    img.save(path)
    print(f"[GridVision] Saved observation to {path}")


def save_diff(prev_frame: np.ndarray, curr_frame: np.ndarray, path: str, scale: int = 8):
    """Save a side-by-side diff as PNG."""
    img = render_diff(prev_frame, curr_frame, scale=scale)
    img.save(path)
    print(f"[GridVision] Saved diff to {path}")


# ─── Test harness (run manually, not from Claude Code) ────────────────
if __name__ == "__main__":
    """
    Test usage — run this yourself:
        python grid_observation.py

    Requires: pip install arc-agi open-clip-torch scipy pillow
    """
    import arc_agi, sys, time
    sys.stdout.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("  GridObservation Producer — Test Harness")
    print("=" * 60)
    print()
    print("Run this script manually to test.")
    print("  python grid_observation.py")
    print("  python grid_observation.py --no-clip  (skip CLIP, use handcrafted)")
    print("  python grid_observation.py --game vc33-9851e02b")
    print()

    use_clip = "--no-clip" not in sys.argv
    game_id = "g50t-5849a774"
    for i, arg in enumerate(sys.argv):
        if arg == "--game" and i + 1 < len(sys.argv):
            game_id = sys.argv[i + 1]

    # Initialize
    producer = GridObservationProducer(use_clip=use_clip)

    # Load game
    arcade = arc_agi.Arcade()
    env = arcade.make(game_id)
    state = env.reset()

    print(f"Game: {game_id}")
    print(f"Actions: {state.available_actions}")
    print(f"Levels to win: {state.win_levels}")
    print()

    # Observe initial frame
    t0 = time.time()
    obs = producer.observe(state.frame[0], step_number=0, action_taken=0, level_id=f"{game_id}-L1")
    t_obs = (time.time() - t0) * 1000

    print(f"Frame 0:")
    print(f"  Objects: {obs.num_objects}")
    print(f"  Colors: {obs.num_colors} (background={obs.background_color})")
    print(f"  Changes: {obs.num_changes}")
    print(f"  Embedding: {obs.embedding.shape} (norm={np.linalg.norm(obs.embedding):.4f})")
    print(f"  Time: {t_obs:.1f}ms")
    if obs.perception_notes:
        print(f"  Notes: {obs.perception_notes}")
    print()

    # Take some actions and observe
    for step in range(1, 6):
        actions = state.available_actions
        if not actions:
            break
        action = actions[step % len(actions)]
        state = env.step(action)

        t0 = time.time()
        obs = producer.observe(state.frame[0], step_number=step, action_taken=action,
                               level_id=f"{game_id}-L1")
        t_obs = (time.time() - t0) * 1000

        print(f"Frame {step} (action={action}):")
        print(f"  Objects: {obs.num_objects}, Changes: {obs.num_changes}, Moved: {len(obs.moved)}")
        print(f"  Time: {t_obs:.1f}ms")
        if obs.perception_notes:
            print(f"  Notes: {obs.perception_notes}")

        # Show moved objects
        for m in obs.moved:
            print(f"    Moved: color={m['color']} delta={m['delta']}")

        # Save annotated PNG + comparison view
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_observation(obs, f"obs_{ts}_step{step:03d}.png", scale=8)
        comp = render_comparison(obs, scale=8)
        comp.save(f"compare_{ts}_step{step:03d}.png")
        print(f"  Saved obs_{ts}_step{step:03d}.png + compare_{ts}_step{step:03d}.png")

        # Save diff if there were changes
        if obs.changes and producer.prev_frame is not None:
            # prev_frame was updated, but we can show current vs initial
            pass  # diff saved below

    # Save a final annotated observation
    save_observation(obs, "obs_final.png", scale=8)

    print()
    print(f"PNGs saved to current directory. Open obs_step*.png to see annotated grids.")
    print("Done. Observations ready for SAGE.")
