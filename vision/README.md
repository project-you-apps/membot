# Membot Vision

Grid perception and visual memory tools for spatial reasoning tasks.

## GridObservation Producer

Turns a 64×64 discrete color grid into a structured observation for downstream reasoning engines.

### What it does

1. **Connected components** — flood fill per color → objects with id, color, cells, bbox, centroid, size
2. **Frame diff** — cell-level changes between frames (what changed, from what to what)
3. **Object tracking** — match objects across frames by color + nearest centroid → movement deltas
4. **Embedding** — CLIP or hand-crafted features for similarity search across game states
5. **Visual output** — clean game view, annotated analysis view, side-by-side comparison PNGs

### Quick start

```python
from grid_observation import GridObservationProducer

producer = GridObservationProducer()  # loads CLIP once
obs = producer.observe(frame, step_number=0, action_taken=0, level_id="game-1")

# obs.objects      → detected objects with bounding boxes
# obs.changes      → cells that changed since last frame
# obs.moved        → objects that moved + delta vectors
# obs.embedding    → 512-dim CLIP vector for cart search
```

### Swappable encoders

| Encoder | Dims | Speed | Size | Use case |
|---------|------|-------|------|----------|
| CLIPGridEncoder | 512 | ~50-100ms | ~400MB | General purpose (Phase 1) |
| HandcraftedGridEncoder | 128 | ~0.1ms | 0 | Competition fallback, constrained hardware |
| (Future CNN) | TBD | ~5ms | ~5-10MB | Custom-trained on game grids |

### Visual debugging

```bash
# Run the test harness (generates timestamped PNGs)
python grid_observation.py
python grid_observation.py --no-clip          # skip CLIP, use handcrafted
python grid_observation.py --game vc33-9851e02b  # specific game

# Output files:
#   obs_YYYYMMDD_HHMMSS_step001.png       — annotated analysis view
#   compare_YYYYMMDD_HHMMSS_step001.png   — side-by-side game vs analysis
```

### Render modes

- `render_clean(frame)` — game as played, optional gridlines
- `render_observation(obs)` — bounding boxes, diff highlights, info bar
- `render_comparison(obs)` — side-by-side clean + annotated
- `render_diff(prev, curr)` — three-panel: previous / current / diff

### Dependencies

```
scipy          # connected component analysis
pillow         # image rendering
open-clip-torch  # CLIP embedding (optional, skip with --no-clip)
arc-agi        # game SDK (for test harness only)
```

### Integration

This module produces `GridObservation` dataclass instances. Downstream consumers (SAGE, Membot carts) receive the observation via a push buffer. See the interface spec for the full data contract.
