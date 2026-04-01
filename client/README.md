# Membot Client Library

Python client for interacting with Membot federated cartridges via HTTP API.

## Features

### Core Cartridge Operations
- Read/write cartridges via HTTP or filesystem fallback
- Track winning sequences and action effectiveness
- Store strategic insights with confidence scores
- Persist experience across sessions

### Visual Memory (New)
- **Frame Storage**: Store numpy arrays as base64-encoded PNG in JSON
- **Frame Retrieval**: Lossless reconstruction of stored frames
- **Action Outcomes**: Track before/after visual pairs
- **Visual Similarity**: Pixel-wise comparison for pattern matching
- **Visual Search**: Find similar frames by threshold

## Installation

```bash
# Install dependencies
pip install numpy pillow requests

# Copy client to your project
cp client/membot_cartridge.py your_project/
```

## Quick Start

### Basic Usage

```python
from membot_cartridge import MembotCartridge

# Create cartridge for your agent/game
cart = MembotCartridge("my-agent-session")
cart.read()

# Store winning sequences
cart.add_winning_sequence(
    level=1,
    actions=[1, 1, 6, 2, 6],  # Action integers
    state_hashes=["abc123", "def456", ...]  # Optional
)

# Add strategic insights from your reasoning
cart.add_strategic_insight(
    "Pattern X enables action Y",
    confidence=0.85
)

# Update action effectiveness from experience
cart.update_action_effectiveness(driver)

# Persist to membot server or filesystem
cart.write()
```

### Visual Memory

```python
import numpy as np
from membot_cartridge import MembotCartridge

cart = MembotCartridge("vision-agent")
cart.read()

# Store a frame snapshot (e.g., initial state, goal state)
frame = np.random.randint(0, 16, (64, 64), dtype=np.uint8)
cart.store_frame_snapshot(
    "level_1_initial",
    frame,
    metadata={"level": 1, "step": 0, "description": "Initial state"}
)

# Store action visual outcome (before/after pair)
before_frame = get_current_frame()
execute_action(6)
after_frame = get_current_frame()

cart.store_action_visual_outcome(
    action=6,
    before_frame=before_frame,
    after_frame=after_frame,
    level=1,
    step=5
)

# Compute visual similarity
similarity = cart.compute_visual_similarity(frame_a, frame_b)
print(f"Similarity: {similarity:.3f}")  # 0.0 = completely different, 1.0 = identical

# Find similar snapshots
similar = cart.find_similar_snapshots(
    current_frame,
    threshold=0.85  # 85% similarity
)
for label, score in similar:
    print(f"{label}: {score:.3f}")

# Retrieve stored frame
retrieved = cart.get_frame_snapshot("level_1_initial")
assert np.array_equal(frame, retrieved)  # Lossless reconstruction
```

## Visual Memory Architecture

### Frame Encoding
1. Input: numpy array (H, W) with values 0-15 (ARC colors)
2. Scale: 0-15 → 0-255 for PNG visibility
3. Encode: Grayscale PNG → base64 string
4. Store: JSON-safe string in cartridge

### Frame Decoding
1. Decode: base64 string → PNG bytes
2. Load: PNG → PIL Image → numpy array
3. Scale back: 0-255 → 0-15
4. Output: Original numpy array (lossless)

### Storage Efficiency
- Base64 PNG provides ~10-20x compression vs raw numpy
- Keeps cartridges JSON-serializable for git/federation
- Works with membot HTTP API and filesystem fallback

## Cartridge Structure

```json
{
  "game_family": "example",
  "winning_sequences": [
    {
      "level": 1,
      "actions": [1, 1, 6],
      "action_names": ["UP", "UP", "CLICK"],
      "state_hashes": ["abc", "def", "ghi"],
      "learned": 1234567890.0
    }
  ],
  "action_effectiveness": {
    "global": {"UP": 0.75, "CLICK": 0.45},
    "state_dependent": {
      "state_hash_abc": {"UP": 0.9, "CLICK": 0.3}
    }
  },
  "strategic_insights": [
    "Pattern X enables Y (confidence: 85.0%)"
  ],
  "visual_memory": {
    "snapshots": {
      "level_1_initial": {
        "frame_b64": "iVBORw0KGgo...",
        "metadata": {"level": 1, "step": 0},
        "timestamp": 1234567890.0
      }
    },
    "action_outcomes": [
      {
        "action": 6,
        "before_b64": "iVBORw0KGgo...",
        "after_b64": "iVBORw0KGgo...",
        "level": 1,
        "step": 5,
        "timestamp": 1234567890.0
      }
    ]
  },
  "total_attempts": 42,
  "best_score": {"levels": 3, "steps": 150},
  "created": 1234567890.0,
  "last_updated": 1234567890.0
}
```

## Testing

```bash
# Run visual memory test suite
python client/test_vision_memory.py
```

Expected output:
```
======================================================================
MEMBOT VISION MEMORY TEST
======================================================================
✓ Frame storage (numpy → base64 PNG)
✓ Frame retrieval (base64 PNG → numpy)
✓ Action outcome tracking (before/after)
✓ Visual similarity computation
✓ Visual search (find similar frames)
```

## Use Cases

### Spatial Reasoning Games
- Store goal states visually
- Compare current frame to goals
- Track which actions move toward goals
- Learn rotation/transformation patterns

### Vision-Based RL
- Store successful trajectory frames
- Identify similar states for transfer learning
- Visual similarity for reward shaping
- Before/after action outcome analysis

### Debug & Visualization
- Snapshot critical moments
- Track visual state evolution
- Identify when agent gets stuck (high similarity to initial)
- Visual diff for debugging

## API Reference

### MembotCartridge

#### Core Methods
- `read()` → dict: Load cartridge from HTTP or filesystem
- `write(data=None)`: Save cartridge to HTTP or filesystem
- `add_winning_sequence(level, actions, state_hashes=None)`: Record winning path
- `add_strategic_insight(hypothesis, confidence)`: Store high-confidence patterns
- `update_action_effectiveness(driver)`: Update from Driver stats
- `update_best_score(levels, steps)`: Track personal best
- `increment_attempts()`: Count total runs

#### Visual Memory Methods
- `store_frame_snapshot(label, frame, metadata=None)`: Store labeled frame
- `get_frame_snapshot(label)` → np.ndarray: Retrieve stored frame
- `store_action_visual_outcome(action, before, after, level, step)`: Track action effects
- `compute_visual_similarity(frame_a, frame_b)` → float: Compare frames [0,1]
- `find_similar_snapshots(target, threshold=0.8)` → list: Search by similarity

### Configuration

**Environment:**
- `MEMBOT_URL`: HTTP endpoint (default: `http://localhost:8000`)
- Filesystem fallback: `~/.membot/cartridges/` or configurable `CARTRIDGE_DIR`

**Cartridge Naming:**
- Format: `arc-agi-3/{game_family}` for membot HTTP paths
- Filesystem: `{game_family}.json` in cartridge directory

## License

Same as parent membot project (check root LICENSE file).
