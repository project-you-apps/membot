# Membox Phase 1 — Implementation Spec

**Date**: 2026-04-08
**Status**: Implementation-ready
**Author**: Andy Grossberg + Claude (Opus 4.6)
**Parent spec**: `membox-multiuser-dbms-spec.md` (full Membox vision, 5 phases)
**Built on**: `multi-cart-query-spec.md` (Phase 1 shipped 2026-04-08)
**Sibling**: `federated-cart-spec.md` (Phase 1 shipped 2026-04-08)

---

## Why this spec exists

The full Membox spec covers all five phases (locking → version chains → permissions → admin agent → VPS integration). This document is **just Phase 1**, written tight enough that someone can sit down with it and ship the lock+tagging layer in 1-2 days without revisiting the bigger spec for every decision.

Phase 1 is the smallest possible thing that lets two agents safely write to the same cart without clobbering each other AND records WHO wrote each pattern. That's it. No version chains, no dispute detection, no permissions, no UI. Those come in Phase 2+.

Phase 1 is enough to unblock the demo path:

> Two textareas on a webpage. One says "agent A". One says "agent B". Both write to the same cart. Both writes are visible. Both are attributed. Lock indicator shows who's holding the write mutex right now. That's the 30-second elevator demo of Membox.

---

## What Phase 1 ships

### 1. `membot/membox.py` — new module

A parallel layer alongside `multi_cart.py`. Reuses the multi-cart layer for the underlying mount/search; adds the lock + tagging machinery.

```python
import membox

# Mount a regular brain cart as a Membox-enabled cart
membox.mount("./team_kb.cart", cart_id="team", role="working")

# Two agents writing concurrently
with membox.acquire_lock("team", agent_id="alice", timeout_ms=5000):
    membox.imprint("team", text="The project uses React", agent_id="alice")

with membox.acquire_lock("team", agent_id="bob", timeout_ms=5000):
    membox.imprint("team", text="We migrated to Vue last week", agent_id="bob")

# Read never blocks
results = membox.search("team", "what frontend framework?")
# Each result has metadata.agent_id showing who wrote it
```

### 2. New MCP tools (in `membot_server.py`)

- `membox_mount(cart_path, cart_id, role)` — mount a cart in Membox mode
- `membox_unmount(cart_id)`
- `membox_list()` — list Membox-enabled mounted carts + current lock state
- `membox_acquire_lock(cart_id, agent_id, timeout_ms)`
- `membox_release_lock(cart_id, agent_id)`
- `membox_lock_holder(cart_id)` — returns the agent_id currently holding the lock or null
- `membox_imprint(cart_id, text, agent_id, tags, reasoning)` — convenience wrapper that acquires the lock, imprints, releases. The "do the right thing" path.
- `membox_search(cart_id, query, top_k, agent_id)` — read API; never blocks. Returns results with the per-pattern `agent_id`, `timestamp`, and `reasoning` fields visible.
- `membox_status(cart_id)` — status: lock holder, last write time, write count per agent, pattern count

### 3. Tests

`tests/test_membox.py` exercises:

- **mount/unmount** — same shape as multi_cart tests
- **single-agent imprint + read** — happy path
- **two-agent concurrent imprint** — both succeed, both attributed correctly, ordering preserved by lock
- **lock contention** — agent B's `acquire_lock` blocks until agent A releases
- **lock timeout** — agent B's `acquire_lock(timeout_ms=100)` returns False after 100ms if A still holds
- **lock holder query** — returns correct agent_id mid-write, None when free
- **read never blocks** — agent A holds the lock, agent B's `search` succeeds immediately
- **lock auto-release on crash** — if the holding process dies, the lock becomes available within timeout window
- **agent_id stamping** — every pattern has `agent_id` in its per-pattern metadata after write
- **invalid lock release** — agent B trying to release agent A's lock errors cleanly

---

## Architecture (just enough detail)

### Lock implementation

We use a process-local `threading.Lock` per `cart_id`, wrapped in a class that adds:
- Timeout-aware acquire (`Lock.acquire(timeout=...)`)
- Holder tracking (which `agent_id` currently holds the lock)
- Lease expiration (auto-release if holder hasn't renewed within N seconds — handles crashes)
- Read-never-blocks invariant (reads don't touch the lock at all)

```python
class CartLock:
    def __init__(self, cart_id: str, lease_seconds: int = 30):
        self._cart_id = cart_id
        self._mutex = threading.Lock()
        self._holder: str | None = None
        self._holder_acquired_at: float | None = None
        self._lease_seconds = lease_seconds

    def acquire(self, agent_id: str, timeout_ms: int = 5000) -> bool:
        # 1. Check if current holder's lease has expired (crash recovery)
        if self._holder is not None and self._holder_acquired_at is not None:
            if time.time() - self._holder_acquired_at > self._lease_seconds:
                self._mutex.release()  # forcibly free the lock
                self._holder = None
                self._holder_acquired_at = None

        # 2. Try to acquire within timeout
        acquired = self._mutex.acquire(timeout=timeout_ms / 1000.0)
        if acquired:
            self._holder = agent_id
            self._holder_acquired_at = time.time()
        return acquired

    def release(self, agent_id: str) -> None:
        if self._holder != agent_id:
            raise PermissionError(
                f"Agent {agent_id!r} tried to release lock held by {self._holder!r}"
            )
        self._holder = None
        self._holder_acquired_at = None
        self._mutex.release()

    def holder(self) -> str | None:
        return self._holder
```

**Why threading.Lock and not file-based locking**: Phase 1 is single-process. Multi-process / multi-machine locking is a Phase 4 problem. The Membot server runs in one process and that process serializes access to its mounted Membox carts. If multiple Membot servers ever need to share a cart, that's a *future* concern that needs Redis or a database advisory lock. Don't build it now.

**Why a lease**: handles the "agent crashed while holding the lock" case without manual intervention. Lease default of 30 seconds means a crashed agent's lock auto-releases after 30s. Live agents that need to hold the lock longer can call a `renew_lock` operation (Phase 2 addition, not Phase 1).

### agent_id stamping

Every Membox `imprint` call writes the `agent_id` into the per-pattern metadata blob (the same `per_pattern_meta` field we already use for tags/owner/description/etc.). New required fields per pattern:

```json
{
  "agent_id": "alice",
  "written_at": "2026-04-08T15:30:00Z",
  "origin": "agent",         // "agent" | "human" | "system"
  "reasoning": "Updating React → Vue based on PR #234"  // optional
}
```

These coexist with the existing per-pattern metadata fields (`tags`, `owner`, `description`, `source`, etc.). Membox doesn't replace anything — it adds.

### Cart format compatibility

A Membox-enabled cart is a **regular brain cart** with extra per-pattern metadata fields. There's no separate "Membox cart format." Any existing brain cart can be opened in Membox mode by passing `membox=True` to `mount`. Existing patterns just won't have `agent_id` stamps, which is fine — the search API treats missing `agent_id` as `"unknown"`.

This means:
- **No migration** for existing carts. Any cart works in Membox mode out of the box.
- **No fork** of the cart format. Membox carts can be shared with non-Membox tools (which just ignore the extra fields).
- **Cart Builder doesn't need changes** for Phase 1. It can keep producing regular carts; Membox is purely a runtime layer on top.

### Read API — non-blocking

`membox_search(cart_id, query, ...)` calls through to `multi_cart.search()` for the actual retrieval. The lock is **only** held during writes. Reads bypass the lock entirely and use the same path as a normal multi-cart search. This is the "many readers, one writer" classic concurrency pattern.

The read result format adds the `agent_id`, `written_at`, `origin`, and `reasoning` fields to each result so the consuming agent can see who wrote what.

### How the Membox layer wraps multi_cart

```python
# membox.py module-global state
_membox_carts: dict[str, CartLock] = {}   # cart_id → lock
_membox_metadata: dict[str, dict] = {}    # cart_id → membox config (membox.txt parsed)

def mount(cart_path: str, cart_id: str, role: str = None) -> dict:
    # Just calls through to multi_cart.mount, then registers a lock
    import multi_cart as mc
    result = mc.mount(cart_path, cart_id=cart_id, role=role)
    _membox_carts[cart_id] = CartLock(cart_id)
    return result

def unmount(cart_id: str) -> dict:
    import multi_cart as mc
    if cart_id in _membox_carts:
        del _membox_carts[cart_id]
    return mc.unmount(cart_id)

def imprint(cart_id: str, text: str, agent_id: str, tags: str = "",
            reasoning: str = "") -> dict:
    """Acquire the lock, write the pattern with agent_id stamping, release."""
    if cart_id not in _membox_carts:
        raise ValueError(f"Cart {cart_id!r} not mounted in Membox mode")

    lock = _membox_carts[cart_id]
    if not lock.acquire(agent_id, timeout_ms=5000):
        return {
            "ok": False,
            "error": "lock_timeout",
            "current_holder": lock.holder(),
        }
    try:
        # Build per-pattern metadata with agent_id
        meta = {
            "agent_id": agent_id,
            "written_at": datetime.now(timezone.utc).isoformat(),
            "origin": "agent",
            "reasoning": reasoning or None,
            "tags": tags.split(",") if tags else [],
        }
        # Call into multi_cart to do the actual imprint
        # (multi_cart.imprint with per_pattern_meta does NOT yet exist —
        # see "Adjacent change" below)
        result = mc.imprint_with_meta(cart_id, text, meta)
        return {"ok": True, "agent_id": agent_id, **result}
    finally:
        lock.release(agent_id)
```

### Adjacent change required: `multi_cart.imprint_with_meta`

`multi_cart.py` doesn't currently have an `imprint` function. Phase 1 needs to add one. Signature:

```python
def imprint_with_meta(cart_id: str, text: str, per_pattern_meta: dict) -> dict:
    """Imprint a single passage into a mounted cart with per-pattern metadata.
    Returns {"local_addr": int, "fingerprint_changed": True}.
    """
```

This is ~20 lines: embed the text, append to the cart's `embeddings`/`texts`/`compressed_texts`, append to `per_pattern_meta`, recompute `sign_bits` for the new entry, update the manifest. The save-to-disk happens at the end (with the same fingerprint update we do in `_save_federated_cart`).

This is a small addition and lives naturally in `multi_cart.py` because every Membot subsystem (single-cart, federated, Membox) needs the ability to imprint into a mounted cart. Phase 1 of Membox just *uses* it; the function is general-purpose.

---

## What Phase 1 deliberately does NOT include

| Feature | Phase | Why deferred |
|---|---|---|
| Version chains (SUPERSEDED edges) | 2 | Needs the dispute detection design first |
| Dispute detection (semantic conflict) | 2 | Needs version chains as the storage substrate |
| Tombstone-based delete | 2 | Comes with version chains naturally |
| `membox.txt` permission file | 3 | Needs an auth layer first; not blocking the demo |
| Agent registration / trust scoring | 3 | Same |
| Admin agent template | 4 | Polish, not core |
| HITL dispute resolution UI | 5 | Needs VPS frontend, separate work |
| Multi-process / multi-machine locking | Future | Not needed for Phase 1 demo |
| Lock renewal API | 2 | Lease auto-release covers Phase 1 needs |

---

## File checklist

**New files**:
- `membot/membox.py` (~250 lines)
- `tests/test_membox.py` (~200 lines)

**Modified files**:
- `membot/multi_cart.py` — add `imprint_with_meta()` (~30 lines)
- `membot/membot_server.py` — add 9 new MCP tools wrapping the membox API (~200 lines)
- `docs/RFC/membox-multiuser-dbms-spec.md` — Amendment 2 noting Phase 1 ship + linking to this spec (~10 lines)
- `README.md` — extend the "What's New" section with the Membox row (~20 lines)
- `membot/docs/DEVLOG.md` — entry for the Membox Phase 1 ship

**Total**: ~700 lines of new code + ~250 lines of tests + docs.

---

## Test plan

```bash
cd membot
python tests/test_membox.py
```

Expected output: 9-10 tests pass. The concurrent-writer test uses Python's `threading` module to spin up two threads, both calling `imprint` against the same cart, and asserts:
- Both writes succeed
- The cart has both patterns (verified by search)
- Each pattern has the correct `agent_id` in its metadata
- The writes were serialized (timestamps don't overlap during the actual write window)

The lock timeout test asserts that `acquire(timeout_ms=100)` returns `False` within ~100ms if the lock is held.

The crash recovery test acquires a lock, deletes the lock object (simulating crash), waits for the lease to expire, and verifies a new agent can acquire the lock without error.

---

## Demo path after Phase 1 lands

1. **Phase 1 (1-2 days)**: this spec
2. **Cart Builder UX in VPS (1-2 days)**: existing Cart Builder GUI ported into the VPS React frontend so users can build a cart from documents in the browser. Lives at `project-you.app/studio/build`.
3. **Phase 2 (3-5 days)**: version chains + dispute detection. The bigger lift.
4. **Tiny VPS UI for Membox visualization (1-2 days)**: two textareas labeled "Agent A" and "Agent B" that POST to `membox_imprint`. A lock indicator showing the current holder. After Phase 2 lands, also a side-by-side dispute resolution panel.
5. **90-second demo video (1 day)**: screen capture of building a cart, two agents writing, dispute appearing, resolution.

**Total: ~10 working days to a pitch-ready demo.**

---

## Open questions for Phase 2

- **Dispute detection threshold**: how high should the cosine similarity be before two patterns are flagged as "potentially in conflict"? Probably configurable per cart via `membox.txt`. Start at 0.75, tune based on test data.
- **Version chain navigation**: do we walk forward (latest first) or backward (original first) by default in search results? Probably "latest first, with `?include_history=true` to get the chain."
- **Resolution UI affordances**: HITL is the v1 resolver. Agent vote is v2. Do we want a "majority of N agents" auto-resolve as v1.5, or is HITL good enough until V1 demo lands?

These are Phase 2 problems. File and move on.

---

*Phase 1 is the smallest meaningful Membox. Two agents, one cart, real lock, real attribution. Anything beyond that is Phase 2 or later.*
