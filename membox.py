"""
membox.py — Multi-user shared brain cart layer for Membot.

The third mode of the three-mode framework (single-user / federated / multiuser).
Lets multiple agents safely write to the same brain cart with per-agent
attribution and a write mutex that guarantees serialization without blocking
reads.

Spec: docs/RFC/membox-phase1-implementation.md
Parent spec: docs/RFC/membox-multiuser-dbms-spec.md
Built on: multi_cart.py (the multi-cart pool is the substrate)

Phase 1 scope (this file)
=========================
- Per-cart write mutex (CartLock) with lease-based crash recovery
- agent_id stamping on every write (per-pattern metadata)
- Read API that never blocks (delegates to multi_cart.search)
- Mount/unmount/list parallel to multi_cart's API
- imprint() convenience that acquires lock → writes → releases atomically

NOT in Phase 1
==============
- Version chains (SUPERSEDED edges) — Phase 2
- Dispute detection (semantic conflict) — Phase 2
- Tombstone delete — Phase 2
- membox.txt permissions — Phase 3
- Multi-process / multi-machine locking — future

Architecture
============
Membox carts are regular brain carts with extra per-pattern metadata fields.
Any existing cart works in Membox mode out of the box. The Membox layer is a
RUNTIME wrapper around the multi-cart pool — there's no separate cart format.
A cart becomes "Membox-enabled" by being mounted via membox.mount() instead
of multi_cart.mount(); this attaches a CartLock and routes writes through
the agent_id stamping path.

Reads use the same multi_cart.search() that everything else uses. Writes are
serialized via the CartLock. Lock acquisition has a configurable timeout and
auto-releases on crash via lease expiration.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

# Lazy import multi_cart inside functions to avoid circular import at load time
log = logging.getLogger(__name__)

# Default lock lease (seconds before a held lock auto-releases for crash recovery)
DEFAULT_LEASE_SECONDS = 30
# Default acquire timeout in milliseconds
DEFAULT_TIMEOUT_MS = 5000


# =============================================================================
# CART LOCK — write mutex with lease-based crash recovery
# =============================================================================

class CartLock:
    """Per-cart write mutex.

    Multiple writers serialize through this lock. Readers never touch it
    (they go through multi_cart.search directly). Lock holders are tracked
    by agent_id, and a lease timeout means a crashed agent's lock auto-
    releases after N seconds without manual intervention.

    Phase 1 is single-process — uses threading.Lock. Multi-process locking
    is a future concern.
    """

    def __init__(self, cart_id: str, lease_seconds: int = DEFAULT_LEASE_SECONDS):
        self._cart_id = cart_id
        self._mutex = threading.Lock()
        self._holder: Optional[str] = None
        self._holder_acquired_at: Optional[float] = None
        self._lease_seconds = lease_seconds
        # Tracks total successful acquires for diagnostics
        self._acquire_count = 0
        self._wait_count = 0  # how many times someone had to wait

    def acquire(self, agent_id: str, timeout_ms: int = DEFAULT_TIMEOUT_MS) -> bool:
        """Try to acquire the lock for `agent_id` within `timeout_ms` milliseconds.

        Returns True if acquired, False on timeout. Performs lease-expiration
        cleanup if the current holder's lease has expired (crash recovery).
        """
        # 1. Crash recovery: if the current holder's lease has expired,
        #    forcibly release the lock so it can be re-acquired.
        if self._holder is not None and self._holder_acquired_at is not None:
            held_for = time.time() - self._holder_acquired_at
            if held_for > self._lease_seconds:
                log.warning(
                    f"[membox] Lock on {self._cart_id} held by {self._holder!r} "
                    f"for {held_for:.0f}s (lease={self._lease_seconds}s) — forcibly releasing"
                )
                # Forcibly release. We can't safely call self._mutex.release()
                # if we don't own it, so we re-create the mutex instead.
                self._mutex = threading.Lock()
                self._holder = None
                self._holder_acquired_at = None

        # 2. Try to acquire within timeout
        if self._mutex.locked():
            self._wait_count += 1
        acquired = self._mutex.acquire(timeout=timeout_ms / 1000.0)
        if acquired:
            self._holder = agent_id
            self._holder_acquired_at = time.time()
            self._acquire_count += 1
        return acquired

    def release(self, agent_id: str) -> None:
        """Release the lock. Only the current holder may release.

        Raises PermissionError if `agent_id` doesn't match the holder.
        Raises RuntimeError if the lock isn't held.
        """
        if self._holder is None:
            raise RuntimeError(f"Lock on {self._cart_id} is not held; cannot release")
        if self._holder != agent_id:
            raise PermissionError(
                f"Agent {agent_id!r} tried to release lock on {self._cart_id} "
                f"held by {self._holder!r}"
            )
        self._holder = None
        self._holder_acquired_at = None
        self._mutex.release()

    def holder(self) -> Optional[str]:
        """Return the agent_id currently holding the lock, or None if free."""
        # Lease check inline so callers see accurate state without acquiring
        if self._holder is not None and self._holder_acquired_at is not None:
            if time.time() - self._holder_acquired_at > self._lease_seconds:
                return None  # lease expired — effectively free
        return self._holder

    def stats(self) -> dict:
        """Return diagnostic stats for this lock."""
        held_for = None
        if self._holder_acquired_at is not None:
            held_for = round(time.time() - self._holder_acquired_at, 2)
        return {
            "cart_id": self._cart_id,
            "holder": self._holder,
            "held_for_seconds": held_for,
            "lease_seconds": self._lease_seconds,
            "acquire_count": self._acquire_count,
            "wait_count": self._wait_count,
            "is_locked": self._mutex.locked(),
        }


# =============================================================================
# MEMBOX STATE
# =============================================================================

# cart_id → CartLock
_membox_locks: dict[str, CartLock] = {}

# cart_id → membox config (parsed membox.txt — Phase 3, empty dict for Phase 1)
_membox_config: dict[str, dict] = {}

# cart_id → write log for diagnostics (recent writes per cart, ring buffer)
_write_log: dict[str, list[dict]] = {}
_WRITE_LOG_MAX = 100  # keep the last 100 writes per cart


# =============================================================================
# MOUNT / UNMOUNT / LIST
# =============================================================================

def mount(cart_path: str, cart_id: Optional[str] = None,
          role: Optional[str] = None,
          lease_seconds: int = DEFAULT_LEASE_SECONDS,
          verify_integrity: bool = True) -> dict:
    """Mount a brain cart in Membox mode.

    Same shape as multi_cart.mount() — delegates to it for the actual cart
    loading — but also registers a CartLock for the cart so writes can be
    serialized.

    Args:
        cart_path: Filesystem path to the cart file
        cart_id: Stable identifier for the mount (defaults to filename)
        role: Optional semantic tag (passed through to multi_cart)
        lease_seconds: Lock lease before auto-release for crash recovery
        verify_integrity: Pass through to multi_cart.mount()

    Returns:
        dict with mount details + Membox-specific fields
    """
    import multi_cart as mc

    result = mc.mount(
        cart_path, cart_id=cart_id, role=role,
        verify_integrity=verify_integrity,
    )

    final_cart_id = result["cart_id"]
    if final_cart_id in _membox_locks:
        # Defensive — multi_cart.mount should have errored on collision already
        raise ValueError(
            f"cart_id {final_cart_id!r} is already Membox-mounted"
        )

    _membox_locks[final_cart_id] = CartLock(final_cart_id, lease_seconds=lease_seconds)
    _membox_config[final_cart_id] = {"membox_enabled": True, "lease_seconds": lease_seconds}
    _write_log[final_cart_id] = []

    log.info(
        f"[membox] Mounted '{final_cart_id}' in Membox mode "
        f"(lease={lease_seconds}s, n_patterns={result['n_patterns']})"
    )
    return {**result, "membox": True, "lease_seconds": lease_seconds}


def unmount(cart_id: str) -> dict:
    """Unmount a Membox cart. Releases the lock if held."""
    import multi_cart as mc

    if cart_id in _membox_locks:
        lock = _membox_locks[cart_id]
        if lock.holder() is not None:
            log.warning(
                f"[membox] Unmounting {cart_id!r} while lock is held by "
                f"{lock.holder()!r} — forcing release"
            )
        del _membox_locks[cart_id]
    if cart_id in _membox_config:
        del _membox_config[cart_id]
    if cart_id in _write_log:
        del _write_log[cart_id]

    return mc.unmount(cart_id)


def list_mounts() -> list[dict]:
    """List every Membox-mounted cart with current lock state and stats."""
    import multi_cart as mc

    out = []
    for cart_id in sorted(_membox_locks.keys()):
        lock = _membox_locks[cart_id]
        cart_state = mc.get_cart(cart_id)
        if cart_state is None:
            # Out of sync — Membox thinks it's mounted but multi_cart doesn't
            continue
        out.append({
            "cart_id": cart_id,
            "role": cart_state.get("role"),
            "n_patterns": cart_state.get("n_patterns", 0),
            "lock": lock.stats(),
            "recent_writes": len(_write_log.get(cart_id, [])),
        })
    return out


# =============================================================================
# LOCK API (low-level — most callers should use imprint() instead)
# =============================================================================

def acquire_lock(cart_id: str, agent_id: str,
                 timeout_ms: int = DEFAULT_TIMEOUT_MS) -> bool:
    """Acquire the write lock on a Membox cart.

    Returns True on success, False on timeout. Use imprint() for the
    common acquire-write-release pattern.
    """
    if cart_id not in _membox_locks:
        raise ValueError(f"Cart {cart_id!r} is not Membox-mounted")
    return _membox_locks[cart_id].acquire(agent_id, timeout_ms=timeout_ms)


def release_lock(cart_id: str, agent_id: str) -> None:
    """Release the write lock. Only the current holder may release."""
    if cart_id not in _membox_locks:
        raise ValueError(f"Cart {cart_id!r} is not Membox-mounted")
    _membox_locks[cart_id].release(agent_id)


def lock_holder(cart_id: str) -> Optional[str]:
    """Return the agent_id currently holding the lock, or None if free."""
    if cart_id not in _membox_locks:
        raise ValueError(f"Cart {cart_id!r} is not Membox-mounted")
    return _membox_locks[cart_id].holder()


def lock_stats(cart_id: str) -> dict:
    """Return diagnostic stats for the lock on a cart."""
    if cart_id not in _membox_locks:
        raise ValueError(f"Cart {cart_id!r} is not Membox-mounted")
    return _membox_locks[cart_id].stats()


# =============================================================================
# IMPRINT (the main write API)
# =============================================================================

def imprint(cart_id: str, text: str, agent_id: str,
            tags: str = "", reasoning: str = "",
            origin: str = "agent",
            timeout_ms: int = DEFAULT_TIMEOUT_MS) -> dict:
    """Write a new pattern to a Membox cart with agent_id stamping.

    Acquires the write lock, imprints the pattern with full per-agent metadata,
    releases the lock. Returns the new local_addr and metadata snapshot.

    Args:
        cart_id: Membox-mounted cart to write to
        text: Pattern text content
        agent_id: Who's writing (REQUIRED — Membox writes must be attributed)
        tags: Optional comma-separated tags
        reasoning: Optional explanation of WHY this is being written
        origin: 'agent' (default) | 'human' | 'system'
        timeout_ms: How long to wait for the lock before giving up

    Returns:
        dict with keys: ok, local_addr (if ok), agent_id, written_at, error (if not)
    """
    import multi_cart as mc

    if cart_id not in _membox_locks:
        raise ValueError(f"Cart {cart_id!r} is not Membox-mounted")
    if not agent_id:
        raise ValueError("agent_id is required for Membox imprint (no anonymous writes)")
    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    lock = _membox_locks[cart_id]

    if not lock.acquire(agent_id, timeout_ms=timeout_ms):
        return {
            "ok": False,
            "error": "lock_timeout",
            "current_holder": lock.holder(),
            "timeout_ms": timeout_ms,
        }

    try:
        # Build per-pattern metadata for this write
        meta = {
            "agent_id": agent_id,
            "written_at": datetime.now(timezone.utc).isoformat(),
            "origin": origin,
            "tags": [t.strip() for t in tags.split(",") if t.strip()] if tags else [],
        }
        if reasoning:
            meta["reasoning"] = reasoning

        # Delegate the actual imprint to multi_cart
        # (multi_cart.imprint_with_meta is the parallel addition for this work)
        result = mc.imprint_with_meta(cart_id, text, meta)

        # Append to the write log (ring buffer)
        log_entry = {
            "agent_id": agent_id,
            "written_at": meta["written_at"],
            "local_addr": result.get("local_addr"),
            "origin": origin,
            "text_preview": text[:120],
        }
        _write_log[cart_id].append(log_entry)
        if len(_write_log[cart_id]) > _WRITE_LOG_MAX:
            _write_log[cart_id] = _write_log[cart_id][-_WRITE_LOG_MAX:]

        log.info(
            f"[membox] {cart_id}: agent {agent_id!r} wrote pattern at "
            f"local_addr={result.get('local_addr')}"
        )
        return {
            "ok": True,
            "agent_id": agent_id,
            "local_addr": result.get("local_addr"),
            "written_at": meta["written_at"],
            **result,
        }
    finally:
        lock.release(agent_id)


# =============================================================================
# SEARCH (read API — never blocks)
# =============================================================================

def search(cart_id: str, query: str, top_k: int = 10,
           agent_id: Optional[str] = None) -> dict:
    """Search a Membox cart. Never blocks (reads bypass the write lock).

    Args:
        cart_id: Membox-mounted cart to search
        query: Natural language query
        top_k: Max results to return
        agent_id: Optional — who's reading (for audit logging in Phase 4)

    Returns:
        dict with results list. Each result includes the per-pattern Membox
        metadata fields (agent_id, written_at, origin, reasoning, tags) so
        the consumer can see who wrote what.
    """
    import multi_cart as mc

    if cart_id not in _membox_locks:
        raise ValueError(f"Cart {cart_id!r} is not Membox-mounted")

    # Reads use the multi_cart search path directly. No lock contact at all.
    result = mc.search(query, top_k=top_k, scope=cart_id)

    # multi_cart.search doesn't currently surface per_pattern_meta in result
    # objects — it just returns text/local_addr/score. For Phase 1 we can
    # enrich the results here by reading per_pattern_meta directly off the
    # cart state for each result.
    cart_state = mc.get_cart(cart_id)
    if cart_state is not None:
        per_pattern_meta_list = _extract_per_pattern_meta_list(cart_state)
        for r in result.get("results", []):
            addr = r.get("local_addr")
            if addr is not None and addr < len(per_pattern_meta_list):
                membox_meta = per_pattern_meta_list[addr]
                if isinstance(membox_meta, dict):
                    r["membox_meta"] = membox_meta

    return result


def _extract_per_pattern_meta_list(cart_state: dict) -> list:
    """Pull per_pattern_meta out of a mounted cart's state and parse it.

    The on-disk format is a 0-d ndarray holding a JSON string (a list of dicts).
    This helper handles the unwrap and falls back to an empty list if the
    field is absent or unparseable.
    """
    import json
    # multi_cart.mount stores the cart state but doesn't currently parse
    # per_pattern_meta — it leaves it in the raw npz form. We need to read
    # the cart_path again, OR multi_cart needs to expose it. For Phase 1
    # we read it on demand from the cart_path.
    cart_path = cart_state.get("cart_path")
    if not cart_path:
        return []
    try:
        data = np.load(cart_path, allow_pickle=True)
        if "per_pattern_meta" not in data.files:
            return []
        meta = data["per_pattern_meta"]
        s = meta.item() if meta.ndim == 0 else str(meta)
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return parsed
    except Exception as e:
        log.debug(f"[membox] Could not load per_pattern_meta from {cart_path}: {e}")
    return []


# =============================================================================
# STATUS / DIAGNOSTICS
# =============================================================================

def status(cart_id: str) -> dict:
    """Return Membox status for a cart: lock state, write count per agent,
    recent write log, total pattern count.
    """
    import multi_cart as mc

    if cart_id not in _membox_locks:
        raise ValueError(f"Cart {cart_id!r} is not Membox-mounted")

    lock_stats_dict = _membox_locks[cart_id].stats()
    cart_state = mc.get_cart(cart_id)
    n_patterns = cart_state.get("n_patterns", 0) if cart_state else 0

    # Count writes per agent from the write log
    writes_by_agent: dict[str, int] = {}
    for entry in _write_log.get(cart_id, []):
        a = entry.get("agent_id", "unknown")
        writes_by_agent[a] = writes_by_agent.get(a, 0) + 1

    return {
        "cart_id": cart_id,
        "n_patterns": n_patterns,
        "lock": lock_stats_dict,
        "writes_by_agent": writes_by_agent,
        "recent_writes": _write_log.get(cart_id, [])[-10:],  # last 10
        "membox_enabled": True,
    }


def unmount_all() -> int:
    """Unmount every Membox cart. Returns count. Useful for tests."""
    count = len(_membox_locks)
    for cart_id in list(_membox_locks.keys()):
        try:
            unmount(cart_id)
        except Exception as e:
            log.warning(f"[membox] unmount_all: error unmounting {cart_id}: {e}")
    return count
