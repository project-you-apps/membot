"""Cart generation counter: detect a stale view, refuse a stale write.

THE PROBLEM THIS SOLVES
-----------------------
`mount_cartridge` loads a cart into PER-SESSION RAM. Two people mounting the same cart get
two independent copies. So today:

    Andy deletes a passage and saves  ->  my session still holds the old arrays
    I then save from my session       ->  I write his deleted passage back

Whole-cart, last-writer-wins, silent. That is not a stale view, it is a data-loss path, and
it is exactly what CC_single_cart_substrate_assumption predicted in July.

This module does not fix the copies -- that is the shared-mount change. It makes the
resulting conflict VISIBLE and REFUSED instead of silent and destructive, which is the
property that actually matters and costs about a tenth as much.

WHY A SIDECAR AND NOT A KEY IN THE CART
---------------------------------------
The whole point is asking "has this changed?" cheaply. A generation stored inside the cart
would mean loading 183 MB to answer a yes/no question, so every check would be expensive
enough that callers would skip it -- and a check people skip is not a check.

WHY THE SIDECAR RECORDS THE CART'S SIZE AND MTIME
-------------------------------------------------
Because a sidecar can drift from the file it describes. `droplet_sync`, a manual copy, an
scp, another process -- all write the cart without going through us, and a naive counter
sails straight past them still reading "generation 7, all good."

So the sidecar stamps the cart's size and mtime at bump time. If the cart on disk no longer
matches that stamp, the cart was modified OUT OF BAND and the generation is not to be
trusted. That case is reported explicitly rather than folded into "current", because
"someone changed this behind our back" and "you are two saves behind" want different
responses from a human.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

SIDECAR_SUFFIX = ".generation.json"
FORMAT = "vps-cart-generation"
VERSION = 1


class StaleCartError(RuntimeError):
    """Raised when a write is attempted from a view that is no longer current."""

    def __init__(self, message: str, status: dict):
        super().__init__(message)
        self.status = status


def sidecar_path(cart_path) -> Path:
    cart = Path(cart_path)
    return cart.with_name(cart.name + SIDECAR_SUFFIX)


def _cart_stamp(cart_path) -> dict:
    """Size + mtime of the cart, or zeros when it does not exist yet."""
    try:
        st = Path(cart_path).stat()
        return {"cart_size": st.st_size, "cart_mtime": st.st_mtime}
    except OSError:
        return {"cart_size": 0, "cart_mtime": 0.0}


def read_generation(cart_path) -> dict:
    """Current generation for a cart. NEVER raises.

    A cart with no sidecar is generation 0 -- not an error. Every cart in existence
    predates this module, and a migration that makes old carts unreadable to answer a
    question about concurrency would be a poor trade.
    """
    path = sidecar_path(cart_path)
    stamp = _cart_stamp(cart_path)
    base = {
        "generation": 0,
        "updated_at": 0.0,
        "updated_by": "",
        "has_sidecar": False,
        "out_of_band": False,
        **stamp,
    }
    if not path.exists():
        return base
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A corrupt sidecar must not make the cart unusable. Report it as out-of-band --
        # we genuinely do not know what generation this is, which is the same practical
        # situation as someone having written the cart behind our back.
        base["out_of_band"] = True
        return base

    recorded_size = data.get("cart_size", 0)
    recorded_mtime = data.get("cart_mtime", 0.0)
    out_of_band = bool(
        recorded_size and (
            recorded_size != stamp["cart_size"]
            # mtime comparison is tolerant: copies and filesystems round differently, and a
            # false "someone tampered" every time a file is moved would train people to
            # ignore the warning.
            or abs(recorded_mtime - stamp["cart_mtime"]) > 2.0
        )
    )
    return {
        "generation": int(data.get("generation", 0)),
        "updated_at": float(data.get("updated_at", 0.0)),
        "updated_by": data.get("updated_by", ""),
        "has_sidecar": True,
        "out_of_band": out_of_band,
        **stamp,
    }


def bump_generation(cart_path, seat: str = "", now: float = 0.0) -> dict:
    """Record that the cart just changed. Call AFTER the cart is written, never before.

    Order matters: bumping first and then failing the write would leave every other session
    believing it is behind something that never happened.
    """
    now = now or time.time()
    current = read_generation(cart_path)
    payload = {
        "format": FORMAT,
        "version": VERSION,
        "generation": current["generation"] + 1,
        "updated_at": now,
        "updated_by": seat or "",
        **_cart_stamp(cart_path),
    }

    path = sidecar_path(cart_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return payload


def check_stale(cart_path, loaded_generation: int) -> dict:
    """How far behind a mounted view is. Pure read; safe to call on every request."""
    current = read_generation(cart_path)
    behind = max(0, current["generation"] - int(loaded_generation or 0))
    return {
        "stale": behind > 0 or current["out_of_band"],
        "behind": behind,
        "current": current["generation"],
        "loaded": int(loaded_generation or 0),
        "out_of_band": current["out_of_band"],
        "updated_by": current["updated_by"],
        "updated_at": current["updated_at"],
    }


def describe_stale(status: dict) -> str:
    """One sentence a human can act on. Staleness is NEWS, not usually a decision."""
    if status.get("out_of_band"):
        return ("This cart was changed outside the app (a sync, a copy, or another "
                "process). Reload before writing.")
    behind = status.get("behind", 0)
    if behind <= 0:
        return "Your view is current."
    who = status.get("updated_by") or "someone else"
    saves = "save" if behind == 1 else "saves"
    return f"Changed by {who} — your view is {behind} {saves} behind. Reload to catch up."


def assert_writable(cart_path, loaded_generation: int, seat: str = "") -> dict:
    """Guard a save. Raises StaleCartError rather than letting a stale write land.

    THIS REFUSES; IT DOES NOT MERGE. Merge logic built on the per-session-copy model would
    be thrown away by the shared-mount change, so the deliberate scope here is: make the
    conflict visible, lose nothing, and let a human decide. A refused save costs someone a
    reload. A silent one costs whatever the other person deleted.
    """
    status = check_stale(cart_path, loaded_generation)
    if status["stale"]:
        raise StaleCartError(describe_stale(status), status)
    return status
