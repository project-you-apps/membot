"""Storage for sub-cartridge manifests: create, list, get, delete.

`subcart.py` defines the FORMAT and can resolve one against a parent. This module is the
missing verb -- until now nothing in the product could actually make one.

Manifests live in `<cart dir>/subcarts/`, beside the cart they slice, so a sub-cart travels
with its parent and a deployment that has never heard of them simply has no such directory.

TWO NAMES, TWO JOBS
-------------------
The FILENAME is the immutable `id`. The human `name` lives inside the file and is mutable,
because people rename things -- so a name can never be identity. Renaming is then a one-field
edit that breaks no references, and aliases fall out for free.

DELETE MOVES TO TRASH, IT DOES NOT UNLINK
-----------------------------------------
A manifest is a few kilobytes, but it can represent real curatorial work: someone chose those
passages by hand. Deleting the JSON destroys the selection, not the passages -- which is
exactly the kind of loss that feels survivable right up until it is not. `delete()` moves the
file to `subcarts/trash/` with a timestamp suffix, so recovery is a file move rather than a
re-curation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import subcart as _sc

SUBCART_DIRNAME = "subcarts"
TRASH_DIRNAME = "trash"
SUFFIX = ".subcart.json"


class SubcartStore:
    """Manifests for one parent cart directory."""

    def __init__(self, cart_path, parent_cart: str = ""):
        cart = Path(cart_path)
        # Accept either the cart file or its directory, because callers have both and
        # guessing wrong yields a silently empty listing rather than an error.
        self.dir = (cart if cart.is_dir() else cart.parent) / SUBCART_DIRNAME
        self.parent_cart = parent_cart or (cart.name if not cart.is_dir() else "")

    # ------------------------------------------------------------------ paths

    def path_for(self, subcart_id: str) -> Path:
        safe = "".join(c for c in (subcart_id or "") if c.isalnum() or c in "-_")
        if not safe:
            raise ValueError(f"sub-cart id {subcart_id!r} has no path-safe characters")
        return self.dir / f"{safe}{SUFFIX}"

    # ------------------------------------------------------------------ read

    def list(self) -> list:
        """Every manifest here. A manifest that will not parse is REPORTED, not skipped.

        Silently omitting an unreadable file makes a corrupted sub-cart indistinguishable
        from one that was never created, which is the worst possible way to lose curation.
        """
        if not self.dir.exists():
            return []
        out = []
        for path in sorted(self.dir.glob(f"*{SUFFIX}")):
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                problems = _sc.validate_manifest(manifest)
                out.append({
                    "id": manifest.get("id"),
                    "name": manifest.get("name"),
                    "title": manifest.get("title"),
                    "mode": (manifest.get("membership") or {}).get("mode"),
                    "authored_by": (manifest.get("membership") or {}).get("authored_by"),
                    "keys": len((manifest.get("membership") or {}).get("keys") or []),
                    "created_at": manifest.get("created_at"),
                    "file": path.name,
                    "problems": problems,
                })
            except Exception as e:                     # noqa: BLE001
                out.append({"id": None, "name": None, "file": path.name,
                            "problems": [f"unreadable: {e}"]})
        return out

    def get(self, id_or_name: str) -> dict:
        """Resolve by id first, then by name. Raises if absent or ambiguous."""
        direct = self.path_for(id_or_name) if id_or_name else None
        if direct is not None and direct.exists():
            return _sc.load_manifest(direct)

        matches = [row for row in self.list() if row.get("name") == id_or_name]
        if not matches:
            raise KeyError(f"no sub-cart with id or name {id_or_name!r}")
        if len(matches) > 1:
            # Should be impossible -- create() enforces name uniqueness -- but if it
            # happens, guessing is worse than refusing.
            raise KeyError(f"{len(matches)} sub-carts named {id_or_name!r}; use the id")
        return _sc.load_manifest(self.dir / matches[0]["file"])

    def name_taken(self, name: str) -> bool:
        return any(row.get("name") == name for row in self.list())

    # ------------------------------------------------------------------ write

    def create(self, manifest: dict) -> dict:
        """Persist a new manifest. Refuses duplicate names and duplicate ids."""
        problems = _sc.validate_manifest(manifest)
        if problems:
            raise ValueError("invalid manifest: " + "; ".join(problems))

        name = manifest.get("name")
        if self.name_taken(name):
            # A friendly refusal, not a silent suffix. If we auto-renamed, a user would
            # end up with `notes` and `notes-2` and no idea which one they just built.
            raise ValueError(f"a sub-cart named {name!r} already exists here; "
                             f"choose another name or delete that one first")

        path = self.path_for(manifest["id"])
        if path.exists():
            raise ValueError(f"sub-cart id {manifest['id']} already exists")

        self.dir.mkdir(parents=True, exist_ok=True)
        _sc.save_manifest(manifest, path)
        return manifest

    def delete(self, id_or_name: str) -> dict:
        """Move a manifest to `subcarts/trash/`. Never unlinks.

        Deleting the JSON destroys a SELECTION, not any passages -- but a hand-curated
        selection is real work, so recovery should be a file move rather than redoing it.
        """
        manifest = self.get(id_or_name)
        path = self.path_for(manifest["id"])
        trash = self.dir / TRASH_DIRNAME
        trash.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        destination = trash / f"{path.stem}.{stamp}{SUFFIX}"
        path.rename(destination)
        return {"id": manifest["id"], "name": manifest.get("name"),
                "moved_to": str(destination)}

    def rename(self, id_or_name: str, new_name: str) -> dict:
        """Rename is a one-field edit precisely because the FILE is keyed by id."""
        if not new_name or not new_name.strip():
            raise ValueError("new name is required")
        new_name = new_name.strip()
        manifest = self.get(id_or_name)
        if manifest.get("name") != new_name and self.name_taken(new_name):
            raise ValueError(f"a sub-cart named {new_name!r} already exists here")
        manifest["name"] = new_name
        _sc.save_manifest(manifest, self.path_for(manifest["id"]))
        return manifest
