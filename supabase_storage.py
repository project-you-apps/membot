"""
supabase_storage.py — Mempack persistence layer.

Wraps supabase-py for Mempack CRUD against:

- public.mempacks               row metadata (one row per cart)
- public.mempack_patterns       per-pattern H-block, normalized columns + bytea
- public.mempack_provisions_log audit trail for auto-provision events
- Storage bucket 'mempacks'     the actual .cart.npz blobs

Membot uses the service role key to bypass RLS — it acts on behalf of any
user at mount/store time, scoped by owner_id passed in API calls. User-side
JWTs are verified separately in the REST handler layer.

Required env vars in /opt/membot/.env:
  SUPABASE_URL                  e.g. https://uikdknfxcqklldmfshug.supabase.co
  SUPABASE_SERVICE_ROLE_KEY     JWT with service_role claim (from Supabase Studio)

See docs/PATTERN-ANATOMY.md §3 for H-block field semantics, and
vector-plus-studio-repo/db/002_mempacks_schema.sql for the table definitions
this module reads/writes.

Andy + Claude 2026-05-13.
"""

import base64
import os
import struct
import warnings
from typing import Optional

# supabase-py 2.30+ emits DeprecationWarnings from inside its own client
# construction about `timeout` and `verify` kwargs being moved to the http
# client. We don't pass those kwargs ourselves — the library does it internally —
# so we can't fix them from here. Suppress just those specific warnings
# rather than letting them clutter every log line.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"supabase\..*",
)

try:
    from supabase import create_client, Client
except ImportError as e:
    raise ImportError(
        "supabase-py not installed in this venv. "
        "Run: /opt/membot/venv/bin/pip install supabase"
    ) from e

# H-block format constants live in cartridge_builder; re-import here so callers
# can import everything Mempack-related from one place.
from cartridge_builder import (  # noqa: F401 — re-exported for callers
    HIPPO_FORMAT,
    HIPPO_SIZE,
    FLAG_TOMBSTONE, FLAG_PINNED, FLAG_HAS_PARENT, FLAG_HAS_CHILD, FLAG_HAS_SIBLING,
    FLAG_PERISH_MASK, FLAG_PERISH_VOLATILE, FLAG_PERISH_REPLACEABLE, FLAG_PERISH_ARCHIVAL,
    PERM_R, PERM_W, PERM_X, PERM_DEFAULT,
    FORMAT_VERSION_CANONICAL,
)


BUCKET = "mempacks"


# ---------------------------------------------------------------------------
# Client init (lazy, singleton)
# ---------------------------------------------------------------------------

_client: Optional["Client"] = None


def get_client() -> "Client":
    """Lazy-init the supabase client. Reads creds from env on first call.

    Raises RuntimeError if SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY missing —
    a clear failure mode rather than a confusing client-init crash later.
    """
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in env. "
                "See /opt/membot/.env on the droplet. "
                "Service role key comes from Supabase Studio -> Project Settings -> API."
            )
        _client = create_client(url, key)
    return _client


def reset_client() -> None:
    """Drop the cached client (useful in tests, or after env changes)."""
    global _client
    _client = None


# ---------------------------------------------------------------------------
# Storage bucket ops — the .cart.npz blobs
# ---------------------------------------------------------------------------

def storage_path(owner_id: str, name: str) -> str:
    """Construct the canonical storage path for a user's Mempack blob.

    Maps to the bucket RLS policies: first path segment MUST equal auth.uid()
    for the requesting user. Service role bypasses RLS so membot reads/writes
    any owner's blob.
    """
    return f"{owner_id}/{name}.cart.npz"


def upload_blob(owner_id: str, name: str, blob_bytes: bytes) -> None:
    """Upload (or replace) a Mempack blob in Storage. Uses upsert so re-runs
    are idempotent (cart-update overwrites the previous version).
    """
    sb = get_client()
    path = storage_path(owner_id, name)
    sb.storage.from_(BUCKET).upload(
        path,
        blob_bytes,
        file_options={
            "content-type": "application/octet-stream",
            "upsert": "true",
        },
    )


def download_blob(owner_id: str, name: str) -> bytes:
    """Fetch a Mempack blob from Storage. Returns the raw .npz bytes."""
    sb = get_client()
    path = storage_path(owner_id, name)
    return sb.storage.from_(BUCKET).download(path)


def delete_blob(owner_id: str, name: str) -> None:
    """Remove a Mempack blob from Storage."""
    sb = get_client()
    path = storage_path(owner_id, name)
    sb.storage.from_(BUCKET).remove([path])


# ---------------------------------------------------------------------------
# Postgres ops on public.mempacks
# ---------------------------------------------------------------------------

def get_mempack_by_name(owner_id: str, name: str) -> Optional[dict]:
    """Return the mempack row for (owner_id, name), or None if not present."""
    sb = get_client()
    res = (
        sb.table("mempacks")
        .select("*")
        .eq("user_id", owner_id)
        .eq("name", name)
        .maybe_single()
        .execute()
    )
    return res.data if res else None


def list_mempacks(owner_id: str) -> list[dict]:
    """Return all mempack rows for a user, oldest first."""
    sb = get_client()
    res = (
        sb.table("mempacks")
        .select("*")
        .eq("user_id", owner_id)
        .order("created_at")
        .execute()
    )
    return res.data or []


def insert_mempack(
    owner_id: str,
    name: str,
    pattern_count: int,
    size_bytes: int,
    briefing: str = "",
    pattern_i_text: str = "",
    manifest: dict | None = None,
    cart_type: str = "agent-memory",
    status: str = "pending",
) -> dict:
    """Insert a new mempack row. Returns the inserted row (with id, created_at).

    Atomicity model: row is inserted with storage_status='pending' before blob
    upload, then updated to 'ready' after upload succeeds. Reconciliation cron
    sweeps stale 'pending' rows older than N minutes.
    """
    sb = get_client()
    row = {
        "user_id": owner_id,
        "name": name,
        "cart_type": cart_type,
        "storage_bucket": BUCKET,
        "storage_path": storage_path(owner_id, name),
        "storage_status": status,
        "pattern_count": pattern_count,
        "size_bytes": size_bytes,
        "briefing": briefing,
        "pattern_i_text": pattern_i_text,
        "manifest": manifest or {},
        "format_version": FORMAT_VERSION_CANONICAL,
    }
    res = sb.table("mempacks").insert(row).execute()
    return res.data[0] if res.data else {}


def mark_mempack_ready(mempack_id: str) -> None:
    """Flip storage_status pending → ready after a successful blob upload."""
    sb = get_client()
    sb.table("mempacks").update({"storage_status": "ready"}).eq("id", mempack_id).execute()


def touch_mempack_mount(mempack_id: str) -> None:
    """Bump last_mounted_at + mount_count on a successful mount."""
    sb = get_client()
    sb.rpc("increment_mempack_mount", {"mempack_id": mempack_id}).execute()
    # Note: increment_mempack_mount is a Postgres function we'll add in 003
    # if we want atomic counter increments. For v1, naive update is fine:
    # sb.table("mempacks").update({...}).eq("id", mempack_id).execute()


def delete_mempack_row(mempack_id: str) -> None:
    """Cascade-delete the mempack row. mempack_patterns rows go with it via FK."""
    sb = get_client()
    sb.table("mempacks").delete().eq("id", mempack_id).execute()


# ---------------------------------------------------------------------------
# Postgres ops on public.mempack_patterns
# ---------------------------------------------------------------------------

def insert_pattern_rows(
    mempack_id: str,
    hippocampus_bytes: bytes,
    texts: list[str],
) -> int:
    """Explode an H-block byte array into rows of public.mempack_patterns.

    Args:
        mempack_id:        UUID of the parent mempack row
        hippocampus_bytes: concatenated 64-byte H-blocks, N * HIPPO_SIZE bytes
        texts:             parallel list of N pattern bodies (for text_preview + text_length)

    Returns: number of pattern rows inserted.

    Idempotency: caller should delete existing rows for this mempack_id first
    if doing a full re-sync (cart update path). Insert-only here.
    """
    sb = get_client()
    n_patterns = len(hippocampus_bytes) // HIPPO_SIZE
    if n_patterns == 0:
        return 0

    rows = []
    for i in range(n_patterns):
        chunk = hippocampus_bytes[i * HIPPO_SIZE:(i + 1) * HIPPO_SIZE]
        vals = struct.unpack(HIPPO_FORMAT, chunk)
        text = texts[i] if i < len(texts) else ""
        rows.append({
            "mempack_id":     mempack_id,
            "pattern_idx":    i,
            "pattern_id":     vals[0],
            "format_version": vals[1],
            "cartridge_type": vals[2],
            "parent_ptr":     vals[3],
            "child_ptr":      vals[4],
            "sibling_ptr":    vals[5],
            "source_hash":    vals[6],
            "sequence_num":   vals[7],
            "ts_unix":        vals[8],
            "flags":          vals[9],
            "perms_byte":     vals[10],
            "text_preview":   (text or "")[:200],
            "text_length":    len((text or "").encode("utf-8")),
            # PostgREST serializes payloads to JSON; bytea columns accept hex
            # `\x...` strings, which Postgres decodes on insert.
            "hippocampus_raw": "\\x" + chunk.hex(),
        })

    # Batch insert in chunks of 200 (supabase-py / PostgREST default payload cap)
    inserted = 0
    for batch_start in range(0, len(rows), 200):
        batch = rows[batch_start:batch_start + 200]
        res = sb.table("mempack_patterns").insert(batch).execute()
        inserted += len(res.data) if res.data else 0
    return inserted


def delete_pattern_rows_for(mempack_id: str) -> None:
    """Wipe all per-pattern rows for a mempack. Used before cart-update re-sync."""
    sb = get_client()
    sb.table("mempack_patterns").delete().eq("mempack_id", mempack_id).execute()


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def log_provision(
    owner_id: str,
    mempack_id: Optional[str],
    trigger_source: str,
    outcome: str,
    error: Optional[str] = None,
) -> None:
    """Append one row to public.mempack_provisions_log.

    trigger_source: 'lazy_list' | 'manual' | 'signup_trigger' | 'migration'
    outcome:        'created' | 'already_existed' | 'failed'
    """
    sb = get_client()
    sb.table("mempack_provisions_log").insert({
        "user_id":        owner_id,
        "mempack_id":     mempack_id,
        "trigger_source": trigger_source,
        "outcome":        outcome,
        "error_message":  error,
    }).execute()


# ---------------------------------------------------------------------------
# Auth lookups (service role can read auth.users via the admin API)
# ---------------------------------------------------------------------------

def find_user_uuid_by_email(email: str) -> Optional[str]:
    """Look up a Supabase auth user's UUID by email. Service role only.

    Useful for the one-shot migration script when we need to map a synthetic
    owner_id (smoke-test mempack) to a real user. Iterates paged user list.
    """
    sb = get_client()
    page = 1
    per_page = 200
    while True:
        try:
            resp = sb.auth.admin.list_users(page=page, per_page=per_page)
        except Exception:
            return None
        # supabase-py returns either a list-like or an object with `.users`
        users = getattr(resp, "users", None) or (resp if isinstance(resp, list) else [])
        if not users:
            return None
        for u in users:
            uemail = getattr(u, "email", None) or (u.get("email") if isinstance(u, dict) else None)
            if uemail == email:
                return getattr(u, "id", None) or (u.get("id") if isinstance(u, dict) else None)
        if len(users) < per_page:
            return None
        page += 1


# ---------------------------------------------------------------------------
# High-level convenience: provision a fresh starter Mempack for a user
# ---------------------------------------------------------------------------

def auto_provision_primary(
    owner_id: str,
    starter_blob_bytes: bytes,
    starter_hippocampus: bytes,
    starter_texts: list[str],
    pattern_count: int,
    briefing: str,
    pattern_i_text: str,
    manifest: dict,
    trigger_source: str = "lazy_list",
) -> dict:
    """Idempotent provision of a user's primary Mempack.

    If `primary` already exists for owner_id: log 'already_existed', return
    the existing row.

    Otherwise: insert row (status='pending') -> upload blob -> mark 'ready'
    -> insert per-pattern rows -> log 'created' -> return row.

    On any failure mid-way: log 'failed' with error message; caller decides
    whether to clean up the partial state or leave for reconciliation.
    """
    existing = get_mempack_by_name(owner_id, "primary")
    if existing:
        log_provision(owner_id, existing["id"], trigger_source, "already_existed")
        return existing

    try:
        row = insert_mempack(
            owner_id=owner_id,
            name="primary",
            pattern_count=pattern_count,
            size_bytes=len(starter_blob_bytes),
            briefing=briefing,
            pattern_i_text=pattern_i_text,
            manifest=manifest,
            cart_type="agent-memory",
            status="pending",
        )
        mempack_id = row["id"]

        upload_blob(owner_id, "primary", starter_blob_bytes)
        insert_pattern_rows(mempack_id, starter_hippocampus, starter_texts)
        mark_mempack_ready(mempack_id)

        log_provision(owner_id, mempack_id, trigger_source, "created")
        # Refetch with status='ready' set
        return get_mempack_by_name(owner_id, "primary") or row
    except Exception as e:
        log_provision(owner_id, None, trigger_source, "failed", error=str(e))
        raise
