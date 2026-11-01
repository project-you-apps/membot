"""Re-derive the line numbers cited by docs/AUDIT-membot-mutation-paths-2026-11-01.md.

WHY THIS EXISTS
---------------
The audit enumerates every path that mutates durable cart state, with a line number for
each. `membot_server.py` grew 10,393 -> 10,448 -> 10,501 -> 10,580 lines during the hour
the audit was written, and the offsets were NOT uniform -- `_write_blocked` moved +64
while `memory_store` moved +108 -- so no single correction factor exists and arithmetic
cannot repair a stale table.

⛔ A wrong line number is worse than no line number: it sends the next reader to a
function that is not the one under discussion, and they have no way to tell. So the audit
cites FUNCTION NAMES as the identifiers and treats the numbers as generated output.
This script is the generator. Run it; paste the result; the audit is current again.

    python audit_anchors.py                  # resolve against membot_server.py
    python audit_anchors.py --check          # exit 1 if any anchor is missing/ambiguous

An anchor that stops matching is itself a finding: either the code moved (fine, the
number updates) or the call site is GONE (not fine, the audit entry needs rewriting).
`--check` is what makes the difference visible instead of silently printing a short list.

Stdlib only.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

DEFAULT_TARGET = Path(__file__).resolve().parent / "membot_server.py"

# (section, label, regex). Order follows the audit.
#
# Each pattern is anchored on text that identifies the SITE, not on text that merely
# appears near it -- a pattern matching a docstring mention would resolve to prose and
# report a line that looks plausible and is wrong.
ANCHORS = [
    # ---- the three structural facts
    ("facts", "_log_activity def", r"^def _log_activity\("),
    ("facts", "_log_activity ring buffer", r"^_depot_activity: collections\.deque"),
    ("facts", "_log_activity CALL SITES (all read-path)", r"^\s+_log_activity\("),
    ("facts", "MCP auth middleware (on_call_tool)", r"async def on_call_tool\("),
    ("facts", "  bearer check", r'if not auth_header\.startswith\("Bearer "\)'),
    ("facts", "  default start-up: no auth", r"rate limiting enabled \(no auth"),
    ("facts", "is_logging_enabled gate", r"if sbs\.is_logging_enabled\("),
    ("facts", "redact.append_review_log call", r"redact\.append_review_log\("),

    # ---- gates
    ("gates", "_write_blocked def", r"^def _write_blocked\("),
    ("gates", "_write_blocked CALL SITES", r"^\s+(blocked = |return )?_write_blocked\("),
    ("gates", "read_only default (= not --writable)", r'_server_config\["read_only"\] = not args\.writable'),
    ("gates", "Mempack carve-out of server read_only", r'_server_config\.get\("read_only"\) and not is_mempack'),
    ("gates", "CORS allow-origin *", r"Access-Control-Allow-Origin"),
    ("gates", "_require_authenticated def", r"^def _require_authenticated\("),
    ("gates", "_require_authenticated CALL SITES", r"^\s+user_id, err = _require_authenticated\(request\)"),
    ("gates", "_require_mempack_owner CALL SITES", r"^\s+user_id, mp, err = _require_mempack_owner\("),

    # ---- tier 1
    ("tier1", "1. rest_sync def", r"^async def rest_sync\("),
    ("tier1", "   sync makedirs", r'^        cart_dir = os\.path\.join\(BASE_DIR, "cartridges"\)'),
    ("tier1", "   sync open append", r'open\(sync_file, "a", encoding="utf-8"\)'),
    ("tier1", "   sync _log_rest", r'_log_rest\(request, "SYNC"'),
    ("tier1", "2. rest_store def", r"^async def rest_store\("),
    ("tier1", "   rest_store -> memory_store", r"_call = getattr\(memory_store, 'fn', memory_store\)"),
    ("tier1", "3. rest_save def", r"^async def rest_save\("),
    ("tier1", "4. rest_mount def", r"^async def rest_mount\("),
    ("tier1", "   mount -> _create_empty_cartridge", r"_create_empty_cartridge\(name\)"),
    ("tier1", "   _create_empty_cartridge def", r"^def _create_empty_cartridge\("),
    ("tier1", "   empty makedirs", r"^    os\.makedirs\(target_dir, exist_ok=True\)"),

    # ---- tier 2
    ("tier2", "5. memory_store def", r"^def memory_store\("),
    ("tier2", "   memory_store mount_read_only gate", r'^    if state\.get\("mount_read_only"\):$'),
    ("tier2", "   memory_store embeddings vstack", r'^            state\["embeddings"\] = np\.vstack\(\[state\["embeddings"\]'),
    ("tier2", "   memory_store binary vstack", r'^            state\["binary_corpus"\] = np\.vstack\(\[state\["binary_corpus"\]'),
    ("tier2", "   memory_store persist call", r"_mempack_persist_new_pattern\(state, stored_text, emb\)"),
    ("tier2", "6. mempack_update_pattern_i def", r"^def mempack_update_pattern_i\("),
    ("tier2", "   previous_text capture", r"^        previous_text = texts\[PATTERN_I_IDX\]"),
    ("tier2", "   pattern_i IN-PLACE overwrite", r"^            texts\[PATTERN_I_IDX\] = text"),
    ("tier2", "   pattern_i NOT _write_blocked (comment)", r"deliberately NOT gated by _write_blocked"),
    ("tier2", "   pattern_i persist call", r"persist_msg = _mempack_persist_pattern_update\("),
    ("tier2", "7. save_cartridge def", r"^def save_cartridge\("),
    ("tier2", "   save_cartridge assert_writable", r"_cart_generation\.assert_writable\("),
    ("tier2", "   save_cartridge bump_generation", r"_cart_generation\.bump_generation\("),
    ("tier2", "8. seat_hide def", r"^def seat_hide\("),
    ("tier2", "   seat_hide set_mark", r"^        store\.set_mark\(session_id"),
    ("tier2", "9. propose_correction def", r"^def propose_correction\("),
    ("tier2", "   _ps.propose", r'_ps\.propose\(state\["cartridge_path"\]'),
    ("tier2", "10. subcart_create def", r"^def subcart_create\("),
    ("tier2", "    store.create", r"^        store\.create\(manifest\)"),
    ("tier2", "11. subcart_delete def", r"^def subcart_delete\("),
    ("tier2", "    store.delete", r"^        out = store\.delete\(name\)"),
    ("tier2", "12. subcart_export def", r"^def subcart_export\("),
    ("tier2", "13. membox_imprint def", r"^def membox_imprint\("),
    ("tier2", "14. federate_publish def", r"^def federate_publish\("),
    ("tier2", "15. federate_consolidate def", r"^def federate_consolidate\("),
    ("tier2", "16. federate_migrate_jsonl def", r"^def federate_migrate_jsonl\("),

    # ---- tier 3
    ("tier3", "17. rest_mempack_create def", r"^async def rest_mempack_create\("),
    ("tier3", "    insert_mempack", r"row = sbs\.insert_mempack\("),
    ("tier3", "    insert_pattern_rows", r"sbs\.insert_pattern_rows\("),
    ("tier3", "    mark_mempack_ready", r"sbs\.mark_mempack_ready\("),
    ("tier3", "    log_provision created", r'sbs\.log_provision\(owner_id, mempack_id, "manual", "created"\)'),
    ("tier3", "18. rest_copy def", r"^async def rest_copy\("),
    ("tier3", "    copy dst_texts.append (local cart)", r"^        dst_texts\.append\(new_text\)"),
    ("tier3", "19. rest_mempack_pattern_i_update def", r"^async def rest_mempack_pattern_i_update\("),
    ("tier3", "    _mempack_replace_blobless call", r"result = _mempack_replace_blobless\("),
    ("tier3", "    replace IN-PLACE texts[idx]", r"^    texts\[idx\] = new_text"),
    ("tier3", "20. rest_mempack_dispatch def", r"^async def rest_mempack_dispatch\("),
    ("tier3", "21. rest_mempack_settings def", r"^async def rest_mempack_settings\("),
    ("tier3", "    settings logging toggle", r'activity_logging_enabled"\] = bool'),
    ("tier3", "    settings DIRECT supabase update", r'sb\.table\("mempacks"\)\.update\(patch\)'),
    ("tier3", "    settings append_activity", r'event_type="settings_update"'),
    ("tier3", "    settings log failure is non-fatal", r"Settings activity-log append failed"),
    ("tier3", "22. rest_mempacks_list def", r"^async def rest_mempacks_list\("),
    ("tier3", "    auto_provision default true", r"auto_provision_param = request\.query_params\.get"),
    ("tier3", "    auto_provision_primary (a GET that writes)", r"sbs\.auto_provision_primary\($"),

    # ---- tier 4 / 5
    ("tier4", "23. _overlay_record_hits def", r"^def _overlay_record_hits\("),
    ("tier4", "    overlay touch_many", r"store\.touch_many\(session_id"),
    ("tier4", "    overlay flush (read path)", r"^        store\.flush\(\)"),
    ("tier5", "24. __main__ --mount save_as_npz", r"^            save_as_npz\(empty_path, empty_emb, \[\]\)"),
    ("tier5", "25. _ensure_mempack_cached def", r"^def _ensure_mempack_cached\("),
    ("tier5", "    _ensure_mempack_cached CALL SITES", r"^\s+_ensure_mempack_cached\("),

    # ---- shared persist helpers (the §6.2 choke points)
    ("helpers", "save_as_npz def", r"^def save_as_npz\("),
    ("helpers", "save_as_npz CALL SITES", r"^\s+save_as_npz\("),
    ("helpers", "save_manifest def", r"^def save_manifest\("),
    ("helpers", "save_manifest CALL SITES", r"^\s+(manifest = )?save_manifest\("),
    ("helpers", "_mempack_repack_and_persist def", r"^def _mempack_repack_and_persist\("),
    ("helpers", "_mempack_repack_and_persist CALL SITES", r"blob_size = _mempack_repack_and_persist\("),
    ("helpers", "_mempack_append_blobless def", r"^def _mempack_append_blobless\("),
    ("helpers", "_mempack_append_blobless CALL SITES", r"result = _mempack_append_blobless\("),
    ("helpers", "_mempack_replace_blobless def", r"^def _mempack_replace_blobless\("),
    ("helpers", "_mempack_persist_new_pattern def", r"^def _mempack_persist_new_pattern\("),
    ("helpers", "_mempack_persist_pattern_update def", r"^def _mempack_persist_pattern_update\("),
    ("helpers", "_detach_for_write def", r"^def _detach_for_write\("),
    ("helpers", "_detach_for_write CALL SITES", r"^        _detach_for_write\(state, session_id\)"),

    # ---- listed as NOT mutations, so a reader does not chase them
    ("not-mutations", "multi_mount def", r"^def multi_mount\("),
    ("not-mutations", "multi_unmount def", r"^def multi_unmount\("),
    ("not-mutations", "multi_mount_directory def", r"^def multi_mount_directory\("),
    ("not-mutations", "membox_mount def", r"^def membox_mount\("),
    ("not-mutations", "membox_unmount def", r"^def membox_unmount\("),
    ("not-mutations", "membox_acquire_lock def", r"^def membox_acquire_lock\("),
    ("not-mutations", "membox_release_lock def", r"^def membox_release_lock\("),
    ("not-mutations", "firm_hot_stack def (pure view)", r"^def firm_hot_stack\("),
    ("not-mutations", "rest_embed def", r"^async def rest_embed\("),
    ("not-mutations", "load_signatures def", r"^def load_signatures\("),
    ("not-mutations", "patterns_export temp unlink", r"os\.unlink\(tmp_pdf_path\)"),
    ("not-mutations", "rest_auth_signout def", r"^async def rest_auth_signout\("),
    ("not-mutations", "_log_rest def (log.info only)", r"^def _log_rest\("),
]

# Anchors in the helper modules. (module, label, regex)
MODULE_ANCHORS = [
    ("overlay.py", "save_overlay atomic write", r"fd, tmp = tempfile\.mkstemp\("),
    ("overlay_store.py", "set_mark", r"^    def set_mark\("),
    ("overlay_store.py", "touch_many", r"^    def touch_many\("),
    ("overlay_store.py", "flush (interval-gated, not permission-gated)", r"^    def flush\("),
    ("overlay_store.py", "flush return payload", r'^        return \{"written": len\(written\)'),
    ("subcart_store.py", "create", r"^    def create\("),
    ("subcart_store.py", "delete (moves to trash, never unlinks)", r"^    def delete\("),
    ("proposal_store.py", "_write", r"^def _write\("),
    ("cart_generation.py", "bump_generation (counter, overwrites)", r"^def bump_generation\("),
    ("cart_generation.py", "assert_writable", r"^def assert_writable\("),
    ("multi_cart.py", "imprint_with_meta", r"^def imprint_with_meta\("),
    ("multi_cart.py", "_persist_cart (OUTSIDE save_as_npz)", r"^def _persist_cart\("),
    ("multi_cart.py", "  savez", r"^    np\.savez_compressed\(cart_path"),
    ("federate.py", "_save_federated_cart (OUTSIDE save_as_npz)", r"^def _save_federated_cart\("),
    ("federate.py", "_save_federated_cart CALL SITES", r"save_result = _save_federated_cart\("),
    ("membox.py", "imprint (requires agent_id)", r"^def imprint\("),
    ("membox.py", "CartLock", r"^class CartLock"),
    ("membox.py", "_write_log (2nd in-memory ring buffer)", r"^_write_log: dict"),
    ("membox.py", "_write_log dropped on unmount", r"del _write_log\[cart_id\]"),
    ("supabase_storage.py", "append_activity (unconditional write path)", r"^def append_activity\("),
    ("supabase_storage.py", "is_logging_enabled (fail-open)", r"^def is_logging_enabled\("),
    ("supabase_storage.py", "set_logging_enabled (can disable the trail)", r"^def set_logging_enabled\("),
    ("supabase_storage.py", "auto_provision_primary", r"^def auto_provision_primary\("),
    ("redact.py", "review_log_path (env-overridable)", r"^def review_log_path\("),
    ("redact.py", "append_review_log (durable JSONL, UNCHAINED)", r"^def append_review_log\("),
    ("redact.py", "  open append + fsync", r'with open\(path, "a", encoding="utf-8"\) as fh:'),
]


def stamp(path: Path) -> str:
    raw = path.read_bytes()
    n = len(raw.decode("utf-8", errors="replace").splitlines())
    return (f"{path.name}  lines {n}  bytes {len(raw)}  "
            f"sha256 {hashlib.sha256(raw).hexdigest()[:16]}")


def resolve(path: Path, anchors) -> list:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for group, label, pat in anchors:
        rx = re.compile(pat)
        hits = [i for i, ln in enumerate(lines, start=1) if rx.search(ln)]
        out.append((group, label, hits))
    return out


def main(argv) -> int:
    check = "--check" in argv
    target = Path(argv[1]) if len(argv) > 1 and not argv[1].startswith("-") \
        else DEFAULT_TARGET
    here = target.parent

    missing = []
    print(stamp(target))
    print()
    last_group = None
    for group, label, hits in resolve(target, ANCHORS):
        if group != last_group:
            print(f"-- {group}")
            last_group = group
        if not hits:
            missing.append(f"{target.name}: {label}")
            print(f"   {label:46s} ** NOT FOUND **")
        else:
            print(f"   {label:46s} {', '.join(str(h) for h in hits)}")

    print()
    by_module: dict = {}
    for module, label, pat in MODULE_ANCHORS:
        by_module.setdefault(module, []).append((module, label, pat))
    for module, group in by_module.items():
        p = here / module
        if not p.exists():
            missing.append(f"{module}: FILE MISSING")
            print(f"-- {module}  ** FILE MISSING **")
            continue
        print(f"-- {stamp(p)}")
        for _, label, hits in resolve(p, group):
            if not hits:
                missing.append(f"{module}: {label}")
                print(f"   {label:46s} ** NOT FOUND **")
            else:
                print(f"   {label:46s} {', '.join(str(h) for h in hits)}")

    print()
    if missing:
        # An anchor that no longer matches is a FINDING, not a formatting problem: the
        # call site either moved in a way the pattern cannot follow, or it is gone and the
        # audit entry describing it is now false.
        print(f"VERDICT: {len(missing)} anchor(s) no longer match — the audit entry for "
              f"each needs re-reading, not just renumbering:")
        for m in missing:
            print(f"  - {m}")
        return 1 if check else 0
    print("VERDICT: every anchor resolved; the audit's numbers can be refreshed from "
          "the listing above.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
