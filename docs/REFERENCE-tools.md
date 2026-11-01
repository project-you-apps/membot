# Tool reference

Every tool Membot exposes over MCP. There are **44**, enumerated by matching
`@mcp.tool()` in `membot_server.py`; all 44 are documented below. The same set is
available on both transports (stdio and HTTP) — the transport changes how a caller is
identified, not which tools exist.

Every tool returns a **string**. Most return human-readable text intended for an agent to
read; `get_pattern_meta` and `walk_associate` return JSON encoded as a string. There is no
structured-object return type anywhere on this surface.

## How to read the tables

- **Effect** is one of:
  - `read` — returns information, changes no durable state.
  - `write` — changes state on disk (a cart, or a sidecar beside it).
  - `process` — changes only in-memory server state (what is mounted, who holds a lock).
    Nothing on disk changes and the change is lost on restart.
- **Read-only** says whether the tool is refused when the server runs without
  `--writable`. The gate is a single function with exactly four call sites, so the honest
  answer for most write tools is **no**.

### What `--read-only` actually blocks

Read-only mode is the default. It refuses exactly these:

| Refused under read-only | |
| --- | --- |
| `memory_store` | except when the mounted cart is a Mempack, which is deliberately carved out |
| `save_cartridge` | |
| `subcart_export` | because it writes a new cart file |
| `POST /api/mount` with `auto_create` | the REST cart-creation path |

Every other state-changing tool below runs regardless of the flag, including
`mempack_update_pattern_i` (an in-place overwrite), `note_add`, `note_share`, `seat_hide`,
`subcart_create`, `subcart_delete`, `propose_correction`, `membox_imprint` and the three
`federate_*` writers. Treat `--read-only` as "this server will not grow or rewrite a
knowledge cart", **not** as "this server is immutable". A separate, stronger gate exists
per mount: `mount_cartridge(..., read_only=True)` refuses all writes to that mount for the
life of the mount, and there is deliberately no way to unlock it short of remounting.

---

## Carts: list, mount, read

| Tool | Parameters (defaults) | Effect | Read-only | Returns |
| --- | --- | --- | --- | --- |
| `list_cartridges` | none | read | allowed | Available carts in the search directories, with size and capabilities. |
| `mount_cartridge` | `name`, `session_id=""`, `owner_id=""`, `read_only=False` | process | allowed | Loads a cart into this session by exact or partial name, verifying it against its manifest if one exists. `owner_id` scopes a Mempack lookup to one user. `read_only=True` refuses all writes to this mount regardless of server flags, permanently for that mount. |
| `unmount` | `session_id=""` | process | allowed | Drops this session's mount and frees its memory. |
| `get_status` | `session_id=""` | read | allowed | One line: mounted cart, passage count, embedding dimension, whether the Hamming index and navigation links are present, GPU availability, active session count, this session id, whether this view is stale relative to the cart on disk. |
| `get_passage` | `idx`, `session_id=""` | read | allowed | The full untruncated text of one passage, plus its navigation links. The companion to `memory_search`, which returns truncated previews. |
| `passage_links` | `idx`, `session_id=""` | read | allowed | Previous and next pointers, source hash, sequence position, flags, and the text of the linked passages. |
| `get_pattern_meta` | `cartridge`, `idx`, `session_id=""` | read | allowed | JSON string. The full per-pattern metadata record for one index in a Cart-Builder-authored cart: content type (`document`, `graphic`, `table`), source filename, page, and for graphics a base64 PNG, for tables the HTML. One record per call, because a single image can be hundreds of kilobytes. `cartridge` accepts a multi-cart pool id, a session-mounted cart name, or `""` for whatever this session holds. |

## Search

| Tool | Parameters (defaults) | Effect | Read-only | Returns |
| --- | --- | --- | --- | --- |
| `memory_search` | `query`, `top_k=5`, `session_id=""`, `verbose=False` | read | allowed | Ranked previews of roughly 550 characters each, with an index per hit and previous/next hints. `verbose=True` adds the per-result signal breakdown. **Note:** a search that returns hits also updates this seat's attention overlay on disk, so searching is a read that has a durable per-seat side effect. |
| `walk_associate` | `query`, `top_k=10`, `walk_top_k=0`, `walk_min_hits=2`, `walk_max_show=10`, `temperature=0.0`, `session_id=""`, `cart_id=""` | read | allowed | JSON string. Direct matches plus a "may have missed" set found by re-querying from each primary result's own embedding and keeping items that recur. `walk_top_k=0` means auto. Use for exploration; use `memory_search` for direct lookup. |

## Writing and saving

| Tool | Parameters (defaults) | Effect | Read-only | Returns |
| --- | --- | --- | --- | --- |
| `memory_store` | `content`, `tags=""`, `session_id=""` | write | **refused** (except Mempacks) | Embeds and appends one passage to the mounted cart, in memory; `save_cartridge` persists it. `content` is capped at `MEMBOT_MAX_TEXT_LENGTH`. Byte-identical re-stores are skipped as duplicates. Credential redaction runs before the text is embedded, and the store is **refused** if the redaction scan itself fails. This is the one tool that appends a provenance journal record for the write itself. The reply names both a count (`#N`) and the usable address (`idx:N`); the two differ by one, so use the `idx:` value. |
| `save_cartridge` | `name=""`, `session_id=""` | write | **refused** | Writes the in-memory cart to disk as `.npz` and regenerates the SHA-256 manifest. `name=""` saves over the mounted name. Bumps the generation counter, which **overwrites** the previous writer stamp rather than appending history. |

## The why layer: notes and proposals

| Tool | Parameters (defaults) | Effect | Read-only | Returns |
| --- | --- | --- | --- | --- |
| `note_add` | `idx`, `kind`, `body`, `scope="private"`, `surface="CLI"`, `session_id=""` | write | allowed | Attaches a sticky note to a passage. `kind` is one of `why`, `context`, `caution`, `decision`, `prediction`. Notes are append-only and born private; nothing but the author's own seat can see a private note. `surface` must be one of `CLI`, `VSC`, `WEB` and is taken from the caller without verification. Requires a cart that exists on disk. Appends a journal record. |
| `notes` | `idx=-1`, `session_id=""` | read | allowed | Every note this seat may see — its own private notes plus all shared notes — for the whole cart, or for one passage when `idx` is given. Never another seat's private notes. |
| `note_share` | `note_id`, `session_id=""` | write | allowed | Promotes one of the caller's own private notes to cart-wide visibility. Only the author may do this. Appends a journal record with operation `perm_change`. |
| `propose_correction` | `idx`, `kind`, `body`, `suggested_text=""`, `session_id=""` | write | allowed | Files a proposal against a passage and changes nothing else. `kind` is one of `correction`, `outdated`, `duplicate`, `context`, `question`. `suggested_text` is never applied automatically. This exists because the agent surface has no destructive verbs: corrections are reviewed by a human. |
| `proposals` | `status="open"`, `session_id=""` | read | allowed | Proposals filed against the mounted cart. `status` is `open`, `accepted`, `rejected` or `all`. Accepting or rejecting is an owner action elsewhere and is deliberately not reachable from this surface. |

## Per-seat views

| Tool | Parameters (defaults) | Effect | Read-only | Returns |
| --- | --- | --- | --- | --- |
| `seat_hot_stack` | `top_k=8`, `session_id=""` | read | allowed | What this seat has been working on, ranked by this seat's own attention. Scored only over passages this seat has actually touched. |
| `firm_hot_stack` | `top_k=10`, `session_id=""` | read | allowed | The roll-up across all seats with an overlay on this cart, ranked by **distinct seats** first and volume second, so forty people reading something once outranks one person reading it forty times. Shows only passages present in the mounted cart, and can surface passages the caller has never opened. |
| `seat_hide` | `indices`, `unhide=False`, `session_id=""` | write | allowed | Removes passages from this seat's own search results and hot stack. `indices` is a comma-separated list of `idx` values, for example `"3,7,12"`. Nothing is deleted, no other seat's view changes, and `unhide=True` reverses it. Writes to this seat's overlay sidecar. |
| `seat_hidden` | `session_id=""` | read | allowed | What this seat has hidden, so it can be found and undone. |

## Sub-cartridges

| Tool | Parameters (defaults) | Effect | Read-only | Returns |
| --- | --- | --- | --- | --- |
| `subcart_create` | `name`, `indices`, `title=""`, `authored_by="user"`, `session_id=""` | write | allowed | Creates a named selection over the mounted cart, stored as content keys rather than copies. `indices` is a comma-separated list of `idx` values, or the literal `"all"`. `authored_by` records how the selection was made and nothing branches on it. **Not a security boundary** — a sub-cart scopes what you see; separating what someone must not read means putting it in a different cart. |
| `subcart_list` | `session_id=""` | read | allowed | Sub-cartridges over the mounted cart and how many passages each currently resolves to. Reports manifests that fail to parse rather than skipping them. |
| `subcart_export` | `name`, `out_name=""`, `session_id=""` | write | **refused** | Materializes a sub-cart into a standalone cart: copies passages and embeddings in, remaps internal link indices, verifies the result and refuses to write if verification fails. Will not overwrite an existing file. Default output name is `<subcart>-export`. Reports unresolved keys and severed links. |
| `subcart_delete` | `name`, `session_id=""` | write | allowed | Deletes a sub-cart definition by moving its manifest into a `trash/` directory; it is never unlinked. The parent cart and its passages are untouched. |

## Mempacks

Mempacks require Supabase credentials. With `SUPABASE_URL` or the secret key unset, these
tools cannot operate on a Mempack.

| Tool | Parameters (defaults) | Effect | Read-only | Returns |
| --- | --- | --- | --- | --- |
| `mempack_read_pattern_i` | `session_id=""` | read | allowed | The text of Pattern I — the agent's own behavioral instructions, reserved at index 1 of an agent-memory cart — or an explanatory message if the mounted cart is not a Mempack or has no Pattern I yet. |
| `mempack_update_pattern_i` | `text`, `session_id=""` | write | **allowed** | Overwrites Pattern I **in place**, then round-trips to Supabase: the blob is repacked, the pattern row at index 1 is updated, and the denormalized preview on the Mempack row is refreshed. `text` is capped at 10,000 characters. This is the one destructive overwrite on this surface; it is deliberately *not* gated by read-only mode, and the previous text is preserved only in the Supabase activity log, which the Mempack owner can switch off. |

## Multi-cart pool

A process-global pool, shared across sessions, for querying many carts at once. These
tools take no `session_id`.

| Tool | Parameters (defaults) | Effect | Read-only | Returns |
| --- | --- | --- | --- | --- |
| `multi_mount` | `cart_path`, `cart_id=""`, `role=""`, `verify_integrity=True` | process | allowed | Adds a cart to the pool by filesystem path. `cart_id` defaults to the filename. `role` is a free-text tag used to filter searches. `verify_integrity=False` accepts a stale or mismatched manifest. |
| `multi_unmount` | `cart_id` | process | allowed | Removes one cart from the pool. |
| `multi_list` | none | read | allowed | Every cart currently in the pool. |
| `multi_mount_directory` | `dir_path`, `role=""`, `pattern="*.cart"` | process | allowed | Adds every matching cart file in a directory with one shared role. |
| `multi_search` | `query`, `top_k=10`, `scope="all"`, `role_filter=""`, `scope_mode="global"` | read | allowed | Searches the pool with results attributed to their source cart. `scope` is `all`, `local` (first cart only) or a specific `cart_id`. `scope_mode` controls cross-cart ranking: `global` (true top-K overall), `per_cart` (top-K from each, up to K times the cart count), `balanced` (top-K candidates per cart, then globally reranked), `diagnostic` (top-K from every cart, unmerged). |

## Federation

For fleets of machines that each publish their own learning cart. All three writers below
run regardless of read-only mode and write cart files through a code path that the shared
persistence helpers do not cover.

| Tool | Parameters (defaults) | Effect | Read-only | Returns |
| --- | --- | --- | --- | --- |
| `federate_publish` | `session_file`, `machine_id`, `fleet_dir` | write | allowed | Appends the entries in a `.jsonl` (or a `.json` with an `entries`/`learning_entries` key) to one machine's federated cart under `fleet_dir`. |
| `federate_consolidate` | `fleet_dir`, `output_dir=""`, `similarity_threshold=0.85`, `mode="preserve"` | write | allowed | Mounts every machine cart in the fleet, finds cross-machine matches above the threshold, and writes a consolidated cart. `mode="preserve"` keeps all variants and records cross-cart agreement as metadata; `mode="collapse"` keeps one representative per matched group, which is smaller and loses the individual machines' versions. `output_dir` defaults to a `consolidated` directory beside the fleet. |
| `federate_migrate_jsonl` | `jsonl_dir`, `output_dir=""`, `in_place=False` | write | allowed | One-time conversion of per-machine JSONL learning files into carts. The source JSONL files are not removed. |
| `federate_load` | `fleet_dir` | process | allowed | Mounts every machine's federated cart into the multi-cart pool with `role="federated"`, so one `multi_search` reaches the whole fleet. |

## Membox: shared carts with locking

Membox is the multi-writer path for a single shared cart file. It is the best-attributed
write surface on the server: `agent_id` is required, there are no anonymous writes, and a
real file lock is held across the write.

| Tool | Parameters (defaults) | Effect | Read-only | Returns |
| --- | --- | --- | --- | --- |
| `membox_mount` | `cart_path`, `cart_id=""`, `role=""`, `lease_seconds=30`, `verify_integrity=True` | process | allowed | Mounts a cart in Membox mode. `lease_seconds` is the auto-release timeout if a lock holder crashes. |
| `membox_unmount` | `cart_id` | process | allowed | Unmounts and releases the lock if held. Also discards that cart's in-memory write log. |
| `membox_list` | none | read | allowed | Every Membox-mounted cart with its lock state and write statistics. |
| `membox_imprint` | `cart_id`, `text`, `agent_id`, `tags=""`, `reasoning=""`, `origin="agent"`, `timeout_ms=5000` | write | **allowed** | Acquires the lock, writes one pattern with per-agent metadata (`agent_id`, write time, `origin`, `tags`, `reasoning`), releases the lock. `agent_id` is required. `origin` is `agent`, `human` or `system`. Returns the new address or an error if the lock could not be taken in time. **`agent_id` is self-declared and not verified.** |
| `membox_search` | `cart_id`, `query`, `top_k=10`, `agent_id=""` | read | allowed | Searches without blocking on the write lock. Results carry the per-pattern Membox metadata, so a consumer can see who wrote what. `agent_id` here is advisory and currently has no audit effect. |
| `membox_acquire_lock` | `cart_id`, `agent_id`, `timeout_ms=5000` | process | allowed | Takes the write lock manually, for several writes in sequence. Most callers should use `membox_imprint`, which acquires, writes and releases as one operation. |
| `membox_release_lock` | `cart_id`, `agent_id` | process | allowed | Releases the lock. Only the current holder may release it. |
| `membox_lock_holder` | `cart_id` | read | allowed | The `agent_id` holding the lock, or `(idle)`. |
| `membox_status` | `cart_id` | read | allowed | Lock state, per-agent write counts, pattern count and the recent write log. The write log is an in-memory ring buffer capped at 100 entries per cart and does not survive a restart or an unmount. |

---

## Count and completeness

44 tools, grouped as follows:

| Group | Count | Names |
| --- | --- | --- |
| Carts | 7 | `list_cartridges`, `mount_cartridge`, `unmount`, `get_status`, `get_passage`, `passage_links`, `get_pattern_meta` |
| Search | 2 | `memory_search`, `walk_associate` |
| Write and save | 2 | `memory_store`, `save_cartridge` |
| Notes and proposals | 5 | `note_add`, `notes`, `note_share`, `propose_correction`, `proposals` |
| Per-seat views | 4 | `seat_hot_stack`, `firm_hot_stack`, `seat_hide`, `seat_hidden` |
| Sub-cartridges | 4 | `subcart_create`, `subcart_list`, `subcart_export`, `subcart_delete` |
| Mempacks | 2 | `mempack_read_pattern_i`, `mempack_update_pattern_i` |
| Multi-cart pool | 5 | `multi_mount`, `multi_unmount`, `multi_list`, `multi_mount_directory`, `multi_search` |
| Federation | 4 | `federate_publish`, `federate_consolidate`, `federate_migrate_jsonl`, `federate_load` |
| Membox | 9 | `membox_mount`, `membox_unmount`, `membox_list`, `membox_imprint`, `membox_search`, `membox_acquire_lock`, `membox_release_lock`, `membox_lock_holder`, `membox_status` |

The module docstring at the top of `membot_server.py` lists only seven tools under a
"Tools:" heading. That list is an early subset and was never extended; this document is
the complete surface.

## The REST surface is not the same surface

Separately from the 44 MCP tools, the server registers HTTP routes under `/api/`,
`/depot/` and `/app`. These are used by browser clients and are convenient for scripting,
but they are **not** gated by the MCP middleware: they receive neither the
`MEMBOT_API_KEY` Bearer check nor the per-IP rate limit, and four of them change state.
They are covered in [`GUIDE-administration.md`](GUIDE-administration.md) under Transports
and in [`COMPLIANCE-posture.md`](COMPLIANCE-posture.md) under what is not covered. If you
expose Membot beyond loopback, read both sections before you decide what to put in front
of it.
