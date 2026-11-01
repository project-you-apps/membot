# Administration guide

Running Membot for other people: configuration, deployment, isolation, backup, deletion
and the checks an operator should run. Read
[`README-product.md`](README-product.md) first for the concepts, and
[`COMPLIANCE-posture.md`](COMPLIANCE-posture.md) for what is and is not recorded.

---

## 1. Command-line flags

These are the flags the argument parser actually accepts, read from the `__main__` block
of `membot_server.py`. The top-of-file module docstring and the repository `README.md`
both describe a smaller or older set; where they disagree with the list below, this list
is correct.

| Flag | Default | What it does |
| --- | --- | --- |
| `--transport` | `stdio` | One of `stdio`, `http`, `sse`. |
| `--host` | `127.0.0.1` | HTTP bind address. Loopback by default, so a fresh install is not reachable from the network. `localhost` and `localhost.localdomain` are recognized as loopback alongside numeric addresses. |
| `--port` | `8000` | HTTP port. |
| `--writable` | off | Enables `memory_store`, `save_cartridge` and `subcart_export`. |
| `--read-only` | on | **Has no effect.** It is declared with `default=True` while the server computes its mode as `read_only = not args.writable`, so read-only is the default whether you pass this flag or not, and passing it does not override `--writable`. It is accepted for backward compatibility; `--writable` is the only control. |
| `--mount <name>` | none | Auto-mount a cart at startup into the default session. **Creates an empty cart if the name does not exist.** If the cart fails its manifest integrity check the server logs the failure and exits with status 3; if the cart can be neither found nor created it exits with status 4. |
| `--allow-unverified-mount` | off | Start anyway when the `--mount` cart fails its integrity check. The failure is still logged at ERROR. Use only for a known-stale manifest on a cart you trust. |
| `--max-concurrent-searches <n>` | half the core count, clamped to 2–8 | Ceiling on simultaneous searches. Callers past the ceiling wait up to 20 seconds and are then told the server is busy. |
| `--max-cart-ram-mb <n>` | half of physical RAM | Ceiling on the total RAM the shared cart pool may hold. Mounts that would exceed it are refused. A cart already resident always mounts, since an additional holder costs nothing. |

There is also a start-up refusal worth knowing about. If the server would be **writable
and reachable from the network and unauthenticated** — that is, `--writable` plus a
non-loopback `--host` plus no `MEMBOT_API_KEY` — it logs the three conditions and exits
with status 5 rather than starting. Any one of the three changes resolves it. A
read-only public instance and a writable loopback instance both start normally.

---

## 2. Environment variables

### Access control and limits

| Variable | Unset behavior |
| --- | --- |
| `MEMBOT_API_KEY` | **No authentication on the HTTP transport.** Rate limiting still applies; the start-up log says `rate limiting enabled (no auth — set MEMBOT_API_KEY to require)`. When set, MCP tool calls must carry `Authorization: Bearer <key>`. Two limits: it is a single shared secret with no per-caller identity, and it is enforced in MCP middleware only, so the `/api/` REST routes are not covered by it. |
| `MEMBOT_RATE_LIMIT` | `60` requests per window, per client IP. |
| `MEMBOT_RATE_WINDOW_SEC` | `60` seconds. |
| `MEMBOT_RATE_LIMIT_LOOPBACK_EXEMPT` | Loopback clients are rate-limited like anyone else. Set it to exactly `1` to exempt them. This is opt-in rather than default for a specific reason: behind a reverse proxy on the same host, every public request arrives from loopback, so a default exemption would silently disable the limiter on a public instance. On a single-user local box the exemption is usually right — otherwise the agent, its tools and the REST bridge share one budget and throttle each other, and a throttled `save_cartridge` loses work that looks like a rate-limit message. |
| `MEMBOT_MAX_ENTRIES` | `3000000` passages per cart. |
| `MEMBOT_MAX_TEXT_LENGTH` | `10000` characters per stored passage. Longer content is refused, so if you are ingesting real documents, raise this rather than chopping the source. |
| `MEMBOT_MAX_QUERY_LENGTH` | `2000` characters per query. |

### Embedding backend

| Variable | Unset behavior |
| --- | --- |
| `MEMBOT_EMBED_BACKEND` | `auto` — use Ollama if it is reachable, otherwise load SentenceTransformers in-process. Set to `ollama` or `st` to force one. The chosen backend is named in the start-up log, deliberately, so a silent fallback cannot be mistaken for a configuration. |
| `OLLAMA_HOST` | `http://localhost:11434`. |
| `MEMBOT_OLLAMA_EMBED_MODEL` | `nomic-embed-text`. |
| `MEMBOT_OLLAMA_EMBED_NUM_GPU` | Ollama's own default GPU layer count. |
| `MEMBOT_OLLAMA_EMBED_KEEP_ALIVE` | Ollama's own default model keep-alive. |

Using Ollama rather than the in-process model saves roughly 2 GB of resident memory and
the model download, because the already-running Ollama process holds the weights.

### Redaction

| Variable | Unset behavior |
| --- | --- |
| `MEMBOT_REDACTION_LOG` | The review log is written to `membot/redaction-review.jsonl`. Set this to relocate it — onto a separate volume, or somewhere with different permissions from the cart directory. |

### Mempacks and web authentication

All four are needed only for the Mempack and signed-in-web path. **Leave `SUPABASE_URL`
unset and every Mempack feature switches itself off cleanly** — "not configured" is a
supported state, not an error.

| Variable | Unset behavior |
| --- | --- |
| `SUPABASE_URL` | Supabase is treated as unconfigured; Mempack features are unavailable. |
| `SUPABASE_SECRET_KEY` | Falls back to `SUPABASE_SERVICE_ROLE_KEY`. If neither is set, Supabase is unconfigured. Both names are accepted on purpose, so a key rename cannot silently make the server conclude Supabase is absent. |
| `SUPABASE_SERVICE_ROLE_KEY` | The legacy name for the above. |
| `SUPABASE_JWT_SECRET` | Symmetrically signed (HS256) user tokens cannot be verified and are rejected. Asymmetrically signed tokens (RS256, ES256) are verified against the published key set at `SUPABASE_URL` and do not need this secret. Token verification fails closed: an unverifiable token is not treated as trusted. |
| `MEMBOT_PROD` | Sign-out cookie clearing uses no cookie domain and does not set `Secure`. Set it to any non-empty value on a public HTTPS deployment, or make sure your proxy sends `X-Forwarded-Proto: https`, which the server also accepts as the production signal. |

---

## 3. Transports, and what each is for

**stdio** is for one user on one machine. The agent host launches Membot as a child
process; there is no listener and no network surface. Identity is not verified because
there is nothing to verify — the pipe is the user. This is the mode to recommend to a
single-practitioner install.

**HTTP** (`--transport http`) serves the MCP endpoint at `/mcp` plus the REST and
dashboard routes. This is the mode for several clients, a browser front end, or a remote
agent. `sse` is also accepted as a transport value, for older MCP clients that expect
Server-Sent Events.

### What the HTTP middleware covers, and what it does not

The authentication and rate-limiting middleware hooks **MCP tool calls**. The server also
registers plain HTTP routes with `@mcp.custom_route`, and those are not MCP tool calls, so
they pass through neither the Bearer check nor the limiter:

- Read routes: `/api/status`, `/api/search`, `/api/filter`, `/api/cartridges`,
  `/api/passage`, `/api/embed`, `/depot/status`, `/depot`, `/app`.
- **Routes that change state with no authentication of their own:** `POST /api/store`,
  `POST /api/save`, `POST /api/mount` (with `auto_create`), and `POST /api/sync`. Of
  these, `/api/store`, `/api/save` and `/api/mount` inherit the read-only gate, so a
  default-configured server refuses them. `POST /api/sync` does not: it appends
  caller-supplied JSON lines to a file under `cartridges/` whose name the caller chooses
  (sanitized for path characters), with no authentication, no read-only gate and no size
  or line limit.
- Routes that do authenticate, against a verified Supabase token and an ownership check:
  the `/api/mempack*` family, `/api/mempacks`, `/api/copy`, `/api/whoami`,
  `/api/auth/signout`.
- CORS on the REST routes is `Access-Control-Allow-Origin: *`.

**Operational consequence.** If you expose Membot off loopback, put a reverse proxy in
front of it and allow only the paths you intend to serve. Do not rely on
`MEMBOT_API_KEY` to protect `/api/`. For a deployment where only MCP agents should reach
the server, proxying `/mcp` alone and blocking `/api/`, `/depot/` and `/app` is the
smallest configuration that matches the authentication model.

---

## 4. Deployment

### systemd

The unit shipped as `membot/membot.service`:

```ini
[Unit]
Description=Membot Vector+ Memory Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/membot
ExecStart=/opt/membot/venv/bin/python membot_server.py --transport http --read-only --port 8000
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

Four things to know about it before you deploy it unchanged:

1. **It binds loopback.** It passes no `--host`, so the server listens on `127.0.0.1`
   only and is not reachable from other machines. That is a sensible default behind a
   reverse proxy on the same host; if you expect it to serve the network directly, it
   will appear to be down.
2. **`--read-only` in the `ExecStart` line does nothing** (see the flag table above).
   The server is read-only because `--writable` is absent, not because of that flag.
   Removing `--read-only` changes nothing; adding `--writable` is what makes it writable.
3. **It runs as `root`.** For a regulated deployment, create a dedicated service account
   that owns `/opt/membot` and the cart directory, and set `User=` to it. Nothing in
   Membot needs privileged ports or root-owned paths.
4. **It sets no environment variables other than `PYTHONUNBUFFERED`.** Add an
   `EnvironmentFile=` pointing at a root-owned, mode-0600 file for `MEMBOT_API_KEY` and
   any Supabase keys, rather than putting secrets in the unit, which is world-readable.

### Reverse proxy

A proxy terminating TLS on the same host and forwarding to `127.0.0.1:8000` is the
expected shape. Two consequences follow from it and are easy to miss: every request will
appear to come from loopback, so leave `MEMBOT_RATE_LIMIT_LOOPBACK_EXEMPT` unset; and the
proxy should send `X-Forwarded-Proto: https`, which the server uses as its production
signal for cookie handling.

### Logging

The server logs to stderr and to `membot/membot.log`. **The file handler opens in `w`
mode, so the log is truncated on every start.** If you need operational history across
restarts, capture stderr — under systemd, the journal already does — or rotate the file
before service start. Do not treat `membot.log` as a retained record of anything.

---

## 5. Per-seat access: what isolates one user from another

Isolation in Membot is layered, and it is important to know which layer is a security
boundary and which is a view.

**Mount isolation (a real boundary, within a process).** Each session has its own mount.
A signed-in caller's session id is derived from their verified token, so they get their
own mount slot. Mempack lookup by name is scoped to the owner, so another user's Mempack
is not reachable by name. The REST read routes additionally re-check ownership before
returning content from a mounted Mempack and return 403 on a mismatch.

**Seat views (not a security boundary).** Hot stacks, hidden passages and sub-cartridges
are per-seat so that several people can share one cart without disturbing each other's
view. They scope what you see; they do not stop anyone who can mount the cart from
reading all of it.

**Private notes (a read-time rule, enforced in one place).** A seat sees its own private
notes plus every shared note, never another seat's private note. The filter lives in a
single function that every read path goes through. An unidentified caller — empty seat —
sees shared notes only, which is the fail-closed reading of "I do not know who you are".

**The cart is the actual boundary.** The rule the design states plainly, and the one to
give your users: *a cart someone must not read is a cart they cannot mount.* Separate
anything confidential into its own cart and control access to the file. Do not use a
sub-cartridge, a hidden passage or a private note as a confidentiality control.

**Where this is weakest.** On the HTTP transport with only `MEMBOT_API_KEY`, every caller
holds the same secret. Such a caller may name any seat it likes, except the reserved
`user_` and `anon_` prefixes, which are refused precisely because those are minted from
verified facts. So a shared-key caller cannot impersonate a signed-in user, but shared-key
callers are not distinguishable from one another and can read each other's seat state. If
you need per-person separation over HTTP, the token-authenticated path is the one that
provides it.

---

## 6. Backup: exactly which files are "the data"

Carts and their sidecars are separate files on purpose. A backup that captures only the
`.npz` restores the content and loses the notes, the provenance and every seat's history.

**Back up the whole cart directory tree.** For a cart at
`cartridges/<name>.cart.npz`, the complete set is:

| File | Lose it and you lose |
| --- | --- |
| `<name>.cart.npz` | the content and its vectors |
| `<name>.cart_manifest.json` | the ability to detect that the cart was altered |
| `<name>.cart.npz.journal.jsonl` | the provenance record |
| `<name>.cart.npz.journal.jsonl.hwm` | detection of entries removed from the end of the journal |
| `<name>.cart.npz.notes.json` | the "why" layer |
| `<name>.cart.npz.proposals.json` | open and resolved correction proposals |
| `<name>.cart.npz.generation.json` | stale-view detection (regenerates; lowest value) |
| `overlays/*.overlay.json` | every seat's attention history and hidden list — one file per seat |
| `subcarts/*.subcart.json` and `subcarts/trash/` | named selections, including recoverable deleted ones |

Also back up, once per installation rather than per cart:

- `redaction-review.jsonl` (or wherever `MEMBOT_REDACTION_LOG` points) — the only record
  of what credential redaction removed at ingest.
- Your environment file, separately and with its own access controls, since it holds
  `MEMBOT_API_KEY` and any Supabase keys.

**Back up the journal and its high-water-mark file so that they can be verified
together**, and prefer a destination the Membot host cannot write to. Both the journal
and its mark are files in the cart directory, so anyone who can write there can alter
both; an off-host copy is what turns that from "detectable by the operator" into
"detectable by someone checking the operator". See
[`COMPLIANCE-posture.md`](COMPLIANCE-posture.md) for the precise limit.

**Mempacks are not in this tree.** Their content lives in your Supabase project
(`mempacks`, `mempack_patterns`, `mempack_activity`, `mempack_provisions_log`, and the
`mempacks` storage bucket). Backing up the Membot host does not back them up; the local
copies on disk are a cache. If you use Mempacks, Supabase backup is a separate task with
its own schedule.

**Consistency.** Carts are written whole and replaced; sidecars other than the journal are
written to a temporary file and renamed atomically; the journal is appended under an
exclusive lock and fsynced. A copy taken mid-write can still catch the journal one entry
ahead of or behind its mark, which `verify` reports as either a torn tail or a stale mark
rather than as tampering. A snapshot of a quiesced server is cleanest.

---

## 7. Retention and deletion, stated honestly

**There is no delete.** This is a design decision, not a gap, and it has consequences you
must plan for.

What the agent-facing surface can do:

- `propose_correction` files a proposal and changes nothing. The reasoning is explicit:
  an agent's corpus can be manipulated into instructing the agent to delete things, so
  destructive verbs are not reachable from that surface at all. Proposals are resolved by
  a human elsewhere.
- `seat_hide` removes a passage from one seat's view. Nothing is deleted and no other
  seat is affected.
- `subcart_delete` moves a selection's manifest into a `trash/` subdirectory. It is never
  unlinked.
- Notes are append-only. A note is corrected by writing a new one that supersedes it; the
  original stays exactly as written, and a correction cannot widen who could see the
  original.
- The provenance journal has no compaction, no delete and no amend function, by explicit
  design. History is corrected by appending an erratum, which is itself a chained entry.

The only overwrite on the surface is `mempack_update_pattern_i`, which replaces one
reserved pattern in place.

**What that means for a retention policy.** A policy of the form "records of type X are
destroyed after N years" is not satisfiable today through Membot's own interface. What is
available is:

1. **Delete the files.** Carts, sidecars and journals are plain files. An operator with
   filesystem access can remove a cart and everything beside it. This is the only
   complete deletion mechanism and it is all-or-nothing at cart granularity.
2. **Separate by cart from the start.** Because the cart is the real boundary and the only
   deletable unit, retention classes should be separate carts. Deciding this at ingest is
   much cheaper than discovering it at a deletion request.
3. **Removing one passage from a cart is not supported.** It would mean rewriting the
   `.npz`, regenerating the manifest, and leaving the journal describing a passage that is
   no longer there. No tooling does this. If you need per-record deletion, say so during
   evaluation rather than assuming it.

**Also note what survives a cart deletion.** The redaction review log and the server log
live outside the cart directory. The review log holds kinds, offsets and one-way
fingerprints — never a credential value, and never passage text — but it is a separate
file with its own lifetime, and a thorough deletion procedure should account for it.

---

## 8. Operational checks

### Verify the provenance journal

The one check to run on a schedule. It prints a computed verdict line first and exits
non-zero on a break:

```bash
python journal.py verify cartridges/<name>.cart.npz.journal.jsonl
```

A clean result looks like `OK: 1284 entries, seq 1..1284, chain intact`. A break prints
`BROKEN at line N (seq S): <reasons> -- <detail>` followed by the offending record, and
names the failure precisely rather than reporting a generic integrity error. The
distinguishable cases are: a mid-file line that is no longer valid JSON; a last line torn
by a power failure, which is tolerated when the high-water mark confirms it; a missing
sequence number, meaning a record was deleted; a repeated or out-of-order sequence number,
meaning records were reordered or duplicated; a record that does not chain to its
predecessor; a record whose own hash no longer covers its own contents, meaning it was
edited in place; and entries removed from the end of the file, which only the high-water
mark can see.

Treat a non-zero exit as an incident. There is deliberately no repair path: `verify` names
the break and a human decides, because silent repair is indistinguishable from tampering.

Shape, separately from integrity:

```bash
python journal.py stats cartridges/<name>.cart.npz.journal.jsonl
python journal.py tail  cartridges/<name>.cart.npz.journal.jsonl 20
```

`stats` reports entry counts broken down by operation, writer and surface, and the first
and last timestamps. Its `gapless` field is arithmetic over the sequence range, not a
chain check — `verify` is the only thing that can say the journal is intact.

### Verify cart integrity

Each cart carries a manifest holding a SHA-256 fingerprint of its embeddings and a passage
count. It is checked when a cart is mounted, and the result is acted on: a cart that fails
is not served. If a cart auto-mounted with `--mount` fails, the server exits 3 rather than
starting and quietly serving nothing. A cart with no manifest is reported as "legacy" and
is allowed; `scripts/rebuild_manifests.py` exists for generating them.

Be clear about what this check proves. It detects corruption, an edit made without
regenerating the manifest, and a truncated file. It does not detect an edit where the
manifest was regenerated too, because both files sit in the same directory.

### Liveness and visibility

```bash
curl http://127.0.0.1:8000/api/status
```

`/depot/status` returns a JSON snapshot of sessions, carts and recent activity, and
`/depot` serves the same as an HTML board. Two cautions: the recent-activity list is an
in-memory ring buffer of the last 200 events that does not survive a restart, and the
events it holds are read-path events only — searches, mounts, passage fetches. It is not
an audit trail and must not be presented as one.

The start-up log is worth reading once per deployment, because it states as computed
facts: the chosen embedding backend and why, the cart pool memory budget, the search
concurrency ceiling, read-only versus writable, the cart search directories, the entry and
text limits, whether the bind address is loopback, and whether authentication is required.
If an instance is behaving unexpectedly, that block usually says why.

### Audit anchors

`python audit_anchors.py --check` re-resolves the code locations cited by the internal
mutation-path audit and exits non-zero if any no longer match. This is a maintenance tool
for whoever keeps that audit current rather than a production check, but it is the
mechanism that stops the audit's claims from silently going stale, which is worth knowing
if you are relying on that document.

### Routine checklist

| Frequency | Check |
| --- | --- |
| Per deployment | Read the start-up log block. Confirm the bind address, the write mode and whether a key is required are what you intended. |
| Per deployment | Confirm the service account is not `root` and that the environment file holding `MEMBOT_API_KEY` is not readable by other accounts. |
| Daily or per backup | `journal.py verify` on every active cart. Non-zero exit is an incident. |
| Per backup | Confirm the backup contains the sidecars and the `overlays/` directory, not just the `.npz`. |
| Weekly | Review `redaction-review.jsonl` for new entries. Each line means a credential-shaped string was stripped at ingest; review by distinct fingerprint, not by occurrence. |
| On any integrity failure | Stop writing to the cart, preserve the journal and its high-water-mark file, and compare against the off-host copy. |
