# Membot

**A portable memory store that agents and people read through the same tools, with attribution on every write, and no LLM in the retrieval path.**

Membot is an [MCP](https://modelcontextprotocol.io/) server. Point an agent at it and the agent
gets a searchable store it can read, annotate and extend. Point a person at the same store
through [Vector+ Studio](https://github.com/project-you-apps/vector-plus-studio) and they see the
same records, the same notes, and the same history.

A **cartridge** — a cart — is one file. It holds passages, their embeddings, and a compact binary
signature per passage. You can copy it, back it up, hand it to someone, or keep it on a machine
that never touches the internet. Carts are built either by drag-and-drop in the Vector+ Studio
front end, or with the Desktop Builder for large local builds and OCR. Both ingest your
documents and embed them locally; building a cart does not send anything anywhere.

---

## Why this instead of a vector database

**Retrieval costs no tokens.** Search is a sentence-transformer embedding plus binary matching
plus keyword reranking. No model is called to answer a query. For an agent making hundreds of
retrievals in a run, that is the difference between a background cost and a line item.

**It stores WHY, which almost nothing does.** Every record system stores who touched what and
when. Almost none store **why**, because why has to be captured **at the moment of the decision
and cannot be reconstructed afterward.** Six months on, the document is still in the folder and
the person who chose it has moved on. Membot has a field for the part that leaves with them.

**Your reasoning is yours until you publish it.** A note is born **private**. Nothing else can
see it until you explicitly share it, because a shared "why" that is on by default is
surveillance rather than memory. ⇒ People write honestly only when they are not performing.

**Every write carries who, where and when.** Not an audit log bolted beside the data but an
invariant in the format. A note written through a web browser and a note written by an agent at
3am are both attributed, and both say which surface they came from.

**Corrections never erase.** When a note is corrected, the old one is marked superseded and
**stays** -- rendered struck through, still attributed. Nothing in the normal path removes a
record. A reader can see that someone was wrong, who said so, and when.

**It is a file.** No cluster, no managed service, no egress. The store is portable by
construction, which matters when the data is not allowed to leave a building.

---

## Quick start

### Prerequisites

- Python 3.10+
- An MCP-compatible client — [Claude Code](https://claude.com/claude-code), Claude Desktop,
  [OpenClaw](https://github.com/anthropics/openclaw), or your own harness

### Install

```bash
git clone https://github.com/project-you-apps/membot.git
cd membot
pip install -r requirements.txt
```

### Run

```bash
# Local (stdio) — for agents that launch Membot as a subprocess.
# Read-only by default; add --writable to enable stores and saves.
python membot_server.py

# Remote (HTTP) — for any MCP client over the network.
python membot_server.py --transport http --port 8000

# Writable, for a personal or team server.
python membot_server.py --transport http --port 8000 --writable
```

| Flag | Default | Description |
| --- | --- | --- |
| `--transport` | `stdio` | `stdio`, `http`, or `sse` |
| `--host` | `0.0.0.0` | Bind address (HTTP/SSE) |
| `--port` | `8000` | Listen port (HTTP/SSE) |
| `--writable` | off | Enables `memory_store` and `save_cartridge` |

**stdio** is JSON-RPC over stdin/stdout. **HTTP** serves Streamable HTTP at
`http://{host}:{port}/mcp`.

> ⚠ **Read [Security posture](#security-posture) before exposing an HTTP server to anything but
> loopback.** The API key and rate limit do not cover every route.

### If HTTPS fails but your browser works

A first search, build, sign-in or mount failing with `CERTIFICATE_VERIFY_FAILED` — while a
browser on the same machine reaches the same address fine — means the machine is **intercepting
TLS**. Consumer antivirus and corporate proxies do this by presenting their own certificate.
Browsers accept it because they verify against the OS trust store; Python verifies against a
bundled CA set that has never heard of it.

`requirements.txt` includes [`truststore`](https://pypi.org/project/truststore/), which routes
verification through the OS store. **Verification stays fully on** — this changes where trust
comes from, not whether it is checked.

⚠ Watch for it when a mount returns **503**: a TLS failure inside an access check is
indistinguishable from a permissions outage, and the error will not mention certificates.
Full diagnosis: [docs/TROUBLESHOOTING-TLS-INTERCEPTION.md](../docs/TROUBLESHOOTING-TLS-INTERCEPTION.md)

---

## What you can do with it

Membot exposes about fifty MCP tools. Grouped by the job rather than alphabetically:

| Job | Tools |
| --- | --- |
| **Find things** | `memory_search` · `multi_search` · `walk_associate` · `get_passage` · `passage_links` |
| **Mount and manage carts** | `mount_cartridge` · `unmount` · `list_cartridges` · `multi_mount` · `multi_mount_directory` · `multi_list` · `multi_unmount` · `get_status` |
| **Annotate** | `note_add` · `notes` · `note_share` · `propose_correction` · `proposals` |
| **Write and persist** | `memory_store` · `save_cartridge` · `get_pattern_meta` |
| **Per-seat attention** | `seat_hot_stack` · `firm_hot_stack` · `seat_hide` · `seat_hidden` |
| **Scoped subsets** | `subcart_create` · `subcart_list` · `subcart_export` · `subcart_delete` |
| **Agent-owned memory (Mempack)** | `mempack_read_pattern_i` · `mempack_update_pattern_i` |
| **Shared writable carts (Membox)** | `membox_mount` · `membox_search` · `membox_imprint` · `membox_acquire_lock` · `membox_release_lock` · `membox_lock_holder` · `membox_list` · `membox_status` · `membox_unmount` |
| **Watch for change** | `watch` · `unwatch` · `what_changed` |
| **Federate across machines** | `federate_publish` · `federate_consolidate` · `federate_load` · `federate_migrate_jsonl` |

⭐ **For agents, there is a companion document:** [AGENTS.md](AGENTS.md) that describes which tool for which
job, what the returns mean, and the failure modes worth handling.

---

## How search works

Three signals, combined:

- **Cosine** over sentence-transformer embeddings (Nomic) — semantic similarity.
- **Hamming** over a 768-bit binary signature per passage (96 bytes) — fast, cheap prefiltering
  and a second opinion on ranking.
- **Keyword reranking** — surface matches that embedding geometry misses, like identifiers and
  proper nouns.

ⓘ **An honest note on "associative" search.** Membot can return neighbours a literal query would
not have found. Having tested it, we attribute that to the **embedding model's** representation
quality rather than to anything downstream of it. We no longer claim an architectural
differentiator in retrieval, and we would rather say so than have you discover it in a bake-off.

The repository includes an optional GPU engine under a separate licence. It is **not required**
and is not part of the retrieval path described above.

---

## Notes, attribution and history

The features customers actually asked for, in order of how often they ask:

**Notes on passages.** A note carries a `kind` — why, context, caution, decision, prediction,
recommendation — plus its author, surface and timestamp. Notes live beside the cart in a sidecar
file, not inside it, so annotating does not rewrite the data.

**Private by default, shared on purpose.** `note_add` creates a private note; `note_share` is a
separate, deliberate act. ⛔ There is no setting that makes new notes public. ⚠ And a correction
**cannot widen who can see the original** — correcting something must never leak it.

**Correction without erasure.** A correcting note supersedes its predecessor. The superseded note
renders **struck through**, still attributed. ⛔ Nothing in the normal path deletes a note.

**A hash-chained journal.** Every note event appends a line carrying the previous line's hash, so
an edited, reordered or interior-deleted entry is detectable. A **high-water mark** in a second
file closes trailing deletion, which the chain alone cannot catch.

⭐ **Keep the journal in a repo your reviewer also clones** and the last gap closes too: every
clone is an independent witness, so a rewrite is detectable by anyone holding one.

---

## The Hot Stack

A shared cart does not have *a* hot stack. Two people searching the same store build different
hierarchies over it, and both are correct. So Membot keeps attention per seat, and rolls it up
separately.

**`seat_hot_stack`** returns what **you** have been working on in the mounted cart, ranked by
your own attention. It scores only passages you have actually touched, so a passage another seat
has opened forty times is cold for you if you have never opened it. ⇒ Untouched material is
cold, not absent: a thing you could read but never have simply does not rise.

**`firm_hot_stack`** returns what the **organization** is warm on, which is a different question
and not the sum of the first one. ⭐ The signal that exists only here is **how many distinct
people touched a passage**. One person reading something forty times and forty people reading it
once produce the same access count and mean completely different things about a company.
⇒ Ranked by distinct seats, that difference becomes visible.

**`seat_hide` / `seat_hidden`** let a seat drop something out of its own view without removing it
for anyone else — the free rung of the delete ladder.

ⓘ **What this is and is not.** It is attention over a mounted cart, maintained per seat, and it
works today. ⛔ It is *not* yet a personal memory that follows you across carts: **promotion** —
moving what you worked on in a session into a durable personal or organizational store — is
designed and not built, and a personal cart is not auto-mounted when you sign in. ⇒ Both are on
the roadmap below.

---

## Security posture

Stated plainly, because the previous version of this file was wrong about it.

**What the API key covers.** `MEMBOT_API_KEY` and the 60 req/min rate limit are enforced in the
MCP protocol hook, which means they apply to **`/mcp` tool calls only**. ⛔ The server also serves
REST routes under `/api/*`, `/depot/*` and `/app` as ordinary HTTP endpoints — **those get
neither the key check nor the rate limit.** ⚠ The middleware reads `Authorization: Bearer <key>`
only; `X-API-Key` is ignored.

⇒ **Do not expose an HTTP Membot to an untrusted network.** If you need remote access, proxy
`/mcp` and block the rest.

**How identity works.** The seat is resolved rather than accepted:

- **stdio** — the claim is trusted. A local pipe *is* the user.
- **a verified token present** — the seat is derived from it, and a disagreeing claim is recorded
  as overridden.
- **no usable identity** — the claim is honored and recorded as `claimed`, except in namespaces
  the server reserves for itself.

⇒ Journal entries carry `identity_source: verified | claimed | default`. ⭐ **A reader should
treat `claimed` as present but unverified, never as equivalent to `verified`.** Binding a write
to an author cryptographically is designed and not built.

**Cartridges are data, and should be treated as data you trust.** Loading a cart from an
untrusted source is not currently a safe operation. ⚠ Hardening the loader so that admissibility
is checked before anything is materialized is in progress; until it lands, **only open carts you
or your organization produced.**

Further hardening is underway; this section will say more as it lands.

---

## Roadmap, stated as roadmap

| | state |
| --- | --- |
| Notes, correction-by-supersession, hash-chained journal | ✅ shipped |
| Per-seat hot stack and hidden set | ✅ shipped |
| Mempack — agent-owned writable carts | ✅ shipped |
| Membox — shared writable carts with a lock | ✅ phase 1 |
| Federation across machines | ✅ shipped |
| Hot Stack — per seat and firm-wide | ✅ shipped |
| **Promotion — session work into a durable personal store** | ⬜ designed |
| **Auto-mounted personal cart on sign-in** | ⬜ designed |
| **Verified identity binding (signatures)** | ⬜ designed |
| **Sealed carts — a one-way, append-only posture** | ⬜ designed |
| **Loader hardening** | ⬜ in progress |
| **Change notification to subscribers** | ⬜ designed |

---

## Related

- **[Vector+ Studio](https://github.com/project-you-apps/vector-plus-studio)** — the browser
  front end. Drag and drop to build a cart, search it, read and write the same notes.
- **[AGENTS.md](AGENTS.md)** — the agent-facing guide to these tools.

## Licence

| Component | Licence |
| --- | --- |
| Server, tools and utilities | MIT |
| Optional GPU engine (`bin/*`) | Proprietary — see [bin/LICENSE](bin/LICENSE) |

The GPU engine is free for personal, educational and non-commercial use. Commercial use requires
a separate licence — [contact us](mailto:andy@project-you.app).
