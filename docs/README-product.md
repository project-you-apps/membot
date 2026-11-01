# Membot

Membot is a memory server for AI agents and the people who work alongside them. It
stores text, embeds it so it can be found by meaning rather than by keyword, and keeps a
record of who stored what and why. It runs on your own hardware, stores its data in plain
files in a directory you choose, and speaks the Model Context Protocol (MCP), so any
MCP-capable agent host can use it as a memory tool.

The problem it solves is institutional forgetting. Documents survive staff turnover;
the reasoning behind them does not. Most record systems capture who touched what and
when. Membot additionally captures *why*, as a first-class annotation attached to the
record, and keeps an append-only provenance journal of the writes that pass through its
memory path, so a reviewer can reconstruct how a body of knowledge came to look the way
it does.

Membot is the server. Vector Plus Studio is a separate web front end that talks to the
same data. This document covers the server.

> **On claims in this document.** Everything here was checked against the source in this
> repository. Where a capability is partial, the limits are stated rather than omitted.
> Two documents you should read before relying on Membot for a regulated workload are
> [`COMPLIANCE-posture.md`](COMPLIANCE-posture.md) (what is and is not recorded) and
> [`GUIDE-administration.md`](GUIDE-administration.md) (configuration, backup, deletion).

---

## Core concepts

These are introduced in the order you meet them.

**Cartridge (or "cart").** A single file holding a body of text plus the vectors that
make it searchable. A cart is the unit you load, back up, copy and hand to someone else.
The current format is a NumPy `.npz` archive — no pickled Python, so loading a cart
cannot execute code. A cart is typically named `something.cart.npz`.

**Passage.** One stored item inside a cart: a memory, a paragraph, a chunk of a document.
Passages are addressed by a zero-based integer index, `idx`, which is what search results
return and what the read tools take.

**Mount.** Loading a cart into the server so it can be searched. A mount is per-session
and lives in memory; it does not modify the cart on disk. The same cart mounted by
several callers is loaded once into a shared pool.

**Seat.** One person or agent, as far as the server is concerned. Seats exist so that
several people can share one cart without sharing each other's view of it: each seat has
its own attention history, its own hidden passages, its own private notes and its own
named selections. Seats are not an authentication system — see
[Identity](#identity-in-one-paragraph) below and the compliance document for exactly what
a seat does and does not prove.

**Sidecar.** A small file next to the cart holding something about the cart rather than
the cart's content: the integrity manifest, the provenance journal, the notes, the
per-seat attention overlays. Sidecars are deliberately separate files so they can be
backed up, shipped or permissioned independently of the cart. The full list is in the
administration guide under Backup.

**Mempack.** A per-user writable cart that lives in a Supabase project (Postgres plus
object storage) instead of, or in addition to, the local filesystem. Mempacks are the
multi-user hosted path: a signed-in user owns their Mempack, ownership is checked against
a verified token, and the server keeps a server-side activity log for them. A purely
local Membot install does not use Mempacks at all and needs no Supabase credentials.
**If your data must not leave your own infrastructure, do not enable Mempacks** — see
[Where data lives](#where-data-lives-on-disk).

Two more concepts appear in the tool list and are worth a line each:

- **Sub-cartridge.** A named selection over a cart, stored as a list of content keys
  rather than a copy of the passages. Cheap to create, and `subcart_export` turns one
  into a standalone cart you can hand to someone else.
- **Note.** A sticky note attached to a passage, carrying the reasoning behind it. Notes
  are born private to the seat that wrote them and become visible to the cart only when
  the author explicitly shares them.

---

## Installing

Requirements:

- Python 3.10 or later.
- The packages in `requirements.txt`. The embedding model (`nomic-embed-text-v1.5`) is
  downloaded on first use unless you point the server at a local Ollama instance.
- No GPU is required. A CUDA GPU enables an additional associative-recall path; search
  works without one.

```bash
git clone <your-copy-of-the-membot-repository>
cd membot
pip install -r requirements.txt
```

Two notes from `requirements.txt` that matter on real machines:

- `sentence-transformers` pulls in PyTorch, and by default the CUDA build, which is
  several gigabytes. On a machine with no NVIDIA GPU, install the CPU build first:
  `pip install --index-url https://download.pytorch.org/whl/cpu torch`.
- On machines whose security software or corporate proxy intercepts TLS, the `truststore`
  package (declared in `requirements.txt`) routes certificate verification through the
  operating system trust store. Verification stays on; only the source of trust changes.
  Without it, HTTPS calls fail with `CERTIFICATE_VERIFY_FAILED` while a browser on the
  same machine works fine.

---

## Starting the server

Membot has two transports. Both expose the same tools.

### stdio — for a local agent host

The agent host launches Membot as a child process and talks JSON-RPC over its standard
input and output. This is the mode to use for a single user on their own machine: there is
no network listener, and the operating system's own process boundary is the access
boundary.

```bash
# Read-only (the default)
python membot_server.py

# Writable
python membot_server.py --writable
```

You configure this in your agent host's MCP settings, not on the command line. A minimal
entry looks like this:

```json
{
  "mcpServers": {
    "membot": {
      "command": "python",
      "args": ["/absolute/path/to/membot/membot_server.py", "--writable"]
    }
  }
}
```

### HTTP — for remote or multi-client use

```bash
# Loopback only (the default bind address), read-only
python membot_server.py --transport http --port 8000

# Loopback, writable — for a local sidecar serving several clients on this machine
python membot_server.py --transport http --port 8000 --writable

# Reachable from the network, writable — requires a key, or the server refuses to start
MEMBOT_API_KEY=replace-with-your-own-secret \
  python membot_server.py --transport http --host 0.0.0.0 --port 8000 --writable
```

The MCP endpoint is `http://<host>:<port>/mcp`. Clients authenticate, when a key is set,
with an `Authorization: Bearer <key>` header. There is also a small REST surface under
`/api/` used by browser clients and by the walkthrough below.

Two defaults are worth stating plainly because they are load-bearing:

- **Read-only is the default.** `memory_store`, `save_cartridge` and `subcart_export` are
  refused unless you pass `--writable`. Note that read-only does *not* gate every write;
  the exact list is in [`REFERENCE-tools.md`](REFERENCE-tools.md) and the gaps are
  catalogued in [`COMPLIANCE-posture.md`](COMPLIANCE-posture.md).
- **The bind address defaults to `127.0.0.1`.** Reaching the dangerous configuration —
  writable, reachable from the network, no authentication — takes two deliberate flags,
  and the server refuses to start in that state and exits with status 5. (The current
  `README.md` in the repository root still documents the old `0.0.0.0` default. The
  argument parser is correct; the README is stale.)

---

## Five-minute walkthrough: store something, search it, read it back

This uses the HTTP transport and the REST surface, so every step is a single `curl` you
can paste. Run the server in one terminal:

```bash
python membot_server.py --transport http --port 8000 --writable
```

**1. Confirm it is up and see what carts it can find.**

```bash
curl http://127.0.0.1:8000/api/status
curl http://127.0.0.1:8000/api/cartridges
```

**2. Create and mount a cart.** `auto_create` makes an empty cart if the name is new.
Pass a session header so this walkthrough gets its own seat rather than the shared default
one.

```bash
curl -X POST http://127.0.0.1:8000/api/mount \
  -H "Content-Type: application/json" \
  -H "X-Membot-Session: walkthrough" \
  -d '{"name": "scratch", "auto_create": true}'
```

**3. Store two passages.**

```bash
curl -X POST http://127.0.0.1:8000/api/store \
  -H "Content-Type: application/json" \
  -H "X-Membot-Session: walkthrough" \
  -d '{"content": "We moved the quarterly close to the fifth business day because the bank feed does not settle until the fourth.", "tags": "close,policy"}'

curl -X POST http://127.0.0.1:8000/api/store \
  -H "Content-Type: application/json" \
  -H "X-Membot-Session: walkthrough" \
  -d '{"content": "Expense reports over 5,000 need a second approver. This was a response to the 2025 audit finding, not a general policy change.", "tags": "expenses,policy"}'
```

Each response contains the stored index, for example `Stored memory #1 (idx:0)`. Note the
two numbers: `#1` is a count and `idx:0` is the address. **Use the `idx:` value**; it is
what every read tool takes.

**4. Search by meaning.** The query does not share words with either passage.

```bash
curl -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -H "X-Membot-Session: walkthrough" \
  -d '{"query": "why do we wait before closing the books", "top_k": 5}'
```

Search returns ranked previews — roughly 550 characters each — with an `index` for every
hit. Previews are truncated on purpose, to keep an agent's context small.

**5. Read the full text back.**

```bash
curl "http://127.0.0.1:8000/api/passage?idx=0" \
  -H "X-Membot-Session: walkthrough"
```

**6. Persist the cart to disk.** Until you do this, the two passages exist only in the
server's memory and are lost on restart.

```bash
curl -X POST http://127.0.0.1:8000/api/save \
  -H "Content-Type: application/json" \
  -H "X-Membot-Session: walkthrough" \
  -d '{}'
```

**7. Check the provenance journal.** The store in step 3 appended a record to the cart's
journal sidecar.

```bash
python journal.py tail cartridges/scratch.cart.npz.journal.jsonl 10
python journal.py verify cartridges/scratch.cart.npz.journal.jsonl
```

`verify` prints a one-line verdict first and exits non-zero if the chain is broken.

The same seven steps through an MCP agent host are the tools `get_status`,
`mount_cartridge`, `memory_store`, `memory_search`, `get_passage` and `save_cartridge`.
Every tool is listed in [`REFERENCE-tools.md`](REFERENCE-tools.md).

---

## Where data lives on disk

Everything is a file in a directory you control. There is no hidden database.

Relative to the `membot/` directory:

| Path | What it is |
| --- | --- |
| `cartridges/` | Carts. The first entry in the search path and where new carts are written. |
| `cartridges/<name>.cart.npz` | A cart: embeddings plus passage text, compressed, no pickle. |
| `cartridges/<name>.cart_manifest.json` | Integrity manifest: SHA-256 fingerprint of the embeddings plus a passage count. |
| `cartridges/<name>.cart.npz.journal.jsonl` | The provenance journal for that cart, one JSON object per line. |
| `cartridges/<name>.cart.npz.journal.jsonl.hwm` | High-water mark: the sequence number and hash of the newest journal entry. |
| `cartridges/<name>.cart.npz.notes.json` | Sticky notes. |
| `cartridges/<name>.cart.npz.proposals.json` | Filed correction proposals. |
| `cartridges/<name>.cart.npz.generation.json` | A write counter used to detect a stale view. |
| `cartridges/overlays/<seat>.overlay.json` | One file per seat: that seat's attention history and hidden passages. |
| `cartridges/subcarts/*.subcart.json` | Sub-cartridge definitions, plus a `trash/` directory for deleted ones. |
| `data/` | A second cart search directory, scanned read-only alongside `cartridges/`. |
| `membot.log` | The server log. Truncated on every start. |
| `redaction-review.jsonl` | One line per credential redacted at ingest: kind, fingerprint, offset. Never the value. |

Two things live outside that tree:

- **The embedding model cache**, in the usual Hugging Face or Ollama location for the
  account running the server.
- **Mempacks**, if you enable them. These live in a Supabase project: the `mempacks`,
  `mempack_patterns`, `mempack_activity` and `mempack_provisions_log` tables, and a
  `mempacks` storage bucket. The server also keeps a local cache copy of downloaded
  Mempack blobs. **Mempacks mean your data is in a third-party cloud account**; if that
  is unacceptable, leave `SUPABASE_URL` unset and every Mempack feature switches itself
  off.

---

## How to get your data out

Three ways, in increasing order of independence from Membot.

**1. Read the cart directly with NumPy.** A cart is a plain `.npz`. No Membot code is
needed and nothing in it executes:

```python
import numpy as np

z = np.load("cartridges/scratch.cart.npz", allow_pickle=True)
texts = list(z["passages"])          # the passage text, in index order
vecs = z["embeddings"]               # one row per passage
print(len(texts), vecs.shape)
```

The sidecars are JSON and JSONL and need no tooling at all.

**2. Export a slice as a standalone cart.** `subcart_create` names a selection,
`subcart_export` materializes it — copying the passages and embeddings in, remapping
internal links, and refusing to write if the result fails verification. The output is an
ordinary cart file the recipient can open without access to the original.

**3. Export Mempack contents as a document.** For a Mempack, the owner can download the
filtered pattern list as Markdown or plain text from
`GET /api/mempack/<id>/patterns/export?format=md`. The same endpoint offers `docx` and
`pdf`, which require Pandoc on the server host; whether your host has it is a deployment
question, so treat Markdown and text as the formats you can rely on.

There is deliberately no bulk delete and no destructive edit on the agent-facing surface.
Corrections are filed as proposals and acted on by a human. What that means for a
retention policy is set out in the administration guide; read it before you promise a
regulator a deletion window.

---

## Identity in one paragraph

How a caller's identity is established depends entirely on the transport, and the
difference matters more than anything else in this document.

On **stdio**, the local pipe is the user: there is nothing to verify and no one else's
seat to reach, so the claimed seat is accepted. On **HTTP with a verified Supabase
token**, the seat is derived from the token's subject claim and a conflicting claim is
overridden and recorded. On **HTTP with only the shared API key**, the seat is whatever
the caller says it is, except that the reserved `user_` and `anon_` namespaces are
refused — so a key holder cannot impersonate a signed-in user, but all key holders are
otherwise indistinguishable from one another. A shared key is a door key, not a name
badge. If you need per-person accountability on the HTTP transport, you need the
token-authenticated path.

---

## Where to go next

- [`REFERENCE-tools.md`](REFERENCE-tools.md) — all 44 tools, their parameters and
  defaults, and whether each reads or writes.
- [`GUIDE-administration.md`](GUIDE-administration.md) — configuration, deployment,
  per-seat isolation, backup, retention, operational checks.
- [`COMPLIANCE-posture.md`](COMPLIANCE-posture.md) — what is recorded, what the
  provenance journal does and does not guarantee, what credential redaction catches, and
  what is not covered.
