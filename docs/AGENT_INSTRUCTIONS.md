# Membot — Agent Instructions

Hello, agent. You now have access to **Membot** the neuromorphic brain cartridge server granting you swappable, searchable memory without requiring an LLM call to do the search. This doc covers how to use it across three access patterns so you can pick the one that fits your host.

---

## What Membot is

A memory substrate. Mount a brain cartridge, search it with three-signal ranking (cosine + Hamming + keywords), store new memories, swap to a different cartridge. The whole pipeline runs without LLM inference — embeddings come from a sentence transformer, search is binary math, no GPU required.

**What you can do:**

- **List** available cartridges
- **Mount** one (loads it into the substrate as the active search target)
- **Search** the mounted cartridge with a natural-language query
- **Store** new memories (writable carts only; auth required for the hosted droplet)
- **Swap** cartridges as your task domain changes

**What you have access to on the hosted droplet:**

Five bundled sample cartridges are publicly accessible at `https://project-you.app/...` for read-only search:

| Cart | Patterns | Size | Topic |
| --- | --- | --- | --- |
| `gutenberg-classics` | varies | 74 MB | Classic literature |
| `gutenberg-poetry` | 60,523 | 190 MB | Poetry from Project Gutenberg |
| `neuroscience-and-AI-papers` | 6,407 | 28 MB | Neuroscience + AI literature |
| `attention-is-all-you-need` | 24 | 0.1 MB | The Transformer paper |
| `arxiv_2400k_index` | 2,400,000 | 359 MB | arXiv abstracts (Hamming-only) |

If your host has provisioned a **Mempack** for you (your personal cart for accumulating findings), Pattern 0 of that cart will tell you what it is and how to use it. Read your Mempack's Pattern I (= Pattern 1) first to remind yourself what kind of agent you are.

---

## Three ways to access Membot

Pick the one that matches your host. All three reach the same server; they differ in protocol.

### A. Direct MCP (Claude Code, Cursor, Windsurf, custom MCP clients)

Drop this into your MCP host's config (typically `.mcp.json` or `mcp.json`):

```json
{
  "mcpServers": {
    "membot": {
      "type": "sse",
      "url": "http://137.184.227.79:8000/sse"
    }
  }
}
```

(For write access, swap the URL to `http://137.184.227.79:8040/sse` and add an `X-API-Key` header — talk to your operator about getting a key.)

Once configured, tools are exposed under the `membot.*` namespace. Call them like any other MCP tool:

- `membot.list_cartridges()`
- `membot.mount_cartridge(name="neuroscience-and-AI-papers")`
- `membot.memory_search(query="hippocampal replay", top_k=5)`
- `membot.memory_store(content="...", tags="optional,tags")` *(write only)*
- `membot.get_status()`

See `mcp.json.example` in this directory for a copy-paste template.

### B. mcporter CLI (Bash-based MCP calls)

If your host can run Bash but doesn't expose MCP tools natively, use `mcporter`:

```bash
mcporter call membot.list_cartridges
mcporter call membot.mount_cartridge name="neuroscience-and-AI-papers"
mcporter call membot.memory_search query="your query" top_k=5
mcporter call membot.memory_store content="your finding"
mcporter call membot.get_status
```

This is the pattern documented in `SOUL-research-bot-merged.md`.

### C. REST API (no MCP support required)

Plain HTTP. Useful for custom agents, scripts, or hosts without MCP plumbing.

```bash
# List cartridges
curl https://project-you.app/membot/api/cartridges

# Mount a cart
curl -X POST https://project-you.app/membot/api/mount \
  -H "Content-Type: application/json" \
  -d '{"name":"neuroscience-and-AI-papers"}'

# Search the mounted cart
curl -X POST https://project-you.app/membot/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"hippocampal replay","top_k":5}'

# Store (write access required)
curl -X POST https://project-you.app/membot/api/store \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-key>" \
  -d '{"content":"your finding","tags":"optional,tags"}'

# Status
curl https://project-you.app/membot/api/status
```

---

## Workflow

Regardless of which access pattern you use, the general workflow is the same:

1. **`list_cartridges`** to see what's available
2. **`mount_cartridge`** the one most relevant to your task (only one mounted at a time per session)
3. **`memory_search`** with natural-language queries; results come back ranked with cosine + Hamming + keyword scores
4. **Fall back to web search** if the cartridge doesn't have what you need
5. **`memory_store`** new findings back to a writable cart (your Mempack, if you have one) so the next session can build on them
6. **Synthesize** your answer using the cart's results as authoritative context

---

## Your Mempack (if provisioned)

A Mempack is your personal writable cartridge. It travels with you across sessions, hosts, and (eventually) other agents you collaborate with. If your operator has provisioned one for you, it lives at a known path on the droplet and you can mount it like any other cart.

**Pattern structure:**

- **Pattern 0** — Cart identity manifest (who owns this Mempack, when it was created, permissions, briefing). Read-only metadata.
- **Pattern I** (= Pattern 1) — *Your* behavior. Personality, learned preferences, dos and don'ts, active threads. **Read this first.** Update it when your behavior meaningfully changes.
- **Pattern N+** — Accumulated material. Anything you've gathered worth keeping.

When you find something worth remembering, store it to your Mempack. When you start a new session, mount your Mempack and read Pattern I to remind yourself who you are.

---

## A few principles

- **One cart per session for single-mount workflows, multi-mount via Federate when you need more.** Each session has one "active" cart for the simple `mount_cartridge → memory_search` flow. The Federate layer (`multi_mount` / `multi_search`) lets agents hold many carts simultaneously and search across them with namespaced result attribution — useful when you want your Mempack mounted alongside a reference cart.
- **Don't trust search results blindly** — the substrate finds *related* passages, which is powerful but means lexical near-misses can rank high. Read the snippet; verify the source.
- **Hamming-only carts** (like `arxiv_2400k_index`) work great for retrieval but don't carry full embeddings, so semantic recall is slightly coarser than full-cosine carts.
- **Write sparingly.** Your Mempack is meant to be a curated personal index, not a dumping ground. Store findings that have ongoing relevance; let trivia stay in transient context.

---

## Help / troubleshooting

If `get_status` works but `memory_search` returns an empty array, you probably haven't mounted a cart yet (`get_status` will show `cartridge: null`).

If you get a 401 on `memory_store`, you don't have write access. Read access is free; write access requires an API key. Talk to your operator.

If the droplet is slow to respond, it's probably loading a large cart from disk; first search after mount can take a few seconds.

If something else is broken: [andy@project-you.app](mailto:andy@project-you.app).
