# Draft — Mempack README additions (2026-05-18 / 19 ship)

> Proposed insertions for [`membot/README.md`](../README.md) covering what shipped on the
> 2026-05-18 evening + 2026-05-19 day push. Drop into the existing
> **"What's New (May 2026) → Mempack — Per-Agent Writable Brain Cartridges"** section
> after `#### Path C Lazy Auto-Provision` and before
> `#### Canonical 12-Field H-Block Format`.
>
> Voice is intentionally matched to the existing README style: concrete, table-heavy,
> code-block-bias, no marketing language.

---

#### Dashboard at `/membot/app`

A web surface for managing your Mempack — no terminal, no MCP host, no Python required.
Sign in once at `project-you.app` (Supabase OAuth via Google / GitHub / email-magic-link),
land on the dashboard, your Mempacks auto-list. The dashboard is host-agnostic about
the agent runtime: it's just the **operator's** view of the cart.

Surfaces:

| Section | What it does |
|---|---|
| **Auth chip** (header, both tabs) | Avatar + dropdown with email + sign-out; "Sign in" button when signed out. Pop-up to `/vps/app/` for the actual sign-in form (shared `.project-you.app` cookie). |
| **Owner row** | "Signed in as `<email>`" prominent, full UUID kept available but de-emphasized as a small mono input below. Override button for admin/debug switching. |
| **Connect an agent** (collapsible) | Two peer paths: (1) `mcp.json` snippet for Claude Desktop / Cursor / Claude Code / Windsurf with `owner_id` baked in + starter prompt + common config paths, (2) Download button for `mempack_local_agent.py` + Ollama install + pre-filled run command. |
| **Dispatch box** | Textarea + Send button. Auto-prefills with the most recent `DISPATCH`-tagged pattern in the Mempack. Text persists after Send so a repeat dispatch is one click, not a re-type. |
| **Pattern I editor** | Textarea + Save + "Load template…" dropdown (Default / Researcher / "More coming soon…"). Re-applying a template re-substitutes `{owner_id_short}` / `{owner_id}` / `{name}` / `{created_at}` with current account values. |
| **Patterns Browser** | Tag filter, debounced text search, scrollable list of every pattern in the cart (newest-first by idx), expandable rows with full body + Copy + Download (.md / .txt / .docx / .pdf). Per-pattern download or filtered-set bulk download. Pattern 0 and Pattern I hidden by default (they have dedicated surfaces above). |
| **Activity feed** | Polls every 5s. IMPRINT (accent) / MOUNT (green) / PATTERN_I_UPDATE (purple, with `▸ View previous version` expander) / SETTINGS_UPDATE / COPY_IN / DISPATCH (blue) chips. Real-time as the agent works. |
| **Settings** | Activity-logging on/off toggle (per-Mempack opt-out). |

Hard-refresh after deploys to bust the cached HTML/JS.

#### Passive DISPATCH — the Mempack as a job queue

A Mempack isn't just memory; it's the **asynchronous handoff layer** between user and agent.
Most agent platforms treat tasks as active dispatch — the agent's runtime fires immediately
or fails. The Mempack model is **passive dispatch + active dispatch**:

| Active-dispatch platforms | Mempack passive-DISPATCH |
|---|---|
| Agent must be running when task arrives | Task waits in the queue for any agent |
| One agent, one session, one task | Multi-agent, multi-session, queue-of-tasks |
| User must be available to fire the dispatch | Fire-and-forget; tomorrow's agent picks up |
| State = chat context | State = Mempack (durable, portable, queryable) |

The mechanics:

1. User authors a task in the dashboard's Dispatch box → server-side, that body is stored
   as a `[DISPATCH] <user's instruction>` pattern. The server appends an
   `[dispatched: <ISO 8601 UTC timestamp>]` line at imprint time so agents never have to
   guess what "today" means (local models in particular are trained to hedge on temporal
   grounding).
2. Some agent — now or later, same or different host — mounts the Mempack via
   `mount_cartridge`.
3. The agent reads Pattern I, which instructs it to `memory_search("DISPATCH", top_k=10)`
   on every session.
4. The agent acks the dispatch, works it, and stores findings back via `memory_store`
   with appropriate tags (FINDING / SUMMARY / EVIDENCE / METHOD / etc.).
5. The user sees IMPRINT chips populate the activity feed in real time, and the new
   patterns become available in the Patterns Browser within ~5 seconds of each write.

No real-time push channel; the Mempack **is the bus**, agents are the workers, dispatch
survives session boundaries. Multi-agent runs leave audit trails via the per-pattern
signature line agents append at memory_store time:

```
[FINDING] Accelerator #4: Oregon AI Accelerator
Name: Oregon AI Accelerator
Type: Accelerator (hybrid cohort)
Location: Portland, OR
...
[signed: claude-sonnet-4-6@claude-desktop, 2026-05-18T21:52:40Z]
```

#### Two agent paths

Mempack is host- AND runtime-agnostic. Two paths today, both surfaced from the
**Connect an agent** panel on the dashboard:

**1. MCP host (frontier model)**

Drop the auto-generated snippet into your client's `mcp.json` / Connectors UI:

```json
{
  "mcpServers": {
    "membot": {
      "type": "streamableHttp",
      "url": "https://project-you.app/membot/mcp"
    }
  }
}
```

Hosts confirmed working today: **Claude Desktop** (via in-app Connectors), **Claude Code**,
**Cursor**, **Windsurf**. The dashboard generates a starter prompt with your `owner_id`
baked in:

```text
Mount the Mempack named "primary" with owner_id 3579e6ee-...
Then read Pattern I (mempack_read_pattern_i) and search for tag DISPATCH
(memory_search query="DISPATCH" top_k=10). Acknowledge any dispatches you find
back to me with a one-liner before starting work.

If there are no DISPATCH patterns, tell me in chat — do not invent work.

When you complete the dispatch, store every finding via memory_store with
an appropriate tag (FINDING / SUMMARY / METHOD / etc.). Sign each stored
pattern by appending: [signed: <model>@<host>, <timestamp>]

Do NOT report completion until memory_store has actually been called and
returned a "Stored as passage #N" confirmation.
```

**2. Mempack Local Agent (Python + Ollama, no API key)**

Download [`tools/mempack_local_agent.py`](../tools/mempack_local_agent.py)
from the dashboard's Connect panel (also available at
`https://project-you.app/membot/downloads/mempack_local_agent.py`). Single file,
one dep (`requests`), one optional (`truststore` for AV-intercepted TLS on Windows).

```bash
pip install requests truststore
ollama pull qwen2.5:14b   # or hermes3:8b, qwen2.5-coder:14b, llama3.1:8b

python mempack_local_agent.py \
    --owner-id <your-supabase-uuid> \
    --model qwen2.5:14b \
    --user-label "Your Name" \
    --prompt "Mount my primary Mempack and tell me what is queued."

# Or interactive REPL (no --prompt):
python mempack_local_agent.py --owner-id <uuid> --model qwen2.5:14b
```

The harness:
- Hand-rolled MCP-over-HTTP client (no `mcp` SDK dependency)
- Dynamic `tools/list` fetch — auto-syncs with whatever membot exposes
- Pipes Ollama's structured `tool_calls` straight into MCP `tools/call`
- Bounded agent loop (default 20 turns) with pretty-printed conversation + tool I/O
- Tool-capable Ollama models verified: `qwen2.5:14b`, `qwen2.5:7b`, `qwen3:8b`,
  `hermes3:8b`, `llama3.1:8b`, `qwen2.5-coder:14b`. (Gemma family models lack
  tool-call support in Ollama's default template — pick from the list above.)

**Smoke test** included at [`tools/mempack_local_agent_smoke.py`](../tools/mempack_local_agent_smoke.py):
generates a unique marker, asks the agent to store one pattern containing it, then
verifies via MCP search. Exit 0 = PASS, 1 = FAIL with model-specific diagnostics.

#### Pattern I templates

Two starter templates ship today; more coming. Pick from the dashboard's
"Load template…" dropdown above the Pattern I editor.

| Template | Use case | Distinctive features |
|---|---|---|
| **Default** | Generic personal Mempack | Tag-store-search loop. No specialization. Good baseline. |
| **Researcher** | Investigates dispatched topics, gathers evidence, synthesizes findings | Mount-time checklist (search DISPATCH, search ACTIVE, ack before work), tag vocabulary (FINDING / EVIDENCE / OPEN_QUESTION / DEAD_END / SUMMARY / ACTIVE / METHOD), OBSERVATION-vs-INFERENCE distinction, six-mode disambiguation when existing work product is present (first-time / refresh / second opinion / extend / audit / skip), "Sign your work" + "Reporting completion is forbidden until memory_store has actually been called" strict guards. |

Pattern I templates use placeholder substitution at apply-time:

| Placeholder | Substitution preference |
|---|---|
| `{owner_id_short}` | `full_name` (from OAuth metadata) → email local-part → UUID prefix |
| `{owner_id}` | `email` → `full_name` → UUID |
| `{name}` | Mempack name |
| `{created_at}` | ISO 8601 UTC at apply time |

So a Pattern I instantiated for a user with Google-OAuth `full_name = "Andy Grossberg"`
greets the agent with *"You are a research agent operating on behalf of Andy Grossberg"*,
not *"…operating on behalf of 3579e6ee"*. The UUID stays as the database key in every
API call and storage path; only the agent-facing text changes.

#### Previous-version recoverability (Fix A)

Pattern I overwrites are destructive in-place SQL `UPDATE` operations — the prior text
is gone after a save. To make this recoverable without schema work, every
`pattern_i_update` and `pattern_update` activity row now captures the **previous**
text body (capped at 8 KB) in `mempack_activity.metadata.previous_text`.

The activity feed surfaces this as an inline `▸ View previous version (N chars)`
expander on those rows. Click → scrollable `<pre>` with the prior text + Copy
button. If the previous body exceeded 8 KB, an amber notice flags the truncation.

Both dashboard saves (via `POST /api/mempack/<id>/pattern-i`) and agent-side
`mempack_update_pattern_i` MCP calls capture previous_text — so an agent that
rewrites its own Pattern I leaves a recovery trail too.

True tombstone semantics (old row marked tombstoned, new row inserted, time-travel
reads via `?include_tombstoned=true`) are the next iteration — Fix B in the roadmap.

#### Activity event types

| Event | Chip color | Captures |
|---|---|---|
| `imprint` | accent | New pattern stored; metadata has preview + length |
| `dispatch` | blue | DISPATCH-tagged pattern from the dashboard's Dispatch box |
| `mount` | green | Agent mounted the cart; metadata has session_id + pattern count |
| `pattern_i_update` | purple | Pattern I overwritten; metadata has **previous_text** (Fix A) |
| `pattern_update` | purple | Non-Pattern-I in-place pattern overwrite; same previous_text capture |
| `copy_in` | cyan | Pattern copied in from another cart with provenance footer |
| `settings_update` | dim | Per-Mempack toggle changed (e.g., activity-logging opt-out) |
| `create` | amber | Mempack provisioned |

---

(End of proposed insertions; the existing `#### Canonical 12-Field H-Block Format`
section continues unchanged below.)

---

## Optional: light edits to the existing Mempack section

Two small tweaks to consider in the existing copy:

1. The opening paragraph at L221 says "the briefing and behavior load automatically.
   No prompt-stuffing, no per-session re-briefing. The cart bootstraps the agent."
   That's still accurate, but worth adding a forward-pointer:
   *"See **Two agent paths** below for the two supported ways to actually run an
   agent against a Mempack today: MCP-host frontier models, or the Python harness
   for local Ollama models."*

2. The `#### MCP-Native Access` section at L238 currently lists `/api/mempacks` +
   `mempack_read_pattern_i` + `mount_cartridge` as the example flow. Worth noting
   that the **`/membot/app` dashboard** is the friendlier path for non-developers
   and acts as the canonical example of all surfaces working together.

(These are nits — the existing copy is fine; the additions above are the real delta.)
