# AGENTS.md — using Membot as an agent

> For an agent **consuming** Membot through MCP. If you are working **on** this codebase, read
> [README.md](README.md) first.

Membot gives you a searchable store you can read, annotate and extend. **Retrieval calls no
model**, so searching is cheap enough to do liberally — the cost is latency, not tokens.

---

## 1 · The shape of a session

```
get_status            → is anything mounted? what?
list_cartridges       → what is available
mount_cartridge(name) → mount it
memory_search(query)  → search the mounted cart
```

⛔ **Search before mounting returns nothing useful.** If a search comes back empty, call
`get_status` before concluding the store is empty — an unmounted cart and an empty cart look
identical from the outside.

⚠ **The server is read-only unless it was started with `--writable`.** `memory_store` and
`save_cartridge` will refuse. Notes are a separate path and are usually available.

---

## 2 · Which tool for which job

| You want to… | Use | Not |
| --- | --- | --- |
| Find passages matching an idea | `memory_search` | — |
| Search several mounted carts at once | `multi_search` | calling `memory_search` per cart |
| Find what is *adjacent* to a topic | `walk_associate` | `memory_search` with a vaguer query |
| Read a result in full | `get_passage` | assuming the snippet is the whole passage |
| Record **why** something is the way it is | `note_add(kind="why")` | `memory_store` |
| Say a note is wrong | `propose_correction` | editing or re-storing |
| Add a new fact to the store | `memory_store` | `note_add` |
| See what changed since last time | `what_changed` | re-searching and diffing yourself |
| Keep your own working set | `seat_hot_stack` | a local scratch file |

⭐ **`note_add` versus `memory_store` is the distinction worth getting right.** A **note** is
commentary *about* a passage — reasoning, a caution, a decision. A **store** adds a new passage
to the corpus. ⇒ If what you are writing would be meaningless without the thing it points at,
it is a note.

### `walk_associate`, specifically

It re-queries from each primary result's own embedding and returns what surfaced repeatedly.
⇒ Use it for *"what else bears on this?"*, not for *"find X."* ⚠ It costs more calls and returns
looser matches by design. ⛔ Do not use it as a fallback when `memory_search` returns nothing —
that is usually a mount problem, not a recall problem.

---

## 3 · Reading the returns

**Snippets are truncated.** Search results carry a snippet, not the passage. Call `get_passage`
before quoting, summarizing or making a judgment that depends on wording.

**Notes carry provenance, and you should surface it.** A note has an author, a surface, a
timestamp, a `kind`, and may be marked **superseded**.

⇒ ⛔ **A superseded note has not been retracted — it has been corrected.** It stays in the record
on purpose. When you present notes to a human, show superseded ones as superseded rather than
dropping them. Dropping them silently destroys the thing the format exists to preserve.

**`identity_source` tells you how much the attribution is worth:**

| value | means |
| --- | --- |
| `verified` | derived from a verified token |
| `claimed` | the caller said so and nothing checked it |
| `default` | no identity was available |

⭐ **Treat `claimed` as present but unverified — never as absent, and never as equivalent to
`verified`.** If you are producing anything a reviewer will rely on, say which it was.

---

## 4 · Writing well

**Always give a note a `kind`.** `why` · `context` · `caution` · `decision` · `prediction` ·
`recommendation`. The kind is what makes a note findable later by someone who does not know it
exists.

⭐ **Write the reasoning, not the conclusion.** The conclusion is usually recoverable from the
data; the reasoning is not. *"Chose the 2024 schedule because the 2023 one predates the
restatement"* is worth storing. *"Used the 2024 schedule"* is not.

**Correcting:** `propose_correction` rather than writing a second contradictory note. The link
between them is what lets a reader see that the question was settled rather than argued twice.

⚠ **You are accountable to a principal, and the record says so.** A write from an agent is
attributed to the agent as **writer** — ownership resolves to the accountable human or
organization behind it. ⇒ Write as though the attribution will be read by someone deciding
whether to trust the record. It will be.

---

## 4b · The Hot Stack — why you would use it

**What it is.** A working set you never maintain. Every passage you open or search into is
recorded against *your* seat, and `seat_hot_stack` returns the top of that by your own
attention. ⇒ You did not build the list; your work built it.

**Why it is worth a call.** You lose your context. That is the normal case, not the failure
case. When it happens, the choice is to re-derive what you were doing by searching and guessing,
or to ask. ⭐ **One `seat_hot_stack` call returns what you were actually working on** — cheaper
than three speculative searches and more accurate, because it reflects what you touched rather
than what you think you touched.

⇒ Good opening move for a resumed session:

```
get_status            → what is mounted
seat_hot_stack(10)    → what I was working on here
```

**Does it change over time?** It **warms with use**: a passage you return to rises. ⚠ In the
current tool it is ranked by **access count, then recency** — ⛔ **it does not cool.** Over a
long-lived seat it therefore reflects *cumulative* attention rather than *current* focus. If you
need "lately", read the timestamps rather than trusting the order.

**`firm_hot_stack` answers a different question** — what the **organization** is warm on, ranked
by **how many distinct people** touched a passage. ⇒ Use it when deciding what to raise with a
human, or what matters beyond your own task. ⭐ One person reading something forty times and
forty people reading it once produce the same access count and mean completely different things.

**`seat_hide`** removes something from *your* view without touching anyone else's. ⇒ Use it for
genuine noise. It is reversible and it is not a delete.

### ⛔ Two cautions that matter more than the feature

**Hot means looked-at. It does not mean true, current or important.** ⇒ **Access-recency is not
truth-recency.** A superseded policy that everyone keeps opening *because* it is wrong will sit
at the top of a hot stack. ⛔ Never present hot-stack order to a human as a ranking of
correctness or relevance — say what it is: *what has been getting attention.*

**It is attention, not a record.** ⇒ Do not store anything in it, do not treat its contents as
durable, and do not reconstruct state from it that you should have written down as a note. If it
matters tomorrow, `note_add` it.

⚠ **Per-seat separation is a convention, not yet a boundary.** Your hot stack is distinct from
another agent's only insofar as your `session_id` is distinct — and on HTTP that identity is
self-asserted (§3). ⇒ Fine for keeping two cooperating agents from muddling each other's
attention. ⛔ Not a security property.

---

## 5 · Failure modes worth handling

| Symptom | Likely cause | Do this |
| --- | --- | --- |
| Search returns nothing | nothing mounted | `get_status`, then mount |
| `memory_store` refused | server started read-only | do not retry; report it |
| **Mount returns 503** | ⚠ **may be TLS interception, not permissions** | see the TLS note in [README.md](README.md) — the error will not mention certificates |
| Note write rejected | the cart is not registered, or the seat lacks `annotate` | report which; do not downgrade to a `memory_store` |
| Results look stale | a different cart is mounted than you think | `get_status` |
| A tool is missing entirely | the server is older than this document | `get_status` reports the version |

⛔ **Do not paper over a refusal by finding another route to the same write.** A refused write is
a decision. Routing around it produces a record that says something different from what the
system intended, and that is worse than not writing at all.

---

## 6 · Things not to do

- ⛔ **Do not mount a cart from an untrusted source.** Loading is not yet a safe operation on
  arbitrary input. Only open carts your organization produced.
- ⛔ **Do not assume `claimed` identity is proof of authorship.** See §3.
- ⛔ **Do not hide superseded notes** when presenting to a human. See §3.
- ⚠ **Do not call `walk_associate` in a loop.** It fans out; a loop fans out exponentially.
- ⚠ **Do not treat the hot stack as durable storage.** It is attention, not a record — it moves.
- ⚠ **Do not re-search to detect change.** `watch` and `what_changed` exist and are cheaper.

---

## 7 · If you are deciding whether to use Membot at all

**Good fit:** you need the same records readable by agents and people · attribution and
reasoning must survive the session · retrieval volume is high and token budget matters · the
data cannot leave a machine or a building.

**Poor fit:** you need transactional writes with rollback · you need sub-millisecond lookups at
scale · you want a managed service with an SLA · you need the store to be authoritative for
something safety-critical today. ⇒ Verified identity binding and sealed carts are designed, not
built, and we would rather you knew that now. They are upcoming features but not yet present.
