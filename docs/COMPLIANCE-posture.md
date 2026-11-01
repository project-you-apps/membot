# Compliance posture

What Membot records, what the records guarantee, and what they do not. This document is
written to be tested. Every claim below was checked against the source in this repository,
and where a control is partial the limit is stated rather than left for you to discover.

Read it alongside [`GUIDE-administration.md`](GUIDE-administration.md), which covers
backup, retention and the operator checks referenced here.

---

## 1. Summary for an evaluator

| Question | Short answer |
| --- | --- |
| Where does our data live? | Plain files in a directory on your own host. Nothing leaves it unless you enable Mempacks, which store content in a Supabase project. |
| Is there a tamper-evident record of changes? | Yes, for writes that go through the memory path: an append-only hash-chained journal per cart, plus a high-water-mark sidecar. It is tamper-**evident**, not tamper-proof. |
| Is every write recorded? | **No.** Three write paths append journal records today. Others do not. Section 4 enumerates this. |
| Can we tell which individual made a change? | Only where the caller presented a verified token. Callers holding the shared API key have no individual identity. |
| Are credentials kept out of the corpus? | Known credential formats are stripped before text is embedded, on the `memory_store` path. It is a filter for known credential shapes, not a PII or general secret detector, and it is wired into one ingest path, not all of them. |
| Can records be deleted on a retention schedule? | Not through Membot's interface. Deletion is file-level, at cart granularity, by an operator. |
| Does an outside party need to trust the operator? | Yes, today. Nothing anchors the journal's head hash outside the host, so tampering is detectable by the operator but not by someone who does not trust the operator. |

---

## 2. What is recorded about who did what and when

### The provenance journal

Each cart has a journal sidecar at `<cart>.journal.jsonl`: one JSON object per line,
append-only, greppable, no database. Every entry carries the same fields, in this order:

| Field | Meaning |
| --- | --- |
| `seq` | Monotonic, gapless counter. A gap is a detected deletion. |
| `event_ts` | When the event happened. A caller may state it for a backdated import. |
| `ingest_ts` | When Membot recorded it, sampled independently under the write lock. |
| `writer` | The seat the write was attributed to. See section 3 on what this proves. |
| `surface` | Where it came from. Closed set: `CLI`, `VSC`, `WEB`, `MCP`. |
| `op` | What happened. Closed set: `store`, `tombstone`, `corroborate`, `supersede`, `perm_change`, `flag`, `note`. |
| `target` | What it happened to, for example `{"cart": "...", "idx": 41234}`. |
| `payload` | Operation-specific detail. |
| `prev_hash` | The hash of entry `seq - 1`. The first entry chains to a fixed genesis value no SHA-256 can produce. |
| `hash` | SHA-256 over every preceding field, canonically serialized. |

Two details worth an auditor's attention. First, `event_ts` and `ingest_ts` are two
independent clock readings of two different events, never one value copied twice; the gap
between them is real information about queueing. Second, `surface` and `op` are
deliberately closed vocabularies — an unrecognized value is **refused**, not stored — so
that a question like "show me every permission change" has an answer a reviewer can trust
to be complete over the records that exist.

Writes are durable: a single line is written under an OS-level exclusive lock, flushed and
`fsync`ed. There is no rewrite-to-append, because rewriting the file to add a line is
rewriting history by definition.

### Where identity provenance is recorded

Every journal entry carries, in its payload, how the writer was established:

- `identity_source`: `verified` (derived from a verified token's subject claim),
  `claimed` (caller-supplied and accepted), or `default` (nothing presented).
- `identity_overridden`: present when a caller claimed one seat and a verified token said
  another. The request is not failed; the disagreement is recorded.
- `identity_refused`: present when a claim was dropped for naming a reserved namespace.

This is the field to read before relying on a `writer` value. A record that cannot
distinguish a verified identity from a string somebody typed is not evidence about a
person, and this journal says which it is on every entry.

### What has no delete, amend or compaction

By explicit design, `journal.py` has no `delete()`, no `amend()` and no `compact()`, and
the design document records that these should not be added. The cart is treated as a
materialized view of the journal, rebuildable by replay. History is corrected by
**appending an erratum** — an entry with `op="supersede"` pointing at the record that was
wrong — which is itself chained. There is also deliberately no repair path for a broken
chain: `verify()` names the break and a human decides, because silent repair is
indistinguishable from tampering.

### The "why" layer

Sticky notes are the second record, and the one that distinguishes Membot from an ordinary
audit log. Every note carries the write-attribution triple — who, which surface, when —
plus a `kind` (`why`, `context`, `caution`, `decision`, `prediction`) and a body. Notes
are append-only; a note is corrected by writing a new one that supersedes it, and a
correction cannot widen who could see the original. The `prediction` kind exists so an
expectation can be recorded *before* an outcome, which is what makes it falsifiable later.

Notes live in a sidecar at `<cart>.notes.json` and are content-keyed to the passage rather
than index-keyed, so a note still points at the right passage after the cart is reindexed.

Two honest limits on notes: the `surface` value is supplied by the caller and is not
verified, and the author recorded on a note is this session's seat identifier, with the
same caveats as section 3.

---

## 3. Identity: how a seat is established

Identity depends entirely on the transport, and this is the single most important thing to
understand before making an accountability claim.

| Situation | What happens |
| --- | --- |
| **stdio** (local pipe, no HTTP request) | The claimed seat is accepted. The pipe is the user; there is nothing to verify and no one else's seat to reach. `identity_source` is recorded as `claimed` or `default`. |
| **HTTP with a verified Supabase token** | The seat is **derived** from the token's `sub` claim as `user_<sub>`. A caller-supplied seat that disagrees is overridden, and the override is recorded on the journal entry. `identity_source` is `verified`. |
| **HTTP with only the shared API key** | The claimed seat is honored, **except** in the reserved `user_` and `anon_` namespaces, which are refused because the server mints those from verified facts. `identity_source` is `claimed`, `default` or refused. |

Token verification fails closed. An expired, malformed or wrongly signed token is not
treated as anonymous-but-trusted; it is a failed assertion of identity. The verifier picks
its branch from the token's declared algorithm and then verifies with that branch's own
fixed algorithm list and its own key, so a token forged by signing with a published public
key as an HMAC secret does not verify.

### What this means in plain terms

**Callers authenticating with the shared API key have no individual identity.**
`MEMBOT_API_KEY` is one string, the same for every caller. It is a door key, not a name
badge. A key holder cannot impersonate a signed-in user, because the reserved namespaces
are refused — but two key holders are indistinguishable from one another, and each can
name any unreserved seat it likes. If your control requires attributing an action to a
named person, that requires the token-authenticated path.

### Attribution across the write paths

Counting all durable write paths in the server, as of the internal mutation-path audit:

- **Six carry a verified identity.** These are the authenticated Mempack and copy
  endpoints, which check a Supabase token and then an ownership match.
- **One requires a self-declared `agent_id`** and refuses anonymous writes:
  `membox_imprint`. The value is not verified, but it must be supplied, and it is stored
  alongside the pattern it wrote.
- **The remainder derive the writer from a session identifier or a caller-supplied
  string.** For these, `writer` is a **claim, not evidence**, and must not be described as
  provenance of a person. Journalling them is still worth doing — an unforgeable record of
  an unattested claim beats no record — but the distinction is recorded in
  `identity_source` precisely so it cannot be blurred.

A hash chain proves a record was not altered. It does not prove that `"writer": "andy"`
was Andy.

---

## 4. What is NOT covered

This section represents the conclusion of `docs/AUDIT-membot-mutation-paths-2026-11-01.md`
in the project root, which enumerated every code path that changes durable state.

### Not every write produces a journal record

**The accurate claim is that writes through the memory path are recorded, not that every
write is recorded.**

The journal helper is called from three places in the server today:

| Tool | Journal operation |
| --- | --- |
| `memory_store` | `store` — with the passage index, character count, tags, and whether anything was redacted (never the redacted value) |
| `note_add` | `note` |
| `note_share` | `perm_change` |

The audit identified 26 distinct durable-mutation paths in total. The remainder — among
them `save_cartridge`, `subcart_create`, `subcart_delete`, `subcart_export`, `seat_hide`,
`propose_correction`, `mempack_update_pattern_i`, `membox_imprint`, the three
`federate_*` writers, the authenticated Mempack endpoints, and the per-seat attention
overlay written as a side effect of searching — do not append a journal record.

The reason is architectural and stated in the code itself: the design requirement is that
every mutation produces a record *because there is no other way to mutate*, which is
achieved by funnelling every write through one door. That funnel does not exist yet, so
today each journal call is a courtesy at its own call site. The audit found that 24 of the
26 paths reach disk through one of eleven shared persistence helpers, which is what makes
completing the funnel tractable; two paths bypass the helpers entirely.

Two further specifics an evaluator should know:

- **A journal append cannot fail the operation it records.** A failure is logged at ERROR
  and the write proceeds. A silent gap is how a ledger develops holes, so the log entry
  matters, but the write is not blocked.
- **A mount with no file on disk has nowhere to put a journal.** A Mempack, or a cart that
  has been created in memory and never saved, produces no journal record for its writes —
  not because the write is hidden, but because the sidecar does not exist yet.

### The other records that exist, and their limits

Several things look like audit trails and are not. Stated so they are not mistaken for
the journal:

| Record | Limit |
| --- | --- |
| Depot activity list (`/depot/status`, `/depot`) | In-memory ring buffer of 200 events, gone on restart. Every one of its call sites is on a **read** path — searches, mounts, passage fetches — so it never held mutations in the first place. Not an audit trail. |
| Membox write log (`membox_status`) | In-memory ring buffer capped at 100 per cart; unmounting a cart discards it. |
| Cart generation counter (`<cart>.generation.json`) | Stamps the latest writer and **overwrites** the previous stamp. It answers "is my view stale"; it cannot answer "who wrote what, when". No history. |
| Mempack activity log (Supabase `mempack_activity`) | Real and persistent, but most call sites are conditional on a per-Mempack `activity_logging_enabled` flag that the **Mempack owner can set to false** through `PATCH /api/mempack/{id}`. It covers Mempacks only: nothing records a mutation to a non-Mempack cart. The settings change itself is logged unconditionally, but inside an exception handler that logs a warning and lets the state change stand if the logging fails. |
| Mempack provisioning log (Supabase `mempack_provisions_log`) | Durable and unconditional. Mempack creation is the best-recorded path in the system. |
| Redaction review log (`redaction-review.jsonl`) | Durable JSONL, fsynced, one line per finding. But it has no sequence number, no hash chain and no lock, and its path is overridable by `MEMBOT_REDACTION_LOG`. It is a review trail, not a tamper-evident one. |
| `membot.log` | Opened in truncate mode on every start. Not retained history. |

### The REST surface is not behind the API key

`MEMBOT_API_KEY` and the rate limiter are enforced in middleware that hooks **MCP tool
calls**. The server's plain HTTP routes are not MCP tool calls, so they receive neither
check. Four of them change state, and `POST /api/sync` in particular performs no
authentication of its own and is not covered by the read-only gate: it appends
caller-supplied JSON lines to a file in the cart directory whose name the caller chooses,
with no size or line limit. CORS on these routes is `Access-Control-Allow-Origin: *`.

Operationally this means: do not treat the API key as a perimeter. If Membot is reachable
beyond loopback, a reverse proxy should allow only the paths you intend to serve. See
section 3 of the administration guide.

### Read-only mode is narrower than it sounds

The read-only gate has four call sites. It blocks `memory_store` (except on a Mempack),
`save_cartridge`, `subcart_export` and REST cart creation. It does **not** block
`mempack_update_pattern_i` — an in-place overwrite — nor `note_add`, `note_share`,
`seat_hide`, `subcart_create`, `subcart_delete`, `propose_correction`, `membox_imprint` or
the `federate_*` writers. A per-mount `read_only=True` flag is the stronger control: it
refuses all writes to that mount for the life of the mount, regardless of server flags,
with no way to unlock short of remounting.

### No retention-driven deletion

There is no delete verb anywhere on the agent-facing surface, by design: an agent's corpus
can be manipulated into instructing the agent to delete things, so the capability is not
reachable. Corrections are filed as proposals and resolved by a human. Deletion is
file-level, at cart granularity, by an operator with filesystem access. A policy of the
form "records of class X are destroyed after N years" needs those records to have been
placed in their own cart at ingest. The administration guide covers this in full; it is
listed here because it is the question a records-retention reviewer asks first.

### Data residency

Local carts and all their sidecars stay on your host. **Mempacks do not**: their content
lives in a Supabase project — Postgres tables plus an object-storage bucket — and the
Membot host keeps only a cache copy. If your data must not leave your own infrastructure,
leave `SUPABASE_URL` unset; Mempack features then switch themselves off cleanly, and
"not configured" is a supported state rather than an error.

---

## 5. The journal's guarantees, stated precisely

This is the section to read closely, because it is the one most easily overstated.

### What a verification run detects

`python journal.py verify <journal>` walks the whole chain and prints a computed verdict
line first, exiting non-zero on a break. It names the first break by line, sequence number
and reason:

| Detected | Meaning |
| --- | --- |
| `malformed_json` | A line in the middle of the file is no longer valid JSON. Someone edited it. |
| `truncated_tail` | The last line is incomplete. Distinguished from tampering because it is the one break a power failure can cause; the entries before it verify. |
| `missing_field` | An entry is missing a required field. |
| `seq_gap` | A sequence number is higher than expected. **A record was deleted from the interior.** |
| `seq_regression` | A sequence number repeats or goes backwards. Records were reordered or duplicated. |
| `prev_hash_mismatch` | A record does not chain to its predecessor. |
| `hash_mismatch` | A record's own hash no longer covers its own contents. It was edited in place. |
| `entries_deleted_from_tail` | The high-water mark records a later sequence number than the file ends at. **Entries were removed from the end.** |
| `tail_entry_rewritten` | Same length, different content: the last entry was replaced. |

Deletion and reordering are told apart by which sequence numbers are missing: a deletion
leaves a hole in the set, a reorder does not. Both break the chain; only one loses data.

### The gap the high-water mark closes

A hash chain alone detects a tampered entry, a reordered pair and a deleted interior entry,
because all three break the link to the next record. **It does not detect deletion of
complete trailing entries**: drop the last two lines and what remains is a shorter chain
that verifies perfectly, and nothing inside the file can know the removed entries ever
existed. That matters more than an average gap, because truncating the tail is both the
easiest alteration and the most useful one — it is how someone would hide the thing they
just did.

The high-water-mark sidecar, `<cart>.journal.jsonl.hwm`, holds the sequence number and
hash of the newest entry. It is written **after** the journal line is fsynced, never
before, so a process killed between the two steps leaves a torn last line with the mark
one ahead — which `verify` reports as a crash, not as deletion. A mark behind the file is
reported as stale rather than as a break, because it means an append succeeded and its
mark write failed.

With the mark in place, removing trailing entries requires editing a **second** file to
stay invisible, and the two files can be backed up, shipped and permissioned separately.

### The limit, stated without hedging

**This is tamper-evidence, not tamper-proofing, and the difference should never be
blurred.**

A hash chain plus the high-water mark detects tampering, reordering, interior deletion and
trailing deletion. But **anyone with write access to the cart directory can update both
files**, because both files are in that directory. An operator who alters a journal and
re-chains it, updating the mark to match, produces a journal that verifies.

The only construction that makes tampering detectable by a party who does **not** trust
the operator is **external anchoring**: periodically publishing the head hash somewhere
the operator does not control, so a later journal can be checked against a value that was
fixed before the alteration. **That is not built.** It is a deliberate decision recorded
in the design, not an oversight.

What this means in practice, for each kind of reader:

- **An operator checking their own system** gets a real control. Accidental corruption, a
  torn write, a mistaken edit, a script that rewrote a file: all detected and named.
- **An outside reviewer who trusts the operator's access controls** gets a real control,
  on the strength of those controls. Restricting write access to the cart directory, and
  keeping the journal and its mark in an off-host copy the Membot host cannot write to,
  is what carries the weight.
- **An outside reviewer who does not trust the operator** does not get a control yet. If
  your use case requires that, external anchoring is the missing piece and you should ask
  for it explicitly rather than assume the chain covers it.

The repository's own test suite pins both the control and the hole, rather than letting
either be assumed. In `test_journal.py`:
`test_deleting_the_tail_is_detected_by_the_high_water_mark` asserts that a cleanly
truncated tail fails verification; `test_without_the_mark_a_truncated_tail_is_still_undetectable`
asserts that without the mark it verifies clean; and
`test_a_crash_torn_tail_is_distinguished_from_a_deletion` asserts that a power cut and a
deliberate truncation are reported differently rather than both as tampering.

---

## 6. Credential redaction at ingest

### What it does and where it runs

When text is stored through `memory_store`, it is scanned for known credential formats and
every hit is replaced with a one-way fingerprint marker **before the text is embedded**.
The ordering is the point: an embedding computed from the original still encodes the
secret, and embedding inversion can recover approximate text, so masking the string while
keeping the original vector would leave a partial leak that looks clean.

It runs at ingest and nowhere else. A retrieval-side filter would leave the value on disk
and in the vectors, one bug away from being served, while producing the feeling of a
control without the control.

**It fails closed.** If the scan itself raises, the store is **refused** and the caller is
told. Refusing is recoverable; storing a credential is not, because the embedding has
already learned it.

The surrounding passage is kept, with the value replaced. That is deliberate: "we set the
API key and deploys started working" is useful institutional memory and should not be
destroyed to protect a string; the fingerprint answers "was this value ever in our
corpus?" without the value being in the corpus; and the marker leaves a visible trace, so
a false positive that ate content is detectable rather than silent.

### Detected versus suspected

Confidence is a two-valued fact about the format, never a score. There is no third value
and no number, because a number would be a classifier.

- **Detected** — the format carries a checksum or a decodable structure, and it checked
  out. This is "that is a credential", not "that looks like one".
- **Suspected** — the shape matched but the format offers nothing to verify against. Still
  redacted; the caller is told the difference so it can auto-accept detected hits and
  queue suspected ones for review.

Formats recognized by prefix, with the verification each offers:

| Format | Prefix | Verification |
| --- | --- | --- |
| Anthropic API key | `sk-ant-` | none — suspected |
| OpenAI API key | `sk-proj-`, `sk-` | none — suspected. The bare `sk-` rule requires an alphanumeric tail of at least 32 characters, which keeps it off ordinary prose |
| GitHub token | `ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_` | checksum — detected when it validates |
| GitHub fine-grained token | `github_pat_` | documented shape, no documented checksum — suspected |
| AWS access key id | `AKIA`, `ASIA` | exact length and boundary |
| Google API key | `AIza` | exact length and boundary |
| Slack token | `xoxb-`, `xoxa-`, `xoxp-`, `xoxr-`, `xoxs-` | none — suspected |
| Stripe secret key | `sk_live_`, `rk_live_` | none — suspected |
| JSON Web Token | `eyJ` | structural decode of the header and payload |
| PEM private key block | `-----BEGIN` followed by a space | block structure |

The scan operates on **bytes**, not on parsed documents, which is why it is
container-independent: the original incident that motivated it had tokens sitting inside a
pickle, where every text-level search reported only "binary file matches".

There is a second tier that keys on **provenance** rather than content — a path like
`.env`, `id_rsa`, `.aws/`, `.ssh/`, `.netrc`, `.pgpass`, or a command whose output is
credentials such as `env`, `printenv`, `gh auth token`, `aws configure`, `docker login` —
and quarantines the whole buffer, because a credentials file has no surrounding context
worth keeping. **This tier is deliberately disabled on the `memory_store` path**, where no
real provenance exists and a passage legitimately tagged "env setup" would otherwise have
its entire body quarantined. It is intended for a transcript or tool-result ingest path.

### What it structurally cannot do

Stated here rather than left to be discovered:

- **It is not a PII detector.** A birthdate is not PII; a birthdate next to a name is.
  Byte- and pattern-level filters cannot see that context. This is a secrets filter, and
  calling it a PII filter would be false.
- **A low-entropy secret with no label is prose.** A password typed into a sentence is
  indistinguishable from any other word. No tier proposed changes this; not typing it
  remains the only control.
- **Compressed or encrypted containers defeat it.** Gzip, zip deflate and anything
  encrypted destroy the byte sequence being scanned for. Container independence means
  container-agnostic, not codec-defeating. Decompress first, then scan.
- **It does not reach data nobody ingests.** Crash dumps and local secret stores on the
  same machine are a separate inventory.
- **It is a lookup, not a judgment.** There is no classifier and none is planned for these
  tiers, on the stated reasoning that a 90-percent-accurate filter at a boundary is worse
  than none, because people trust it and stop being careful.

### Two further limits in the current wiring

1. **It is wired into one ingest path.** `memory_store` — and therefore
   `POST /api/store`, which calls it — is the only caller. `membox_imprint`,
   `federate_publish`, the Mempack copy and dispatch endpoints, and the offline cart
   builders do **not** run it. If you ingest through any of those, scan the source first.
2. **It does not clean an existing cart.** Redaction is an ingest control. Masking a cart
   that already contains a credential requires re-embedding the masked text, because the
   stored vectors were computed from the original; masking the strings and keeping the old
   vectors would produce something that looks clean and is not. No tooling does this today.

### What the review log holds

Every redaction appends a line to `redaction-review.jsonl` (relocatable via
`MEMBOT_REDACTION_LOG`): timestamp, label, tier, kind, confidence, what verified it, the
reason, a truncated SHA-256 fingerprint, and the offset and length. **It never holds the
value.** The origin string is itself run through the scanner before being written, because
a provenance string can contain a credential. Review it by distinct fingerprint rather
than by occurrence — one credential appearing a hundred times is one decision, not a
hundred.

---

## 7. Controls summary

| Control | Status |
| --- | --- |
| Data on your own host, in plain inspectable files | Yes, for local carts. Not for Mempacks. |
| Cart loading cannot execute code | Yes — current format is NumPy `.npz`, no pickle. Legacy pickle loading is restricted to trusted directories. |
| Cart integrity fingerprint, checked on mount, acted on | Yes. A cart that fails is refused, not logged and served. Detects corruption and unrecorded edits; does not detect an edit where the manifest was regenerated too. |
| Append-only hash-chained provenance journal | Yes, per cart, for the three write paths listed in section 4. |
| Detection of edit, reorder, interior deletion | Yes, from the chain. |
| Detection of trailing deletion | Yes, from the high-water-mark sidecar. |
| Detection of tampering by someone with directory write access | **No.** Requires external anchoring, which is not built. |
| Every write produces a record | **No.** Three paths of 26. |
| Per-person attribution | Only on the token-authenticated path. Shared-key callers have no individual identity. |
| Why-level capture (reasoning, decisions, cautions, predictions) | Yes, append-only, private by default, per-passage. |
| Credential stripping before embedding | Yes, on the `memory_store` path, for known formats, failing closed. |
| General PII or secret detection | **No**, and not claimed. |
| No destructive verbs on the agent surface | Yes, deliberately. Corrections are proposals reviewed by a human. |
| Retention-driven deletion | **No.** File-level deletion by an operator, at cart granularity. |
| Rate limiting | Yes, per client IP, on MCP tool calls. Not on the REST routes. |
| Authentication on all HTTP paths | **No.** The API key covers MCP tool calls; the REST routes are not behind it. |
| Refuses the dangerous default configuration | Yes. Writable plus network-reachable plus unauthenticated refuses to start. |
