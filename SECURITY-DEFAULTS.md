# Security-relevant defaults, and when each changed

A fork or a stale checkout cannot tell, from the code it is running, which of its defaults have
since moved. This file lists the settings that affect exposure, what each defaults to today, and
the date and commit where that default last changed — so an operator can diff **intent** instead
of reading every commit in between.

> **Why this file exists.** On 2026-11-13 two machines in a peer fleet were found listening on all
> interfaces. Neither was misconfigured for the version it was running: both predated the commit
> that made loopback the default. The fix existed upstream and did nothing for the machines that
> were actually listening. ⇒ That is a distribution problem, and this file is the remedy we owe.

**If you run a fork, check this list against your revision first.** `git log` on a hand-deployed
box often describes very little about what is actually running there.

---

## The defaults

| setting | default today | changed | commit |
|---|---|---|---|
| `--transport` | `stdio` — **no network listener at all** unless you ask for one | since v1.0, 2026-02-08 | `c2199e3` |
| `--host` | **`127.0.0.1`** — loopback only. Was `0.0.0.0` | **2026-10-18** | `9d7c471` |
| writable + non-loopback + no `MEMBOT_API_KEY` | **the server refuses to start** | **2026-10-18** | `9d7c471` |
| `--port` | `8000` | since 2026-02-10 | `cad1975` |
| writability | **read-only.** `--writable` is the only thing that enables `memory_store` and `save_cartridge` | since 2026-02-10 | `cad1975` |
| mount integrity | **a failed verification refuses the mount.** It used to be logged and ignored | **2026-10-18** | `a70340a` |
| `--allow-unverified-mount` | **off** — the explicit, visible way to accept the risk | **2026-10-18** | `a70340a` |
| a cart with **no manifest** | **refused.** It used to verify successfully | **2026-11-12** | `bec3ece` |
| integrity fingerprint | **`sha256-full-v2`** — every embedding with shape and dtype, every passage length-prefixed, full SHA-256, algorithm named in the manifest. Was a 64-bit digest over the first embedding, the last embedding and a count | **2026-11-12** | `bec3ece` |
| trusted-directory test | **component-aware ancestor match.** A bare string-prefix match previously admitted a sibling directory sharing the prefix | **2026-11-12** | `a7fa937` |
| cart name matching | an **ambiguous** name is refused and the candidates are named. It previously took the first match | **2026-11-12** | `b6eac68` |
| pickle-format carts | opened only from a configured trusted directory | since v1.0, 2026-02-08 | `c2199e3` |
| `MAX_ENTRIES` / `MAX_TEXT` | `3,000,000` / `10,000` | since v1.0, 2026-02-08 | `c2199e3` |
| `MEMBOT_API_KEY` | **unset, meaning no authentication.** Setting it enables the check | since 2026-02-10 | `cad1975` |

The boot log prints the live values of the last several on every start — read it rather than
inferring them from the unit file.

---

## Four things whose names promise more than they do

These are not defaults that changed. They are defaults that **mislead**, which is worse, because
reading the flag is what convinces you not to check.

### 1 · `--read-only` does nothing

Only `--writable` is consulted. `--read-only` is accepted, is the argparse default, and has no
effect on anything. A unit file containing it looks like it is asserting a posture it merely
inherits. ⇒ **To be sure a server is read-only, confirm `--writable` is absent.**

### 2 · The API key is enforced at the MCP layer

`MEMBOT_API_KEY` is checked where MCP tool calls are handled. **Do not assume it covers every HTTP
path the process serves.** ⇒ If the process is reachable from anything but loopback, put a reverse
proxy in front of it and expose only the MCP endpoint. Treat the key as one layer, not the
boundary.

### 3 · Only `Authorization: Bearer` is honored

`X-API-Key` is ignored, and a request sent that way is rejected with nothing indicating the header
was the problem. ⇒ A client that "has the key" and still gets rejected is usually sending the
wrong header.

### 4 · The committed unit file is not the running unit file

The unit in this repo is a template. Deployed units routinely add an API key, a different bind
address, or different mounts. ⇒ **Read the live unit** (`systemctl cat <unit>`) before concluding
anything about a running server's exposure. A repo file is evidence about the repo.

---

## The one-command check

```bash
# What is this process actually listening on?
ss -ltn "( sport = :8000 )"          # repeat for any other port your unit uses

# Does the unit pass --host at all? Absent means the DEFAULT applies --
# which is loopback only if your revision includes 9d7c471 (2026-10-18).
systemctl cat <your-membot-unit> | grep -- --host
```

⇒ If your revision predates `9d7c471` and the unit passes no `--host`, **you are listening on all
interfaces.** Upgrading is also the test: on a current revision a writable, exposed server with no
key refuses to start and says why.

---

*Maintained because a peer fleet asked for it, 2026-11-13. If a security-relevant default changes,
add a row here in the same commit.*
