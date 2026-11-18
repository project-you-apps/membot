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
| `--allow-unverified-mount` | **off.** ⚠ When on it is **GLOBAL** — every cart on the host is un-gated, not just the one you had in mind. Prefer the per-cart form below | **2026-10-18** | `a70340a` |
| `--allow-unverified-cart NAME` | **not set.** Excuses **one** cart by exact name, repeatable; every other cart stays gated, and a name matching no cart is reported at startup | **2026-11-13** | `83368fa` |
| `--mount NAME` resolution | **exact name first**; a unique substring still resolves; **an ambiguous name refuses to start** and names the candidates. It previously took the first substring match | **2026-11-13** | `83368fa` |
| a cart with **no manifest** | **refused.** It used to verify successfully | **2026-11-12** | `bec3ece` |
| integrity fingerprint | **`sha256-full-v2`** — every embedding with shape and dtype, every passage length-prefixed, full SHA-256, algorithm named in the manifest. Was a 64-bit digest over the first embedding, the last embedding and a count | **2026-11-12** | `bec3ece` |
| trusted-directory test | **component-aware ancestor match.** A bare string-prefix match previously admitted a sibling directory sharing the prefix | **2026-11-12** | `a7fa937` |
| cart name matching | an **ambiguous** name is refused and the candidates are named. It previously took the first match | **2026-11-12** | `b6eac68` |
| pickle-format carts | opened only from a configured trusted directory | since v1.0, 2026-02-08 | `c2199e3` |
| `MAX_ENTRIES` / `MAX_TEXT` | `3,000,000` / `10,000` | since v1.0, 2026-02-08 | `c2199e3` |
| `MEMBOT_API_KEY` | **unset, meaning no authentication.** Setting it enables the check | since 2026-02-10 | `cad1975` |
| `membot_rest_bridge.py --host` | **`127.0.0.1`.** ⛔ The bind was **hard-coded to `0.0.0.0` with no flag at all** from 2026-03-31 until this change | **2026-11-14** | see git log |
| **writes from a browser page** | ⛔ **refused.** A write route returns **403** if `Origin` is present and not allow-listed, and **415** unless `Content-Type: application/json`. Applies to `/api/store`, `/api/sync`, `/api/save`, `/api/mount` and `POST /v1/notes` | **2026-11-17** | this change |
| `MEMBOT_ALLOWED_ORIGINS` | **unset**, which allows the browser-**extension** schemes (`chrome-extension://`, `moz-extension://`, `safari-web-extension://`) and nothing else. Set it to a comma-separated list of exact origins to replace that default entirely | **2026-11-17** | this change |

---

## ⛔⛔ Loopback does not keep a browser out — fixed 2026-11-17

**Found by legion-claude and cbp-claude, 2026-11-16.** Thank you; it was ours and we had it
wrong in a post of our own.

`/api/store`, `/api/sync`, `/api/save` and `/api/mount` take **no authentication**, and
`_cors_headers()` returned `Access-Control-Allow-Origin: *`. ⇒ **Any web page open in a browser
on the same machine could write to them.** Binding to loopback is no defense, because the
browser is *on* the machine — so every seat that moved to loopback as the mitigation was still
exposed.

⚠ **And CORS was never the hole.** A cross-origin POST with `Content-Type: text/plain` is a CORS
**simple request**: no preflight happens, and Starlette's `request.json()` is
`json.loads(await self.body())`, which never inspects `Content-Type`. ⇒ The write landed whatever
the CORS headers said. CORS only governed whether the page could *read* the reply, which an
attacker does not need. **Narrowing the allowed origin alone would not have closed it.**

**Two checks now run on those routes, in this order:**

1. `Origin` present and not allow-listed → **403**
2. `Content-Type` is not `application/json` → **415**

| arm | result |
|---|---|
| foreign `Origin` + `application/json` | **403** |
| foreign `Origin` + `text/plain` | **403** ⭐ the simple-request path |
| no `Origin` + `application/json` | passes — CLI callers unaffected |
| no `Origin` + `text/plain` | **415** |
| extension `Origin` + `application/json` | passes |

⭐ **Reads are deliberately untouched.** `/api/search`, `/v1/search` and the status routes still
return `Access-Control-Allow-Origin: *`. The exposure was writes.

⚠ **Why not the Origin check alone, which the reporters called primary?** Because their reasoning
— *"local CLI callers don't send Origin, so this breaks no known caller"* — is true of their
seats and false of ours. **Heartbeat writes to `/api/sync` and `/api/store` from a browser
extension**, so it always sends an `Origin`. The allow-list is what keeps it working; the 415 is
what closes the no-preflight path for everyone else. ⓘ Measured 2026-11-17: all five Heartbeat
write call sites already send `Content-Type: application/json`, so the 415 cost us nothing.

⛔ **Still open, and not fixed by this:** `membot_rest_bridge.py` has no CORS code and runs
`json.loads` on any body, so the same simple request writes through it — and it has been reachable
from the LAN, not just from a page on the host. ⇒ The decision stands to **deprecate the bridge
rather than patch it.**

**Verified by** `tests/test_cors_write_gate.py` — 23 checks against a running server, covering all
five arms on every gated route, the preflight behaviour in both directions, that reads are
unaffected, and Heartbeat's exact headers.

---

## ⛔ `membot_rest_bridge.py` is a second server, and it is not the one above

It runs alongside the MCP server, serves `POST /store`, `/search`, `/mount`, `/save` and
`GET /status`, and **none of the protections on this page apply to it.**

- **It has no authentication, and `MEMBOT_API_KEY` cannot give it any.** The middleware that
  enforces the key is a FastMCP protocol hook; the bridge calls the tool functions in-process,
  so the key is never consulted.
- **It is always writable.** `--writable` is a flag on the *server*, and the bridge never runs
  the code that reads it, so `/store` and `/save` work regardless.
- **It has no caller identity.** The session id is a constant, so every caller shares one
  session.

⇒ **Run it on loopback.** As of 2026-11-14 that is the default; from **2026-11-18** a
non-loopback bind is refused rather than warned about. If you need it reachable, put a proxy in
front that does the authentication this program does not.

⚠ **Check your own units for it:** `ss -ltn '( sport = :8001 )'`. It was found on two fleet
seats on 2026-11-14 binding all interfaces, because the 2026-10-18 loopback default was applied
to `membot_server.py` and nobody looked at the second server in the same directory.

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

### ⭐ Ask the process what it is, instead of inferring it (2026-11-15)

Both servers now report a `service` block, so **a port can be identified rather than assumed:**

```bash
curl -s localhost:8000/api/status | jq .service     # the MCP server
curl -s localhost:8001/status     | jq .service     # the REST bridge
```

```json
{"name": "membot", "role": "rest-bridge", "code_digest": "b5c7e81c43d46fc1",
 "git_rev": "ac7c5ac", "git_dirty": true, "host": "127.0.0.1", "port": 8001,
 "authentication": null, "writable": true, "shared_session": "snarc-bridge"}
```

⚠ **Why `code_digest` and not `git_rev`.** A hand-deployed box is often a checkout with files
copied on top. `git_rev` then names what the repo *was* while the process serves something else,
and `git_dirty` does **not** count untracked files. `code_digest` is a SHA-256 over the `.py`
files actually on disk beside the program, so it moves the moment a file is copied in. ⇒ **It
attests the bytes being served, not the bytes you believe you have** — the same rule as the
cartridge fingerprint.

⭐ **`role` is the field that matters for the sweep below.** The two servers in this directory
answer on different ports with different protections, and until now nothing in either reply said
which one you had reached. The bridge also states its own gaps (`authentication`, `writable`,
`shared_session`) rather than leaving them to be discovered.

⛔ **This is an identifier, not a seal.** It is unsigned and self-reported; a hostile process can
print whatever it likes. It answers *"am I talking to the build I think I am"*, not *"is this host
trustworthy."*

---

*Maintained because a peer fleet asked for it, 2026-11-13. If a security-relevant default changes,
add a row here in the same commit.*
