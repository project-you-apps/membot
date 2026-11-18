"""A seat the request layer derived must survive being handed to a tool. 2026-11-15.

THE BUG, MEASURED. A REST route derives the seat itself -- JWT first, then the
sanitized X-Membot-Session tab header -- and hands it to an @mcp.tool() as
`session_id`. seat_identity.resolve_seat refuses any claim in a reserved namespace
as impersonation, because over HTTP the tools share ONE bearer token. But the
server's own `anon_<tab>` is reserved too, and by then a seat the server minted and
a seat a caller typed are the same string in the same argument.

  => The server mints a name and then refuses to believe its own name.

With a tab header on a writable server, before the fix:

    /api/mount   seated in 'default'
    /api/store   WROTE into 'default'      <- the tab could not see its own write,
    /api/status  reported 'default'           every other anon caller could
    /api/search  looked in 'anon_<tab>'    <- the one route that was right

Reads isolated, writes pooled: the isolation protected nothing and leaked
everything. And the journal recorded `writer: default` for the tab's write, a false
authorship record in the subsystem whose whole job is provenance. It did set
identity_refused, so it was loudly wrong in a field nobody reads.

⛔ WHY NOT A CORE/WRAPPER SPLIT PER TOOL. _journal_event calls _resolve_seat on its
own, deep in the call stack. Splitting a tool fixes that tool and never reaches the
journal. The contextvar reaches both.

⛔ WHY NOT A TOOL PARAMETER. A parameter on an @mcp.tool() signature is part of its
public schema, so it would hand every MCP client the bypass the 2026-11-01 change
closed.

THE CHECK THAT MATTERS is RS6: an end-to-end store under a tab header, asserting
the write is visible to that tab and NOT to the default slot. Everything above it
is the mechanism; that is the behaviour.
"""
import json
import os
import socket
import subprocess
import sys

# ⚠ The cart this runs against is NOT hardcoded. A live demo cart is named for
# whoever it was built for, and that name does not belong in a public test.
# ⭐ Defaults to the cart that SHIPS IN THIS REPO, so the live section actually runs
# for anyone who clones it. ⛔ It used to name a live demo cart built for a specific
# customer -- that name does not belong in a public test -- and a placeholder that
# exists nowhere is no better: the live checks SKIP and the suite still says "passed",
# just with a smaller number nobody reads. Measured 2026-11-18: 14/14 instead of
# 32/32, and 24/24 instead of 27/27.
_TEST_CART = os.environ.get("MEMBOT_TEST_CART", "attention-is-all-you-need")
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import seat_identity  # noqa: E402

RESULTS = []
E2E_RAN = False


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


def note(label, detail):
    print(f"  ----  {label}   [{detail}]")


print("\nRS1 - unset by default, so nothing changes for stdio or MCP callers")
# The regression that would matter most: MCP tool calls never set the contextvar,
# so every branch they used before must still be reached.
check("no derived seat outside a block", seat_identity.current_derived_seat() == "")
r = seat_identity.resolve_seat("someseat")
check("an unreserved claim is still honored (stdio/local path)",
      r["seat"] == "someseat" and r["source"] == "claimed", f"{r['seat']}/{r['source']}")
r = seat_identity.resolve_seat("")
check("nothing claimed still means the default seat",
      r["seat"] == "default" and r["source"] == "default")
r = seat_identity.resolve_seat("anon_someoneelse", request=object(),
                               read_jwt=lambda _: None, decode_jwt=lambda _: {})
check("a CALLER claiming a reserved seat is still REFUSED",
      r["seat"] == "default" and r["refused"] is True,
      "the 2026-11-01 protection is intact")

print("\nRS2 - inside a block, a server-derived anon seat is honored")
with seat_identity.derived_seat("anon_tab7"):
    check("current_derived_seat reports it",
          seat_identity.current_derived_seat() == "anon_tab7")
    r = seat_identity.resolve_seat("anon_tab7", request=object(),
                                   read_jwt=lambda _: None, decode_jwt=lambda _: {})
    check("resolve_seat returns the derived seat, not 'default'",
          r["seat"] == "anon_tab7", r["seat"])
    check("and does NOT mark it refused", r["refused"] is False)
    # ⛔⛔ The trust label must NOT improve. The tab header is attacker-supplied --
    # one tab may name another -- so this restores ROUTING and asserts nothing about
    # identity. Per-seat separation stays a convention, not a boundary.
    check("source stays 'claimed' -- routing restored, trust NOT upgraded",
          r["source"] == "claimed", r["source"])
    check("verified_sub stays empty", r["verified_sub"] == "")

print("\nRS3 - it is deliberately narrow")
# Only anon_ is honored from the contextvar. user_ falls through so branch 2 mints it
# from the live JWT -- the seat arrives verified by this function's own check rather
# than on a promise from its caller.
with seat_identity.derived_seat("user_deadbeef"):
    r = seat_identity.resolve_seat("", request=object(),
                                   read_jwt=lambda _: None, decode_jwt=lambda _: {})
    check("a derived user_ seat is NOT taken on trust from the contextvar",
          r["seat"] != "user_deadbeef",
          f"got {r['seat']!r} -- falls through to JWT verification")
    check("and is not reported as verified", r["source"] != "verified", r["source"])

with seat_identity.derived_seat("plainname"):
    check("a non-reserved derived value also falls through (that path already worked)",
          seat_identity.resolve_seat("plainname")["seat"] == "plainname")

print("\nRS4 - the block always resets, so a seat cannot leak into the next request")
try:
    with seat_identity.derived_seat("anon_leaky"):
        raise RuntimeError("boom")
except RuntimeError:
    pass
check("reset after an exception", seat_identity.current_derived_seat() == "",
      f"got {seat_identity.current_derived_seat()!r}")
with seat_identity.derived_seat("anon_outer"):
    with seat_identity.derived_seat("anon_inner"):
        inner = seat_identity.current_derived_seat()
    outer_after = seat_identity.current_derived_seat()
check("nests and restores the outer value", inner == "anon_inner" and outer_after == "anon_outer",
      f"{inner} then {outer_after}")
check("fully unset after the outermost block", seat_identity.current_derived_seat() == "")
with seat_identity.derived_seat("  anon_spaced!!  "):
    check("the seat is sanitized on the way in",
          seat_identity.current_derived_seat() == "anon_spaced",
          seat_identity.current_derived_seat())

print("\nRS5 - every route that calls a tool actually wraps it")
# A fix nothing calls is the failure this project keeps hitting. Check the call
# sites, with a positive control, rather than trusting that writing the helper was
# the job.
src = open(os.path.join(ROOT, "membot_server.py"), encoding="utf-8").read()
check("the helper exists", "def _rest_seat(" in src)
check("routes derive through it", src.count("_rest_seat(request") >= 5,
      f"{src.count('_rest_seat(request')} call sites")
wrapped = src.count("with seat_identity.derived_seat(")
check("the three tool-calling routes wrap the call", wrapped >= 3, f"{wrapped} blocks")
for tool in ("memory_store", "mount_cartridge", "save_cartridge"):
    i = src.find(f"getattr({tool},")
    window = src[i:i + 600] if i >= 0 else ""
    check(f"{tool} is invoked inside a derived_seat block",
          "with seat_identity.derived_seat(" in window,
          "indirect getattr(...,'fn') call -- the hiding place static analysis missed")
check("/api/status no longer re-resolves the seat it derived",
      "sid = _resolve_session_id(sid)" not in src,
      "that line cancelled its own repair four lines after making it")
# POSITIVE CONTROL for the two checks above.
check("the scan can actually see a wrap",
      "with seat_identity.derived_seat(" in
      "        with seat_identity.derived_seat(session_id):\n")

print("\nRS6 - END TO END: a tab's write lands in the tab, not the shared slot")
PORT = 8061
TAB = "tabRS6"
MARKER = f"rs6marker{os.getpid()}"
proc = None
try:
    devnull = open(os.devnull, "w")
    proc = subprocess.Popen(
        [sys.executable, "membot_server.py", "--transport", "http", "--writable",
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=ROOT, stdout=devnull, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, text=True)
    up = False
    for _ in range(90):
        if proc.poll() is not None:
            break
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=2):
                up = True
                break
        except OSError:
            time.sleep(2)

    if not up:
        note("E2E NOT RUN - server did not start", "mechanism checks above still stand")
    else:
        def post(path, payload, hdrs=None):
            h = {"Content-Type": "application/json"}
            if hdrs:
                h.update(hdrs)
            req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                         data=json.dumps(payload).encode(),
                                         method="POST", headers=h)
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())

        TABH = {"X-Membot-Session": TAB}
        m = post("/api/mount", {"name": _TEST_CART}, TABH)
        res = str(m.get("result", ""))
        seat = res.split("Session:")[-1].strip().rstrip(".") if "Session:" in res else "?"

        if "Mounted" not in res:
            note("E2E NOT RUN - mount did not take", res[:70])
        else:
            E2E_RAN = True
            check("mount seats in the TAB, not 'default'", seat == f"anon_{TAB}", seat)
            post("/api/store", {"content": f"{MARKER} planted by the tab", "tags": "t"}, TABH)

            q_tab = post("/api/search", {"query": MARKER, "top_k": 3}, TABH)
            q_def = post("/api/search", {"query": MARKER, "top_k": 3})

            def found(r):
                return any(MARKER in (x.get("full_text") or x.get("text") or "")
                           for x in r.get("results", []))

            check("the tab can see its OWN write", found(q_tab),
                  f"{len(q_tab.get('results', []))} results")
            check("the shared 'default' slot CANNOT see it", not found(q_def),
                  q_def.get("error") or f"{len(q_def.get('results', []))} results")
finally:
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()

passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print("\n" + "=" * 70)
print(f"VERDICT: {passed}/{total} checks passed" + ("" if passed == total else "   <<< FAILURES"))
if not E2E_RAN:
    print("RS6 DID NOT RUN. The mechanism is checked; the BEHAVIOUR is not.")
for label, ok, detail in RESULTS:
    if not ok:
        print(f"   FAILED: {label}   {detail}")
print("=" * 70)
sys.exit(0 if passed == total else 1)
