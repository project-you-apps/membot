"""The service identity block. 2026-11-15.

WHAT THIS GUARDS. On 2026-11-15 a test pointed at 127.0.0.1:8000 expecting membot.
The port answered: /api/status returned a plausible membot-shaped body and
/api/search returned an empty result set, read as "no cartridge mounted". The
process was Vector Plus Studio, which serves routes of the same names over a
different engine. The wrong diagnosis survived three debugging steps because
nothing in either reply said which program had answered.

=> The block exists so a caller can ASSERT the program before trusting a number.

THE CHECK THAT MATTERS is SI3: code_digest must MOVE when a .py file beside the
program changes. The droplet is a 2026-07-20 checkout with hand-scp'd files on
top -- 9 modified, 30 untracked -- so git_rev there names a July commit while the
process serves something else. A digest that did not notice an edited file would
be the same lie with more steps, so it is tested by editing a file and watching
the digest change rather than by asserting the function returns a string.

⛔ NOT a security test. The block is unsigned and self-reported; a hostile process
can print anything. It answers "am I talking to the build I think I am".
"""
import importlib
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import service_identity  # noqa: E402

RESULTS = []


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


def note(label, detail):
    print(f"  ----  {label}   [{detail}]")


print("\nSI1 - the block names the program and the role")
b = service_identity.service_block("mcp-server", host="127.0.0.1", port=8200)
check("name is membot", b.get("name") == "membot", repr(b.get("name")))
check("role is reported", b.get("role") == "mcp-server", repr(b.get("role")))
check("extras pass through", b.get("host") == "127.0.0.1" and b.get("port") == 8200)
check("contract_version present", bool(b.get("contract_version")), b.get("contract_version"))
check("pid is this process", b.get("pid") == os.getpid())
for k in ("code_digest", "code_files", "git_rev", "git_dirty", "python", "started_at"):
    check(f"key present: {k}", k in b)

print("\nSI2 - the SAME module serves both roles, and they are distinguishable")
# The two servers in this directory are the pair that caused the 2026-11-14
# exposure: the loopback default was applied to one and nobody looked at the
# second. An identity format that could drift between them would reproduce that
# in the one place meant to detect it. So: one function, and the role differs.
bridge = service_identity.service_block("rest-bridge", host="127.0.0.1", port=8001)
check("roles differ", b["role"] != bridge["role"], f"{b['role']} vs {bridge['role']}")
check("but the build identity is IDENTICAL",
      b["code_digest"] == bridge["code_digest"] and b["git_rev"] == bridge["git_rev"],
      "same module, same digest")
check("a caller can tell the two servers apart",
      bridge["role"] == "rest-bridge" and bridge.get("port") == 8001)

print("\nSI3 - code_digest MOVES when a .py file beside the program changes")
# ⭐ THE TEST THAT EARNS THE FEATURE. A digest that did not notice an scp'd file
# would be exactly as misleading as the git_rev it is meant to replace.
with tempfile.TemporaryDirectory() as d:
    a_py = os.path.join(d, "alpha.py")
    with open(a_py, "w", encoding="utf-8") as f:
        f.write("x = 1\n")
    d1, n1 = service_identity.code_digest(d)
    check("digest over a one-file directory", d1 is not None and n1 == 1, f"{d1} / {n1} files")

    # same content, recomputed: stable
    d1b, _ = service_identity.code_digest(d)
    check("stable across calls on unchanged files", d1 == d1b)

    # content changed by one byte
    with open(a_py, "w", encoding="utf-8") as f:
        f.write("x = 2\n")
    d2, _ = service_identity.code_digest(d)
    check("MOVES on a one-byte content change", d2 != d1, f"{d1} -> {d2}")

    # a file ADDED, which is the scp case git does not see as a change
    with open(os.path.join(d, "beta.py"), "w", encoding="utf-8") as f:
        f.write("y = 1\n")
    d3, n3 = service_identity.code_digest(d)
    check("MOVES when a file is ADDED (the scp case)", d3 != d2 and n3 == 2,
          f"{n3} files, {d2} -> {d3}")

    # a RENAME with identical content. The name is hashed, so this must move too --
    # otherwise two different module layouts could share one build id.
    os.rename(os.path.join(d, "beta.py"), os.path.join(d, "gamma.py"))
    d4, n4 = service_identity.code_digest(d)
    check("MOVES on a rename with unchanged content", d4 != d3, f"{d3} -> {d4}")

    # a non-.py file must NOT move it: the digest is about code, and a log or a
    # cart landing in the directory is not a different build.
    with open(os.path.join(d, "notes.txt"), "w", encoding="utf-8") as f:
        f.write("not code\n")
    d5, n5 = service_identity.code_digest(d)
    check("does NOT move for a non-.py file", d5 == d4 and n5 == n4, f"{n5} files")

print("\nSI4 - it degrades instead of raising")
# An identity helper that can throw becomes a denial of service on every endpoint
# that reports it. Same discipline as seat_identity.resolve_seat.
missing = os.path.join(tempfile.gettempdir(), "definitely-not-a-real-dir-91823")
d6, n6 = service_identity.code_digest(missing)
check("unreadable directory -> (None, 0), not an exception", d6 is None and n6 == 0)
rev, dirty = service_identity.git_rev(missing)
check("git on a non-repo -> (None, None), not an exception",
      rev is None and dirty is None)

print("\nSI5 - describe() is one ASCII line fit for a cp1252 console")
line = service_identity.describe("rest-bridge", host="0.0.0.0", port=8001)
check("single line", "\n" not in line)
check("pure ASCII", all(ord(c) < 128 for c in line))
check("names the role", "rest-bridge" in line)
check("carries the build id", (b["code_digest"] or "unknown") in line)
note("describe()", line[:110])

print("\nSI6 - the real tree answers, and git_rev is SECONDARY to code_digest")
real_digest, real_n = service_identity.code_digest(ROOT)
check("digests the actual membot directory", real_digest is not None and real_n > 5,
      f"{real_n} .py files, digest {real_digest}")
rrev, rdirty = service_identity.git_rev(ROOT)
if rrev:
    note("git_rev here", f"{rrev}{' (dirty)' if rdirty else ''}")
    note("why it is secondary",
         "git_dirty ignores UNTRACKED files; the droplet has ~30 of them")
else:
    note("git_rev here", "unavailable -- code_digest still answers")

print("\nSI7 - the two servers actually wire it in")
# A block nothing reports is a block that does not exist. Check the call sites
# rather than trusting that writing the module was the job.
srv = open(os.path.join(ROOT, "membot_server.py"), encoding="utf-8").read()
brg = open(os.path.join(ROOT, "membot_rest_bridge.py"), encoding="utf-8").read()
check("membot_server imports it", "import service_identity" in srv)
check("membot_server reports it on /api/status",
      'service_identity.service_block(\n                "mcp-server"' in srv
      or 'service_identity.service_block("mcp-server"' in srv)
check("membot_server logs it at boot", 'service_identity.describe(' in srv)
check("get_status reports a build id (AGENTS.md section 5 promises a version)",
      "f\"Service: {svc['name']}/{svc['role']} \"" in srv)
check("the bridge imports it", "import service_identity" in brg)
check("the bridge reports role=rest-bridge",
      "'rest-bridge'" in brg or '"rest-bridge"' in brg)
check("the bridge states its missing protections",
      "authentication=None" in brg and "writable=True" in brg)

passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print("\n" + "=" * 68)
print(f"VERDICT: {passed}/{total} checks passed" + ("" if passed == total else "   <<< FAILURES"))
for label, ok, detail in RESULTS:
    if not ok:
        print(f"   FAILED: {label}   {detail}")
print("=" * 68)
sys.exit(0 if passed == total else 1)
