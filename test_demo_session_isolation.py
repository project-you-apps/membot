"""The demo is per-visitor, end to end over HTTP. 2026-10-18.

Covers the mount prompt work and the bug it exposed.

⛔ THE BUG THIS EXISTS FOR: /api/status read session_id from query params only -- no
JWT, no X-Membot-Session -- so it reported the SERVER's boot cart to every caller.
The demo page uses exactly that field to decide whether a cart is mounted, so a fresh
visitor had the mount prompt suppressed and the search box enabled, and then got
"No cartridge mounted" from /api/search. Two routes, one header, different answers.

Tested over real HTTP against a real server rather than by calling the handlers,
because the defect lived in how the ROUTE derived its session -- a unit test holding
a fake request would have been written with the same wrong assumption.

Boots on a spare port with a tiny cart so the boot-cart leak is actually testable.
~20 lines of output.
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8299
BASE = f"http://127.0.0.1:{PORT}"
BOOT_CART = "attention-is-all-you-need"   # 24 patterns, boots in seconds

RESULTS = []


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


def call(path, tab=None, body=None, timeout=120):
    req = urllib.request.Request(BASE + path, method="POST" if body is not None else "GET")
    if tab:
        req.add_header("X-Membot-Session", tab)
    data = None
    if body is not None:
        req.add_header("Content-Type", "application/json")
        data = json.dumps(body).encode()
    with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
        return json.loads(r.read().decode())


proc = subprocess.Popen(
    [sys.executable, os.path.join(HERE, "membot_server.py"),
     "--transport", "http", "--port", str(PORT), "--mount", BOOT_CART, "--writable"],
    cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL)
try:
    up = False
    for _ in range(45):
        try:
            call("/api/status")
            up = True
            break
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(2)
    if not up:
        print("  FAIL  server never came up on port", PORT)
        sys.exit(1)

    print(f"\n§1 · the server booted with --mount {BOOT_CART}")
    srv = call("/api/status")
    check("default session holds the boot cart", srv.get("cartridge") == BOOT_CART,
          str(srv.get("cartridge")))

    # Printed text stays ASCII-safe: stdout redirects to a cp1252 file on this box,
    # and a UnicodeEncodeError in a print would fail the run for a reason that has
    # nothing to do with what is being tested.
    print("\n.2 - THE REGRESSION: a visitor must NOT inherit the boot cart")
    a = call("/api/status", tab="visitorA")
    check("visitorA sees NO cartridge", a.get("cartridge") in (None, ""),
          repr(a.get("cartridge")))
    check("visitorA sees 0 memories", a.get("memories") == 0, str(a.get("memories")))
    check("and reports its own session id", str(a.get("session_id", "")).startswith("anon_"),
          str(a.get("session_id")))

    print("\n§3 · search agrees with status — the two must not disagree")
    s = call("/api/search", tab="visitorA", body={"query": "anything", "top_k": 3})
    check("search says no cartridge", "no cartridge" in (s.get("error") or "").lower(),
          repr(s.get("error")))
    check("status and search agree for the same header",
          (a.get("cartridge") in (None, "")) and bool(s.get("error")),
          "both say unmounted")

    print("\n§4 · mounting is per visitor")
    call("/api/mount", tab="visitorB", body={"name": BOOT_CART}, timeout=240)
    b = call("/api/status", tab="visitorB")
    a2 = call("/api/status", tab="visitorA")
    srv2 = call("/api/status")
    check("visitorB now has the cart", b.get("cartridge") == BOOT_CART, str(b.get("cartridge")))
    check("visitorA STILL has none", a2.get("cartridge") in (None, ""), repr(a2.get("cartridge")))
    check("server default unchanged", srv2.get("cartridge") == BOOT_CART)
    check("one shared payload, not two", srv2.get("shared_carts") == 1,
          f"shared_carts={srv2.get('shared_carts')}")

    print("\n§5 · a header-less client still reaches 'default' (backward compatible)")
    legacy = call("/api/status")
    check("no header -> default session", legacy.get("session_id") == "default",
          str(legacy.get("session_id")))
    check("no header -> boot cart", legacy.get("cartridge") == BOOT_CART)

    print("\n§6 · the page itself tells a new visitor to pick a cart")
    # Read the served string in-process rather than over HTTP: the page is ~110 KB
    # and fetching it mid-run raced the server's shutdown. The content is identical
    # either way, and SESSIONS -- the thing HTTP was actually needed for -- are
    # covered in full above.
    sys.path.insert(0, HERE)
    import membot_server as _M
    page = _M._APP_HTML
    check("mount prompt present", "Pick a cartridge to explore" in page)
    check("per-session reassurance present", "your session only" in page)
    check("the old misleading invitation is GONE",
          "Search your brain cartridge" not in page)
    check("emptyStateHTML defined once", page.count("function emptyStateHTML") == 1)
    check("search is gated on a mount", "Pick a cartridge first" in page)

finally:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()

passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print("\n" + "=" * 62)
print(f"VERDICT: {passed}/{total} checks passed" + ("" if passed == total else "  <<< FAILURES"))
for label, ok, detail in RESULTS:
    if not ok:
        print(f"   FAILED: {label}   {detail}")
print("=" * 62)
sys.exit(0 if passed == total else 1)
