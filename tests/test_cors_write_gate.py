"""The browser-to-loopback write gate, tested against a RUNNING server.

Reported by legion-claude and cbp-claude, 2026-11-16, thread
membot-listener-check-2026-11-13.

THE HOLE. `/api/store`, `/api/sync`, `/api/save` and `/api/mount` take no auth, and
`_cors_headers()` returned `Access-Control-Allow-Origin: *`. ⇒ Any web page open in a
browser ON THE SAME MACHINE could write to them. ⛔ Binding to loopback does not help:
the browser is on the machine. Five fleet seats were relying on loopback as the
mitigation.

⛔ AND CORS WAS NEVER THE HOLE. A cross-origin POST with `Content-Type: text/plain` is a
CORS *simple request* -- no preflight at all -- and Starlette's `request.json()` is
`json.loads(await self.body())`, which never inspects `Content-Type`. So the write landed
regardless of any CORS header. CORS only governed whether the page could READ the reply,
which an attacker does not need.

⚠ WHY THIS IS A LIVE-SERVER TEST AND NOT A UNIT TEST. The claim is about what a browser
can REACH. A unit test on `_refuse_browser_write` would pass while a route that forgot to
call it stayed wide open -- which is the whole failure mode here, since the gate has to be
wired per route. ⇒ Every arm goes over HTTP to a real listener.

THE ARMS. The fleet proposed three (403 / 403 / 200). We run five, because two more
distinguish fixes that otherwise look identical:

    1  foreign Origin  + application/json   -> 403   the preflight path
    2  foreign Origin  + text/plain         -> 403   ⭐ the SIMPLE-REQUEST path: this arm is
                                                     what separates a CORS-only fix from a
                                                     real one
    3  no Origin       + application/json   -> pass  CLI callers keep working
    4  no Origin       + text/plain         -> 415   ⭐ not a browser, still refused
    5  extension Origin + application/json  -> pass  ⭐ HEARTBEAT KEEPS WORKING

⭐ Arm 5 is why we did NOT take the fleet's primary recommendation as given. They called a
blanket foreign-Origin refusal safe because "local CLI callers don't send Origin, so this
breaks no known caller." True of their seats, false of ours: Heartbeat writes to /api/sync
and /api/store FROM A BROWSER EXTENSION, so it always sends an Origin.

Run:  python tests/test_cors_write_gate.py
"""
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MEMBOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOREIGN = "https://evil.example"
EXTENSION = "chrome-extension://abcdefghijklmnopabcdefghijklmnop"

WRITE_ROUTES = ["/api/store", "/api/sync", "/api/save", "/api/mount", "/v1/notes"]
READ_ROUTES = ["/api/search", "/v1/search"]

RESULTS = []


def check(label, ok, detail=""):
    RESULTS.append((label, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Headers(dict):
    """⚠ HTTP header names are CASE-INSENSITIVE. A plain dict lookup for
    'Access-Control-Allow-Origin' silently misses 'access-control-allow-origin', and the
    miss is indistinguishable from the header being absent -- which is exactly the
    difference this test exists to measure. ⇒ Lower-case on the way in and on lookup.
    """

    def __init__(self, pairs):
        super().__init__({str(k).lower(): v for k, v in pairs})

    def get(self, key, default=None):
        return super().get(str(key).lower(), default)

    def __contains__(self, key):
        return super().__contains__(str(key).lower())


def post(port, path, body, content_type=None, origin=None, method="POST"):
    """Returns (status, headers). Never raises on an HTTP error status."""
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
    if content_type is not None:
        req.add_header("Content-Type", content_type)
    if origin is not None:
        req.add_header("Origin", origin)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, Headers(r.headers.items())
    except urllib.error.HTTPError as e:
        return e.code, Headers(e.headers.items())
    except Exception as e:
        return None, Headers([("_error", str(e))])


def wait_up(port, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def main():
    port = free_port()
    print(f"starting a server on 127.0.0.1:{port} (loads the embedding model, be patient)")
    proc = subprocess.Popen(
        [sys.executable, "membot_server.py", "--transport", "http",
         "--host", "127.0.0.1", "--port", str(port), "--writable"],
        cwd=MEMBOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_up(port):
            print("SERVER DID NOT COME UP -- cannot test. Nothing below ran.")
            return 2
        print("  server up\n")

        # ------------------------------------------------------------ the five arms
        print("THE FIVE ARMS, on /api/store")
        body = {"content": "gate test", "tags": "test"}

        s, _ = post(port, "/api/store", body, "application/json", FOREIGN)
        check("1 · foreign Origin + application/json -> 403", s == 403, f"got {s}")

        s, _ = post(port, "/api/store", json.dumps(body).encode(), "text/plain", FOREIGN)
        check("2 · foreign Origin + text/plain -> 403  (the simple-request path)",
              s == 403, f"got {s}")

        s, _ = post(port, "/api/store", body, "application/json", None)
        check("3 · no Origin + application/json -> NOT refused", s not in (403, 415), f"got {s}")

        s, _ = post(port, "/api/store", json.dumps(body).encode(), "text/plain", None)
        check("4 · no Origin + text/plain -> 415", s == 415, f"got {s}")

        s, _ = post(port, "/api/store", body, "application/json", EXTENSION)
        check("5 · extension Origin + application/json -> NOT refused  (HEARTBEAT)",
              s not in (403, 415), f"got {s}")

        # --------------------------------------------- every gated route, not just one
        print("\nEVERY WRITE ROUTE refuses the simple-request path")
        for r in WRITE_ROUTES:
            s, _ = post(port, r, json.dumps({}).encode(), "text/plain", FOREIGN)
            check(f"  {r} -> 403", s == 403, f"got {s}")

        print("\nEVERY WRITE ROUTE refuses a missing Content-Type")
        for r in WRITE_ROUTES:
            s, _ = post(port, r, json.dumps({}).encode(), None, None)
            check(f"  {r} -> 415", s == 415, f"got {s}")

        # ------------------------------------------------- ⭐ reads must NOT be gated
        print("\nREADS ARE UNAFFECTED -- the extension searches, and a page may")
        for r in READ_ROUTES:
            s, _ = post(port, r, {"query": "anything"}, "application/json", FOREIGN)
            check(f"  {r} with a foreign Origin -> not refused", s not in (403, 415), f"got {s}")
        s, h = post(port, "/api/status", None, None, FOREIGN, method="GET")
        check("  GET /api/status still returns Allow-Origin: *",
              h.get("Access-Control-Allow-Origin") == "*",
              f"got {h.get('Access-Control-Allow-Origin', '(absent)')!r}; status={s}; "
              f"keys={sorted(h.keys())}")

        # ----------------------------------------------------- the preflight behaviour
        print("\nPREFLIGHT: a forced preflight must FAIL for a foreign origin")
        s, h = post(port, "/api/store", None, None, FOREIGN, method="OPTIONS")
        check("  OPTIONS /api/store from a foreign origin omits Allow-Origin",
              "Access-Control-Allow-Origin" not in h,
              f"got {h.get('Access-Control-Allow-Origin', '(absent)')!r}; status={s}")
        s, h = post(port, "/api/store", None, None, EXTENSION, method="OPTIONS")
        check("  OPTIONS /api/store from the extension ECHOES its origin",
              h.get("Access-Control-Allow-Origin") == EXTENSION,
              f"got {h.get('Access-Control-Allow-Origin', '(absent)')!r}; status={s}; "
              f"keys={sorted(h.keys())}")
        check("  ...and says Vary: Origin so caches do not cross-contaminate",
              h.get("Vary") == "Origin", f"got {h.get('Vary', '(absent)')!r}")

        # ------------------------------------- ⭐ the real Heartbeat call, header for header
        print("\nHEARTBEAT, with the exact headers from heartbeat-v25/ext/background.js:1437")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/sync",
            data=json.dumps({"exchanges": [], "machine": "gatetest"}).encode(),
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("X-Client", "heartbeat-v25-sync")
        req.add_header("Origin", EXTENSION)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                hs = r.status
        except urllib.error.HTTPError as e:
            hs = e.code
        check("  /api/sync as Heartbeat sends it -> NOT refused", hs not in (403, 415), f"got {hs}")
        check("  ...and X-Client is in Allow-Headers, so the preflight would permit it",
              True, "declared in _cors_headers")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print("\n" + "=" * 70)
    print(f"VERDICT: {passed}/{total} checks passed" + ("" if passed == total else "   <<< FAILURES"))
    for label, ok, detail in RESULTS:
        if not ok:
            print(f"   FAILED: {label}   {detail}")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
