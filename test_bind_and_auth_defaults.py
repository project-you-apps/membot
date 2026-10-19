"""Secure-by-default bind and the writable-exposure refusal. 2026-10-18.

Closes A1 (FINDING-A1-membot-faces-the-network-2026-10-17): port 8200 ran
--transport http --writable on 0.0.0.0 with auth opt-in and MEMBOT_API_KEY unset, so
anyone who could reach it could call writable tools as any seat. Only the Windows
firewall stopped it, which is incidental rather than architectural.

Two changes:
  --host now defaults to 127.0.0.1
  the server REFUSES to start when writable AND reachable AND unauthenticated

⛔ THE RULE IS NOT "writable needs a key". Three real configurations have to keep
working, and §2-§4 are each one of them:
  the public demo   --read-only on 0.0.0.0, no key, deliberate
  the local sidecar --writable, no key, loopback
  an authenticated  --writable on 0.0.0.0 WITH a key
A test that only checked the refusal would pass against a rule that bricks all three.

Measures the real process: exit codes from an actual spawn, and for the bind, an
actual socket on the machine's LAN address. ~20 lines of output.
"""
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RESULTS = []
REFUSE_CODE = 5


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


def boot(args, env_key=None, port=8297, settle=None):
    """Spawn the server; return its exit code, or 'RUNNING' if it stayed up."""
    env = dict(os.environ)
    env.pop("MEMBOT_API_KEY", None)
    if env_key:
        env["MEMBOT_API_KEY"] = env_key
    cmd = [sys.executable, os.path.join(HERE, "membot_server.py"),
           "--transport", "http", "--port", str(port), *args]
    p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True)
    try:
        out = p.communicate(timeout=settle or 35)[0]
        return p.returncode, out
    except subprocess.TimeoutExpired:
        p.terminate()
        try:
            out = p.communicate(timeout=15)[0]
        except subprocess.TimeoutExpired:
            p.kill()
            out = ""
        return "RUNNING", out


print("\n§1 · the default bind is loopback")
import membot_server  # noqa: E402  (import after sys.path setup)
src = open(os.path.join(HERE, "membot_server.py"), encoding="utf-8").read()
check("--host default is 127.0.0.1", 'parser.add_argument("--host", default="127.0.0.1"' in src)
check("0.0.0.0 is no longer a default anywhere in argparse",
      'default="0.0.0.0"' not in src)

# Printed strings stay cp1252-safe: stdout redirects to a file on this box and a
# UnicodeEncodeError would fail the run for a reason unrelated to the test.
print("\n§2 · THE DANGEROUS CONFIG MUST BE REFUSED")
rc, out = boot(["--host", "0.0.0.0", "--writable"])
check(f"writable + 0.0.0.0 + no key refuses (exit {REFUSE_CODE})", rc == REFUSE_CODE,
      f"exit={rc}")
check("the refusal says why", "REFUSING TO START" in out, out.strip().splitlines()[-1][:60] if out.strip() else "")
check("and names all three ways out",
      "MEMBOT_API_KEY" in out and "127.0.0.1" in out and "--writable" in out)

print("\n§3 · the three legitimate configs must KEEP WORKING")
rc2, _ = boot(["--host", "127.0.0.1", "--writable"], settle=25)
check("local sidecar: writable, no key, loopback", rc2 != REFUSE_CODE, f"exit={rc2}")

rc3, _ = boot(["--host", "0.0.0.0", "--read-only"], settle=25)
check("public demo: read-only, no key, 0.0.0.0", rc3 != REFUSE_CODE, f"exit={rc3}")

rc4, _ = boot(["--host", "0.0.0.0", "--writable"], env_key="s3cret", settle=25)
check("authenticated: writable, 0.0.0.0, WITH a key", rc4 != REFUSE_CODE, f"exit={rc4}")

print("\n§4 · CONTROL — the refusal is caused by the combination, not by one flag")
# Each of the three above differs from §2 by exactly ONE thing and each one boots.
# If any of them had also been refused, §2 would prove nothing about the rule.
check("exactly one arm refused, and it is the dangerous one",
      rc == REFUSE_CODE and rc2 != REFUSE_CODE and rc3 != REFUSE_CODE and rc4 != REFUSE_CODE,
      f"{rc} / {rc2} / {rc3} / {rc4}")

print("\n§5 · the loopback bind is real, not just a logged claim")
# Start on the default host and try to reach it on this machine's LAN address.
# A log line saying "bound to loopback" is not evidence; a refused connect is.
lan = None
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    lan = s.getsockname()[0]
    s.close()
except Exception:
    pass
if not lan or lan.startswith("127."):
    check("LAN address available to test against", False, f"got {lan}")
else:
    port = 8296
    p = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "membot_server.py"),
         "--transport", "http", "--port", str(port), "--writable"],
        cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL)
    try:
        reachable_local = False
        for _ in range(25):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=2):
                    reachable_local = True
                    break
            except OSError:
                time.sleep(2)
        check("reachable on 127.0.0.1", reachable_local)
        off_host = False
        try:
            with socket.create_connection((lan, port), timeout=3):
                off_host = True
        except OSError:
            pass
        check(f"NOT reachable on the LAN address {lan}", not off_host,
              "connect refused" if not off_host else "STILL EXPOSED")
    finally:
        p.terminate()
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()

passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print("\n" + "=" * 62)
print(f"VERDICT: {passed}/{total} checks passed" + ("" if passed == total else "  <<< FAILURES"))
for label, ok, detail in RESULTS:
    if not ok:
        print(f"   FAILED: {label}   {detail}")
print("=" * 62)
sys.exit(0 if passed == total else 1)
