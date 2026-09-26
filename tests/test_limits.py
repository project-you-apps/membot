#!/usr/bin/env python3
"""Request-admission limits: the public default stays bounded; a local box opts out explicitly.

Pins GPT's HOLD on membot #6 (2026-09-25): the loopback exemption, the opt-in that controls it,
non-loopback staying throttled, env overrides for text/query/rate/window, and loopback
spellings that do not widen to arbitrary hosts.

One change from what the HOLD asked to pin. The exemption is opt-in
(MEMBOT_RATE_LIMIT_LOOPBACK_EXEMPT=1), not the default. README's "Behind a Reverse Proxy" puts
nginx on the same host with proxy_pass http://127.0.0.1:8000, so every public request arrives
from loopback. A default exemption would have disabled the limiter for exactly the public
instance that "must remain bounded by default". Case 1 is that proxy case.

The limits are read at import, so every environment runs in a fresh interpreter.

Run from membot/:  python tests/test_limits.py
"""
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MEMBOT_DIR = os.path.dirname(_HERE)

failures = []


def check(label, ok, detail=""):
    if not ok:
        failures.append(label)
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if detail and not ok else ""))


PROBE = r"""
import json, sys
sys.path.insert(0, %r)
import membot_server as m
ips = ["127.0.0.1", "::1", "127.0.0.2", "::ffff:127.0.0.1",
       "10.0.0.5", "192.168.1.20", "2001:db8::1", "0.0.0.0",
       "localhost", "unknown", "", "127.0.0.1.evil.example"]
applies = {ip: m._rate_limit_applies(ip) for ip in ips}
# Drive the real limiter for one non-loopback client up to and past RATE_LIMIT.
seq = [m._check_rate_limit("10.0.0.5") for _ in range(m.RATE_LIMIT + 1)]
print(json.dumps({
    "applies": applies, "seq": seq,
    "RATE_LIMIT": m.RATE_LIMIT, "RATE_WINDOW_SEC": m.RATE_WINDOW_SEC,
    "MAX_TEXT_LENGTH": m.MAX_TEXT_LENGTH, "MAX_QUERY_LENGTH": m.MAX_QUERY_LENGTH,
    "MAX_ENTRIES": m.MAX_ENTRIES, "EXEMPT": m.RATE_LIMIT_LOOPBACK_EXEMPT,
}))
""" % _MEMBOT_DIR


def probe(**env_overrides):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MEMBOT_")}
    env.update(env_overrides)
    r = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True,
                       cwd=_MEMBOT_DIR, timeout=300)
    if r.returncode != 0:
        raise SystemExit(f"probe failed rc={r.returncode}\n{r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


LOOPBACK = ["127.0.0.1", "::1", "127.0.0.2", "::ffff:127.0.0.1"]
NOT_LOOPBACK = ["10.0.0.5", "192.168.1.20", "2001:db8::1", "0.0.0.0",
                "localhost", "unknown", "", "127.0.0.1.evil.example"]

# 1. Default: EVERY client is limited, loopback included (the same-host reverse-proxy case).
d = probe()
check("1. default: loopback is rate-limited (a proxied public instance stays bounded)",
      all(d["applies"][ip] for ip in LOOPBACK), f"applies={d['applies']}")
check("1b. default: non-loopback is rate-limited", all(d["applies"][ip] for ip in NOT_LOOPBACK))
check("1c. default limits are the public-server values",
      (d["RATE_LIMIT"], d["RATE_WINDOW_SEC"], d["MAX_TEXT_LENGTH"], d["MAX_QUERY_LENGTH"],
       d["MAX_ENTRIES"], d["EXEMPT"]) == (60, 60, 10_000, 2_000, 3_000_000, False), str(d))

# 2. Opt-in: loopback ADDRESSES are exempt; nothing else is.
x = probe(MEMBOT_RATE_LIMIT_LOOPBACK_EXEMPT="1")
check("2. opt-in: every loopback spelling is exempt (127/8, ::1, IPv4-mapped)",
      not any(x["applies"][ip] for ip in LOOPBACK), f"applies={x['applies']}")
check("2b. opt-in: non-loopback, hostnames and junk are STILL limited",
      all(x["applies"][ip] for ip in NOT_LOOPBACK),
      f"widened to: {[ip for ip in NOT_LOOPBACK if not x['applies'][ip]]}")
check("2c. only the exact value '1' opts in", not probe(MEMBOT_RATE_LIMIT_LOOPBACK_EXEMPT="true")["EXEMPT"]
      and not probe(MEMBOT_RATE_LIMIT_LOOPBACK_EXEMPT="0")["EXEMPT"])

# 3. Env overrides are parsed and applied, and the limiter enforces the overridden rate.
o = probe(MEMBOT_RATE_LIMIT="3", MEMBOT_RATE_WINDOW_SEC="7", MEMBOT_MAX_TEXT_LENGTH="400000",
          MEMBOT_MAX_QUERY_LENGTH="8000", MEMBOT_MAX_ENTRIES="12345")
check("3. env overrides applied (rate, window, text, query, entries)",
      (o["RATE_LIMIT"], o["RATE_WINDOW_SEC"], o["MAX_TEXT_LENGTH"], o["MAX_QUERY_LENGTH"],
       o["MAX_ENTRIES"]) == (3, 7, 400_000, 8_000, 12_345), str(o))
check("3b. a non-loopback client is throttled after RATE_LIMIT requests",
      o["seq"] == [True, True, True, False], f"seq={o['seq']}")
check("3c. default limiter throttles at 60", d["seq"] == [True] * 60 + [False],
      f"allowed={d['seq'].count(True)}")

# 4. The middleware routes admission through the one decision, and the size checks read the
#    overridable constants (source pins: the middleware is not drivable without a live server).
src = open(os.path.join(_MEMBOT_DIR, "membot_server.py"), encoding="utf-8").read()
check("4. middleware admits through _rate_limit_applies",
      "if _rate_limit_applies(client_ip) and not _check_rate_limit(client_ip):" in src)
check("4b. no second, string-matched loopback set survives", "_LOOPBACK = {" not in src)

print(f"\nfailures={len(failures)}")
for f in failures:
    print(f"  - {f}")
sys.exit(1 if failures else 0)
