"""No undefined names in the modules that serve requests. 2026-11-15.

⛔ WHY THIS EXISTS. On 2026-11-15, consolidating six REST routes onto one seat
helper deleted `user_id = _get_user_id_from_request(request)` from rest_search,
which left `user_id` UNDEFINED inside this line:

    if state.get("mempack_id") and state.get("owner_id") and user_id and ...

That is the cross-user Mempack ownership refusal. Every test in the repo passed,
because `state.get("mempack_id")` is falsy for a knowledge cart and `and`
short-circuits before reaching the undefined name.

  => THE ONLY PATH THAT WOULD HAVE RAISED IS THE ONE THE CHECK EXISTS TO REFUSE.
     A NameError in a security check, invisible to every happy-path test, because
     the guard's own short-circuit hid it.

⭐ AND A HAND-ROLLED SCANNER IS THE WRONG TOOL. The instinct was to write an AST
walk for this one shape. A linter already does it for every shape, across every
function, in under a second -- and the point is the CLASS, not the instance. The
bespoke scanner would have been one more check matching one pattern in one place,
which is the failure mode this project keeps paying for.

ⓘ Unused imports and f-strings without placeholders are NOT failed here. They are
style, they are already numerous, and a test that fails on them would be switched
off -- at which point it stops catching undefined names too.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Modules that serve a request or decide an identity. A NameError here is a 500 at
# best and a skipped security check at worst.
FILES = [
    "membot_server.py",
    "membot_rest_bridge.py",
    "seat_identity.py",
    "service_identity.py",
    "multi_cart.py",
    "cart_io.py",
    "redact.py",
    "journal.py",
    "note_store.py",
]

RESULTS = []


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


present = [f for f in FILES if os.path.exists(os.path.join(ROOT, f))]
missing = [f for f in FILES if f not in present]

print(f"\nUN1 - pyflakes over {len(present)} request-serving modules")
check("the module list resolves to real files", not missing,
      f"missing: {', '.join(missing)}" if missing else "")

try:
    proc = subprocess.run([sys.executable, "-m", "pyflakes"] + present,
                          cwd=ROOT, capture_output=True, text=True, timeout=180)
    ran = True
except (OSError, subprocess.SubprocessError) as e:
    ran = False
    print(f"  ----  pyflakes could not run   [{type(e).__name__}]")

if not ran:
    print("  ----  INCONCLUSIVE: install pyflakes to enable this guard.")
    print("  ----  A guard that cannot run is not a guard; this does NOT pass.")
    sys.exit(2)

out = (proc.stdout or "") + (proc.stderr or "")
lines = [l for l in out.splitlines() if l.strip()]
undefined = [l for l in lines if "undefined name" in l]

check("pyflakes produced output to judge", bool(lines) or proc.returncode == 0,
      f"{len(lines)} lines, exit {proc.returncode}")
check("NO undefined names", not undefined,
      "; ".join(l.strip()[-90:] for l in undefined[:4]) if undefined else
      f"{len(present)} modules clean")

print("\nUN2 - POSITIVE CONTROL: pyflakes really does catch this shape")
# A clean run is indistinguishable from a run that checked nothing. So hand it the
# exact shape that got through -- a name used under a short-circuit -- and require
# that it is reported.
import tempfile  # noqa: E402

probe = (
    "def f(state, request):\n"
    "    seat = request\n"
    "    if state.get('mempack_id') and state.get('owner_id') and user_id:\n"
    "        return 'denied'\n"
    "    return seat\n"
)
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "probe_undefined.py")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(probe)
    r = subprocess.run([sys.executable, "-m", "pyflakes", p],
                       capture_output=True, text=True, timeout=60)
    probe_out = (r.stdout or "") + (r.stderr or "")
    check("the probe's undefined name IS reported",
          "undefined name" in probe_out and "user_id" in probe_out,
          probe_out.strip().splitlines()[0][-70:] if probe_out.strip() else "no output")
    check("and it is the short-circuited shape that actually shipped",
          "user_id" in probe and " and " in probe)

passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print("\n" + "=" * 68)
print(f"VERDICT: {passed}/{total} checks passed" + ("" if passed == total else "   <<< FAILURES"))
for label, ok, detail in RESULTS:
    if not ok:
        print(f"   FAILED: {label}   {detail}")
print("=" * 68)
sys.exit(0 if passed == total else 1)
