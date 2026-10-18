"""--mount must act on its integrity verdict. 2026-10-18.

Before today the boot path computed `ok, verify_msg = verify_manifest(...)` and then
never read `ok` — it logged integrity=<msg> and mounted regardless. Andy: "if it
doesn't see if it's ok what's the point of checking in the first place?"

Spawns the real server as a subprocess so the exit CODE is the measurement, not an
assertion about a function. stdio transport is used because a failed check exits
before mcp.run() is reached; the pass case would block, so it is checked by manifest
verdict rather than by booting (see §3).

Output is 5 lines. No GPU, no network.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from cart_io import verify_manifest, load_cartridge_safe  # noqa: E402

RESULTS = []


def check(label, cond, detail=""):
    RESULTS.append((label, bool(cond), detail))
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + (f"   [{detail}]" if detail else ""))


def boot(cart_name, extra=(), timeout=180):
    """Start the server with --mount and return its exit code.

    A clean boot would block in mcp.run(), so a timeout here means "did not refuse",
    which is itself a verdict rather than an error.
    """
    cmd = [sys.executable, os.path.join(HERE, "membot_server.py"),
           "--transport", "stdio", "--mount", cart_name, *extra]
    try:
        p = subprocess.run(cmd, cwd=HERE, capture_output=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        return p.returncode
    except subprocess.TimeoutExpired:
        return "RUNNING"


# membot resolves --mount against its own cartridges/ dir, so the fixture has to live
# there. Named with a prefix no real cart uses, and removed in the finally block.
CART_DIR = os.path.join(HERE, "cartridges")
os.makedirs(CART_DIR, exist_ok=True)
NAME = "zzz-boot-integrity-fixture"
CART = os.path.join(CART_DIR, f"{NAME}.cart.npz")
MANIFEST = CART.rsplit(".", 1)[0] + "_manifest.json"
created = []

try:
    import membot_server as M

    rng = np.random.default_rng(11)
    emb = rng.standard_normal((25, 768)).astype(np.float32)
    texts = [f"boot fixture passage {i}" for i in range(25)]
    M.save_as_npz(CART, emb, texts)
    M.save_manifest(CART, emb, 25)
    created = [CART, MANIFEST]

    print("\n§1 · a manifest that MATCHES must verify")
    d = load_cartridge_safe(CART)
    ok, msg = verify_manifest(CART, d["embeddings"], len(d["texts"]))
    check("good cart verifies", ok, msg)

    print("\n§2 · a manifest that does NOT match must refuse the boot")
    # Rewrite the cart with different content, leaving the OLD manifest in place --
    # exactly the stale/tampered case, and a real fingerprint mismatch rather than a
    # monkeypatched verdict.
    emb2 = rng.standard_normal((30, 768)).astype(np.float32)
    M.save_as_npz(CART, emb2, [f"different passage {i}" for i in range(30)])
    d2 = load_cartridge_safe(CART)
    ok2, msg2 = verify_manifest(CART, d2["embeddings"], len(d2["texts"]))
    check("fixture now genuinely fails verification", not ok2, msg2[:48])

    rc = boot(NAME)
    check("server REFUSES to start (exit 3)", rc == 3, f"exit={rc}")

    print("\n§3 · the escape hatch must still start it")
    # ⚠ "booted" is exit 0 here, not a hang. stdin is at EOF, so a stdio MCP server
    # reaches mcp.run(), finds no input and returns cleanly. The discriminating fact
    # is only that it did NOT take the exit-3 refusal path, which §2 just showed this
    # same cart does take without the flag.
    rc2 = boot(NAME, extra=["--allow-unverified-mount"], timeout=60)
    check("--allow-unverified-mount does NOT refuse", rc2 != 3, f"exit={rc2}")
    check("the flag is what changed the outcome", rc2 != 3 and RESULTS[2][1],
          "same cart: exit 3 without, past-the-gate with")

finally:
    for p in created:
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
    # save_as_npz may emit sidecars; sweep anything carrying the fixture prefix.
    for f in os.listdir(CART_DIR):
        if f.startswith(NAME):
            try:
                target = os.path.join(CART_DIR, f)
                shutil.rmtree(target) if os.path.isdir(target) else os.remove(target)
            except OSError:
                pass

passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print("\n" + "=" * 62)
print(f"VERDICT: {passed}/{total} checks passed" + ("" if passed == total else "  <<< FAILURES"))
for label, ok, detail in RESULTS:
    if not ok:
        print(f"   FAILED: {label}   {detail}")
print("=" * 62)
sys.exit(0 if passed == total else 1)
