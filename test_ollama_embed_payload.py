#!/usr/bin/env python3
"""The Ollama embed request sends num_gpu and keep_alive ONLY when configured.

Adapted from dp-web4/membot (c31c3c6, 75cc47c), which pinned GPT's review on their #3.
⚠ ONE ASSERTION IS DELIBERATELY INVERTED. Theirs required `options.num_gpu == 0` by
default, forcing CPU, because on their boxes the GPU belongs to a being. Membot ships to
hosts whose hardware we do not know, so our default sends NEITHER FIELD and lets Ollama
decide — the same "unset means not sent" rule their own later keep_alive commit adopted.
A test that asserted their default would fail here for the right reason, so it asserts
ours and says why.

No network. urllib.request.urlopen is replaced before _embed_via_ollama runs, and
OLLAMA_HOST points at a closed port, so a missed patch fails loudly instead of reaching a
live ollama or a live membot. The env var is read at import, so every environment runs in
a fresh interpreter.

Run from membot/:  python test_ollama_embed_payload.py
"""
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Locate the dir that actually holds membot_server.py rather than assuming this file's
# depth — this test lives beside the other script-style tests, not in tests/, because its
# module-level sys.exit() would abort pytest collection for the whole suite.
_MEMBOT_DIR = _HERE if os.path.exists(os.path.join(_HERE, "membot_server.py")) \
    else os.path.dirname(_HERE)

failures = []


def check(label, ok, detail=""):
    if not ok:
        failures.append(label)
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if detail and not ok else ""))


PROBE = r"""
import io, json, sys, urllib.request
sys.path.insert(0, %r)
import membot_server as m

captured = []

class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False

def fake_urlopen(req, timeout=None):
    captured.append({"url": req.full_url, "body": json.loads(req.data)})
    return _Resp(json.dumps({"embeddings": [[0.0, 1.0, 2.0]]}).encode())

urllib.request.urlopen = fake_urlopen
vec = m._embed_via_ollama("hello")
print(json.dumps({"captured": captured, "vec_len": int(vec.shape[0])}))
""" % _MEMBOT_DIR


def probe(**env_overrides):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("MEMBOT_") and k != "OLLAMA_HOST"}
    env["OLLAMA_HOST"] = "http://127.0.0.1:9"  # discard port: never a live server
    env.update(env_overrides)
    r = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True,
                       cwd=_MEMBOT_DIR, timeout=300)
    if r.returncode != 0:
        raise SystemExit(f"probe failed rc={r.returncode}\n{r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


# 1. Default: let Ollama decide. (INVERTED from dp's test — see the module docstring.)
out = probe()
check("default: exactly one request sent", len(out["captured"]) == 1, out)
body = out["captured"][0]["body"]
check("default: NO num_gpu sent, so Ollama's own placement applies",
      "options" not in body or "num_gpu" not in body.get("options", {}), body)
check("default: request goes to /api/embed on OLLAMA_HOST",
      out["captured"][0]["url"] == "http://127.0.0.1:9/api/embed", out["captured"][0]["url"])
check("default: model and input still carried",
      body.get("model") and body.get("input") == "hello", body)
check("default: embedding parsed", out["vec_len"] == 3, out)

# 2. Override: a host whose accelerator is genuinely free.
out = probe(MEMBOT_OLLAMA_EMBED_NUM_GPU="99")
body = out["captured"][0]["body"]
check("override 99: options.num_gpu == 99",
      body.get("options", {}).get("num_gpu") == 99, body)
check("override 99: sent as an int, not a string",
      isinstance(body.get("options", {}).get("num_gpu"), int), body)

# 3. Explicit 0 is the same as the default.
out = probe(MEMBOT_OLLAMA_EMBED_NUM_GPU="0")
check("explicit 0: options.num_gpu == 0",
      out["captured"][0]["body"].get("options", {}).get("num_gpu") == 0, out)

# 4. keep_alive: absent by default (Ollama's own default applies), passed through when set.
out = probe()
check("default: no keep_alive sent", "keep_alive" not in out["captured"][0]["body"], out)
out = probe(MEMBOT_OLLAMA_EMBED_KEEP_ALIVE="0")
check("keep_alive 0: sent as \"0\" (unload right after the embed)",
      out["captured"][0]["body"].get("keep_alive") == "0", out)
out = probe(MEMBOT_OLLAMA_EMBED_KEEP_ALIVE="30s")
check("keep_alive 30s: passed through", out["captured"][0]["body"].get("keep_alive") == "30s", out)
check("keep_alive alone does not start sending num_gpu",
      "num_gpu" not in out["captured"][0]["body"].get("options", {}), out)

# 5. The two knobs are independent and compose.
out = probe(MEMBOT_OLLAMA_EMBED_NUM_GPU="0", MEMBOT_OLLAMA_EMBED_KEEP_ALIVE="0")
b = out["captured"][0]["body"]
check("both set: num_gpu 0 AND keep_alive 0 both present",
      b.get("options", {}).get("num_gpu") == 0 and b.get("keep_alive") == "0", b)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all passed")
