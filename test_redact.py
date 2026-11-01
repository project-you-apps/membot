#!/usr/bin/env python3
"""Ingest redaction, tiers 1 and 2: every check here fails if the filter is disabled.

HOUSE RULE THIS FILE IS WRITTEN AGAINST: existence is not operation. A check that
passes in both states is not a check. So there is no test here that merely calls a
function and asserts it returned something: every tier-1 check names the rule that
must have matched, every tier-2 check asserts the kind AND that the value is gone from
the output, and the negative checks assert byte-identical passthrough rather than
"no exception".

THE TEST THAT IS THE WHOLE POINT is §4, the container tests. On 2026-09-15 real OAuth
tokens reached the corpus and were later found inside a PICKLE, where every text-level
grep reported only "Binary file matches". A filter that understands transcripts misses
that; a filter that understands bytes does not. §4 also pins the LIMIT in the other
direction -- compression destroys the byte sequence, so container independence is not
codec independence -- because a limit nobody wrote down gets claimed away in a sales
conversation.

⛔ EVERY CREDENTIAL IN THIS FILE IS FAKE AND CONSTRUCTED HERE. Nothing was copied from
this repo, its history, a transcript, or any live system. The literal string FAKE
appears inside each one so that a future grep for leaked secrets can clear them at a
glance. The GitHub checksum is computed by a helper in this file rather than imported
from redact.py, so that a change to redact.py's own checksum code cannot make this
test agree with it by construction.

No network, no GPU, no imports outside the stdlib plus redact.py.

Run from membot/:  python test_redact.py > redact-test.log 2>&1   (then grep it)
"""
import gzip
import os
import pickle
import shutil
import sys
import tempfile
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Test output is redirected to a file per CLAUDE_DO_NOT_RUN_TESTS.md, and on Windows a
# redirected stdout defaults to cp1252, which cannot encode the markers used in the
# labels below. A test that dies on its own PASS line teaches nothing.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover
    pass

import redact                                            # noqa: E402

failures = []


def check(label, ok, detail=""):
    if not ok:
        failures.append(label)
    print(f"{'PASS' if ok else 'FAIL'}  {label}"
          + (f"\n        {detail}" if detail and not ok else ""))


# ---------------------------------------------------------------------------
# fake credentials, constructed here
# ---------------------------------------------------------------------------

_B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def github_checksum(secret30: str) -> str:
    """Independent reimplementation of the crc32-base62 checksum, so that this test
    does not borrow redact.py's arithmetic to agree with redact.py."""
    n = zlib.crc32(secret30.encode("ascii"))
    out = ""
    while n:
        out = _B62[n % 62] + out
        n //= 62
    return out.rjust(6, "0")


_GH_SECRET = "FAKEgithubTOKEN000000000000000"       # 30 chars
assert len(_GH_SECRET) == 30
GITHUB = "ghp_" + _GH_SECRET + github_checksum(_GH_SECRET)
# Same shape, checksum deliberately broken. This is the pair that proves the checksum
# is being VERIFIED rather than merely present.
_bad = github_checksum(_GH_SECRET)
_bad = ("Z" if _bad[0] != "Z" else "Y") + _bad[1:]
GITHUB_BADSUM = "ghp_" + _GH_SECRET + _bad

OAUTH = "sk-ant-oat01-" + "F" * 80 + "AA"
ANTHROPIC = "sk-ant-api03-" + "F" * 90 + "AA"
OPENAI = "sk-" + "FAKEopenaiKEY" + "0" * 22
OPENAI_PROJ = "sk-proj-" + "FAKEproj" + "0" * 40
GITHUB_PAT = "github_pat_" + "11FAKEFAKEFAKEFAKE0000_" + "F" * 59
AWS = "AKIA" + "IOSFODNN7EXAMPLE"                   # AWS's own published example value
AWS_TEMP = "ASIA" + "FAKEFAKEFAKEFAKE"
GOOGLE = "AIza" + "FAKEgoogleKEY" + "0" * 22
SLACK = "xoxb-" + "0000000000-0000000000-FAKEFAKEFAKEFAKEFAKE"
STRIPE = "sk_live_" + "FAKEstripe0000000000000"

_PEM_BODY = (
    "Tk9UQUtFWVRISVNJU0ZBS0VQQURESU5HRk9SQVRFU1RPTkxZQUFBQUFBQUFBQUFB\n"
    "QkJCQkJCQkJCQkJDQ0NDQ0NDQ0NDQ0NEREREREREREREREVFRUVFRUVFRUVFRkZG\n"
    "RkZGRkZGRkZHR0dHR0dHR0dHSEhISEhISEhISEhJSUlJSUlJSUlJSkpKSkpKSkpKSg==\n"
)
PEM = ("-----BEGIN RSA PRIVATE KEY-----\n" + _PEM_BODY
       + "-----END RSA PRIVATE KEY-----")

# A JWT whose header really does base64url-decode to JSON carrying `alg` -- that is
# what makes a JWT DETECTABLE without any key.
_JWT_HEADER = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"        # {"alg":"HS256","typ":"JWT"}
_JWT_PAYLOAD = "eyJzdWIiOiJGQUtFLU5PVC1BLVJFQUwtVE9LRU4ifQ"  # {"sub":"FAKE-NOT-A-REA..}
JWT = _JWT_HEADER + "." + _JWT_PAYLOAD + ".FAKEsignatureNOTvalid00000"

ALL_FAKE = (GITHUB, OAUTH, ANTHROPIC, OPENAI, OPENAI_PROJ, GITHUB_PAT, AWS, AWS_TEMP,
            GOOGLE, SLACK, STRIPE, PEM, JWT)


def kinds_in(data_or_text):
    if isinstance(data_or_text, str):
        data_or_text = data_or_text.encode("utf-8")
    return [f["kind"] for f in redact.scan_bytes(data_or_text)]


def one_finding(text):
    found = redact.scan_bytes(text.encode("utf-8"))
    return found[0] if len(found) == 1 else None


# ---------------------------------------------------------------------------
print("\n== 1 · tier 1, paths: every listed credential surface flags ==")
# ---------------------------------------------------------------------------

PATH_CASES = (
    ("C:/Users/andyg/.claude/.credentials.json", "path:credentials"),
    ("/home/andy/.env", "path:dotenv"),
    ("deploy/.env.local", "path:dotenv"),
    ("/etc/ssl/certs/server.pem", "path:pem"),
    ("/home/andy/.ssh/id_rsa", "path:id_rsa"),
    ("/home/andy/.aws/config", "path:aws-dir"),
    ("/root/.ssh/known_hosts", "path:ssh-dir"),
    ("/home/andy/.netrc", "path:netrc"),
    ("/var/lib/postgresql/.pgpass", "path:pgpass"),
    # Windows separators must normalize, or tier 1 is blind on the machine it runs on.
    ("C:\\Users\\andyg\\.ssh\\id_rsa", "path:id_rsa"),
)
for path, want_rule in PATH_CASES:
    verdict = redact.classify_origin(path)
    names = [r["rule"] for r in verdict["rules"]]
    check(f"tier1 path {path!r} -> {want_rule}",
          verdict["flagged"] and want_rule in names,
          f"got flagged={verdict['flagged']} rules={names}")

# The 2026-09-15 origin, specifically. This is the one case that had to work.
v = redact.classify_origin("/home/andy/.claude/.credentials.json")
check("tier1 catches the actual 2026-09-15 origin outright",
      v["flagged"] and "path:credentials" in [r["rule"] for r in v["rules"]],
      str(v))

# ---------------------------------------------------------------------------
print("\n== 2 · tier 1, commands: every listed credential-emitting command flags ==")
# ---------------------------------------------------------------------------

COMMAND_CASES = (
    ("env", "command:env"),
    ("printenv", "command:printenv"),
    ("gh auth token", "command:gh-auth-token"),
    ("aws configure get aws_secret_access_key", "command:aws-configure"),
    ("docker login -u andy --password-stdin", "command:docker-login"),
    # A command is matched per pipeline segment, not only at position 0.
    ("cat notes.txt | printenv", "command:printenv"),
    # ... and by program name, so an absolute path still resolves.
    ("/usr/bin/env", "command:env"),
    # ... and leading VAR=value / sudo noise does not hide the program.
    ("sudo docker login registry.example.com", "command:docker-login"),
    ("FOO=1 printenv", "command:printenv"),
)
for cmd, want_rule in COMMAND_CASES:
    verdict = redact.classify_origin(cmd)
    names = [r["rule"] for r in verdict["rules"]]
    check(f"tier1 command {cmd!r} -> {want_rule}",
          verdict["flagged"] and want_rule in names,
          f"got flagged={verdict['flagged']} rules={names}")

# ---------------------------------------------------------------------------
print("\n== 3 · tier 1 does NOT flag everything ==")
# ---------------------------------------------------------------------------
# A filter that flags all input passes every positive test above and is worthless.
# Two of these are adversarial on purpose: 'environment.py' contains 'env', and
# '.environments/' contains the literal '.env', which is what the boundary rule in
# _PATH_RULES exists to reject.

INNOCUOUS = (
    "src/components/Button.tsx",
    "config/environment.py",
    "src/.environments/readme.md",
    "docs/journal/2026-09-25-what-it-saw-eight-steps-ago.md",
    "membot/proposal_store.py",
    "python manage.py runserver",
    "git status",
    "data/env/notes.txt",
)
for path in INNOCUOUS:
    verdict = redact.classify_origin(path)
    check(f"tier1 quiet on {path!r}", not verdict["flagged"],
          f"rules={[r['rule'] for r in verdict['rules']]}")

check("tier1 quiet on an empty origin", not redact.classify_origin("")["flagged"])

# ---------------------------------------------------------------------------
print("\n== 4 · THE CONTAINER TESTS -- why tier 2 is bytes and not text ==")
# ---------------------------------------------------------------------------

def _is_undecodable(data: bytes) -> bool:
    try:
        data.decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True


blob = pickle.dumps({"accessToken": OAUTH, "note": "written by a tool result"})
check("pickle bytes do not decode as UTF-8, so a text-level filter never looks",
      _is_undecodable(blob), repr(blob[:16]))
check("tier 2 finds the token INSIDE A PICKLE",
      "oauth-token" in kinds_in(blob), str(kinds_in(blob)))

wrapped = b"\x00\x8f\xff\xfe" + OAUTH.encode("ascii") + b"\xfe\x00\x9c" * 4
check("tier 2 finds the token wrapped in arbitrary binary bytes",
      "oauth-token" in kinds_in(wrapped), str(kinds_in(wrapped)))

nested = pickle.dumps({"outer": pickle.dumps({"k": PEM}), "n": 3})
check("tier 2 finds a PEM private key inside a nested pickle",
      "private-key-pem" in kinds_in(nested), str(kinds_in(nested)))

# ⛔ THE LIMIT, asserted so it cannot be claimed away later: container independence is
# not codec independence. Compression and encryption destroy the byte sequence, so the
# ingest path must decompress BEFORE calling scan_bytes.
prose = ("deploy log, repeated for compressibility. " * 40) + OAUTH + (" tail" * 40)
squashed = gzip.compress(prose.encode("utf-8"))
check("DOCUMENTED LIMIT: gzip hides the token from a byte scan (decompress first)",
      "oauth-token" not in kinds_in(squashed), str(kinds_in(squashed)))
check("...and the same bytes uncompressed are found",
      "oauth-token" in kinds_in(gzip.decompress(squashed)))

# ---------------------------------------------------------------------------
print("\n== 5 · tier 2 catches each listed prefix format ==")
# ---------------------------------------------------------------------------

FORMAT_CASES = (
    (OAUTH, "oauth-token"),
    (ANTHROPIC, "anthropic-api-key"),
    (OPENAI, "openai-api-key"),
    (OPENAI_PROJ, "openai-api-key"),
    (GITHUB, "github-token"),
    (GITHUB_PAT, "github-pat"),
    (AWS, "aws-access-key-id"),
    (AWS_TEMP, "aws-access-key-id"),
    (GOOGLE, "google-api-key"),
    (SLACK, "slack-token"),
    (STRIPE, "stripe-secret-key"),
    (PEM, "private-key-pem"),
    (JWT, "jwt"),
)
for value, want_kind in FORMAT_CASES:
    passage = "config says " + value + " end of line"
    found = redact.scan_bytes(passage.encode("utf-8"))
    got = [f["kind"] for f in found]
    check(f"tier2 {want_kind}", want_kind in got, f"got {got}")

# One finding, not two: sk-ant- must not also report as a bare sk- key, or the
# redacted output nests markers inside markers.
f = one_finding("token=" + OAUTH)
check("overlapping prefixes resolve to a single longest finding",
      f is not None and f["kind"] == "oauth-token",
      str(kinds_in("token=" + OAUTH)))

# ---------------------------------------------------------------------------
print("\n== 6 · detected vs suspected is a real distinction, not a label ==")
# ---------------------------------------------------------------------------

f = one_finding("token=" + GITHUB)
check("a VALID github checksum reports DETECTED",
      f is not None and f["confidence"] == redact.DETECTED
      and f["verified_by"] == "crc32-base62", str(f))

f = one_finding("token=" + GITHUB_BADSUM)
check("the SAME SHAPE with a broken checksum is downgraded to SUSPECTED",
      f is not None and f["kind"] == "github-token"
      and f["confidence"] == redact.SUSPECTED, str(f))

f = one_finding("key=" + AWS)
check("an unchecksummable format (AWS key id) reports SUSPECTED",
      f is not None and f["confidence"] == redact.SUSPECTED, str(f))

f = one_finding("auth: " + JWT)
check("a JWT whose header decodes to JSON with `alg` reports DETECTED",
      f is not None and f["confidence"] == redact.DETECTED
      and f["verified_by"] == "base64url-json-header", str(f))

f = one_finding("key:\n" + PEM)
check("a PEM block with a matching footer and base64 body reports DETECTED",
      f is not None and f["confidence"] == redact.DETECTED, str(f))

# Three dotted base64url runs that are not a JWT must not be reported at all.
check("`eyJ` in a non-JWT context is not reported (no guessing)",
      kinds_in("see eyJhbGci.notarealjwt.atall here") == [],
      str(kinds_in("see eyJhbGci.notarealjwt.atall here")))

# A longer uppercase identifier that merely begins with AKIA is not a key id.
check("a longer run beginning AKIA is rejected by the exact-length boundary",
      kinds_in("AKIAIOSFODNN7EXAMPLEEXTRA") == [],
      str(kinds_in("AKIAIOSFODNN7EXAMPLEEXTRA")))

# ---------------------------------------------------------------------------
print("\n== 7 · redaction keeps the passage and replaces only the value ==")
# ---------------------------------------------------------------------------

PREFIX = 'We set "accessToken": "'
SUFFIX = '" and deploys started working.'
out, found = redact.redact_text(PREFIX + OAUTH + SUFFIX)
fp = redact.fingerprint(OAUTH)
expected = PREFIX + f"[REDACTED:oauth-token:sha256:{fp}]" + SUFFIX
check("the value is gone from the redacted text", OAUTH not in out, out[:120])
check("the surrounding passage text survives verbatim",
      out.startswith(PREFIX) and out.endswith(SUFFIX) and "[REDACTED:" in out,
      out[:200])
check("the marker is exactly the shape spec §4 fixes", out == expected, out)
check("the finding carries the fingerprint but never the value",
      len(found) == 1 and found[0]["fingerprint"] == fp
      and not any(OAUTH in str(v) for v in found[0].values()), str(found))

many = "a " + OAUTH + " b " + GITHUB + " c " + AWS + " d"
out, found = redact.redact_text(many)
check("several secrets in one passage are all redacted",
      not any(v in out for v in (OAUTH, GITHUB, AWS)) and len(found) == 3, out)
check("the prose between them is preserved",
      out.startswith("a ") and " b " in out and " c " in out and out.endswith(" d")
      and out.count("[REDACTED:") == 3, out)

# ---------------------------------------------------------------------------
print("\n== 8 · a clean passage comes back byte-identical ==")
# ---------------------------------------------------------------------------
# No false positives AND no silent reformatting. The adversarial parts: 'Task-oriented'
# and 'disk-usage-monitor' both contain the literal 'sk-', which is why the bare sk-
# rule takes an alphanumeric tail and a 32-character floor.

CLEAN = ("We rotated the Supabase key and deploys started working again. "
         "Task-oriented notes on disk-usage-monitor-v2 live in config/environment.py, "
         "and the base64 sample AAAABBBBCCCC is not a credential.\n")
out, found = redact.redact_text(CLEAN)
check("a clean passage is returned byte-identical", out == CLEAN, repr(out))
check("...and reports no findings", found == [], str(found))
check("...and the identical object comes back (no copy-with-changes)", out is CLEAN)

clean_bytes = CLEAN.encode("utf-8") + b"\x00\x8f\xff"
out_b, found_b = redact.redact_bytes(clean_bytes)
check("clean BYTES are returned unchanged too",
      out_b == clean_bytes and found_b == [], str(found_b))

# ---------------------------------------------------------------------------
print("\n== 9 · fingerprints are stable, distinct, and short ==")
# ---------------------------------------------------------------------------

check("the fingerprint is stable across calls",
      redact.fingerprint(OAUTH) == redact.fingerprint(OAUTH))
check("the fingerprint is stable across str and bytes spellings",
      redact.fingerprint(OAUTH) == redact.fingerprint(OAUTH.encode("ascii")))
check("different values fingerprint differently",
      redact.fingerprint(OAUTH) != redact.fingerprint(ANTHROPIC))
check("the fingerprint is 12 hex chars, as in spec §4",
      len(redact.fingerprint(OAUTH)) == 12
      and all(c in "0123456789abcdef" for c in redact.fingerprint(OAUTH)))
def _first_fp(text):
    """Guarded, so that a disabled filter makes this check FAIL rather than abort the
    run -- a suite that dies mid-way hides the rest of its own verdict."""
    found = redact.redact_text(text)[1]
    return found[0]["fingerprint"] if found else ""


_a, _b = _first_fp("x " + GITHUB), _first_fp("completely different wording " + GITHUB)
check("the same value in two different passages gets the same fingerprint",
      bool(_a) and _a == _b, f"{_a!r} vs {_b!r}")

# ---------------------------------------------------------------------------
print("\n== 10 · tier 1 quarantines a credential-surface body ==")
# ---------------------------------------------------------------------------
# A .credentials.json tool result has no surrounding context worth keeping: the body
# IS the secret. This is the path that would have stopped 2026-09-15.

body = '{"claudeAiOauth": {"accessToken": "' + OAUTH + '", "scopes": ["user"]}}'
out, found = redact.redact_text(body, origin="/home/andy/.claude/.credentials.json")
check("a credential-surface origin replaces the whole body", OAUTH not in out, out)
check("...with a single fingerprinted marker",
      out == redact.marker("credential-surface", redact.fingerprint(
          body.encode("utf-8"))), out)
check("...and the tier-1 finding names the rule that fired",
      bool(found) and found[0]["tier"] == 1
      and found[0]["why"].endswith("path:credentials"), str(found[:1]))
check("...and still records what tier 2 saw inside, for review",
      bool(found) and "oauth-token" in found[0].get("tier2_kinds_inside", []),
      str(found[:1]))

out, found = redact.redact_text(body, origin="/home/andy/.claude/.credentials.json",
                                quarantine_on_origin=False)
check("flag-only mode still redacts the tier-2 hit and keeps the JSON scaffolding",
      OAUTH not in out and out.startswith('{"claudeAiOauth"') and "scopes" in out, out)

# ---------------------------------------------------------------------------
print("\n== 11 · the review log never holds the value ==")
# ---------------------------------------------------------------------------
# The review log is a greppable file on the same machine. On 2026-10-19 a search for
# the DECISION RECORD about a key re-emitted the key itself, so the audit trail for a
# leak must not be a second copy of the leak.

tmpdir = tempfile.mkdtemp(prefix="redact-test-")
try:
    logp = os.path.join(tmpdir, "review.jsonl")
    # An origin string can itself carry a secret -- this is the shape that matters.
    origin = 'curl -H "Authorization: Bearer ' + OAUTH + '" https://api.example.com'
    _, found = redact.redact_text("header line " + GITHUB + " trailer", origin="")
    written = redact.append_review_log(found, origin=origin, label="unit-test",
                                       log_path=logp)
    raw = open(logp, "rb").read() if os.path.exists(logp) else b""
    check("the review log was written", written == logp and len(raw) > 0)
    check("the log records the fingerprint",
          redact.fingerprint(GITHUB).encode() in raw)
    check("the log records the kind and the confidence",
          b"github-token" in raw and b"detected" in raw)
    # ⚠ `len(raw)` is part of the condition on purpose: "the value is absent" is
    # trivially true of an empty file, and a verdict on an empty result set is not a
    # verdict.
    check("⛔ the log does NOT contain the secret value",
          len(raw) > 0 and GITHUB.encode() not in raw)
    check("⛔ a secret inside the ORIGIN string is redacted before logging",
          OAUTH.encode() not in raw and b"[REDACTED:oauth-token" in raw)
    check("an empty finding list writes nothing",
          redact.append_review_log([], origin="x", log_path=logp) == "")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

# ---------------------------------------------------------------------------
print("\n== 12 · contracts that keep this on the ingest path ==")
# ---------------------------------------------------------------------------

try:
    redact.scan_bytes("a string")
    check("scan_bytes refuses str (the byte level is the point)", False)
except TypeError:
    check("scan_bytes refuses str (the byte level is the point)", True)

try:
    redact.redact_bytes("a string")
    check("redact_bytes refuses str", False)
except TypeError:
    check("redact_bytes refuses str", True)

try:
    redact.redact_text(b"bytes")
    check("redact_text refuses bytes", False)
except TypeError:
    check("redact_text refuses bytes", True)

s = redact.summarize(redact.redact_text("a " + GITHUB + " b " + GITHUB + " c")[1])
check("summarize counts occurrences and DISTINCT values separately",
      s["occurrences"] == 2 and s["distinct_values"] == 1, str(s))

# No fake credential in this file may survive a redact_text round trip -- i.e. every
# constant above is actually covered by a rule.
uncovered = [v[:12] for v in ALL_FAKE if v in redact.redact_text("x " + v + " y")[0]]
check("every fake credential in this file is caught by some rule",
      uncovered == [], f"uncovered prefixes: {uncovered}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 68)
if failures:
    print(f"VERDICT: FAIL -- {len(failures)} failing checks")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("VERDICT: PASS -- all checks green")
sys.exit(0)
