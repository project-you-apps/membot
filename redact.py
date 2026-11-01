"""Ingest redaction, tiers 1 and 2: credentials structurally cannot enter the corpus.

⛔⛔ THIS RUNS AT INGEST. IT IS NOT A RETRIEVAL FILTER. ⛔⛔
Every public function here is named and documented for the WRITE path. A read-path
filter leaves the secret sitting on disk, one bug away from being served, and gives
the comfortable feeling of a control without the control. If you are reaching for
this module from a search/get/answer path, you are fixing the wrong end: the value
is already stored. Fix the ingest call that stored it.

THE DECISION THIS IMPLEMENTS (SPEC-ingest-redaction-tiers-1-and-2-2026-10-18)
-----------------------------------------------------------------------------
On 2026-09-15 a Claude session read `~/.claude/.credentials.json`. The TOOL RESULT
carried real OAuth tokens into the transcript, and the transcript was ingested.
Nobody pasted anything. There was no bug: the memory system ingests what happens,
and what happened included a credentials file. The behaviour IS the product.

The tokens were later found inside a PICKLE, where every text-level grep said only
"Binary file matches." That is the whole reason tier 2 operates on BYTES rather than
on "conversations" or "messages": bytes do not care what container they are in. A
filter that understands transcripts would never have looked in a pickle, nor in EXIF,
a PDF object stream, or a zip member.

And it does not sit still. On 2026-10-19 a cart search for the DECISION RECORD about
a key returned the key's plaintext value into a live conversation -- which is itself a
transcript, which gets ingested. A credential that enters a searchable corpus once is
thereafter re-emitted by the corpus's own retrieval, indefinitely. "Delete the file"
does not reach it. That is why the only honest place for this filter is the door.

WHAT THIS BUYS: the sentence "credentials structurally cannot enter the memory,"
as opposed to "we are careful." A competitor with no ingest layer cannot say it.

TWO TIERS, AND WHY THERE IS NO THIRD HERE
-----------------------------------------
Tier 1 · PROVENANCE. Does not look at content at all, so no new secret format ever
         defeats it. Smallest, most certain, highest value -- and the one that would
         have caught 2026-09-15 outright, because the tool result carried the path.
Tier 2 · BYTE AUTOMATA. Known prefixed formats, with a checksum or structural check
         where the format has one. Container-independent and fast enough to run on
         everything rather than on a sample.

⛔ NO CLASSIFIER, BY RULING. Nothing here is probabilistic. A 90%-accurate classifier
at the boundary is WORSE than none, because people trust it and stop being careful.
Tiers 1-2 match KNOWN shapes; that is a lookup, not a judgment call. Tier 3 (NER) and
tier 4 (small LM) are PARKED, along with the personal-dictionary product, the
user->model direction, and retroactive masking. Do not add them here.

⛔ WHAT THIS STRUCTURALLY CANNOT DO -- state it, do not discover it:
  * A low-entropy secret with no label. `Fremont2026!` in a sentence is prose. No tier
    proposed in the spec changes that; not typing it remains the only control.
  * PII. A birthdate is not PII; a birthdate next to a name is. Byte- and pattern-level
    filters cannot see context. This is a SECRETS filter. Calling it a PII filter to a
    compliance buyer would be false.
  * A secret inside a COMPRESSED or ENCRYPTED container. gzip, zip deflate, and
    anything encrypted destroy the byte sequence we scan for. Container independence
    means "container-agnostic", not "codec-defeating". Decompress, then scan.
  * Secrets at rest in files nobody ingests (crash dumps, local stack secret stores).
    Ingest redaction does not reach them. That is a separate inventory.

⚠ IF YOU ARE HERE TO MASK AN EXISTING CART: A MASKING PASS MUST RE-EMBED.
Embeddings computed from the original text still encode the secret, and
embedding-inversion recovers approximate text. Masking the strings while keeping the
old vectors leaves a partial leak that LOOKS CLEAN, which is the worst outcome
available. Also: review by DISTINCT VALUE, not by occurrence (on 2026-09-15 that was
111 occurrences of 2 distinct tokens -- two decisions, not 111), and write a masked
COPY; never mutate the original, which is the only copy of parts of 2025. That pass is
NOT built here and is a separate ruling.

Stdlib only. No ML, no dependencies.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import string
import time
import zlib
from pathlib import Path

FORMAT = "vps-redaction"
VERSION = 1

# Confidence is a two-valued FACT about the format, not a score.
#   DETECTED  -- the format carries a checksum or a decodable structure, and it checked
#                out. This is "that is a credential", not "that looks like one".
#   SUSPECTED -- the shape matched but the format offers nothing to verify against (or
#                the verification failed). Still redacted; the caller is told the
#                difference so it can treat the two differently -- e.g. auto-accept
#                detected hits, queue suspected ones for review.
# ⛔ There is no third value and no number. A score would be a classifier.
DETECTED = "detected"
SUSPECTED = "suspected"

FINGERPRINT_CHARS = 12  # per spec §4: [REDACTED:oauth-token:sha256:3974098c8876]

MAX_PEM_BLOCK = 65536

_ALNUM = frozenset((string.ascii_letters + string.digits).encode("ascii"))
_ALNUM_DASH_US = _ALNUM | frozenset(b"-_")
_UPPER_DIGITS = frozenset((string.ascii_uppercase + string.digits).encode("ascii"))
_B64URL = _ALNUM | frozenset(b"-_")
_B64URL_DOT = _B64URL | frozenset(b".")
_B64STD = _ALNUM | frozenset(b"+/=")

# GitHub's token checksum is a CRC32 of the random part, base62-encoded to 6 chars
# (announced with the ghp_/gho_/... formats in 2021).
# ⚠ HONESTY NOTE: this alphabet ordering is believed correct but has NOT been confirmed
# against a real GitHub token here -- confirming it would require a real token in this
# file, which is forbidden. If the ordering is wrong, a real token verifies as
# SUSPECTED instead of DETECTED. That direction is fail-safe: it is still redacted, and
# the finding says WHY it was not upgraded. Do not "fix" this by removing the check.
_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------

def fingerprint(value) -> str:
    """Stable, truncated sha256 of a secret value. Lets us answer "was this value
    ever in our corpus?" WITHOUT the value being in the corpus.

    That question is not hypothetical: on 2026-09-15 it had to be answered by handling
    the real values by hand, because nothing had recorded them in a form that was safe
    to keep.

    ⚠ This is a FINGERPRINT FOR RECALL, not a PCI-DSS token. A token has no
    mathematical relationship to its value; a hash does. For a high-entropy credential
    the distinction is academic -- there is nothing to brute-force. For a low-entropy
    value (a password, a 4-digit code) a truncated unsalted sha256 is reversible by
    anyone with a word list. Do not use this to fingerprint human-chosen secrets and
    then describe it to an auditor as tokenization.
    """
    if isinstance(value, str):
        data = value.encode("utf-8", "surrogatepass")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
    else:
        data = str(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:FINGERPRINT_CHARS]


def marker(kind: str, fp: str) -> str:
    """The replacement string. Shape is fixed by spec §4 and is deliberately
    grep-able, so a human reviewing a cart can find every redaction in it."""
    return f"[REDACTED:{kind}:sha256:{fp}]"


# ---------------------------------------------------------------------------
# TIER 1 -- provenance. The layer that does not decay.
# ---------------------------------------------------------------------------
#
# This is the only tier whose accuracy does not erode as attackers and vendors invent
# new formats, because it never looks at content. A path is a credential surface or it
# is not, and that fact was true before any of these key formats existed.

# (rule name, needle, require a non-letter boundary after the needle)
# The boundary flag is what keeps `.env` from firing on `/src/.environments/` while
# still firing on `.env`, `.env.local` and `.env/`. `.envrc` (direnv) is listed
# explicitly rather than special-cased in the matcher.
_PATH_RULES = (
    ("path:credentials", "credentials", False),
    ("path:dotenv", ".envrc", False),
    ("path:dotenv", ".env", True),
    ("path:pem", ".pem", True),
    ("path:id_rsa", "id_rsa", False),
    ("path:aws-dir", ".aws/", False),
    ("path:ssh-dir", ".ssh/", False),
    ("path:netrc", ".netrc", True),
    ("path:pgpass", ".pgpass", True),
)

# A command whose OUTPUT is credentials. Matched on leading tokens of each pipeline
# segment, not as a substring -- otherwise every path containing "env" would fire.
_COMMAND_RULES = (
    ("command:env", ("env",)),
    ("command:printenv", ("printenv",)),
    ("command:gh-auth-token", ("gh", "auth", "token")),
    ("command:aws-configure", ("aws", "configure")),
    ("command:docker-login", ("docker", "login")),
)

_SEGMENT_SEPARATORS = ("|", "&&", "||", ";", "\n", "\r")


def _normalize_path(s: str) -> str:
    return s.replace("\\", "/").lower()


def _needle_hits(norm: str, needle: str, boundary: bool) -> bool:
    start = 0
    while True:
        i = norm.find(needle, start)
        if i < 0:
            return False
        if not boundary:
            return True
        after = i + len(needle)
        if after >= len(norm) or not norm[after].isalpha():
            return True
        start = i + 1


def _command_segments(s: str):
    buf = s
    for sep in _SEGMENT_SEPARATORS:
        buf = buf.replace(sep, "\x00")
    for seg in buf.split("\x00"):
        seg = seg.strip()
        if seg:
            yield seg


def _leading_tokens(segment: str):
    """Tokens of a command with the noise stripped: `sudo`, and leading VAR=value
    assignments, neither of which changes which program runs."""
    tokens = [t for t in segment.replace("\t", " ").split(" ") if t]
    out = []
    skipping = True
    for tok in tokens:
        if skipping and (tok == "sudo" or ("=" in tok and not tok.startswith("-"))):
            continue
        skipping = False
        out.append(tok)
    if out:
        # /usr/bin/env is `env`. A path like data/env/notes.txt is not, because its
        # last component is notes.txt.
        out[0] = out[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return [t.lower() for t in out]


def classify_origin(path_or_command: str) -> dict:
    """TIER 1, AT INGEST: is the ORIGIN of these bytes a credential surface?

    Takes the path a tool read or the command a tool ran -- whatever provenance the
    ingest call has. Both rule sets are tried, because a tool result's origin may be
    either and we do not want the caller guessing which.

    Returns {"flagged": bool, "tier": 1, "rules": [...], "origin": str}. `rules` names
    every rule that matched, so a human auditing a redaction can see WHY without
    re-deriving it.

    Deliberately conservative: `env FOO=1 python x.py` does not dump the environment,
    but it flags. Over-flagging here costs a redacted passage that keeps its context;
    under-flagging costs a credential in the corpus forever. Those are not symmetric.

    ⛔ Not for the read path. The origin is only known at ingest; by retrieval time
    the bytes have already been stored.
    """
    origin = path_or_command or ""
    rules = []
    norm = _normalize_path(origin)
    for name, needle, boundary in _PATH_RULES:
        if _needle_hits(norm, needle, boundary):
            rules.append({"rule": name, "pattern": needle, "via": "path"})
    for segment in _command_segments(origin):
        tokens = _leading_tokens(segment)
        for name, want in _COMMAND_RULES:
            if tuple(tokens[:len(want)]) == want:
                rules.append({"rule": name, "pattern": " ".join(want),
                              "via": "command"})
    # Dedupe while keeping order; the same rule can match twice in one pipeline.
    seen = set()
    unique = []
    for r in rules:
        key = (r["rule"], r["via"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return {
        "flagged": bool(unique),
        "tier": 1,
        "rules": unique,
        "origin": origin,
    }


# ---------------------------------------------------------------------------
# TIER 2 -- byte automata. Container-independent by construction.
# ---------------------------------------------------------------------------
#
# Implementation note on throughput: this is a prefix scan driven by bytes.find(),
# which is a C-speed substring search, once per literal prefix. A hand-written
# per-byte DFA in Python would be the textbook shape and roughly 50x slower. The
# property that matters -- we can afford to run it on EVERY ingested byte rather than
# on a sample -- comes from staying out of the interpreter loop.


def _verify_none(data, start, end, prefix, body):
    """No checksum, no decodable structure: SUSPECTED is the honest answer."""
    return {"confidence": SUSPECTED,
            "why": "prefixed format with no checksum to verify"}


def _verify_github(data, start, end, prefix, body):
    """GitHub's prefixed tokens carry a CRC32 of the random part, base62-encoded into
    the last 6 characters. This is the discriminator: shape alone would fire on any
    36-character identifier that happened to follow `ghp_`."""
    if len(body) != 36:
        return None
    secret, checksum = body[:30], body[30:]
    crc = zlib.crc32(secret)
    digits = []
    n = crc
    while n:
        digits.append(_BASE62[n % 62])
        n //= 62
    expect = "".join(reversed(digits)).rjust(6, "0")
    try:
        got = checksum.decode("ascii")
    except UnicodeDecodeError:
        return None
    if got == expect:
        return {"confidence": DETECTED, "verified_by": "crc32-base62",
                "why": "crc32 checksum over the 30-byte secret matches"}
    return {"confidence": SUSPECTED, "verified_by": "crc32-base62",
            "why": "github token shape, checksum did not match"}


def _verify_exact_boundary(data, start, end, prefix, body):
    """Formats of fixed length (AWS key ids, Google API keys) have no checksum, but
    they do have an exact length. Requiring that the run STOPS there is a real check:
    it rejects a longer identifier that merely begins with the prefix."""
    nxt = data[end] if end < len(data) else None
    if nxt is not None and nxt in _ALNUM:
        # The run is longer than the format allows, so this is some other identifier
        # that merely starts with the prefix.
        return None
    return {"confidence": SUSPECTED,
            "why": "fixed-length prefixed format; no checksum published"}


def _verify_jwt(data, start, end, prefix, body):
    """A JWT is three base64url segments. The first decodes to JSON with an `alg`
    member -- that is a structural check, not a guess, so a JWT is DETECTED.

    ⚠ "Detected" means "this IS a JWT", not "this signature is valid". Verifying the
    signature needs the key and is not the question here: an expired or forged JWT in
    the corpus is still a credential-shaped secret that must not persist.
    """
    whole = data[start:end]
    parts = whole.split(b".")
    if len(parts) < 3 or not parts[0] or not parts[1] or not parts[2]:
        return None
    # Trim to exactly three segments: a greedy scan happily eats the full stop at the
    # end of a sentence, and anything after it.
    keep = b".".join(parts[:3])
    new_end = start + len(keep)
    header = parts[0]
    try:
        raw = base64.urlsafe_b64decode(header + b"=" * (-len(header) % 4))
        obj = json.loads(raw.decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        # Three dotted base64url runs whose first segment is not JSON is not a JWT --
        # it is prose or an opaque blob. Reporting it would be a guess, and guesses are
        # what this module refuses to make.
        return None
    if isinstance(obj, dict) and "alg" in obj:
        return {"confidence": DETECTED, "end": new_end,
                "verified_by": "base64url-json-header",
                "why": "header decodes to a JSON object carrying `alg`"}
    return {"confidence": SUSPECTED, "end": new_end,
            "why": "header decoded but carries no `alg`"}


def _kind_anthropic(body: bytes) -> str:
    # The 2026-09-15 leak was OAuth tokens, so the two cases are worth telling apart:
    # an api key is rotated in a dashboard, an oauth token is re-minted by a login.
    if body.startswith(b"oat01-"):
        return "oauth-token"
    if body.startswith(b"api03-"):
        return "anthropic-api-key"
    return "anthropic-key"


# (kind | callable, prefixes, charset, min body, max body, verifier)
_PREFIX_RULES = (
    (_kind_anthropic, (b"sk-ant-",), _ALNUM_DASH_US, 30, 200, _verify_none),
    # `sk-proj-` is the current spelling of the same OpenAI prefix the spec lists, not
    # a new format. The bare `sk-` rule takes an alphanumeric tail only and a 32-char
    # floor, which is what keeps it off prose like "disk-usage-monitor-...".
    ("openai-api-key", (b"sk-proj-",), _ALNUM_DASH_US, 40, 200, _verify_none),
    ("openai-api-key", (b"sk-",), _ALNUM, 32, 200, _verify_none),
    ("github-token", (b"ghp_", b"gho_", b"ghu_", b"ghs_", b"ghr_"),
     _ALNUM, 36, 36, _verify_github),
    # Fine-grained PATs: a documented shape, no documented checksum. SUSPECTED, and
    # that is the point of having two words for it.
    ("github-pat", (b"github_pat_",), _ALNUM_DASH_US, 50, 120, _verify_none),
    ("aws-access-key-id", (b"AKIA", b"ASIA"), _UPPER_DIGITS, 16, 16,
     _verify_exact_boundary),
    ("google-api-key", (b"AIza",), _ALNUM_DASH_US, 35, 35, _verify_exact_boundary),
    ("slack-token", (b"xoxb-", b"xoxa-", b"xoxp-", b"xoxr-", b"xoxs-"),
     _ALNUM_DASH_US, 20, 120, _verify_none),
    ("stripe-secret-key", (b"sk_live_", b"rk_live_"), _ALNUM, 20, 60, _verify_none),
    ("jwt", (b"eyJ",), _B64URL_DOT, 20, 8192, _verify_jwt),
)

_PEM_BEGIN = b"-----BEGIN "
_PEM_END = b"-----END "


def _scan_pem(data: bytes) -> list:
    """PEM private keys get their own scanner because the secret is a BLOCK, not a
    token: redacting the header and leaving the base64 body would be theatre."""
    out = []
    pos = 0
    while True:
        i = data.find(_PEM_BEGIN, pos)
        if i < 0:
            return out
        pos = i + 1
        label_start = i + len(_PEM_BEGIN)
        close = data.find(b"-----", label_start)
        if close < 0 or close - label_start > 64:
            continue
        label = data[label_start:close]
        if not label.endswith(b"PRIVATE KEY"):
            continue  # a certificate or public key is not a secret
        footer = _PEM_END + label + b"-----"
        j = data.find(footer, close, min(len(data), close + MAX_PEM_BLOCK))
        if j >= 0:
            end = j + len(footer)
            body = data[close + 5:j]
            stripped = bytes(c for c in body if c in _B64STD)
            try:
                base64.b64decode(stripped + b"=" * (-len(stripped) % 4), validate=True)
                verdict = {"confidence": DETECTED,
                           "verified_by": "pem-footer-and-base64-body",
                           "why": "matching END footer and a body that base64-decodes"}
            except (binascii.Error, ValueError):
                verdict = {"confidence": SUSPECTED,
                           "why": "PEM private-key block, body did not base64-decode"}
        else:
            # Truncated block (a transcript cut it off, a pickle split it). Consume the
            # following base64/whitespace run so the key material still goes, and say
            # plainly that we guessed the extent.
            end = close + 5
            limit = min(len(data), end + MAX_PEM_BLOCK)
            while end < limit and (data[end] in _B64STD or data[end] in b" \t\r\n"):
                end += 1
            verdict = {"confidence": SUSPECTED,
                       "why": "PEM private-key header with no matching END footer; "
                              "extent inferred from the base64 run"}
        out.append(_finding("private-key-pem", data, i, end, verdict))
        pos = max(pos, end)


def _finding(kind, data, start, end, verdict) -> dict:
    end = verdict.get("end", end)
    value = bytes(data[start:end])
    return {
        "tier": 2,
        "kind": kind,
        "start": start,
        "end": end,
        "length": end - start,
        "confidence": verdict["confidence"],
        "verified_by": verdict.get("verified_by", ""),
        "why": verdict.get("why", ""),
        "fingerprint": fingerprint(value),
        # ⛔ The value itself is NEVER put in a finding. A finding travels into review
        # logs, MCP responses and agent context -- all of which are themselves
        # ingested. That is how 2026-10-19 happened: a search for the decision record
        # about a key re-emitted the key.
    }


def scan_bytes(data) -> list:
    """TIER 2, AT INGEST: find known credential formats anywhere in a byte buffer.

    Operates on BYTES, deliberately. The 2026-09-15 tokens were inside a pickle, where
    every text-level grep said "Binary file matches" and nothing looked further. A
    scanner that knows about "messages" or "passages" would have missed them; a
    scanner that knows about bytes does not care that it is holding a pickle, a PDF
    object stream, EXIF or a zip member.

    Returns a list of findings, each with byte offsets, `kind`, and `confidence`
    (DETECTED vs SUSPECTED -- see the constants). Overlaps are resolved in favour of
    the longest match, so `sk-ant-...` is reported once as an Anthropic key and not
    also as a bare `sk-` key. Never returns the matched value.

    ⛔ Not for the read path. Scanning at retrieval tells you the secret is on disk; it
    does not get it off disk.
    """
    if isinstance(data, str):
        raise TypeError("scan_bytes takes bytes -- encode first, or call redact_text. "
                        "The byte level is the point: see the module docstring.")
    buf = bytes(data)
    out = []
    for kind, prefixes, charset, min_len, max_len, verify in _PREFIX_RULES:
        for prefix in prefixes:
            pos = 0
            plen = len(prefix)
            while True:
                i = buf.find(prefix, pos)
                if i < 0:
                    break
                pos = i + 1
                end = i + plen
                limit = min(len(buf), end + max_len)
                while end < limit and buf[end] in charset:
                    end += 1
                body = buf[i + plen:end]
                if len(body) < min_len:
                    continue
                verdict = verify(buf, i, end, prefix, body)
                if verdict is None:
                    continue
                name = kind(body) if callable(kind) else kind
                out.append(_finding(name, buf, i, end, verdict))
    out.extend(_scan_pem(buf))
    return _resolve_overlaps(out)


def _resolve_overlaps(findings: list) -> list:
    """Longest match wins at a given position; a finding inside another is dropped.

    Without this, `sk-ant-oat01-...` reports twice and the redacted output nests
    markers inside markers.
    """
    ordered = sorted(findings, key=lambda f: (f["start"], -f["length"]))
    kept = []
    last_end = -1
    for f in ordered:
        if f["start"] < last_end:
            continue
        kept.append(f)
        last_end = f["end"]
    return kept


# ---------------------------------------------------------------------------
# redaction
# ---------------------------------------------------------------------------

def redact_bytes(data, origin: str = "", quarantine_on_origin: bool = True):
    """AT INGEST: replace every tier-2 hit with a fingerprint marker, keeping the
    surrounding bytes. Returns (redacted_bytes, findings).

    Keeping the passage is a decision, not laziness (spec §4):
      1. "We set the API key and deploys started working" is useful memory and should
         not be destroyed to protect a string.
      2. The fingerprint answers "was this value ever in our corpus?" without the
         value being in the corpus.
      3. It leaves a review trail. Ingest redaction is destructive and unreviewable by
         nature; a false positive that silently eats content and leaves no marker is
         undetectable forever.

    If `origin` is a credential surface (tier 1) the whole buffer is replaced instead,
    because a `.credentials.json` tool result has no surrounding context worth keeping
    -- the body IS the secret. Pass quarantine_on_origin=False to flag without
    quarantining, when the caller has its own policy.
    """
    if isinstance(data, str):
        raise TypeError("redact_bytes takes bytes; use redact_text for text")
    buf = bytes(data)
    findings = []

    origin_verdict = classify_origin(origin) if origin else {"flagged": False,
                                                             "tier": 1, "rules": [],
                                                             "origin": ""}
    tier2 = scan_bytes(buf)

    if origin_verdict["flagged"]:
        # Record what tier 2 saw inside even when quarantining, so the review log can
        # say whether the quarantine actually caught credentials or ate a false alarm.
        findings.append({
            "tier": 1,
            "kind": "credential-surface",
            "start": 0,
            "end": len(buf),
            "length": len(buf),
            "confidence": DETECTED,
            "verified_by": "provenance",
            "why": "origin is a credential surface: "
                   + ", ".join(r["rule"] for r in origin_verdict["rules"]),
            "fingerprint": fingerprint(buf),
            "origin_rules": origin_verdict["rules"],
            "tier2_kinds_inside": sorted({f["kind"] for f in tier2}),
        })

    findings.extend(tier2)

    if origin_verdict["flagged"] and quarantine_on_origin:
        return marker("credential-surface",
                      fingerprint(buf)).encode("ascii"), findings

    if not tier2:
        # Byte-identical return. No reformatting, no normalization, no copy-with-
        # changes: a clean passage must come back exactly as it went in, or the filter
        # is silently rewriting the corpus.
        return buf, findings

    pieces = []
    cursor = 0
    for f in tier2:
        pieces.append(buf[cursor:f["start"]])
        pieces.append(marker(f["kind"], f["fingerprint"]).encode("ascii"))
        cursor = f["end"]
    pieces.append(buf[cursor:])
    return b"".join(pieces), findings


def redact_text(text: str, origin: str = "", quarantine_on_origin: bool = True):
    """AT INGEST: run both tiers over a passage. Returns (redacted_text, findings).

    This is the function an ingest path calls. It encodes to UTF-8 and scans at the
    byte level, because that is the level the formats live at and the level a container
    cannot hide them from.

    ⛔⛔ DO NOT WIRE THIS TO A RETRIEVAL/SEARCH/ANSWER PATH. If it runs there, the
    secret is already on disk and already in the embeddings, and all you have built is
    a thing that looks like a control. Redaction has exactly one correct location: the
    door.
    """
    if not isinstance(text, str):
        raise TypeError("redact_text takes str; use redact_bytes for bytes")
    raw = text.encode("utf-8", "surrogatepass")
    out, findings = redact_bytes(raw, origin=origin,
                                 quarantine_on_origin=quarantine_on_origin)
    if out is raw or out == raw:
        return text, findings
    return out.decode("utf-8", "surrogatepass"), findings


# ---------------------------------------------------------------------------
# review log
# ---------------------------------------------------------------------------
#
# Spec §4.3: ingest redaction is destructive and unreviewable by nature, so every
# redaction logs to a side channel. Without it, a false positive eats content and
# nobody ever knows it happened.
#
# ⚠ HOME: if membot/journal.py exists, THAT is where this belongs -- a redaction is an
# event with a timestamp and an actor, which is what a journal is for. This function is
# a deliberately self-contained JSONL appender so that tiers 1-2 can ship without
# waiting on it; moving it is a small, mechanical change.

DEFAULT_LOG_NAME = "redaction-review.jsonl"


def review_log_path(explicit=None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("MEMBOT_REDACTION_LOG")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / DEFAULT_LOG_NAME


def append_review_log(findings, origin: str = "", label: str = "",
                      log_path=None, now: float = 0.0) -> str:
    """Append one JSONL line per finding so a human can audit what was redacted.

    ⛔ THE LOG NEVER HOLDS THE VALUE -- only the fingerprint, the kind and the offsets.
    A review log is a file on the same machine, backed up and greppable, and a corpus
    is exactly what we are trying to keep secrets out of. Note that `origin` itself can
    carry a secret (`curl -H "Authorization: Bearer ..."` is a provenance string), so
    it is run through tier 2 before being written.
    """
    findings = list(findings or [])
    if not findings:
        return ""
    now = now or time.time()
    path = review_log_path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_origin, _ = redact_text(origin or "", origin="")
    lines = []
    for f in findings:
        lines.append(json.dumps({
            "format": FORMAT,
            "version": VERSION,
            "at": now,
            "label": label or "",
            "origin": safe_origin,
            "tier": f.get("tier"),
            "kind": f.get("kind"),
            "confidence": f.get("confidence"),
            "verified_by": f.get("verified_by", ""),
            "why": f.get("why", ""),
            "fingerprint": f.get("fingerprint"),
            "start": f.get("start"),
            "length": f.get("length"),
        }, sort_keys=True))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return str(path)


def summarize(findings) -> dict:
    """Counts for a one-line verdict at the end of an ingest run. Review happens by
    DISTINCT VALUE, not by occurrence -- on 2026-09-15 that was 111 occurrences of 2
    distinct tokens, which is two decisions rather than 111 -- so distinct fingerprints
    is the number that tells a human how much work they actually have."""
    findings = list(findings or [])
    kinds, confidences = {}, {}
    for f in findings:
        kinds[f.get("kind", "?")] = kinds.get(f.get("kind", "?"), 0) + 1
        c = f.get("confidence", "?")
        confidences[c] = confidences.get(c, 0) + 1
    return {
        "occurrences": len(findings),
        "distinct_values": len({f.get("fingerprint") for f in findings}),
        "by_kind": kinds,
        "by_confidence": confidences,
        "detected": confidences.get(DETECTED, 0),
        "suspected": confidences.get(SUSPECTED, 0),
    }
