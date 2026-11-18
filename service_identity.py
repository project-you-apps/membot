"""Which program is this, and which build of it. 2026-11-15.

WHY THIS EXISTS. On 2026-11-15 a test pointed at 127.0.0.1:8000 expecting membot.
The port answered: /api/status returned a plausible membot-shaped body and
/api/search returned an empty result set, which read as "no cartridge mounted".
The process was Vector Plus Studio, which serves routes of the same names over a
different engine. The wrong diagnosis survived three debugging steps.

  => AN ENDPOINT THAT ANSWERS IS NOT THE ENDPOINT YOU MEANT, and nothing in the
     reply said so. A caller has no way to tell a sibling product, a stale build
     or an impostor from the thing it asked for.

ONE MODULE, TWO SERVERS, deliberately. membot_server.py and
membot_rest_bridge.py both import this. Those two programs are the pair that
caused the 2026-11-14 exposure -- the loopback default was applied to one and
nobody looked at the second server in the same directory -- so an identity
format that can drift between them would reproduce the original problem in the
one place meant to detect it.

-------------------------------------------------------------------------------
THE PART THAT MATTERS: `code_digest`, NOT `git_rev`.
-------------------------------------------------------------------------------
The droplet is a 2026-07-20 git checkout with hand-scp'd files on top -- 9
modified and 30 untracked against its own HEAD. So `git_rev` there reports a
commit from July while the process serves something else entirely. A revision
that names what the repo WAS is worse than no answer, because it invites
confidence.

`code_digest` hashes the .py files that are actually on disk next to the running
program. It moves the moment anyone scp's a file, which is precisely the event
git cannot see.

  => Same principle as the cart fingerprint work (sha256-members-v3): ATTEST THE
     BYTES YOU SERVE, NOT THE BYTES YOU BELIEVE YOU HAVE. A digest over what the
     loader actually read is the only kind that cannot be quietly wrong.

NOT A SECURITY CLAIM. This is an identifier, not a seal. It is unsigned, it is
self-reported, and a hostile process can print whatever it likes here. It
answers "am I talking to the build I think I am", which is a correctness and
operations question. It does not answer "is this host trustworthy".
-------------------------------------------------------------------------------

NEVER RAISES. Same discipline as seat_identity.resolve_seat: an identity helper
that can throw becomes a denial of service on every endpoint that reports it.
Every field degrades to None and says so, because a missing field is readable
and a 500 is not.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time

SERVICE_NAME = "membot"

# Bumped by hand when the WIRE CONTRACT changes in a way a client could notice:
# a field added to or removed from a response, a semantic change to one. Not a
# release version and not maintained per commit -- `code_digest` is what
# identifies a build. This only tells a client which shapes to expect.
CONTRACT_VERSION = "2026-11-15"

_cache: dict | None = None


def _py_files(dir_path: str) -> list[str]:
    """The .py files beside the running program, non-recursive, sorted.

    Non-recursive on purpose. It captures the modules that are actually imported
    at runtime -- cart_io, multi_cart, seat_identity, membox and the servers --
    while excluding tests/ and droplet-snapshot/, which do not affect behaviour
    and would make the digest move for reasons a caller does not care about.
    """
    try:
        names = [n for n in os.listdir(dir_path) if n.endswith(".py")]
    except OSError:
        return []
    return sorted(names)


def code_digest(dir_path: str) -> tuple[str | None, int]:
    """sha256 over the .py files on disk beside the program. (digest, file_count).

    Hashes name + length + bytes per file, in sorted name order, so a rename and
    a content change are both visible and neither can be masked by the other.
    Truncated to 16 hex characters: this is an identifier for humans and logs,
    and the full digest buys nothing a build id needs.

    Returns (None, 0) if the directory cannot be read -- better an absent field
    than a digest over a partial file list, which would look authoritative and
    be wrong.
    """
    names = _py_files(dir_path)
    if not names:
        return None, 0
    h = hashlib.sha256()
    for name in names:
        try:
            with open(os.path.join(dir_path, name), "rb") as f:
                data = f.read()
        except OSError:
            # One unreadable file makes the whole digest unreliable. Refuse
            # rather than silently hash a subset.
            return None, 0
        h.update(name.encode("utf-8"))
        h.update(str(len(data)).encode("ascii"))
        h.update(data)
    return h.hexdigest()[:16], len(names)


def git_rev(dir_path: str) -> tuple[str | None, bool | None]:
    """(short_rev, dirty) from git, or (None, None) when git cannot answer.

    Best effort and explicitly secondary to code_digest. `dirty` covers tracked
    modifications only -- git does not consider an untracked file a change, and
    the droplet has 30 of them, which is exactly why this is not the primary
    identifier.
    """
    def run(args):
        try:
            p = subprocess.run(["git", *args], cwd=dir_path, capture_output=True,
                               text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        if p.returncode != 0:
            return None
        return p.stdout.strip()

    rev = run(["rev-parse", "--short", "HEAD"])
    if not rev:
        return None, None
    status = run(["status", "--porcelain", "--untracked-files=no"])
    return rev, (bool(status) if status is not None else None)


def service_block(role: str, **extra) -> dict:
    """The identity block a server reports. Computed once, then cached.

    Args:
        role:  which program this is -- "mcp-server" or "rest-bridge". The two
               are separate processes with different protections, and conflating
               them is how the 2026-11-14 exposure went unnoticed.
        extra: live facts the caller alone knows, such as the bind address, port
               and transport. Pass what the process is ACTUALLY using, not what
               was requested, so a reader is not inferring exposure from a
               config file.

    ⇒ A caller should assert `name` and `role` before trusting anything else in
      the response, and log `code_digest` next to any result it keeps.
    """
    global _cache
    if _cache is None:
        here = os.path.dirname(os.path.abspath(__file__))
        digest, n_files = code_digest(here)
        rev, dirty = git_rev(here)
        _cache = {
            "name": SERVICE_NAME,
            "contract_version": CONTRACT_VERSION,
            "code_digest": digest,
            "code_files": n_files,
            "git_rev": rev,
            "git_dirty": dirty,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "pid": os.getpid(),
        }
    out = dict(_cache)
    out["role"] = role
    out.update(extra)
    return out


def describe(role: str, **extra) -> str:
    """One line for a boot log. ASCII only -- this prints to a cp1252 console."""
    b = service_block(role, **extra)
    bits = [f"{b['name']}/{role}", f"contract={b['contract_version']}",
            f"code={b['code_digest'] or 'unknown'}({b['code_files']} files)"]
    if b["git_rev"]:
        bits.append(f"git={b['git_rev']}" + ("-dirty" if b["git_dirty"] else ""))
    else:
        bits.append("git=unavailable")
    for k, v in extra.items():
        bits.append(f"{k}={v}")
    bits.append(f"pid={b['pid']}")
    return " ".join(bits)
