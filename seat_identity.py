"""Seat identity: make `session_id` advisory wherever a verified identity exists.

WHAT THIS CLOSES (measured 2026-11-01)
--------------------------------------
Membot has two request surfaces and they do NOT agree about identity.

  1. The web/REST routes (`/api/*`, six call sites) already do this correctly. Each one
     calls `_session_id_for_user(verified_sub, fallback=_anon_session_from_request(...))`,
     so the seat is DERIVED from a verified Supabase JWT and the caller's own
     `session_id` is only a fallback for anonymous visitors. Since 2026-10-25 that
     fallback also refuses the reserved `user_` / `anon_` prefixes, because a REST caller
     sending {"session_id": "user_<someone-else>"} would otherwise land in that user's
     session.

  2. The `@mcp.tool()` functions do not. Every per-seat tool -- seat_hot_stack,
     seat_hidden, seat_hide, subcart_list, subcart_export, subcart_delete,
     mempack_read_pattern_i, proposals, firm_hot_stack -- takes `session_id: str = ""`
     as a plain parameter and believes it. On the HTTP transport the only gate in front
     of them is a single shared Bearer token (MEMBOT_API_KEY), the same string for every
     caller, so any holder of that key can pass `session_id="user_<someone-else's-uuid>"`
     and read that seat's state.

The server's own comment already names both the gap and the fix, above the Mempack
endpoints: "Auth model (MVP): existing write API key required; owner_id passed in body.
Long-term: verify Supabase JWT directly, derive owner_id from `sub` claim."

⭐ THE REASONING IS NOT NEW -- IT IS THE 2026-10-25 FIX, APPLIED TO THE OTHER SURFACE.
   That fix refused the reserved prefixes on the REST body-fallback path for exactly this
   reason. The MCP path has the same hole and never got the same patch.

WHY THIS IS ADDITIVE AND CANNOT BREAK A RUNNING CALLER
------------------------------------------------------
⚠ Deliberately NOT "require a JWT on the MCP path." That would break every agent caller
  today, and agent callers hold a shared key precisely because they have no user identity
  to present. Issuing per-caller credentials is a product decision, not a patch.

So the rule is three-way and each branch leaves working callers working:

    no request at all (stdio)   -> trust the claimed seat. A local pipe IS the user; there
                                   is no identity to verify and nothing to impersonate.
    request + verified JWT      -> DERIVE the seat from the JWT's `sub`. A claimed value
                                   that disagrees is overridden and reported, not honored.
    request, no JWT             -> shared-key caller. Honor the claimed seat EXCEPT in the
                                   reserved `user_` / `anon_` namespaces, which would be
                                   impersonation of a signed-in seat.

⇒ The only call that changes behavior is one that was impersonating a reserved seat, which
  no legitimate caller does. Everything else is untouched.

⛔ WHAT THIS DOES NOT DO, stated so nobody reads it as more than it is: it does not give
   agent callers identities, and it does not make a shared key into per-caller auth. With
   MEMBOT_API_KEY unset the HTTP transport has no auth at all (the middleware logs "rate
   limiting enabled (no auth)"), and this module cannot invent an identity that was never
   presented. It stops impersonation of VERIFIED seats; it does not authenticate anonymous
   ones.

DEPENDENCY INJECTION, AND WHY
-----------------------------
The JWT reader and decoder live in membot_server.py. Importing them here would be
circular, and more importantly it would make this module untestable without booting a
server. Both are injected instead, so every branch above is exercised by a plain unit
test with no transport, no network and no Supabase.

Stdlib only.
"""

from __future__ import annotations

import re

# Namespaces the server assigns itself. A caller may never name a seat in one of these,
# because `user_<uuid>` is minted from a verified JWT and `anon_<tab>` from a sanitized
# header -- both are server-side facts, so a caller claiming one is claiming to be someone.
RESERVED_PREFIXES = ("user_", "anon_")

# Same sanitation the REST path applies: a seat becomes a dict key and a log field, so it
# is constrained to a safe alphabet and a bounded length before it is used as either.
_SAFE = re.compile(r"[^A-Za-z0-9_-]")
MAX_SEAT_CHARS = 64

DEFAULT_SEAT = "default"

# `source` values, so a caller (and the journal) can tell how a seat was established
# rather than having to guess from its shape.
SOURCE_VERIFIED = "verified"   # derived from a verified JWT `sub`
SOURCE_CLAIMED = "claimed"     # caller-supplied and accepted (stdio, or unreserved)
SOURCE_DEFAULT = "default"     # nothing presented, nothing claimed


def sanitize_seat(claimed: str) -> str:
    """Reduce a caller-supplied seat to the safe alphabet, or '' if nothing survives."""
    return _SAFE.sub("", (claimed or "").strip())[:MAX_SEAT_CHARS]


def resolve_seat(claimed: str = "",
                 request=None,
                 read_jwt=None,
                 decode_jwt=None) -> dict:
    """Establish the seat for one call, preferring a verified identity over a claim.

    Args:
        claimed:    the caller-supplied `session_id`. Advisory.
        request:    the live HTTP request, or None on stdio / outside a request.
        read_jwt:   callable(request) -> token string or None. Inject
                    membot_server._read_jwt_from_request.
        decode_jwt: callable(token) -> claims dict. Must RAISE on an invalid or expired
                    token. Inject membot_server._decode_supabase_jwt.

    Returns a dict rather than a bare string, because the caller needs to know HOW the
    seat was established -- an audit record that cannot distinguish a verified seat from a
    claimed one records nothing an auditor will accept:

        {"seat": str, "source": "verified"|"claimed"|"default",
         "overridden": bool,       # a claim was present and disagreed with the JWT
         "refused": bool,          # a claim was dropped for naming a reserved namespace
         "verified_sub": str}      # '' when no JWT was verified

    ⛔ NEVER raises. An identity helper that can throw becomes a denial-of-service on
       every tool that calls it, and the fail-closed behavior we want is "fall back to a
       non-privileged seat", not "take the server down".
    """
    safe_claim = sanitize_seat(claimed)
    out = {
        "seat": safe_claim or DEFAULT_SEAT,
        "source": SOURCE_CLAIMED if safe_claim else SOURCE_DEFAULT,
        "overridden": False,
        "refused": False,
        "verified_sub": "",
    }

    # --- Branch 1: no request. stdio, or a direct in-process call. --------------------
    # The local pipe IS the user. There is no identity to verify and no one else's seat
    # to reach, so the claim stands. This is also what keeps every existing local tool
    # and every test working unchanged.
    if request is None or read_jwt is None or decode_jwt is None:
        return out

    # --- Branch 2: a request with a verifiable JWT. The seat is DERIVED. --------------
    token = None
    try:
        token = read_jwt(request)
    except Exception:
        # A malformed header is not an identity. Fall through to the shared-key branch
        # rather than failing the call.
        token = None

    if token:
        try:
            claims = decode_jwt(token) or {}
            sub = str(claims.get("sub") or "").strip()
        except Exception:
            # Invalid, expired, wrong audience, unsupported algorithm, PyJWT missing.
            # ⛔ An unverifiable token must NOT be treated as anonymous-but-trusted: it
            #    is a failed assertion of identity, so the claim it arrived with is
            #    refused if it names a reserved seat (branch 3 handles that below).
            sub = ""

        if sub:
            derived = f"user_{sanitize_seat(sub)}"
            out["verified_sub"] = sub
            out["seat"] = derived
            out["source"] = SOURCE_VERIFIED
            # A claim that disagrees with a verified identity is the interesting case:
            # it is reported so the journal can record that someone asked for a seat
            # they were not, without the request failing.
            out["overridden"] = bool(safe_claim) and safe_claim != derived
            return out

    # --- Branch 3: a request with no usable identity. Shared-key caller. -------------
    # Honor the claim, except in the reserved namespaces, which would be impersonation
    # of a seat the server mints from verified facts.
    if safe_claim and safe_claim.startswith(RESERVED_PREFIXES):
        out["seat"] = DEFAULT_SEAT
        out["source"] = SOURCE_DEFAULT
        out["refused"] = True
        return out

    return out


def seat_for_journal(resolved: dict) -> dict:
    """The subset of a resolution worth recording on a journal entry.

    Keeps `writer` and how it was established together, because "writer: andy" is only
    evidence if the record also says whether anyone checked.
    """
    return {
        "writer": resolved.get("seat", DEFAULT_SEAT),
        "identity_source": resolved.get("source", SOURCE_DEFAULT),
        "identity_overridden": bool(resolved.get("overridden")),
        "identity_refused": bool(resolved.get("refused")),
    }
