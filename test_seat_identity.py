"""Tests for seat_identity.resolve_seat.

⛔ EXISTENCE IS NOT OPERATION. Every test here is written to FAIL if the override were
   removed -- the ones that matter are the impersonation refusals, not the happy path.
   A test that passes whether or not the check runs is not a test.

No server, no transport, no network: the JWT reader and decoder are injected, which is
the whole reason they are injected.
"""

import seat_identity as si


# --- fakes -----------------------------------------------------------------------------

class FakeRequest:
    """Stands in for a Starlette request. resolve_seat only passes it to read_jwt."""
    def __init__(self, token=None):
        self.token = token


def read_jwt(request):
    return getattr(request, "token", None)


def decode_good(token):
    """A verifier that accepts exactly one token and raises on anything else,
    the way a real JWT decoder does."""
    if token == "good-token":
        return {"sub": "abc-123", "aud": "authenticated"}
    raise ValueError(f"invalid token: {token!r}")


def decode_always_raises(token):
    raise ValueError("expired")


# --- Branch 1: no request (stdio / in-process) -----------------------------------------

def test_stdio_trusts_the_claim():
    """A local pipe IS the user. Nothing to verify, nothing to impersonate."""
    r = si.resolve_seat("alice", request=None)
    assert r["seat"] == "alice"
    assert r["source"] == si.SOURCE_CLAIMED
    assert r["refused"] is False
    assert r["overridden"] is False


def test_stdio_with_no_claim_is_default():
    r = si.resolve_seat("", request=None)
    assert r["seat"] == si.DEFAULT_SEAT
    assert r["source"] == si.SOURCE_DEFAULT


def test_stdio_reserved_claim_is_allowed():
    """⚠ Deliberate, and worth pinning down so a later 'tightening' does not break
    local tools: with no request there is no privilege boundary to cross."""
    r = si.resolve_seat("user_whoever", request=None)
    assert r["seat"] == "user_whoever"
    assert r["refused"] is False


# --- Branch 2: verified JWT derives the seat -------------------------------------------

def test_verified_jwt_derives_the_seat():
    r = si.resolve_seat("", request=FakeRequest("good-token"),
                        read_jwt=read_jwt, decode_jwt=decode_good)
    assert r["seat"] == "user_abc-123"
    assert r["source"] == si.SOURCE_VERIFIED
    assert r["verified_sub"] == "abc-123"


def test_verified_jwt_overrides_a_disagreeing_claim():
    """⭐ THE CENTRAL TEST. A caller asking to be someone else gets their own seat,
    and the disagreement is reported rather than silently dropped."""
    r = si.resolve_seat("user_someone-else", request=FakeRequest("good-token"),
                        read_jwt=read_jwt, decode_jwt=decode_good)
    assert r["seat"] == "user_abc-123", "a claim must never beat a verified identity"
    assert r["source"] == si.SOURCE_VERIFIED
    assert r["overridden"] is True


def test_agreeing_claim_is_not_flagged_as_overridden():
    """`overridden` has to mean something. If it fired on every verified call it would
    be noise in the journal and nobody would read it."""
    r = si.resolve_seat("user_abc-123", request=FakeRequest("good-token"),
                        read_jwt=read_jwt, decode_jwt=decode_good)
    assert r["seat"] == "user_abc-123"
    assert r["overridden"] is False


# --- Branch 3: request, no usable identity (shared-key caller) -------------------------

def test_shared_key_caller_may_use_an_ordinary_seat():
    """The additive half: an agent naming its own non-reserved seat still works, so
    no existing caller breaks."""
    r = si.resolve_seat("sage-3", request=FakeRequest(None),
                        read_jwt=read_jwt, decode_jwt=decode_good)
    assert r["seat"] == "sage-3"
    assert r["source"] == si.SOURCE_CLAIMED
    assert r["refused"] is False


def test_shared_key_caller_cannot_claim_a_user_seat():
    """⭐⭐ THE HOLE THIS MODULE EXISTS TO CLOSE. Over HTTP the only gate is one shared
    bearer token, so without this refusal any holder of it reads any signed-in seat."""
    r = si.resolve_seat("user_abc-123", request=FakeRequest(None),
                        read_jwt=read_jwt, decode_jwt=decode_good)
    assert r["seat"] == si.DEFAULT_SEAT
    assert r["refused"] is True
    assert r["source"] == si.SOURCE_DEFAULT


def test_shared_key_caller_cannot_claim_an_anon_seat():
    """`anon_<tab>` is also server-minted, from a sanitized header."""
    r = si.resolve_seat("anon_sometab", request=FakeRequest(None),
                        read_jwt=read_jwt, decode_jwt=decode_good)
    assert r["seat"] == si.DEFAULT_SEAT
    assert r["refused"] is True


def test_expired_token_does_not_become_a_trusted_anonymous_caller():
    """A failed assertion of identity is not the same as making no assertion. The claim
    it arrived with is still refused if it names a reserved seat."""
    r = si.resolve_seat("user_abc-123", request=FakeRequest("stale"),
                        read_jwt=read_jwt, decode_jwt=decode_always_raises)
    assert r["seat"] == si.DEFAULT_SEAT
    assert r["refused"] is True
    assert r["verified_sub"] == ""


def test_decoder_without_sub_claim_falls_through():
    """A token that verifies but carries no subject identifies nobody."""
    r = si.resolve_seat("user_abc-123", request=FakeRequest("t"),
                        read_jwt=read_jwt, decode_jwt=lambda t: {"aud": "authenticated"})
    assert r["refused"] is True
    assert r["seat"] == si.DEFAULT_SEAT


# --- robustness: this helper sits in front of every per-seat tool ----------------------

def test_never_raises_when_the_reader_explodes():
    """⛔ An identity helper that can throw is a denial of service on every tool that
    calls it. It must degrade to a non-privileged seat instead."""
    def boom(_request):
        raise RuntimeError("header parsing blew up")

    r = si.resolve_seat("user_abc-123", request=FakeRequest("x"),
                        read_jwt=boom, decode_jwt=decode_good)
    assert r["refused"] is True
    assert r["seat"] == si.DEFAULT_SEAT


def test_seat_is_sanitized_and_bounded():
    """A seat becomes a dict key and a log field on every call."""
    r = si.resolve_seat("../../etc/passwd; DROP TABLE", request=None)
    assert r["seat"] == "etcpasswdDROPTABLE"

    r = si.resolve_seat("z" * 500, request=None)
    assert len(r["seat"]) == si.MAX_SEAT_CHARS


def test_sanitation_cannot_be_used_to_smuggle_a_reserved_prefix():
    """⚠ Sanitation runs BEFORE the prefix check, so a claim that only becomes reserved
    after stripping must still be refused. 'user.abc' -> 'userabc' is not reserved, but
    'user_%abc' -> 'user_abc' is -- and that is the one worth pinning."""
    r = si.resolve_seat("user_%abc-123", request=FakeRequest(None),
                        read_jwt=read_jwt, decode_jwt=decode_good)
    assert r["seat"] == si.DEFAULT_SEAT
    assert r["refused"] is True


def test_seat_for_journal_keeps_identity_provenance():
    """'writer: andy' is only evidence if the record also says whether anyone checked."""
    verified = si.resolve_seat("", request=FakeRequest("good-token"),
                               read_jwt=read_jwt, decode_jwt=decode_good)
    row = si.seat_for_journal(verified)
    assert row["writer"] == "user_abc-123"
    assert row["identity_source"] == si.SOURCE_VERIFIED
    assert row["identity_overridden"] is False
    assert row["identity_refused"] is False

    refused = si.resolve_seat("user_abc-123", request=FakeRequest(None),
                              read_jwt=read_jwt, decode_jwt=decode_good)
    assert si.seat_for_journal(refused)["identity_refused"] is True
