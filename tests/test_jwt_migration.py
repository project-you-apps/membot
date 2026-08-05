"""membot's Supabase JWT verification across the legacy/asymmetric migration.

membot verifies user tokens FAIL-CLOSED: an unverifiable token means no user, which means
every Mempack operation refuses. So this is the code that decides whether the legacy JWT
secret can be revoked at all -- without a working asymmetric path, revoking would lock
every authenticated caller out of the droplet.

Mirrors vector-plus-studio-repo/api/tests/test_auth_jwks.py.
"""

import datetime as _dt
import os
import sys
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import membot_server as ms  # noqa: E402

SEAT = "11111111-1111-1111-1111-111111111111"
LEGACY_SECRET = "legacy-shared-secret-value"
PROJECT = "https://uikdknfxcqklldmfshug.supabase.co"


def _claims(**over):
    now = _dt.datetime.now(_dt.timezone.utc)
    base = {"sub": SEAT, "aud": ms._JWT_AUD, "role": "authenticated",
            "exp": now + _dt.timedelta(hours=1), "iat": now}
    base.update(over)
    return base


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    ms._jwks_cache = ()
    yield
    ms._jwks_cache = ()


@pytest.fixture(scope="module")
def rsa_pair():
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = k.private_bytes(serialization.Encoding.PEM,
                           serialization.PrivateFormat.PKCS8,
                           serialization.NoEncryption()).decode()
    pub = k.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return priv, pub


def _install_jwks(monkeypatch, public_pem):
    class _Key:
        key = public_pem

    class _Client:
        def get_signing_key_from_jwt(self, token):
            return _Key()

    monkeypatch.setattr(ms, "_jwks_client_for", lambda url: _Client())


def test_legacy_hs256_still_verifies(monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", LEGACY_SECRET)
    token = jwt.encode(_claims(), LEGACY_SECRET, algorithm="HS256")
    assert ms._decode_supabase_jwt(token)["sub"] == SEAT


def test_asymmetric_verifies_via_jwks(monkeypatch, rsa_pair):
    priv, pub = rsa_pair
    monkeypatch.setenv("SUPABASE_URL", PROJECT)
    _install_jwks(monkeypatch, pub)
    token = jwt.encode(_claims(), priv, algorithm="RS256")
    assert ms._decode_supabase_jwt(token)["sub"] == SEAT


def test_both_kinds_verify_mid_migration(monkeypatch, rsa_pair):
    """The whole point: tokens issued before the rotation keep working after it."""
    priv, pub = rsa_pair
    monkeypatch.setenv("SUPABASE_JWT_SECRET", LEGACY_SECRET)
    monkeypatch.setenv("SUPABASE_URL", PROJECT)
    _install_jwks(monkeypatch, pub)
    old = jwt.encode(_claims(), LEGACY_SECRET, algorithm="HS256")
    new = jwt.encode(_claims(), priv, algorithm="RS256")
    assert ms._decode_supabase_jwt(old)["sub"] == SEAT
    assert ms._decode_supabase_jwt(new)["sub"] == SEAT


def test_hs256_refused_after_the_secret_is_revoked(monkeypatch, rsa_pair):
    """Post-revoke: URL set, secret gone. HS256 is no longer something we issue."""
    _priv, pub = rsa_pair
    monkeypatch.setenv("SUPABASE_URL", PROJECT)
    _install_jwks(monkeypatch, pub)
    token = jwt.encode(_claims(), LEGACY_SECRET, algorithm="HS256")
    with pytest.raises(jwt.PyJWTError):
        ms._decode_supabase_jwt(token)


def test_public_key_cannot_be_used_as_an_hmac_secret(monkeypatch, rsa_pair):
    """Algorithm confusion. A JWKS is PUBLIC -- that key is not a secret.

    Forge by signing HS256 with the published public key as the HMAC secret. A verifier
    that trusts the header's alg against whichever key it holds accepts this and grants the
    attacker any `sub` they name. Hand-rolled because PyJWT refuses to ENCODE it -- that
    guardrail stops us writing the bug, not receiving the attack.
    """
    import base64, hashlib, hmac, json
    _priv, pub = rsa_pair
    monkeypatch.setenv("SUPABASE_JWT_SECRET", LEGACY_SECRET)
    monkeypatch.setenv("SUPABASE_URL", PROJECT)
    _install_jwks(monkeypatch, pub)

    def seg(d):
        return base64.urlsafe_b64encode(
            json.dumps(d, separators=(",", ":"), default=str).encode()).rstrip(b"=")

    signing_input = seg({"alg": "HS256", "typ": "JWT"}) + b"." + seg(_claims(sub="attacker"))
    sig = hmac.new(pub.encode(), signing_input, hashlib.sha256).digest()
    forged = (signing_input + b"." + base64.urlsafe_b64encode(sig).rstrip(b"=")).decode()

    with pytest.raises(jwt.PyJWTError):
        ms._decode_supabase_jwt(forged)


def test_unsupported_algorithm_is_refused(monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", LEGACY_SECRET)
    token = jwt.encode(_claims(), LEGACY_SECRET, algorithm="HS512")
    with pytest.raises(jwt.PyJWTError):
        ms._decode_supabase_jwt(token)


def test_none_algorithm_is_refused(monkeypatch):
    monkeypatch.setenv("SUPABASE_JWT_SECRET", LEGACY_SECRET)
    token = jwt.encode(_claims(), key="", algorithm="none")
    with pytest.raises(jwt.PyJWTError):
        ms._decode_supabase_jwt(token)
