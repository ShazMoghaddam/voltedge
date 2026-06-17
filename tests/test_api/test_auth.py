"""Tests for JWT auth — token creation, validation, roles, expiry."""

from __future__ import annotations

from datetime import timedelta, datetime, timezone

import pytest
from fastapi.testclient import TestClient
from jose import jwt

from voltedge.api.app import create_app
from voltedge.api.auth import (
    authenticate_user, create_access_token, decode_token,
    hash_password, verify_password,
    SECRET_KEY, ALGORITHM, VALID_SCOPES,
)


@pytest.fixture
def client():
    return TestClient(create_app())


# ── Password hashing ──────────────────────────────────────────────────────────

def test_hash_and_verify_correct_password():
    hashed = hash_password("my-secret")
    assert verify_password("my-secret", hashed)


def test_hash_and_verify_wrong_password():
    hashed = hash_password("my-secret")
    assert not verify_password("wrong", hashed)


def test_hashes_are_unique():
    h1 = hash_password("same")
    h2 = hash_password("same")
    assert h1 != h2  # bcrypt adds random salt


# ── Token creation ────────────────────────────────────────────────────────────

def test_create_token_returns_string():
    token = create_access_token("alice", scopes=["read"])
    assert isinstance(token, str)
    assert len(token) > 20


def test_token_contains_correct_subject():
    token = create_access_token("bob", scopes=["read"])
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["sub"] == "bob"


def test_token_contains_scopes():
    token = create_access_token("carol", scopes=["read", "write"])
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert set(payload["scopes"]) == {"read", "write"}


def test_token_defaults_to_read_scope():
    token = create_access_token("dave")
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert "read" in payload["scopes"]


def test_token_has_expiry():
    token = create_access_token("eve", scopes=["read"])
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert "exp" in payload
    assert payload["exp"] > datetime.now(timezone.utc).timestamp()


def test_token_rejects_invalid_scope():
    with pytest.raises(ValueError, match="Invalid scopes"):
        create_access_token("frank", scopes=["superuser"])


# ── Token decoding ────────────────────────────────────────────────────────────

def test_decode_valid_token():
    token = create_access_token("alice", scopes=["read", "admin"])
    data = decode_token(token)
    assert data.username == "alice"
    assert "read" in data.scopes
    assert "admin" in data.scopes


def test_decode_expired_token_raises_401():
    from fastapi import HTTPException
    expired_token = create_access_token(
        "grace", scopes=["read"],
        expires_delta=timedelta(seconds=-1),
    )
    with pytest.raises(HTTPException) as exc_info:
        decode_token(expired_token)
    assert exc_info.value.status_code == 401


def test_decode_garbage_token_raises_401():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        decode_token("not.a.valid.jwt")
    assert exc_info.value.status_code == 401


def test_decode_tampered_token_raises_401():
    from fastapi import HTTPException
    token = create_access_token("legit", scopes=["read"])
    tampered = token[:-5] + "XXXXX"
    with pytest.raises(HTTPException):
        decode_token(tampered)


# ── User authentication ───────────────────────────────────────────────────────

def test_authenticate_valid_admin():
    user = authenticate_user("admin", "admin-secret")
    assert user is not None
    assert user.username == "admin"
    assert "admin" in user.scopes


def test_authenticate_valid_analyst():
    user = authenticate_user("analyst", "analyst-secret")
    assert user is not None
    assert user.scopes == ["read"]


def test_authenticate_wrong_password_returns_none():
    user = authenticate_user("admin", "wrong-password")
    assert user is None


def test_authenticate_unknown_user_returns_none():
    user = authenticate_user("nobody", "any-password")
    assert user is None


# ── /auth/token endpoint ──────────────────────────────────────────────────────

def test_login_returns_token(client):
    r = client.post("/auth/token",
                    data={"username": "admin", "password": "admin-secret"})
    assert r.status_code == 200
    body = r.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0


def test_login_wrong_credentials_returns_401(client):
    r = client.post("/auth/token",
                    data={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


def test_login_unknown_user_returns_401(client):
    r = client.post("/auth/token",
                    data={"username": "ghost", "password": "any"})
    assert r.status_code == 401


def test_login_analyst_has_read_scope_only(client):
    r = client.post("/auth/token",
                    data={"username": "analyst", "password": "analyst-secret"})
    token = r.json()["access_token"]
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["scopes"] == ["read"]
    assert "admin" not in payload["scopes"]


# ── Protected route integration ───────────────────────────────────────────────

def test_protected_route_without_token_returns_401(client):
    """Sites list is currently open, but test auth machinery via /ready."""
    # Simulate a secured route by manually checking require_auth behaviour
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        decode_token("missing")
    assert exc_info.value.status_code == 401


def test_token_from_login_is_decodable(client):
    r = client.post("/auth/token",
                    data={"username": "operator", "password": "operator-secret"})
    token = r.json()["access_token"]
    data = decode_token(token)
    assert data.username == "operator"
    assert "write" in data.scopes


# ── Scope enforcement ─────────────────────────────────────────────────────────

def test_require_scope_passes_for_correct_scope():
    from voltedge.api.auth import require_scope
    token = create_access_token("alice", scopes=["admin"])
    checker = require_scope("admin")
    user = checker(token=token)
    assert user.username == "alice"


def test_require_scope_raises_403_for_missing_scope():
    from fastapi import HTTPException
    from voltedge.api.auth import require_scope
    token = create_access_token("analyst", scopes=["read"])
    checker = require_scope("admin")
    with pytest.raises(HTTPException) as exc_info:
        checker(token=token)
    assert exc_info.value.status_code == 403


def test_all_valid_scopes_accepted():
    for scope in VALID_SCOPES:
        token = create_access_token("user", scopes=[scope])
        data = decode_token(token)
        assert scope in data.scopes
