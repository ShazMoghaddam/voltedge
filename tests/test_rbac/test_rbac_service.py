"""Tests for RBAC service — uses in-memory SQLite per test."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from voltedge.rbac.models import Base, Permission, Role, User
from voltedge.rbac.service import (
    RBACService, PERMISSIONS, ROLES, seed_roles_and_permissions,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    """Fresh in-memory SQLite DB per test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    seed_roles_and_permissions(session)
    yield session
    session.close()


@pytest.fixture
def svc(db):
    return RBACService(db)


# ── Seed tests ────────────────────────────────────────────────────────────────

def test_seed_creates_all_permissions(db):
    perms = db.query(Permission).all()
    assert len(perms) == len(PERMISSIONS)


def test_seed_creates_all_roles(db):
    roles = db.query(Role).all()
    assert len(roles) == len(ROLES)


def test_seed_is_idempotent(db):
    seed_roles_and_permissions(db)   # Second call
    seed_roles_and_permissions(db)   # Third call
    perms = db.query(Permission).all()
    assert len(perms) == len(PERMISSIONS)


def test_admin_has_all_permissions(db):
    admin = db.query(Role).filter_by(name="admin").first()
    perm_names = {p.name for p in admin.permissions}
    for p in PERMISSIONS:
        assert f"{p['resource']}:{p['action']}" in perm_names


def test_viewer_cannot_write(db):
    viewer = db.query(Role).filter_by(name="viewer").first()
    perm_names = {p.name for p in viewer.permissions}
    assert "sites:write" not in perm_names
    assert "pipeline:trigger" not in perm_names
    assert "users:admin" not in perm_names


def test_operator_can_trigger_pipeline(db):
    operator = db.query(Role).filter_by(name="operator").first()
    perm_names = {p.name for p in operator.permissions}
    assert "pipeline:trigger" in perm_names


def test_analyst_can_export_esg(db):
    analyst = db.query(Role).filter_by(name="analyst").first()
    perm_names = {p.name for p in analyst.permissions}
    assert "esg:export" in perm_names


# ── User CRUD ─────────────────────────────────────────────────────────────────

def test_create_user_returns_user(svc):
    user = svc.create_user("alice", "alice@example.com", "secret123", ["analyst"])
    assert user.username == "alice"
    assert user.email == "alice@example.com"
    assert user.is_active is True


def test_create_user_assigns_role(svc):
    user = svc.create_user("bob", "bob@example.com", "pass", ["operator"])
    role_names = {r.name for r in user.roles}
    assert "operator" in role_names


def test_create_user_defaults_to_viewer(svc):
    user = svc.create_user("carol", "carol@example.com", "pass")
    role_names = {r.name for r in user.roles}
    assert "viewer" in role_names


def test_create_duplicate_user_raises(svc):
    svc.create_user("dave", "dave@example.com", "pass")
    with pytest.raises(ValueError, match="already exists"):
        svc.create_user("dave", "dave2@example.com", "pass2")


def test_get_user_returns_active_user(svc):
    svc.create_user("eve", "eve@example.com", "pass", ["analyst"])
    user = svc.get_user("eve")
    assert user is not None
    assert user.username == "eve"


def test_get_user_returns_none_for_missing(svc):
    assert svc.get_user("ghost") is None


def test_password_is_hashed(svc):
    user = svc.create_user("frank", "frank@example.com", "mysecret")
    assert user.hashed_password != "mysecret"
    assert user.hashed_password.startswith("$2b$")


# ── Authentication ────────────────────────────────────────────────────────────

def test_authenticate_correct_password(svc):
    svc.create_user("grace", "grace@example.com", "mypass123")
    user = svc.authenticate("grace", "mypass123")
    assert user is not None
    assert user.username == "grace"


def test_authenticate_wrong_password(svc):
    svc.create_user("hank", "hank@example.com", "correct")
    assert svc.authenticate("hank", "wrong") is None


def test_authenticate_unknown_user(svc):
    assert svc.authenticate("nobody", "pass") is None


# ── Role assignment ───────────────────────────────────────────────────────────

def test_assign_role(svc):
    svc.create_user("ivan", "ivan@example.com", "pass", ["viewer"])
    user = svc.assign_role("ivan", "analyst")
    role_names = {r.name for r in user.roles}
    assert "analyst" in role_names
    assert "viewer" in role_names   # Original role retained


def test_assign_same_role_is_idempotent(svc):
    svc.create_user("jane", "jane@example.com", "pass", ["viewer"])
    svc.assign_role("jane", "viewer")
    user = svc.get_user("jane")
    viewer_count = sum(1 for r in user.roles if r.name == "viewer")
    assert viewer_count == 1


def test_revoke_role(svc):
    svc.create_user("kate", "kate@example.com", "pass", ["analyst", "operator"])
    user = svc.revoke_role("kate", "operator")
    role_names = {r.name for r in user.roles}
    assert "operator" not in role_names
    assert "analyst" in role_names


def test_assign_unknown_role_raises(svc):
    svc.create_user("leo", "leo@example.com", "pass")
    with pytest.raises(ValueError, match="not found"):
        svc.assign_role("leo", "superuser")


# ── Permission checking ───────────────────────────────────────────────────────

def test_check_permission_true_for_valid(svc):
    svc.create_user("mike", "mike@example.com", "pass", ["operator"])
    assert svc.check_permission("mike", "pipeline", "trigger") is True


def test_check_permission_false_for_insufficient_role(svc):
    svc.create_user("nina", "nina@example.com", "pass", ["viewer"])
    assert svc.check_permission("nina", "pipeline", "trigger") is False


def test_admin_has_all_permissions_via_check(svc):
    svc.create_user("oscar", "oscar@example.com", "pass", ["admin"])
    for p in PERMISSIONS:
        assert svc.check_permission("oscar", p["resource"], p["action"]) is True


def test_check_permission_false_for_unknown_user(svc):
    assert svc.check_permission("unknown", "sites", "read") is False


def test_user_all_permissions_set(svc):
    svc.create_user("paula", "paula@example.com", "pass", ["analyst"])
    user = svc.get_user("paula")
    perms = user.all_permissions()
    assert "sites:read" in perms
    assert "esg:export" in perms
    assert "users:admin" not in perms


# ── Deactivation ──────────────────────────────────────────────────────────────

def test_deactivate_user(svc):
    svc.create_user("quinn", "quinn@example.com", "pass")
    svc.deactivate_user("quinn")
    assert svc.get_user("quinn") is None


def test_list_users_excludes_inactive(svc):
    svc.create_user("rex", "rex@example.com", "pass")
    svc.create_user("sue", "sue@example.com", "pass")
    svc.deactivate_user("rex")
    users = svc.list_users()
    usernames = {u.username for u in users}
    assert "rex" not in usernames
    assert "sue" in usernames


# ── Role listing ──────────────────────────────────────────────────────────────

def test_list_roles_returns_all(svc):
    roles = svc.list_roles()
    role_names = {r.name for r in roles}
    for r in ROLES:
        assert r in role_names


def test_privilege_hierarchy(svc):
    """viewer < analyst < operator < admin by permission count."""
    role_perm_counts = {}
    for rname in ("viewer", "analyst", "operator", "admin"):
        role = svc.db.query(Role).filter_by(name=rname).first()
        role_perm_counts[rname] = len(role.permissions)

    assert role_perm_counts["viewer"]   < role_perm_counts["analyst"]
    assert role_perm_counts["analyst"]  < role_perm_counts["operator"]
    assert role_perm_counts["operator"] < role_perm_counts["admin"]
