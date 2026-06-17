"""
VoltEdge RBAC — Service layer

Provides:
  - Database setup (SQLite dev / PostgreSQL prod)
  - Seed built-in roles and permissions
  - User CRUD
  - Permission checking
  - FastAPI dependency: require_permission(resource, action)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, status
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

import bcrypt

from voltedge.rbac.models import Base, Permission, Role, User
from voltedge.utils.logger import get_logger

log = get_logger(__name__)

# ── Built-in permission catalogue ──────────────────────────────────────────────

PERMISSIONS: list[dict] = [
    # Sites
    {"resource": "sites",    "action": "read",    "description": "List and read site data"},
    {"resource": "sites",    "action": "write",   "description": "Write / simulate site data"},
    # Analytics
    {"resource": "analytics","action": "read",    "description": "Run forecast, anomaly, ESG, optimize"},
    {"resource": "analytics","action": "write",   "description": "Configure analytics parameters"},
    # ESG
    {"resource": "esg",      "action": "read",    "description": "View ESG metrics"},
    {"resource": "esg",      "action": "export",  "description": "Export ESG PDF reports"},
    # Pipeline
    {"resource": "pipeline", "action": "trigger", "description": "Trigger data ingestion"},
    {"resource": "pipeline", "action": "admin",   "description": "Configure pipeline settings"},
    # Models
    {"resource": "models",   "action": "read",    "description": "View model metadata and metrics"},
    {"resource": "models",   "action": "retrain", "description": "Trigger model retraining"},
    # Users
    {"resource": "users",    "action": "read",    "description": "List users"},
    {"resource": "users",    "action": "write",   "description": "Create and update users"},
    {"resource": "users",    "action": "admin",   "description": "Assign roles, delete users"},
    # Audit
    {"resource": "audit",    "action": "read",    "description": "View audit logs"},
]

# ── Built-in role definitions ──────────────────────────────────────────────────

ROLES: dict[str, dict] = {
    "viewer": {
        "description": "Read-only access to sites and analytics",
        "permissions": [
            "sites:read", "analytics:read", "esg:read", "models:read",
        ],
    },
    "analyst": {
        "description": "Can run analytics and export ESG reports",
        "permissions": [
            "sites:read", "analytics:read", "analytics:write",
            "esg:read", "esg:export", "models:read",
        ],
    },
    "operator": {
        "description": "Can trigger pipelines and retrain models",
        "permissions": [
            "sites:read", "sites:write",
            "analytics:read", "analytics:write",
            "esg:read", "esg:export",
            "pipeline:trigger", "models:read", "models:retrain",
        ],
    },
    "admin": {
        "description": "Full platform access including user management",
        "permissions": [p["resource"] + ":" + p["action"] for p in PERMISSIONS],
    },
}


# ── Database setup ────────────────────────────────────────────────────────────

def _make_engine(db_url: str = "sqlite:///./voltedge_rbac.db"):
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
        echo=False,
    )
    # Enable WAL mode for SQLite (better concurrent read performance)
    if "sqlite" in db_url:
        @event.listens_for(engine, "connect")
        def set_wal(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
    return engine


@lru_cache(maxsize=1)
def get_engine(db_url: str = "sqlite:///./voltedge_rbac.db"):
    engine = _make_engine(db_url)
    Base.metadata.create_all(engine)
    return engine


def make_session_factory(db_url: str = "sqlite:///./voltedge_rbac.db"):
    return sessionmaker(autocommit=False, autoflush=False, bind=get_engine(db_url))


SessionLocal = make_session_factory()


def get_db():
    """FastAPI dependency: yield a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DBDep = Annotated[Session, Depends(get_db)]


# ── Seed ──────────────────────────────────────────────────────────────────────

def seed_roles_and_permissions(db: Session) -> None:
    """
    Idempotent: creates built-in permissions and roles if they don't exist.
    Safe to call on every startup.
    """
    # Create permissions
    perm_map: dict[str, Permission] = {}
    for p_def in PERMISSIONS:
        existing = db.query(Permission).filter_by(
            resource=p_def["resource"], action=p_def["action"]
        ).first()
        if not existing:
            perm = Permission(
                resource=p_def["resource"],
                action=p_def["action"],
                description=p_def["description"],
            )
            db.add(perm)
            db.flush()
            existing = perm
        perm_map[f"{p_def['resource']}:{p_def['action']}"] = existing

    # Create roles
    for role_name, role_def in ROLES.items():
        existing_role = db.query(Role).filter_by(name=role_name).first()
        if not existing_role:
            role = Role(
                name=role_name,
                description=role_def["description"],
                is_system=True,
            )
            db.add(role)
            db.flush()
            existing_role = role

        # Sync permissions
        target_perms = {perm_map[p] for p in role_def["permissions"] if p in perm_map}
        existing_role.permissions = list(target_perms)

    db.commit()
    log.info("rbac.seeded", roles=list(ROLES.keys()), permissions=len(PERMISSIONS))


# ── User CRUD ─────────────────────────────────────────────────────────────────

class RBACService:
    """High-level RBAC operations."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role_names: list[str] | None = None,
    ) -> User:
        if self.db.query(User).filter_by(username=username).first():
            raise ValueError(f"Username '{username}' already exists.")

        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        user = User(username=username, email=email, hashed_password=hashed)

        roles = role_names or ["viewer"]
        for rn in roles:
            role = self.db.query(Role).filter_by(name=rn).first()
            if role:
                user.roles.append(role)
            else:
                log.warning("rbac.unknown_role", role=rn)

        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        log.info("rbac.user_created", username=username, roles=roles)
        return user

    def get_user(self, username: str) -> User | None:
        return self.db.query(User).filter_by(username=username, is_active=True).first()

    def authenticate(self, username: str, password: str) -> User | None:
        user = self.get_user(username)
        if not user:
            return None
        if not bcrypt.checkpw(password.encode(), user.hashed_password.encode()):
            return None
        return user

    def assign_role(self, username: str, role_name: str) -> User:
        user = self.db.query(User).filter_by(username=username).first()
        if not user:
            raise ValueError(f"User '{username}' not found.")
        role = self.db.query(Role).filter_by(name=role_name).first()
        if not role:
            raise ValueError(f"Role '{role_name}' not found.")
        if role not in user.roles:
            user.roles.append(role)
            self.db.commit()
            log.info("rbac.role_assigned", username=username, role=role_name)
        return user

    def revoke_role(self, username: str, role_name: str) -> User:
        user = self.db.query(User).filter_by(username=username).first()
        if not user:
            raise ValueError(f"User '{username}' not found.")
        role = self.db.query(Role).filter_by(name=role_name).first()
        if role and role in user.roles:
            user.roles.remove(role)
            self.db.commit()
            log.info("rbac.role_revoked", username=username, role=role_name)
        return user

    def list_users(self) -> list[User]:
        return self.db.query(User).filter_by(is_active=True).all()

    def list_roles(self) -> list[Role]:
        return self.db.query(Role).all()

    def check_permission(self, username: str, resource: str, action: str) -> bool:
        user = self.get_user(username)
        if not user:
            return False
        return user.has_permission(resource, action)

    def deactivate_user(self, username: str) -> None:
        user = self.db.query(User).filter_by(username=username).first()
        if user:
            user.is_active = False
            self.db.commit()
            log.info("rbac.user_deactivated", username=username)


# ── FastAPI dependency ────────────────────────────────────────────────────────

def require_permission(resource: str, action: str):
    """
    FastAPI dependency factory: validates JWT + checks RBAC permission.

    Usage:
        @router.get("/esg/report")
        def get_report(user = Depends(require_permission("esg", "export"))):
            ...
    """
    from voltedge.api.auth import decode_token, _oauth2_scheme

    def _checker(
        token: Annotated[str, Depends(_oauth2_scheme)],
        db:    DBDep,
    ):
        token_data = decode_token(token)
        svc = RBACService(db)
        if not svc.check_permission(token_data.username, resource, action):
            log.warning(
                "rbac.permission_denied",
                user=token_data.username,
                resource=resource,
                action=action,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{resource}:{action}' required.",
            )
        return token_data

    return _checker
