"""
VoltEdge RBAC — Data models

Defines the permission model:

    Role  ──< RolePermission >──  Permission
      │
      └── User ──< UserRole >── Role

Hierarchy (least → most privileged):
    viewer  →  analyst  →  operator  →  admin

Permissions follow the resource:action pattern:
    sites:read, sites:write
    analytics:read, analytics:write
    esg:read, esg:export
    users:read, users:write, users:admin
    pipeline:trigger, pipeline:admin
    models:read, models:retrain

Stored in SQLite (dev) or PostgreSQL (prod) via SQLAlchemy.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, String, Table, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ── Many-to-many association tables ───────────────────────────────────────────

role_permissions = Table(
    "role_permissions", Base.metadata,
    Column("role_id",       String, ForeignKey("roles.id"),       primary_key=True),
    Column("permission_id", String, ForeignKey("permissions.id"), primary_key=True),
)

user_roles = Table(
    "user_roles", Base.metadata,
    Column("user_id",  String, ForeignKey("users.id"),  primary_key=True),
    Column("role_id",  String, ForeignKey("roles.id"),  primary_key=True),
    Column("site_id",  String, nullable=True),          # Optional site-scoped role
)


# ── Models ────────────────────────────────────────────────────────────────────

class Permission(Base):
    """
    Atomic capability: resource + action.
    Examples: sites:read, esg:export, models:retrain
    """
    __tablename__ = "permissions"

    id:          Mapped[str] = mapped_column(String, primary_key=True,
                                              default=lambda: str(uuid.uuid4()))
    resource:    Mapped[str] = mapped_column(String, nullable=False)
    action:      Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, default="")

    roles: Mapped[list["Role"]] = relationship(
        "Role", secondary=role_permissions, back_populates="permissions"
    )

    __table_args__ = (UniqueConstraint("resource", "action", name="uq_resource_action"),)

    @property
    def name(self) -> str:
        return f"{self.resource}:{self.action}"

    def __repr__(self) -> str:
        return f"<Permission {self.name}>"


class Role(Base):
    """
    Named collection of permissions.
    Built-in roles: viewer, analyst, operator, admin.
    Custom roles can be created per tenant.
    """
    __tablename__ = "roles"

    id:          Mapped[str] = mapped_column(String, primary_key=True,
                                              default=lambda: str(uuid.uuid4()))
    name:        Mapped[str] = mapped_column(String, unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String, default="")
    is_system:   Mapped[bool] = mapped_column(Boolean, default=False)

    permissions: Mapped[list[Permission]] = relationship(
        Permission, secondary=role_permissions, back_populates="roles"
    )
    users: Mapped[list["User"]] = relationship(
        "User", secondary=user_roles, back_populates="roles"
    )

    def __repr__(self) -> str:
        return f"<Role {self.name}>"


class User(Base):
    """
    Platform user with hashed credentials and role assignments.
    Passwords are hashed with bcrypt (see voltedge.api.auth).
    """
    __tablename__ = "users"

    id:              Mapped[str]      = mapped_column(String, primary_key=True,
                                                       default=lambda: str(uuid.uuid4()))
    username:        Mapped[str]      = mapped_column(String, unique=True, nullable=False)
    email:           Mapped[str]      = mapped_column(String, unique=True, nullable=False)
    hashed_password: Mapped[str]      = mapped_column(String, nullable=False)
    is_active:       Mapped[bool]     = mapped_column(Boolean, default=True)
    created_at:      Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    last_login:      Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    roles: Mapped[list[Role]] = relationship(
        Role, secondary=user_roles, back_populates="users"
    )

    def has_permission(self, resource: str, action: str) -> bool:
        """Check if this user has a specific permission via any of their roles."""
        return any(
            any(p.resource == resource and p.action == action for p in role.permissions)
            for role in self.roles
        )

    def all_permissions(self) -> set[str]:
        """Return the full set of permission names for this user."""
        return {
            p.name
            for role in self.roles
            for p in role.permissions
        }

    def __repr__(self) -> str:
        return f"<User {self.username}>"
