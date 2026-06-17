"""
VoltEdge — Database engine and session factory.

Uses a separate SQLite file (energy.db) so it never conflicts with
the RBAC database (rbac.db). All timestamps stored as UTC ISO-8601 strings.
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Default to data/energy.db inside the project root
_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "energy.db",
)
DB_PATH = os.environ.get("VOLTEDGE_ENERGY_DB", _DEFAULT_PATH)
DB_URL  = f"sqlite:///{DB_PATH}"

engine  = create_engine(DB_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create all tables if they don't exist."""
    from voltedge.db import models  # noqa: F401 — registers models with Base
    Base.metadata.create_all(engine)
