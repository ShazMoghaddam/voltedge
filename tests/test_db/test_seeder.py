"""Tests for the 90-day seeder."""
from __future__ import annotations

import os
import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from voltedge.db.database import Base
from voltedge.db.models import EnergyReading


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Redirect the seeder to a temp database for every test."""
    db_path = str(tmp_path / "seed_test.db")
    monkeypatch.setenv("VOLTEDGE_ENERGY_DB", db_path)
    # Patch engine and Session in the database module
    import voltedge.db.database as dbmod
    engine  = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(dbmod, "engine",  engine)
    monkeypatch.setattr(dbmod, "Session", Session)
    Base.metadata.create_all(engine)
    yield engine, Session


def test_seed_inserts_rows(isolated_db):
    from voltedge.db.seeder import seed
    n = seed()
    assert n > 0


def test_seed_covers_all_sites(isolated_db):
    from voltedge.db.seeder import seed, PROFILES
    engine, Session = isolated_db
    seed()
    with Session() as s:
        sites = {r.site_id for r in s.query(EnergyReading.site_id).distinct()}
    assert sites == set(PROFILES.keys())


def test_seed_90_days_per_site(isolated_db):
    from voltedge.db.seeder import seed, PROFILES, DAYS
    engine, Session = isolated_db
    seed()
    with Session() as s:
        for site_id in PROFILES:
            count = s.query(EnergyReading).filter_by(site_id=site_id).count()
            # Should be close to DAYS * 24; allow ±2 hours for boundary
            assert abs(count - DAYS * 24) <= 2, f"{site_id}: got {count} rows"


def test_seed_is_idempotent(isolated_db):
    from voltedge.db.seeder import seed
    engine, Session = isolated_db
    n1 = seed()
    n2 = seed()   # second call — should skip
    assert n1 > 0
    assert n2 == 0   # skipped — data exists


def test_seed_force_rewipes(isolated_db):
    from voltedge.db.seeder import seed
    engine, Session = isolated_db
    n1 = seed()
    n2 = seed(force=True)
    assert n2 > 0   # re-seeded
    with Session() as s:
        total = s.query(EnergyReading).count()
    assert total == n2   # no duplicates


def test_seed_anomalies_injected(isolated_db):
    from voltedge.db.seeder import seed
    engine, Session = isolated_db
    seed()
    with Session() as s:
        anomalies = s.query(EnergyReading).filter_by(is_anomaly=1).count()
    # 4 sites × ~4 anomalies each = ~16 total
    assert 8 <= anomalies <= 32


def test_seed_kwh_positive(isolated_db):
    from voltedge.db.seeder import seed
    engine, Session = isolated_db
    seed()
    with Session() as s:
        negative = s.query(EnergyReading).filter(EnergyReading.kwh < 0).count()
    assert negative == 0


def test_seed_timestamps_are_utc(isolated_db):
    from voltedge.db.seeder import seed
    engine, Session = isolated_db
    seed()
    with Session() as s:
        row = s.query(EnergyReading).first()
    assert "+00:00" in row.timestamp or row.timestamp.endswith("Z")
