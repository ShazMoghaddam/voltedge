"""Tests for database models and schema."""
from __future__ import annotations

import os
import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from voltedge.db.database import Base
from voltedge.db.models import EnergyReading, AnomalyEvent


@pytest.fixture()
def tmp_session(tmp_path):
    db_path = str(tmp_path / "test_energy.db")
    engine  = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    with Session() as s:
        yield s


def test_tables_created(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/schema.db")
    Base.metadata.create_all(engine)
    tables = inspect(engine).get_table_names()
    assert "energy_readings" in tables
    assert "anomaly_events" in tables


def test_energy_reading_insert(tmp_session):
    r = EnergyReading(
        site_id="LONDON-FACTORY-01",
        timestamp="2026-01-01T08:00:00+00:00",
        kwh=1234.5,
        voltage_v=400.0,
        current_a=88.5,
        power_factor=0.91,
        temperature_c=12.3,
        is_anomaly=0,
    )
    tmp_session.add(r)
    tmp_session.commit()
    assert tmp_session.query(EnergyReading).count() == 1


def test_anomaly_event_insert(tmp_session):
    e = AnomalyEvent(
        site_id="LONDON-FACTORY-01",
        timestamp="2026-01-01T10:00:00+00:00",
        score=0.93,
        label="sudden_spike",
    )
    tmp_session.add(e)
    tmp_session.commit()
    assert tmp_session.query(AnomalyEvent).count() == 1


def test_unique_constraint(tmp_session):
    """Inserting duplicate (site_id, timestamp) should fail."""
    from sqlalchemy.exc import IntegrityError
    r1 = EnergyReading(site_id="S1", timestamp="2026-01-01T00:00:00+00:00", kwh=100.0)
    r2 = EnergyReading(site_id="S1", timestamp="2026-01-01T00:00:00+00:00", kwh=200.0)
    tmp_session.add(r1)
    tmp_session.commit()
    tmp_session.add(r2)
    with pytest.raises(IntegrityError):
        tmp_session.commit()
