"""Tests for the database loader."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from voltedge.db.database import Base
from voltedge.db.models import EnergyReading


@pytest.fixture(autouse=True)
def seeded_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "loader_test.db")
    import voltedge.db.database as dbmod
    engine  = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(dbmod, "engine",  engine)
    monkeypatch.setattr(dbmod, "Session", Session)
    Base.metadata.create_all(engine)
    # Seed a small amount of data
    from voltedge.db.seeder import seed
    seed()
    yield


def test_load_returns_dataframe():
    from voltedge.db.loader import load_site_readings
    import pandas as pd
    df = load_site_readings("LONDON-FACTORY-01", hours=168)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_returns_required_columns():
    from voltedge.db.loader import load_site_readings
    df = load_site_readings("LONDON-FACTORY-01", hours=24)
    for col in ("timestamp", "kwh", "voltage_v", "current_a", "power_factor"):
        assert col in df.columns, f"Missing column: {col}"


def test_load_hours_window():
    from voltedge.db.loader import load_site_readings
    df_24  = load_site_readings("LONDON-FACTORY-01", hours=24)
    df_168 = load_site_readings("LONDON-FACTORY-01", hours=168)
    assert len(df_168) > len(df_24)


def test_load_all_sites():
    from voltedge.db.loader import load_site_readings
    for site in ("LONDON-FACTORY-01", "DUBAI-OFFICE-01",
                 "ROTTERDAM-WAREHOUSE-01", "FRANKFURT-DC-01"):
        df = load_site_readings(site, hours=168)
        assert len(df) > 0, f"No data for {site}"


def test_load_fallback_on_empty(monkeypatch):
    """Unknown site falls back to simulator without raising."""
    from voltedge.db.loader import load_site_readings
    df = load_site_readings("UNKNOWN-SITE-99", hours=24)
    # Fallback returns either empty DF or simulator data — must not raise
    assert df is not None


def test_transformer_accepts_db_output():
    """DB data should pass through the transformer without error."""
    from voltedge.db.loader import load_site_readings
    from voltedge.processing.transformer import EnergyTransformer
    raw = load_site_readings("FRANKFURT-DC-01", hours=168)
    transformed = EnergyTransformer().transform(raw)
    assert "hour" in transformed.columns
    assert "is_business_hour" in transformed.columns
    assert len(transformed) > 0
