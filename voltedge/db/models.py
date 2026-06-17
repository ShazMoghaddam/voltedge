"""
VoltEdge — ORM models for energy readings and anomaly events.
"""
from __future__ import annotations

from sqlalchemy import Column, Float, Index, Integer, String, Text

from voltedge.db.database import Base


class EnergyReading(Base):
    """One hourly energy reading from a site sensor."""
    __tablename__ = "energy_readings"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    site_id       = Column(String(64), nullable=False, index=True)
    timestamp     = Column(String(32), nullable=False)   # UTC ISO-8601
    kwh           = Column(Float, nullable=False)
    voltage_v     = Column(Float, nullable=True)
    current_a     = Column(Float, nullable=True)
    power_factor  = Column(Float, nullable=True)
    temperature_c = Column(Float, nullable=True)
    is_anomaly    = Column(Integer, default=0)           # 0 / 1

    __table_args__ = (
        Index("ix_site_ts", "site_id", "timestamp", unique=True),
    )


class AnomalyEvent(Base):
    """Detected anomaly events (written by the anomaly model, not the seeder)."""
    __tablename__ = "anomaly_events"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    site_id      = Column(String(64), nullable=False, index=True)
    timestamp    = Column(String(32), nullable=False)
    score        = Column(Float, nullable=False)
    label        = Column(String(64), nullable=True)
    acknowledged = Column(Integer, default=0)
    notes        = Column(Text, nullable=True)
