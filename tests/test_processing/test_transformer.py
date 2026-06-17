"""Tests for EnergyTransformer."""

from __future__ import annotations

import asyncio
import numpy as np
import pandas as pd
import pytest

from voltedge.ingestion.simulators import SimulatedSiteConnector
from voltedge.processing.transformer import EnergyTransformer


def _raw_df(hours: int = 200, seed: int = 0) -> pd.DataFrame:
    connector = SimulatedSiteConnector("T", "factory", hours=hours, seed=seed)
    result = asyncio.run(connector.fetch())
    return result.data


def test_transform_adds_temporal_features():
    df = EnergyTransformer().transform(_raw_df())
    for col in ("hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos"):
        assert col in df.columns, f"Missing: {col}"


def test_transform_adds_lag_features():
    df = EnergyTransformer().transform(_raw_df(hours=200))
    assert "kwh_lag_1h" in df.columns
    assert "kwh_lag_24h" in df.columns


def test_transform_adds_rolling_features():
    df = EnergyTransformer().transform(_raw_df())
    assert "kwh_roll_mean_24h" in df.columns
    assert "kwh_roll_std_24h" in df.columns


def test_transform_no_negative_kwh_after_cleaning():
    raw = _raw_df(hours=300)
    # Manually inject some negatives
    raw.loc[raw.index[:5], "kwh"] = -999.0
    df = EnergyTransformer().transform(raw)
    # After interpolation, no negatives should survive
    assert (df["kwh"].dropna() >= 0).all()


def test_transform_preserves_row_count():
    raw = _raw_df(hours=168)
    df = EnergyTransformer().transform(raw)
    assert len(df) == len(raw)


def test_transform_cyclic_features_in_range():
    df = EnergyTransformer().transform(_raw_df())
    assert df["hour_sin"].between(-1, 1).all()
    assert df["hour_cos"].between(-1, 1).all()


def test_transform_empty_input_returns_empty():
    df = EnergyTransformer().transform(pd.DataFrame())
    assert df.empty


def test_transform_is_idempotent():
    """Transforming twice should not double-apply features."""
    transformer = EnergyTransformer()
    raw = _raw_df()
    df1 = transformer.transform(raw)
    # Second transform on already-transformed data should not error
    df2 = transformer.transform(df1)
    assert len(df2) == len(df1)
