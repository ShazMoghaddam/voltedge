"""Tests for the StreamingPipeline."""
from __future__ import annotations
import asyncio
import pytest
from voltedge.streaming.broker import InMemoryBroker
from voltedge.streaming.events import EnergyEvent, EventType
from voltedge.streaming.pipeline import StreamingPipeline, _std

SITE = "PIPE-TEST-01"
TS   = "2026-05-15T14:00:00+00:00"


def _raw(kwh=100.0, ts=TS):
    return EnergyEvent(
        event_type=EventType.RAW, site_id=SITE,
        sensor_id="M1", timestamp=ts, kwh=kwh,
    )


async def _run_one_event(event, threshold=0.5):
    """Helper: publish one raw event, collect processed + anomaly outputs."""
    broker = InMemoryBroker()
    pipeline = StreamingPipeline(broker, anomaly_threshold=threshold)

    results: dict[str, list] = {"processed": [], "anomaly": [], "alert": []}

    async def collect(key, topic, n=1):
        async for evt in broker.subscribe(topic):
            results[key].append(evt)
            if len(results[key]) >= n:
                break

    proc_topic  = broker.topic_for(SITE, "processed")
    anom_topic  = broker.topic_for(SITE, "anomaly")
    alert_topic = broker.topic_for(SITE, "alert")

    async def publish_and_close():
        await asyncio.sleep(0.05)
        await broker.publish(broker.topic_for(SITE, "raw"), event)
        await asyncio.sleep(0.3)
        await broker.close()

    await asyncio.gather(
        collect("processed", proc_topic),
        collect("anomaly",   anom_topic),
        publish_and_close(),
    )
    await pipeline.stop()
    return results


# ── Feature extraction ────────────────────────────────────────────────────────

def test_extract_features_has_cyclic_fields():
    broker   = InMemoryBroker()
    pipeline = StreamingPipeline(broker)
    features = pipeline._extract_features(_raw())
    for key in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"):
        assert key in features, f"Missing: {key}"


def test_extract_features_kwh():
    broker   = InMemoryBroker()
    pipeline = StreamingPipeline(broker)
    features = pipeline._extract_features(_raw(kwh=250.0))
    assert features["kwh"] == 250.0


def test_extract_features_cyclic_in_range():
    broker   = InMemoryBroker()
    pipeline = StreamingPipeline(broker)
    features = pipeline._extract_features(_raw())
    assert -1.0 <= features["hour_sin"] <= 1.0
    assert -1.0 <= features["hour_cos"] <= 1.0


def test_rolling_window_lag_after_two_events():
    """_process_one appends kwh THEN calls _extract_features; mirror that here."""
    broker   = InMemoryBroker()
    pipeline = StreamingPipeline(broker)
    pipeline._windows[SITE].append(100.0)   # previous reading
    pipeline._windows[SITE].append(150.0)   # current (as _process_one would append)
    features = pipeline._extract_features(_raw(kwh=150.0))
    assert "kwh_lag_1" in features
    assert features["kwh_lag_1"] == pytest.approx(100.0)


def test_rolling_window_delta():
    broker   = InMemoryBroker()
    pipeline = StreamingPipeline(broker)
    pipeline._windows[SITE].append(100.0)   # previous
    pipeline._windows[SITE].append(120.0)   # current
    f = pipeline._extract_features(_raw(kwh=120.0))
    assert f["kwh_delta"] == pytest.approx(20.0)


def test_rolling_mean_24_after_24_events():
    broker   = InMemoryBroker()
    pipeline = StreamingPipeline(broker)
    for i in range(24):
        pipeline._windows[SITE].append(float(i * 10))
    f = pipeline._extract_features(_raw(kwh=100.0))
    assert "kwh_roll_mean_24" in f


# ── Rule-based scoring ────────────────────────────────────────────────────────

def test_rule_based_normal_score_is_low():
    score, label = StreamingPipeline._rule_based_score({"kwh": 100.0})
    assert score < 0.5
    assert label == "normal"


def test_rule_based_extreme_spike_high_score():
    features = {"kwh": 1000.0, "kwh_pct_change": 500.0}
    score, label = StreamingPipeline._rule_based_score(features)
    assert score >= 0.90
    assert "extreme" in label or "spike" in label


def test_rule_based_sudden_spike():
    features = {"kwh_pct_change": 120.0}
    score, label = StreamingPipeline._rule_based_score(features)
    assert score >= 0.75


def test_rule_based_flat_line():
    features = {"kwh": 50.0, "kwh_roll_std_24": 0.01}
    score, label = StreamingPipeline._rule_based_score(features)
    assert score >= 0.80
    assert "flat_line" in label


def test_rule_based_extreme_consumption():
    features = {"kwh": 500.0, "kwh_roll_mean_24": 100.0}
    score, label = StreamingPipeline._rule_based_score(features)
    assert score >= 0.85
    assert "extreme_consumption" in label


def test_rule_based_unexpected_low_load():
    features = {
        "kwh": 0.5, "kwh_roll_mean_24": 200.0, "is_business_hour": 1.0,
    }
    score, label = StreamingPipeline._rule_based_score(features)
    assert score >= 0.75
    assert "low_load" in label


# ── Pipeline end-to-end ───────────────────────────────────────────────────────

def test_pipeline_publishes_processed_event():
    broker   = InMemoryBroker()
    pipeline = StreamingPipeline(broker, anomaly_threshold=0.9)
    received = []

    async def run():
        async def sub():
            async for evt in broker.subscribe(broker.topic_for(SITE, "processed")):
                received.append(evt)
                break

        async def flow():
            await asyncio.sleep(0.05)
            await pipeline.start([SITE])
            await asyncio.sleep(0.1)   # let consumer task register its queue
            await broker.publish(broker.topic_for(SITE, "raw"), _raw())
            await asyncio.sleep(0.3)
            await pipeline.stop()
            await broker.close()

        await asyncio.gather(sub(), flow())

    asyncio.run(run())
    assert len(received) == 1
    assert received[0].event_type == EventType.PROCESSED


def test_pipeline_publishes_anomaly_event():
    broker   = InMemoryBroker()
    pipeline = StreamingPipeline(broker, anomaly_threshold=0.9)
    received = []

    async def run():
        async def sub():
            async for evt in broker.subscribe(broker.topic_for(SITE, "anomaly")):
                received.append(evt)
                break

        async def flow():
            await asyncio.sleep(0.05)
            await pipeline.start([SITE])
            await asyncio.sleep(0.1)
            await broker.publish(broker.topic_for(SITE, "raw"), _raw())
            await asyncio.sleep(0.3)
            await pipeline.stop()
            await broker.close()

        await asyncio.gather(sub(), flow())

    asyncio.run(run())
    assert len(received) == 1
    assert received[0].event_type == EventType.ANOMALY
    assert "anomaly_score" in received[0].payload


def test_pipeline_fires_alert_on_extreme_spike():
    broker   = InMemoryBroker()
    pipeline = StreamingPipeline(broker, anomaly_threshold=0.7)

    for _ in range(5):
        pipeline._windows[SITE].append(100.0)

    spike_event = EnergyEvent(
        event_type=EventType.RAW, site_id=SITE,
        sensor_id="M1", timestamp=TS, kwh=5000.0,
    )
    alert_received = []

    async def run():
        async def sub():
            async for evt in broker.subscribe(broker.topic_for(SITE, "alert")):
                alert_received.append(evt)
                break

        async def flow():
            await asyncio.sleep(0.05)
            await pipeline.start([SITE])
            await asyncio.sleep(0.1)
            await broker.publish(broker.topic_for(SITE, "raw"), spike_event)
            await asyncio.sleep(0.4)
            await pipeline.stop()
            await broker.close()

        await asyncio.gather(sub(), flow())

    asyncio.run(run())
    assert len(alert_received) == 1
    assert alert_received[0].event_type == EventType.ALERT


def test_pipeline_stats_structure():
    broker   = InMemoryBroker()
    pipeline = StreamingPipeline(broker)
    stats = pipeline.stats()
    for key in ("running", "sites", "processed", "anomalies", "errors"):
        assert key in stats


def test_pipeline_window_grows():
    broker   = InMemoryBroker()
    pipeline = StreamingPipeline(broker)
    pipeline._windows[SITE].append(100.0)
    pipeline._windows[SITE].append(110.0)
    assert pipeline.window_size(SITE) == 2


def test_pipeline_window_size_zero_unknown_site():
    broker   = InMemoryBroker()
    pipeline = StreamingPipeline(broker)
    assert pipeline.window_size("UNKNOWN") == 0


# ── Utility ───────────────────────────────────────────────────────────────────

def test_std_empty():
    assert _std([]) == 0.0

def test_std_single():
    assert _std([5.0]) == 0.0

def test_std_known():
    assert _std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(2.0, abs=0.01)
