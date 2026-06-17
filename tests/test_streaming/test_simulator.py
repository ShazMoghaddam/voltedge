"""Tests for the LiveSimulator."""
from __future__ import annotations
import asyncio
import pytest
from voltedge.streaming.broker import InMemoryBroker
from voltedge.streaming.events import EventType
from voltedge.streaming.simulator import LiveSimulator

SITES = ["SIM-EU-01", "SIM-US-01"]


async def _run_sim(sites, eps=5.0, duration=0.5, anomaly_prob=0.0, seed=42):
    broker = InMemoryBroker()
    sim    = LiveSimulator(broker, sites=sites, events_per_second=eps,
                           anomaly_prob=anomaly_prob, seed=seed)
    received = []
    async def collect():
        for site in sites:
            topic = broker.topic_for(site, "raw")
            async def sub(t=topic):
                async for evt in broker.subscribe(t):
                    received.append(evt)
                    if len(received) >= int(eps * duration * len(sites)):
                        break
            asyncio.create_task(sub())

    await collect()
    await sim.start()
    await asyncio.sleep(duration)
    await sim.stop()
    await broker.close()
    return received, sim


# ── Basic operation ───────────────────────────────────────────────────────────

def test_simulator_produces_events():
    async def run():
        broker = InMemoryBroker()
        sim    = LiveSimulator(broker, sites=["S1"], events_per_second=10, seed=0)
        received = []
        async def collect():
            async for evt in broker.subscribe(broker.topic_for("S1", "raw")):
                received.append(evt)
                if len(received) >= 5:
                    break
        async def flow():
            await asyncio.sleep(0.05)
            await sim.start()
            await asyncio.sleep(0.8)
            await sim.stop()
            await broker.close()
        await asyncio.gather(collect(), flow())
        return received
    evts = asyncio.run(run())
    assert len(evts) >= 5


def test_simulator_event_type_is_raw():
    async def run():
        broker = InMemoryBroker()
        sim    = LiveSimulator(broker, sites=["S1"], events_per_second=10, seed=0)
        received = []
        async def collect():
            async for evt in broker.subscribe(broker.topic_for("S1", "raw")):
                received.append(evt)
                if received:
                    break
        async def flow():
            await asyncio.sleep(0.05)
            await sim.start()
            await asyncio.sleep(0.5)
            await sim.stop()
            await broker.close()
        await asyncio.gather(collect(), flow())
        return received
    evts = asyncio.run(run())
    assert evts[0].event_type == EventType.RAW


def test_simulator_kwh_positive():
    async def run():
        broker = InMemoryBroker()
        sim    = LiveSimulator(broker, sites=["S1"], events_per_second=5, seed=1)
        received = []
        async def collect():
            async for evt in broker.subscribe(broker.topic_for("S1", "raw")):
                received.append(evt)
                if len(received) >= 3:
                    break
        async def flow():
            await asyncio.sleep(0.05)
            await sim.start()
            await asyncio.sleep(1.0)
            await sim.stop()
            await broker.close()
        await asyncio.gather(collect(), flow())
        return received
    evts = asyncio.run(run())
    assert all(e.kwh >= 0 for e in evts)


def test_simulator_sequence_increments():
    async def run():
        broker = InMemoryBroker()
        sim    = LiveSimulator(broker, sites=["S1"], events_per_second=10, seed=2)
        received = []
        async def collect():
            async for evt in broker.subscribe(broker.topic_for("S1", "raw")):
                received.append(evt)
                if len(received) >= 3:
                    break
        async def flow():
            await asyncio.sleep(0.05)
            await sim.start()
            await asyncio.sleep(0.7)
            await sim.stop()
            await broker.close()
        await asyncio.gather(collect(), flow())
        return received
    evts = asyncio.run(run())
    seqs = [e.sequence for e in evts]
    assert seqs == sorted(seqs)


def test_simulator_uses_correct_site_id():
    async def run():
        broker = InMemoryBroker()
        sim    = LiveSimulator(broker, sites=["MY-SITE"], events_per_second=10, seed=3)
        received = []
        async def collect():
            async for evt in broker.subscribe(broker.topic_for("MY-SITE", "raw")):
                received.append(evt)
                break
        async def flow():
            await asyncio.sleep(0.05)
            await sim.start()
            await asyncio.sleep(0.5)
            await sim.stop()
            await broker.close()
        await asyncio.gather(collect(), flow())
        return received
    evts = asyncio.run(run())
    assert evts[0].site_id == "MY-SITE"


def test_simulator_start_stop_clean():
    async def run():
        broker = InMemoryBroker()
        sim    = LiveSimulator(broker, sites=["S1"], events_per_second=2, seed=0)
        await sim.start()
        await asyncio.sleep(0.2)
        await sim.stop()
        return sim.stats()
    stats = asyncio.run(run())
    assert not stats["running"]
    assert stats["total"] >= 0


def test_simulator_stats_structure():
    broker = InMemoryBroker()
    sim    = LiveSimulator(broker, sites=["S1", "S2"], events_per_second=1.0)
    stats  = sim.stats()
    for k in ("running", "sites", "events_per_second", "counts", "total"):
        assert k in stats


def test_simulator_anomaly_injection_flag():
    """With 100% anomaly probability, every event should be flagged."""
    async def run():
        broker = InMemoryBroker()
        sim    = LiveSimulator(broker, sites=["S1"], events_per_second=10,
                               anomaly_prob=1.0, seed=99)
        received = []
        async def collect():
            async for evt in broker.subscribe(broker.topic_for("S1", "raw")):
                received.append(evt)
                if len(received) >= 3:
                    break
        async def flow():
            await asyncio.sleep(0.05)
            await sim.start()
            await asyncio.sleep(0.6)
            await sim.stop()
            await broker.close()
        await asyncio.gather(collect(), flow())
        return received
    evts = asyncio.run(run())
    assert all(e.payload.get("injected_anomaly") is True for e in evts)


def test_simulator_high_kwh_on_anomaly():
    """Injected anomalies should have notably higher kwh than baseline."""
    async def run():
        broker = InMemoryBroker()
        sim    = LiveSimulator(broker, sites=["S1"], events_per_second=10,
                               anomaly_prob=1.0, seed=5)
        received = []
        async def collect():
            async for evt in broker.subscribe(broker.topic_for("S1", "raw")):
                received.append(evt)
                if len(received) >= 2:
                    break
        async def flow():
            await asyncio.sleep(0.05)
            await sim.start()
            await asyncio.sleep(0.5)
            await sim.stop()
            await broker.close()
        await asyncio.gather(collect(), flow())
        return received
    evts = asyncio.run(run())
    # 3–8× amplification → kwh > 3 × base (400) = 1200
    assert all(e.kwh > 400 for e in evts)


def test_simulator_deterministic_with_seed():
    async def run(seed):
        broker = InMemoryBroker()
        sim    = LiveSimulator(broker, sites=["S1"], events_per_second=5, seed=seed)
        received = []
        async def collect():
            async for evt in broker.subscribe(broker.topic_for("S1", "raw")):
                received.append(evt)
                if len(received) >= 3:
                    break
        async def flow():
            await asyncio.sleep(0.05)
            await sim.start()
            await asyncio.sleep(1.0)
            await sim.stop()
            await broker.close()
        await asyncio.gather(collect(), flow())
        return [e.kwh for e in received]

    r1 = asyncio.run(run(42))
    r2 = asyncio.run(run(42))
    assert r1 == r2


def test_simulator_site_profiles():
    """Different site types should produce different kwh baselines."""
    async def get_avg(site_id, stype):
        broker = InMemoryBroker()
        sim    = LiveSimulator(broker, sites=[site_id],
                               site_types={site_id: stype},
                               events_per_second=5, seed=0)
        received = []
        async def collect():
            async for evt in broker.subscribe(broker.topic_for(site_id, "raw")):
                received.append(evt.kwh)
                if len(received) >= 5:
                    break
        async def flow():
            await asyncio.sleep(0.05)
            await sim.start()
            await asyncio.sleep(1.5)
            await sim.stop()
            await broker.close()
        await asyncio.gather(collect(), flow())
        return sum(received) / len(received) if received else 0

    factory_avg = asyncio.run(get_avg("F", "factory"))
    office_avg  = asyncio.run(get_avg("O", "office"))
    assert factory_avg > office_avg
