"""Tests for InMemoryBroker — no external dependencies."""
from __future__ import annotations
import asyncio
import pytest
from voltedge.streaming.broker import BaseBroker, InMemoryBroker
from voltedge.streaming.events import EnergyEvent, EventType

TOPIC = "voltedge.SITE-01.raw"
TS    = "2026-05-15T12:00:00+00:00"


def _event(kwh=50.0, seq=0):
    return EnergyEvent(
        event_type=EventType.RAW, site_id="SITE-01",
        sensor_id="M1", timestamp=TS, kwh=kwh, sequence=seq,
    )


async def _collect(broker, topic, n):
    """Collect n events from a topic then return them."""
    results = []
    async for evt in broker.subscribe(topic):
        results.append(evt)
        if len(results) >= n:
            break
    return results


# ── Interface ─────────────────────────────────────────────────────────────────

def test_in_memory_broker_is_base_broker():
    assert isinstance(InMemoryBroker(), BaseBroker)

def test_topic_for_helper():
    t = BaseBroker.topic_for("SITE-01", "raw")
    assert t == "voltedge.SITE-01.raw"

# ── Publish & subscribe ───────────────────────────────────────────────────────

def test_publish_with_no_subscribers_does_not_raise():
    broker = InMemoryBroker()
    asyncio.run(broker.publish(TOPIC, _event()))   # no error

def test_single_subscriber_receives_event():
    broker = InMemoryBroker()
    event  = _event(kwh=123.0)

    async def run():
        # Schedule publish after subscriber is listening
        async def pub():
            await asyncio.sleep(0.05)
            await broker.publish(TOPIC, event)
            await broker.close()
        asyncio.create_task(pub())
        return await _collect(broker, TOPIC, 1)

    results = asyncio.run(run())
    assert len(results) == 1
    assert results[0].kwh == 123.0

def test_multiple_events_received_in_order():
    broker = InMemoryBroker()
    events = [_event(seq=i) for i in range(5)]

    async def run():
        async def pub():
            await asyncio.sleep(0.05)
            await broker.publish_batch(TOPIC, events)
            await asyncio.sleep(0.1)
            await broker.close()
        asyncio.create_task(pub())
        return await _collect(broker, TOPIC, 5)

    results = asyncio.run(run())
    assert [r.sequence for r in results] == list(range(5))

def test_fan_out_to_two_subscribers():
    broker  = InMemoryBroker()
    results: dict[str, list] = {"a": [], "b": []}

    async def run():
        async def sub(key):
            async for evt in broker.subscribe(TOPIC):
                results[key].append(evt)
                if len(results[key]) >= 1:
                    break

        async def pub():
            await asyncio.sleep(0.05)
            await broker.publish(TOPIC, _event(kwh=77.0))
            await asyncio.sleep(0.1)
            await broker.close()

        await asyncio.gather(sub("a"), sub("b"), pub())

    asyncio.run(run())
    assert len(results["a"]) == 1
    assert len(results["b"]) == 1
    assert results["a"][0].kwh == 77.0

def test_different_topics_isolated():
    broker = InMemoryBroker()
    received = []

    async def run():
        async def sub():
            async for evt in broker.subscribe("voltedge.SITE-01.processed"):
                received.append(evt)
                break

        async def pub():
            await asyncio.sleep(0.05)
            # Publish to a DIFFERENT topic
            await broker.publish("voltedge.SITE-01.raw", _event())
            await asyncio.sleep(0.2)
            await broker.close()

        await asyncio.gather(sub(), pub())

    asyncio.run(run())
    assert len(received) == 0   # wrong topic — should get nothing

# ── Backpressure ──────────────────────────────────────────────────────────────

def test_queue_depth_drop_oldest():
    """When queue is full, oldest events are dropped to keep latency low."""
    broker = InMemoryBroker(max_queue_depth=3)

    async def run():
        # Add a subscriber but don't consume yet
        queue_ref = asyncio.Queue(maxsize=3)
        broker._queues[TOPIC].append(queue_ref)
        # Publish 5 events — should drop 2
        for i in range(5):
            await broker.publish(TOPIC, _event(seq=i))
        return broker._dropped[TOPIC]

    dropped = asyncio.run(run())
    assert dropped >= 2

# ── Metrics ───────────────────────────────────────────────────────────────────

def test_metrics_structure():
    broker = InMemoryBroker()
    m = broker.metrics()
    for key in ("topics", "published", "dropped", "subscribers"):
        assert key in m

def test_subscriber_count():
    broker  = InMemoryBroker()
    results = []

    async def run():
        async def sub():
            async for _ in broker.subscribe(TOPIC):
                break
        async def check_count():
            await asyncio.sleep(0.05)
            results.append(broker.subscriber_count(TOPIC))
            await broker.close()
        await asyncio.gather(sub(), check_count())

    asyncio.run(run())
    assert results[0] == 1

# ── Close ─────────────────────────────────────────────────────────────────────

def test_close_terminates_subscriber():
    broker = InMemoryBroker()
    count  = []

    async def run():
        async def sub():
            async for _ in broker.subscribe(TOPIC):
                count.append(1)

        async def close_soon():
            await asyncio.sleep(0.1)
            await broker.close()

        await asyncio.gather(sub(), close_soon())

    asyncio.run(run())   # Should not hang

def test_publish_after_close_is_noop():
    broker = InMemoryBroker()
    asyncio.run(broker.close())
    asyncio.run(broker.publish(TOPIC, _event()))   # no error
