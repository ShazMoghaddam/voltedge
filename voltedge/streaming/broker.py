"""
VoltEdge Streaming — Event Broker

The broker is the backbone of the streaming pipeline:
producers publish EnergyEvents, consumers subscribe and receive them.

Two implementations:

  InMemoryBroker   — asyncio.Queue per topic. Zero dependencies.
                     Used in development, tests, and free-tier deployments.
                     Events are lost on process restart (acceptable for live feeds).

  SQSBroker        — AWS SQS as the durable backing store.
                     Used in production. Survives process restarts.
                     Long-polling consumer with auto-delete on ack.

Both expose identical async interfaces so the pipeline never knows
which broker it's talking to.

Topic convention:
    voltedge.{site_id}.raw         — fresh sensor readings
    voltedge.{site_id}.processed   — feature-engineered readings
    voltedge.{site_id}.anomaly     — scored anomaly events
    voltedge.platform.system       — heartbeats and status

Usage (InMemory):
    broker = InMemoryBroker()
    await broker.publish("voltedge.SITE-01.raw", event)
    async for event in broker.subscribe("voltedge.SITE-01.raw"):
        process(event)

Usage (SQS):
    broker = SQSBroker(queue_url="https://sqs.eu-west-1.amazonaws.com/...")
    # Same interface — swap with zero pipeline changes
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from typing import AsyncIterator

from voltedge.streaming.events import EnergyEvent
from voltedge.utils.logger import get_logger

log = get_logger(__name__)


# ── Base interface ────────────────────────────────────────────────────────────

class BaseBroker(ABC):
    """
    Abstract event broker. All streaming components depend on this interface,
    never on a concrete implementation.
    """

    @abstractmethod
    async def publish(self, topic: str, event: EnergyEvent) -> None:
        """Publish an event to a topic."""

    @abstractmethod
    async def subscribe(self, topic: str) -> AsyncIterator[EnergyEvent]:
        """Yield events from a topic indefinitely."""

    @abstractmethod
    async def close(self) -> None:
        """Gracefully shut down the broker."""

    @staticmethod
    def topic_for(site_id: str, event_type: str) -> str:
        """Canonical topic name helper."""
        return f"voltedge.{site_id}.{event_type}"


# ── In-memory broker ──────────────────────────────────────────────────────────

class InMemoryBroker(BaseBroker):
    """
    asyncio.Queue-backed broker for development and testing.

    Features:
      - Multiple subscribers per topic (fan-out via per-subscriber queues)
      - Configurable max queue depth with drop-oldest backpressure
      - Metrics: published / delivered / dropped counts per topic
      - Thread-safe: all operations via event loop
    """

    def __init__(self, max_queue_depth: int = 1000) -> None:
        self._max_depth = max_queue_depth
        # {topic: [queue_per_subscriber]}
        self._queues:    dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._published: dict[str, int]                 = defaultdict(int)
        self._dropped:   dict[str, int]                 = defaultdict(int)
        self._closed:    bool                           = False

    async def publish(self, topic: str, event: EnergyEvent) -> None:
        if self._closed:
            return
        queues = self._queues[topic]
        if not queues:
            return          # No subscribers — drop silently (fire-and-forget)

        self._published[topic] += 1
        for q in queues:
            if q.qsize() >= self._max_depth:
                # Drop oldest to make room (low-latency priority over durability)
                try:
                    q.get_nowait()
                    self._dropped[topic] += 1
                    log.warning("broker.queue_full_drop", topic=topic,
                                depth=self._max_depth)
                except asyncio.QueueEmpty:
                    pass
            await q.put(event)

    async def subscribe(self, topic: str) -> AsyncIterator[EnergyEvent]:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_depth_safe)
        self._queues[topic].append(q)
        log.debug("broker.subscriber_added", topic=topic,
                  total_subscribers=len(self._queues[topic]))
        try:
            while not self._closed:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=1.0)
                    yield event
                    q.task_done()
                except asyncio.TimeoutError:
                    continue   # Check _closed flag on next iteration
        finally:
            if q in self._queues[topic]:
                self._queues[topic].remove(q)
            log.debug("broker.subscriber_removed", topic=topic)

    async def close(self) -> None:
        self._closed = True
        log.info("broker.closed", type="InMemoryBroker")

    def metrics(self) -> dict:
        return {
            "topics":    list(self._queues.keys()),
            "published": dict(self._published),
            "dropped":   dict(self._dropped),
            "subscribers": {t: len(qs) for t, qs in self._queues.items()},
        }

    @property
    def _max_queue_depth_safe(self) -> int:
        return max(1, self._max_depth)

    # ── Test helper ───────────────────────────────────────────────────────────

    async def publish_batch(self, topic: str, events: list[EnergyEvent]) -> None:
        """Publish multiple events atomically (for testing)."""
        for event in events:
            await self.publish(topic, event)

    def subscriber_count(self, topic: str) -> int:
        return len(self._queues.get(topic, []))


# ── SQS broker (production) ───────────────────────────────────────────────────

class SQSBroker(BaseBroker):
    """
    AWS SQS-backed durable event broker for production.

    Uses long-polling (WaitTimeSeconds=20) for efficient consumption.
    Events are deleted from SQS after successful processing.

    Args:
        queue_url:             SQS queue URL.
        region:                AWS region (default eu-west-1).
        max_messages_per_poll: SQS maximum (1–10).
        visibility_timeout:    Seconds before unacked message re-appears.
        aws_access_key_id / aws_secret_access_key: Optional credential override.
    """

    def __init__(
        self,
        queue_url: str,
        region:    str   = "eu-west-1",
        max_messages_per_poll: int = 10,
        visibility_timeout:    int = 30,
        aws_access_key_id:     str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        self._queue_url = queue_url
        self._region    = region
        self._max_msgs  = max_messages_per_poll
        self._vis_timeout = visibility_timeout
        self._aws_key   = aws_access_key_id
        self._aws_secret = aws_secret_access_key
        self._closed    = False
        self._client    = None

    async def publish(self, topic: str, event: EnergyEvent) -> None:
        client = self._get_client()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: client.send_message(
            QueueUrl=self._queue_url,
            MessageBody=event.to_json(),
            MessageAttributes={
                "topic":    {"DataType": "String", "StringValue": topic},
                "site_id":  {"DataType": "String", "StringValue": event.site_id},
                "event_type": {"DataType": "String", "StringValue": event.event_type.value},
            },
        ))
        log.debug("sqs.published", topic=topic, event_id=event.event_id)

    async def subscribe(self, topic: str) -> AsyncIterator[EnergyEvent]:
        client = self._get_client()
        loop = asyncio.get_event_loop()

        while not self._closed:
            response = await loop.run_in_executor(None, lambda: client.receive_message(
                QueueUrl=self._queue_url,
                MaxNumberOfMessages=self._max_msgs,
                WaitTimeSeconds=20,           # Long-poll
                VisibilityTimeout=self._vis_timeout,
                MessageAttributeNames=["topic", "site_id", "event_type"],
            ))

            for msg in response.get("Messages", []):
                attrs = msg.get("MessageAttributes", {})
                msg_topic = attrs.get("topic", {}).get("StringValue", "")

                if topic and msg_topic and msg_topic != topic:
                    continue   # Wrong topic — leave in queue

                try:
                    event = EnergyEvent.from_json(msg["Body"])
                    yield event
                    # Acknowledge: delete from SQS
                    await loop.run_in_executor(None, lambda r=msg["ReceiptHandle"]:
                        client.delete_message(QueueUrl=self._queue_url, ReceiptHandle=r)
                    )
                except Exception as exc:
                    log.error("sqs.parse_error", error=str(exc), body=msg["Body"][:200])

    async def close(self) -> None:
        self._closed = True
        log.info("broker.closed", type="SQSBroker")

    def _get_client(self):
        if self._client is None:
            import boto3
            kwargs: dict = {"region_name": self._region}
            if self._aws_key:
                kwargs["aws_access_key_id"]     = self._aws_key
                kwargs["aws_secret_access_key"] = self._aws_secret
            self._client = boto3.client("sqs", **kwargs)
        return self._client
