"""Redis Streams message bus.

Why Redis Streams and not Kafka: at ~10 events/sec/stream the operational cost
of a Kafka cluster (ZooKeeper/KRaft, partition management, a JVM per broker)
buys nothing. Redis Streams gives us the two properties we actually need --
consumer groups with at-least-once delivery, and a replayable log -- in one
6MB container. The migration path is documented in ``docs/ARCHITECTURE.md``:
:class:`EventBus` is the seam, and a ``KafkaEventBus`` would implement it
without touching a line of service logic.

Delivery semantics are **at-least-once**. Every consumer must therefore be
idempotent or tolerant of duplicates; ``event_id`` is the natural dedup key.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Any, Final, Protocol, TypeVar

import redis.asyncio as aioredis
from redis.exceptions import ResponseError

from amme_shared.events import BaseEvent
from amme_shared.logging import get_logger

log = get_logger(__name__)

E = TypeVar("E", bound=BaseEvent)


class Streams:
    """Canonical stream names. Never hard-code these as literals in a service."""

    LOB_RAW: Final = "lob.raw"
    LOB_TRADES: Final = "lob.trades"
    FEATURES_READY: Final = "features.ready"
    DRIFT_SIGNAL: Final = "drift.signal"
    QUOTE_INTENT: Final = "quote.intent"
    RISK_DECISION: Final = "risk.decision"
    EXEC_FILL: Final = "exec.fill"
    EXEC_PORTFOLIO: Final = "exec.portfolio"

    ALL: Final = (
        LOB_RAW, LOB_TRADES, FEATURES_READY, DRIFT_SIGNAL,
        QUOTE_INTENT, RISK_DECISION, EXEC_FILL, EXEC_PORTFOLIO,
    )


class Message(tuple[str, BaseEvent]):
    """(message_id, event). A tuple subclass so it destructures naturally."""

    __slots__ = ()

    @property
    def message_id(self) -> str:
        return self[0]

    @property
    def event(self) -> BaseEvent:
        return self[1]


class EventBus(Protocol):
    """The seam between service logic and transport."""

    async def publish(self, stream: str, event: BaseEvent) -> str: ...

    async def ensure_group(self, stream: str, group: str) -> None: ...

    def consume(
        self, streams: Sequence[str], group: str, consumer: str, event_type: type[E]
    ) -> AsyncIterator[tuple[str, str, E]]: ...

    async def ack(self, stream: str, group: str, message_id: str) -> None: ...

    async def close(self) -> None: ...


def _encode(event: BaseEvent) -> dict[str, str]:
    return {"type": type(event).__name__, "data": event.model_dump_json()}


def _decode(fields: dict[str, str], event_type: type[E]) -> E:
    return event_type.model_validate_json(fields["data"])


class RedisEventBus:
    """Production bus. One connection pool per process."""

    def __init__(self, url: str, maxlen: int = 100_000, block_ms: int = 1000) -> None:
        self._url = url
        self._maxlen = maxlen
        self._block_ms = block_ms
        self._redis: aioredis.Redis = aioredis.from_url(
            url, decode_responses=True, health_check_interval=30
        )
        self._known_groups: set[tuple[str, str]] = set()

    @property
    def client(self) -> aioredis.Redis:
        return self._redis

    async def ping(self) -> bool:
        result: bool = await self._redis.ping()
        return result

    async def publish(self, stream: str, event: BaseEvent) -> str:
        # approximate=True lets Redis trim on radix-node boundaries -- O(1)
        # amortised instead of O(n) exact trimming on every single XADD.
        message_id: str = await self._redis.xadd(
            stream, _encode(event), maxlen=self._maxlen, approximate=True
        )
        return message_id

    async def publish_many(self, stream: str, events: Sequence[BaseEvent]) -> list[str]:
        if not events:
            return []
        pipe = self._redis.pipeline(transaction=False)
        for event in events:
            pipe.xadd(stream, _encode(event), maxlen=self._maxlen, approximate=True)
        ids: list[str] = await pipe.execute()
        return ids

    async def ensure_group(self, stream: str, group: str) -> None:
        """Create the consumer group, tolerating the already-exists race."""
        if (stream, group) in self._known_groups:
            return
        try:
            await self._redis.xgroup_create(stream, group, id="$", mkstream=True)
            log.info("consumer_group_created", stream=stream, group=group)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._known_groups.add((stream, group))

    async def consume(
        self, streams: Sequence[str], group: str, consumer: str, event_type: type[E]
    ) -> AsyncIterator[tuple[str, str, E]]:
        """Yield ``(stream, message_id, event)`` forever. Caller must :meth:`ack`.

        Reads pending-but-unacked messages (``id=0``) first so that a crashed
        consumer's work is recovered on restart, then switches to new messages
        (``id=>``). Without this a hard kill silently drops in-flight events.
        """
        for stream in streams:
            await self.ensure_group(stream, group)

        recovering = True
        cursor: dict[str, str] = dict.fromkeys(streams, "0")

        while True:
            response = await self._redis.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams=cursor if recovering else dict.fromkeys(streams, ">"),
                count=64,
                block=self._block_ms,
            )
            if not response:
                if recovering:
                    recovering = False
                continue

            drained = True
            for stream_name, entries in response:
                if entries:
                    drained = False
                for message_id, fields in entries:
                    if recovering:
                        cursor[stream_name] = message_id
                    try:
                        event = _decode(fields, event_type)
                    except Exception as exc:  # noqa: BLE001 -- poison pill must not stall
                        log.error(
                            "event_decode_failed",
                            stream=stream_name,
                            message_id=message_id,
                            error=str(exc),
                        )
                        await self.ack(stream_name, group, message_id)
                        continue
                    yield stream_name, message_id, event

            if recovering and drained:
                recovering = False

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        await self._redis.xack(stream, group, message_id)

    async def lag(self, stream: str, group: str) -> int:
        """Undelivered messages for ``group``. -1 when unavailable."""
        try:
            groups: list[dict[str, Any]] = await self._redis.xinfo_groups(stream)
        except ResponseError:
            return -1
        for info in groups:
            if info.get("name") == group:
                raw = info.get("lag")
                return int(raw) if raw is not None else -1
        return -1

    async def close(self) -> None:
        await self._redis.aclose()

    async def __aenter__(self) -> RedisEventBus:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


class InMemoryEventBus:
    """Deterministic in-process bus for unit tests and the backtester.

    Same interface, no Redis, strict FIFO with monotonic integer ids -- which is
    what makes the byte-for-byte replay test in s.6.1 possible.
    """

    def __init__(self, maxlen: int = 100_000) -> None:
        self._maxlen = maxlen
        self._streams: dict[str, deque[tuple[str, BaseEvent]]] = defaultdict(
            lambda: deque(maxlen=maxlen)
        )
        self._offsets: dict[tuple[str, str], int] = {}
        self._acked: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._counter = 0

    async def publish(self, stream: str, event: BaseEvent) -> str:
        self._counter += 1
        message_id = f"{self._counter}-0"
        self._streams[stream].append((message_id, event))
        return message_id

    async def publish_many(self, stream: str, events: Sequence[BaseEvent]) -> list[str]:
        return [await self.publish(stream, e) for e in events]

    async def ensure_group(self, stream: str, group: str) -> None:
        self._offsets.setdefault((stream, group), 0)
        _ = self._streams[stream]

    async def consume(
        self, streams: Sequence[str], group: str, consumer: str, event_type: type[E]
    ) -> AsyncIterator[tuple[str, str, E]]:
        for stream in streams:
            await self.ensure_group(stream, group)
        while True:
            progressed = False
            for stream in streams:
                offset = self._offsets[(stream, group)]
                entries = self._streams[stream]
                if offset < len(entries):
                    message_id, event = entries[offset]
                    self._offsets[(stream, group)] = offset + 1
                    progressed = True
                    if isinstance(event, event_type):
                        yield stream, message_id, event
            if not progressed:
                return

    def drain(self, stream: str, event_type: type[E]) -> list[E]:
        """Everything currently on ``stream``. Test helper, not part of EventBus."""
        return [e for _, e in self._streams[stream] if isinstance(e, event_type)]

    def clear(self) -> None:
        self._streams.clear()
        self._offsets.clear()
        self._acked.clear()
        self._counter = 0

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        self._acked[(stream, group)].add(message_id)

    async def lag(self, stream: str, group: str) -> int:
        return len(self._streams[stream]) - self._offsets.get((stream, group), 0)

    async def close(self) -> None:
        return None


__all__ = ["EventBus", "InMemoryEventBus", "Message", "RedisEventBus", "Streams"]
