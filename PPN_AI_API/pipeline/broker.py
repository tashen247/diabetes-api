"""Brokers for the screening fan-out.

In-process mode is what tests and single-process dev use. RabbitMQ mode is the
deployment in the diagram: one publish onto the fanout exchange, durable
queues, and retry queues that dead-letter back to the model that failed.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from pipeline.errors import BrokerUnavailable
from pipeline.topology import Topology, retry_arguments

log = logging.getLogger("screening")


def _copy_body(body: dict) -> dict:
    return json.loads(json.dumps(body))


def attempt_of(headers: dict | None) -> int:
    headers = headers or {}
    raw = headers.get("attempt", headers.get("x-attempt", 0))
    if isinstance(raw, bytes):
        raw = raw.decode()
    return int(raw)


def model_of(headers: dict | None) -> str | None:
    headers = headers or {}
    raw = headers.get("model", headers.get("x-model"))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode()
    return str(raw)


@dataclass
class Envelope:
    message_id: str
    queue: str
    body: dict
    headers: dict
    ready_at: datetime | None = None
    delivery_tag: int | None = None


@dataclass
class _QueueState:
    ttl_seconds: int | None = None
    target_queue: str | None = None
    messages: deque = field(default_factory=deque)


class InProcessBroker:
    """Fan-out, delay, and dead-letter behaviour without a network."""

    available = True

    def __init__(self, topology: Topology):
        self.topology = topology
        self.fanout_count = 0
        self._lock = threading.Lock()
        self._queues: dict[str, _QueueState] = {}
        self._unacked: dict[str, Envelope] = {}

    def declare(self) -> None:
        with self._lock:
            for test in self.topology.tests:
                self._queues.setdefault(test.queue, _QueueState())
            self._queues.setdefault(self.topology.results_queue, _QueueState())
            self._queues.setdefault(self.topology.dead_queue, _QueueState())
            for retry in self.topology.retry_queues:
                self._queues.setdefault(
                    retry.name,
                    _QueueState(ttl_seconds=retry.ttl_seconds, target_queue=retry.target_queue),
                )

    def publish(self, queue: str, body: dict, headers: dict, now: datetime) -> str:
        message_id = str(uuid.uuid4())
        with self._lock:
            state = self._require(queue)
            ready_at = None
            if state.ttl_seconds:
                ready_at = now + timedelta(seconds=state.ttl_seconds)
            state.messages.append(
                Envelope(
                    message_id=message_id,
                    queue=queue,
                    body=_copy_body(body),
                    headers=dict(headers),
                    ready_at=ready_at,
                )
            )
        return message_id

    def publish_fanout(self, body: dict, headers: dict, now: datetime) -> None:
        """One logical publish. Every model queue receives its own copy."""
        self.fanout_count += 1
        for test in self.topology.tests:
            stamped = dict(headers)
            stamped.setdefault("attempt", 0)
            stamped["study_id"] = body["study_id"]
            self.publish(test.queue, body, stamped, now)

    def pull(self, queue: str, now: datetime) -> Envelope | None:
        with self._lock:
            self._release_due_locked(now)
            state = self._require(queue)
            if state.ttl_seconds:
                return None
            if not state.messages:
                return None
            envelope = state.messages.popleft()
            envelope.queue = queue
            self._unacked[envelope.message_id] = envelope
            return envelope

    def ack(self, envelope: Envelope) -> None:
        with self._lock:
            self._unacked.pop(envelope.message_id, None)

    def nack(self, envelope: Envelope, requeue: bool = True) -> None:
        with self._lock:
            self._unacked.pop(envelope.message_id, None)
            if requeue:
                state = self._require(envelope.queue)
                state.messages.appendleft(envelope)

    def release_due(self, now: datetime) -> int:
        with self._lock:
            return self._release_due_locked(now)

    def _release_due_locked(self, now: datetime) -> int:
        moved = 0
        for state in self._queues.values():
            if not state.ttl_seconds or not state.target_queue:
                continue
            kept: deque = deque()
            target = self._require(state.target_queue)
            for envelope in state.messages:
                due = envelope.ready_at is not None and envelope.ready_at <= now
                if due:
                    forwarded = Envelope(
                        message_id=envelope.message_id,
                        queue=state.target_queue,
                        body=_copy_body(envelope.body),
                        headers=dict(envelope.headers),
                    )
                    target.messages.append(forwarded)
                    moved += 1
                else:
                    kept.append(envelope)
            state.messages = kept
        return moved

    def depth(self, queue: str) -> int:
        with self._lock:
            return len(self._require(queue).messages)

    def browse(self, queue: str) -> list[Envelope]:
        with self._lock:
            return list(self._require(queue).messages)

    def pop_match(self, queue: str, predicate, now: datetime) -> Envelope | None:
        with self._lock:
            self._release_due_locked(now)
            state = self._require(queue)
            for index, envelope in enumerate(state.messages):
                if predicate(envelope):
                    del state.messages[index]
                    return envelope
        return None

    def _require(self, queue: str) -> _QueueState:
        try:
            return self._queues[queue]
        except KeyError as exc:
            raise KeyError(f"undeclared queue {queue}") from exc


class UnavailableBroker:
    available = False

    def __init__(self, reason: str):
        self.reason = reason
        self.fanout_count = 0

    def declare(self) -> None:
        raise BrokerUnavailable(self.reason)

    def publish(self, queue: str, body: dict, headers: dict, now: datetime) -> str:
        raise BrokerUnavailable(self.reason)

    def publish_fanout(self, body: dict, headers: dict, now: datetime) -> None:
        raise BrokerUnavailable(self.reason)

    def pull(self, queue: str, now: datetime) -> Envelope | None:
        raise BrokerUnavailable(self.reason)

    def ack(self, envelope: Envelope) -> None:
        raise BrokerUnavailable(self.reason)

    def nack(self, envelope: Envelope, requeue: bool = True) -> None:
        raise BrokerUnavailable(self.reason)

    def release_due(self, now: datetime) -> int:
        raise BrokerUnavailable(self.reason)

    def depth(self, queue: str) -> int:
        raise BrokerUnavailable(self.reason)

    def browse(self, queue: str) -> list[Envelope]:
        raise BrokerUnavailable(self.reason)

    def pop_match(self, queue: str, predicate, now: datetime) -> Envelope | None:
        raise BrokerUnavailable(self.reason)


class RabbitBroker:
    """Durable RabbitMQ adapter. Queue TTL dead-letters a retry back to its model queue."""

    available = True

    def __init__(self, url: str, topology: Topology):
        self.url = url
        self.topology = topology
        self.fanout_count = 0
        self._lock = threading.Lock()
        self._connection = None
        self._channel = None

    def declare(self) -> None:
        with self._lock:
            channel = self._channel_locked()
            channel.exchange_declare(
                exchange=self.topology.fanout_exchange,
                exchange_type="fanout",
                durable=True,
            )
            for test in self.topology.tests:
                channel.queue_declare(queue=test.queue, durable=True)
                channel.queue_bind(queue=test.queue, exchange=self.topology.fanout_exchange)
            channel.queue_declare(queue=self.topology.results_queue, durable=True)
            channel.queue_declare(queue=self.topology.dead_queue, durable=True)
            for retry in self.topology.retry_queues:
                channel.queue_declare(queue=retry.name, durable=True, arguments=retry_arguments(retry))

    def publish(self, queue: str, body: dict, headers: dict, now: datetime) -> str:
        del now
        message_id = str(uuid.uuid4())
        with self._lock:
            channel = self._channel_locked()
            import pika

            channel.basic_publish(
                exchange="",
                routing_key=queue,
                body=json.dumps(body).encode(),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    content_type="application/json",
                    message_id=message_id,
                    headers=_header_values(headers),
                ),
            )
        return message_id

    def publish_fanout(self, body: dict, headers: dict, now: datetime) -> None:
        del now
        self.fanout_count += 1
        with self._lock:
            channel = self._channel_locked()
            import pika

            channel.basic_publish(
                exchange=self.topology.fanout_exchange,
                routing_key="",
                body=json.dumps(body).encode(),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    content_type="application/json",
                    message_id=str(uuid.uuid4()),
                    headers=_header_values(headers),
                ),
            )

    def pull(self, queue: str, now: datetime) -> Envelope | None:
        del now
        with self._lock:
            channel = self._channel_locked()
            method, properties, body = channel.basic_get(queue, auto_ack=False)
            if method is None:
                return None
            return _envelope_from_rabbit(queue, method, properties, body)

    def ack(self, envelope: Envelope) -> None:
        if envelope.delivery_tag is None:
            return
        with self._lock:
            self._channel_locked().basic_ack(envelope.delivery_tag)

    def nack(self, envelope: Envelope, requeue: bool = True) -> None:
        if envelope.delivery_tag is None:
            return
        with self._lock:
            self._channel_locked().basic_nack(envelope.delivery_tag, requeue=requeue)

    def release_due(self, now: datetime) -> int:
        del now
        return 0

    def depth(self, queue: str) -> int:
        with self._lock:
            declared = self._channel_locked().queue_declare(queue=queue, passive=True)
            return int(declared.method.message_count)

    def browse(self, queue: str) -> list[Envelope]:
        del queue
        return []

    def pop_match(self, queue: str, predicate, now: datetime) -> Envelope | None:
        del now
        with self._lock:
            channel = self._channel_locked()
            declared = channel.queue_declare(queue=queue, passive=True)
            depth = int(declared.method.message_count)
            held: list[int] = []
            found = None
            try:
                for _ in range(depth):
                    method, properties, body = channel.basic_get(queue, auto_ack=False)
                    if method is None:
                        break
                    envelope = _envelope_from_rabbit(queue, method, properties, body)
                    if found is None and predicate(envelope):
                        channel.basic_ack(method.delivery_tag)
                        found = envelope
                    else:
                        held.append(method.delivery_tag)
            except Exception:
                if found is not None:
                    pass
                for tag in held:
                    channel.basic_nack(tag, requeue=True)
                raise
            for tag in held:
                channel.basic_nack(tag, requeue=True)
            return found

    def _channel_locked(self):
        import pika

        if self._connection is None or self._connection.is_closed:
            try:
                params = pika.URLParameters(self.url)
                params.socket_timeout = 3
                params.connection_attempts = 1
                params.retry_delay = 0
                params.blocked_connection_timeout = 3
                self._connection = pika.BlockingConnection(params)
                log.info("rabbitmq connection opened to %s:%s", params.host, params.port)
            except Exception as exc:
                raise BrokerUnavailable(str(exc)) from exc
            self._channel = None
        if self._channel is None or self._channel.is_closed:
            self._channel = self._connection.channel()
        return self._channel


def make_broker(kind: str, topology: Topology, url: str | None = None):
    if kind == "memory":
        broker = InProcessBroker(topology)
        broker.declare()
        return broker
    if kind == "rabbitmq":
        broker = RabbitBroker(url or "amqp://guest:guest@localhost:5673/", topology)
        broker.declare()
        return broker
    raise ValueError(f"unknown broker {kind!r}")


def _header_values(headers: dict) -> dict:
    clean = {}
    for key, value in headers.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def _envelope_from_rabbit(queue, method, properties, body) -> Envelope:
    raw_headers = dict(properties.headers or {})
    headers = {}
    for key, value in raw_headers.items():
        headers[key.decode() if isinstance(key, bytes) else key] = value
    try:
        payload = json.loads(body.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = {"raw": body.decode(errors="replace")}
    return Envelope(
        message_id=properties.message_id or str(uuid.uuid4()),
        queue=queue,
        body=payload,
        headers=headers,
        delivery_tag=method.delivery_tag,
    )
