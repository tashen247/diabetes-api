"""RabbitMQ connection and confirmed publishing.

Every publish waits for the broker's confirm, so "published" means RabbitMQ has the message on
disk. Without confirms a broker restart could silently lose a study.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any

import pika
from pika.adapters.blocking_connection import BlockingChannel

from .settings import Settings
from . import topology

log = logging.getLogger(__name__)

PERSISTENT = pika.DeliveryMode.Persistent


class PublishError(RuntimeError):
    """The broker did not confirm the message, so the caller must not assume it was stored."""


def connect(s: Settings, name: str) -> pika.BlockingConnection:
    params = pika.URLParameters(s.rabbitmq_url)
    params.client_properties = {"connection_name": name}
    params.heartbeat = 30
    params.blocked_connection_timeout = 30
    return pika.BlockingConnection(params)


def open_channel(connection: pika.BlockingConnection, s: Settings, declare: bool = True) -> BlockingChannel:
    channel = connection.channel()
    channel.confirm_delivery()
    if declare:
        topology.declare(channel, s)
    return channel


def publish_json(channel: BlockingChannel, exchange: str, routing_key: str, body: dict[str, Any],
                 headers: dict[str, Any] | None = None, message_id: str | None = None) -> None:
    """Publishes and waits for the confirm. mandatory=True turns 'no queue bound' into an error."""
    try:
        channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=json.dumps(body).encode(),
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=PERSISTENT,
                headers=headers or {},
                message_id=message_id,
            ),
            mandatory=True,
        )
    except (pika.exceptions.UnroutableError, pika.exceptions.NackError) as e:
        raise PublishError(f"broker did not accept message for {exchange!r}/{routing_key!r}: {e}") from e


class SharedPublisher:
    """One connection shared by the intake API's request threads.

    pika connections are not thread-safe, so publishes are serialised with a lock, and a broken
    connection is reopened on the next publish.
    """

    def __init__(self, s: Settings, name: str = "fundus-intake"):
        self._s = s
        self._name = name
        self._lock = threading.Lock()
        self._connection: pika.BlockingConnection | None = None
        self._channel: BlockingChannel | None = None

    def _ensure(self) -> BlockingChannel:
        if self._channel is None or not self._channel.is_open:
            self.close()
            self._connection = connect(self._s, self._name)
            self._channel = open_channel(self._connection, self._s)
        return self._channel

    def publish(self, exchange: str, routing_key: str, body: dict[str, Any], **kwargs: Any) -> None:
        with self._lock:
            try:
                publish_json(self._ensure(), exchange, routing_key, body, **kwargs)
            except PublishError:
                raise
            except pika.exceptions.AMQPError as e:
                self.close()
                raise PublishError(f"RabbitMQ unavailable: {e}") from e

    def close(self) -> None:
        try:
            if self._connection is not None and self._connection.is_open:
                self._connection.close()
        except Exception:  # closing a dead connection can raise; nothing to do about it
            pass
        self._connection = None
        self._channel = None
