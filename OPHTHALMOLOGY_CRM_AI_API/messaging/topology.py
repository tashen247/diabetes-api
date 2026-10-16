"""Declares every exchange and queue the pipeline uses. Safe to run repeatedly.

    intake ──▶ studies (fanout) ──▶ q.<condition> ×8 ──▶ model worker ──▶ results ──▶ aggregator
                                        ▲      │                                       │
                         retry.<c>.<ms> ┘      ├─▶ retry (direct) ─▶ retry.<c>.<ms>    ├─▶ reports (fanout) ─▶ reports.crm
                       (TTL, then back)        └─▶ dlx (direct) ─▶ dead                └─▶ alerts (fanout) ─▶ alerts.ops
"""
from __future__ import annotations

from pika.adapters.blocking_connection import BlockingChannel

from .settings import Settings


def declare(channel: BlockingChannel, s: Settings) -> None:
    # Dead letters: bad images, exhausted retries, and anything a worker rejects.
    channel.exchange_declare(s.dead_letter_exchange, exchange_type="direct", durable=True)
    channel.queue_declare(s.dead_letter_queue, durable=True)
    channel.queue_bind(s.dead_letter_queue, s.dead_letter_exchange, routing_key=s.DEAD_LETTER_ROUTING_KEY)

    # One message per study, copied to every model's queue.
    channel.exchange_declare(s.studies_exchange, exchange_type="fanout", durable=True)
    channel.exchange_declare(s.retry_exchange, exchange_type="direct", durable=True)

    for condition in s.conditions:
        queue = s.model_queue(condition)
        channel.queue_declare(
            queue,
            durable=True,
            arguments={
                # A message the worker rejects (for example, one it can't parse) is dead-lettered,
                # never dropped.
                "x-dead-letter-exchange": s.dead_letter_exchange,
                "x-dead-letter-routing-key": s.DEAD_LETTER_ROUTING_KEY,
            },
        )
        channel.queue_bind(queue, s.studies_exchange)

        # Retry queues have no consumer. A message waits out the TTL, then the default exchange
        # routes it straight back to this model's queue only, so the other seven models never
        # see a retry.
        for delay_ms in s.retry_delays_ms:
            retry_queue = s.retry_queue(condition, delay_ms)
            channel.queue_declare(
                retry_queue,
                durable=True,
                arguments={
                    "x-message-ttl": delay_ms,
                    "x-dead-letter-exchange": "",
                    "x-dead-letter-routing-key": queue,
                },
            )
            channel.queue_bind(retry_queue, s.retry_exchange, routing_key=s.retry_routing_key(condition, delay_ms))

    # Results from every model to the aggregator. Fanout, so an audit log could subscribe too.
    channel.exchange_declare(s.results_exchange, exchange_type="fanout", durable=True)
    channel.queue_declare(s.results_queue, durable=True, arguments={
        "x-dead-letter-exchange": s.dead_letter_exchange,
        "x-dead-letter-routing-key": s.DEAD_LETTER_ROUTING_KEY,
    })
    channel.queue_bind(s.results_queue, s.results_exchange)

    # FINAL reports for the CRM, and alerts for operations.
    channel.exchange_declare(s.reports_exchange, exchange_type="fanout", durable=True)
    channel.queue_declare(s.reports_queue, durable=True)
    channel.queue_bind(s.reports_queue, s.reports_exchange)

    channel.exchange_declare(s.alerts_exchange, exchange_type="fanout", durable=True)
    channel.queue_declare(s.alerts_queue, durable=True)
    channel.queue_bind(s.alerts_queue, s.alerts_exchange)


def delete_all(channel: BlockingChannel, s: Settings) -> None:
    """Removes everything declare() created. Used by the tests to clean up."""
    queues = [s.dead_letter_queue, s.results_queue, s.reports_queue, s.alerts_queue]
    for condition in s.conditions:
        queues.append(s.model_queue(condition))
        queues.extend(s.retry_queue(condition, d) for d in s.retry_delays_ms)
    for q in queues:
        channel.queue_delete(q)
    for x in (s.studies_exchange, s.retry_exchange, s.dead_letter_exchange, s.results_exchange,
              s.reports_exchange, s.alerts_exchange):
        channel.exchange_delete(x)
