"""RabbitMQ topology for one fan-out, eight model queues, retries, and a dead-letter queue.

Retry queues are per model and per delay. RabbitMQ dead-letters each one back
to that model's queue, so a delayed message cannot land on the wrong model and
nothing has to sit in a forwarder process.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline.catalog import TESTS, TestSpec

FANOUT_EXCHANGE = "fundus.fanout"
RESULTS_QUEUE = "q.results"
DEAD_QUEUE = "q.dead-letter"

# attempt 0 -> 30s, attempt 1 -> 2m, attempt 2 -> 10m, attempt 3 -> dead letter
RETRY_STEPS: tuple[tuple[str, int], ...] = (
    ("30s", 30),
    ("2m", 120),
    ("10m", 600),
)


@dataclass(frozen=True)
class RetryQueue:
    name: str
    ttl_seconds: int
    target_queue: str
    test_name: str


@dataclass(frozen=True)
class Topology:
    fanout_exchange: str
    results_queue: str
    dead_queue: str
    tests: tuple[TestSpec, ...]
    retry_queues: tuple[RetryQueue, ...]

    def retry_queue_for(self, test: TestSpec, attempt: int) -> RetryQueue | None:
        if attempt < 0 or attempt >= len(self.retry_queues_for(test)):
            return None
        return self.retry_queues_for(test)[attempt]

    def retry_queues_for(self, test: TestSpec) -> tuple[RetryQueue, ...]:
        return tuple(item for item in self.retry_queues if item.test_name == test.name)


def retry_arguments(retry: RetryQueue) -> dict:
    """Queue arguments that delay a message, then return it to the same model."""
    return {
        "x-message-ttl": int(retry.ttl_seconds * 1000),
        "x-dead-letter-exchange": "",
        "x-dead-letter-routing-key": retry.target_queue,
    }


def build_topology() -> Topology:
    retries: list[RetryQueue] = []
    for test in TESTS:
        for label, seconds in RETRY_STEPS:
            retries.append(
                RetryQueue(
                    name=f"q.retry.{label}.{test.name}",
                    ttl_seconds=seconds,
                    target_queue=test.queue,
                    test_name=test.name,
                )
            )
    return Topology(
        fanout_exchange=FANOUT_EXCHANGE,
        results_queue=RESULTS_QUEUE,
        dead_queue=DEAD_QUEUE,
        tests=TESTS,
        retry_queues=tuple(retries),
    )
