"""Model worker: consumes one condition's queue and runs that model on each study.

    python -m messaging.worker diabetes

Each worker is independent. If the RVO model is down, only q.rvo waits; the other seven keep
going. The worker calls the model's existing HTTP endpoint (the same one /predict-all used), so
the model code itself doesn't change.

For each message, exactly one of these happens before the ack:
  success          result published to the results exchange
  temporary error  copy sent to the next retry queue (10s, 1m, 10m by default)
  retries used up  dead-lettered with the reason, and a FAILED result sent to the aggregator
  bad image        dead-lettered straight away (retrying a corrupt image never helps)
If the broker itself fails mid-way, the message is not acked, so RabbitMQ redelivers it.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from . import broker
from .settings import Settings

log = logging.getLogger(__name__)

ATTEMPT_HEADER = "x-attempt"


class TransientError(Exception):
    """Worth retrying: the model is down, busy, or timed out."""


class PermanentError(Exception):
    """Retrying can't help: the image is missing or unreadable, or the request is invalid."""


ModelCall = Callable[[str, dict[str, Any], bytes, bytes], Any]


def http_model_call(s: Settings) -> ModelCall:
    """Sends both eyes to the model's HTTP endpoint and classifies any failure."""
    session = requests.Session()
    headers = {"Authorization": f"Bearer {s.bearer_token}"} if s.bearer_token else {}

    def call(condition: str, study: dict[str, Any], left: bytes, right: bytes) -> Any:
        url = s.model_urls[condition]
        files = {
            "left_eye_file": (Path(study["left_image"]).name, left, "image/jpeg"),
            "right_eye_file": (Path(study["right_image"]).name, right, "image/jpeg"),
        }
        try:
            resp = session.post(url, params={"referral_id": study["referral_id"]}, files=files,
                                headers=headers, timeout=s.model_timeout_s)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise TransientError(f"model unreachable: {type(e).__name__}") from e
        if resp.status_code in (408, 429) or resp.status_code >= 500:
            raise TransientError(f"model returned HTTP {resp.status_code}")
        if resp.status_code >= 400:
            # 400/415/422 mean the model rejected this input; it will reject it every time.
            raise PermanentError(f"model rejected the images: HTTP {resp.status_code} {resp.text[:200]}")
        try:
            return resp.json()
        except ValueError as e:
            raise TransientError("model returned a response that isn't JSON") from e

    return call


@dataclass
class Outcome:
    action: str  # "done", "retry", "dead"
    detail: str = ""


class ModelWorker:
    def __init__(self, s: Settings, condition: str, model_call: ModelCall | None = None):
        if condition not in s.conditions:
            raise ValueError(f"unknown condition {condition!r}; expected one of {s.conditions}")
        self.s = s
        self.condition = condition
        self.model_call = model_call or http_model_call(s)
        self._stop = threading.Event()
        self._channel: Any = None

    # Image loading -------------------------------------------------------------------
    def _load(self, relative: str) -> bytes:
        root = Path(self.s.image_store_dir).resolve()
        path = (root / relative).resolve()
        if root not in path.parents:
            raise PermanentError(f"image path escapes the image store: {relative}")
        try:
            data = path.read_bytes()
        except FileNotFoundError as e:
            raise PermanentError(f"image missing from the store: {relative}") from e
        if not data:
            raise PermanentError(f"image is empty: {relative}")
        return data

    # Message handling ----------------------------------------------------------------
    def handle(self, channel, method, properties, body: bytes) -> Outcome:
        headers = dict(properties.headers or {})
        attempt = int(headers.get(ATTEMPT_HEADER, 0))
        try:
            study = json.loads(body)
            study_id = study["study_id"]
            _ = study["referral_id"], study["left_image"], study["right_image"]
        except (ValueError, KeyError, TypeError):
            # Unparseable: reject without requeue, and the queue's DLX moves it to the dead-letter queue.
            log.error("[%s] rejecting malformed message", self.condition)
            channel.basic_reject(method.delivery_tag, requeue=False)
            return Outcome("dead", "malformed message")

        try:
            left = self._load(study["left_image"])
            right = self._load(study["right_image"])
            result = self.model_call(self.condition, study, left, right)
        except TransientError as e:
            if attempt < len(self.s.retry_delays_ms):
                delay = self.s.retry_delays_ms[attempt]
                broker.publish_json(channel, self.s.retry_exchange,
                                    self.s.retry_routing_key(self.condition, delay), study,
                                    headers={**headers, ATTEMPT_HEADER: attempt + 1,
                                             "x-last-error": str(e)[:200]},
                                    message_id=properties.message_id)
                channel.basic_ack(method.delivery_tag)
                log.warning("[%s] study %s: %s; retry %d in %d ms", self.condition, study_id, e,
                            attempt + 1, delay)
                return Outcome("retry", str(e))
            return self._dead_letter(channel, method, properties, study, headers, attempt,
                                     f"retries exhausted after {attempt + 1} attempts: {e}")
        except PermanentError as e:
            return self._dead_letter(channel, method, properties, study, headers, attempt, str(e))

        broker.publish_json(channel, self.s.results_exchange, "", {
            "study_id": study_id, "condition": self.condition, "ok": True,
            "result": result, "attempts": attempt + 1,
        }, message_id=f"{study_id}:{self.condition}")
        channel.basic_ack(method.delivery_tag)
        log.info("[%s] study %s done after %d attempt(s)", self.condition, study_id, attempt + 1)
        return Outcome("done")

    def _dead_letter(self, channel, method, properties, study, headers, attempt, reason) -> Outcome:
        study_id = study["study_id"]
        broker.publish_json(channel, self.s.dead_letter_exchange, self.s.DEAD_LETTER_ROUTING_KEY, study, headers={
            **{k: v for k, v in headers.items() if k != ATTEMPT_HEADER},
            "x-condition": self.condition,
            "x-study-id": study_id,
            "x-reason": reason[:400],
            "x-attempts": attempt + 1,
            "x-dead-lettered-at": datetime.now(timezone.utc).isoformat(),
        }, message_id=properties.message_id)
        # Tell the aggregator too, so the report shows NEEDS_ATTENTION instead of waiting silently.
        broker.publish_json(channel, self.s.results_exchange, "", {
            "study_id": study_id, "condition": self.condition, "ok": False,
            "reason": reason, "attempts": attempt + 1,
        }, message_id=f"{study_id}:{self.condition}:failed")
        channel.basic_ack(method.delivery_tag)
        log.error("[%s] study %s dead-lettered: %s", self.condition, study_id, reason)
        return Outcome("dead", reason)

    def _on_message(self, channel, method, properties, body) -> None:
        try:
            self.handle(channel, method, properties, body)
        except broker.PublishError as e:
            # Couldn't hand the outcome on, so keep the message: requeue it for another try.
            log.error("[%s] publish failed, requeueing: %s", self.condition, e)
            channel.basic_nack(method.delivery_tag, requeue=True)

    # Run loop ------------------------------------------------------------------------
    def run(self) -> None:
        connection = broker.connect(self.s, f"fundus-worker-{self.condition}")
        try:
            channel = broker.open_channel(connection, self.s)
            # One study at a time per worker: each one is a GPU inference.
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(self.s.model_queue(self.condition), self._on_message)
            log.info("[%s] consuming %s", self.condition, self.s.model_queue(self.condition))
            while not self._stop.is_set():
                connection.process_data_events(time_limit=0.5)
        finally:
            if connection.is_open:
                connection.close()

    def stop(self) -> None:
        self._stop.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one screening model worker.")
    parser.add_argument("condition")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    worker = ModelWorker(Settings.from_env(), args.condition)
    signal.signal(signal.SIGTERM, lambda *_: worker.stop())
    signal.signal(signal.SIGINT, lambda *_: worker.stop())
    worker.run()


if __name__ == "__main__":
    main()
