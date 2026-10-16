"""Aggregator: records each model's result and releases the report only when all are in.

    python -m messaging.aggregator

A result is acked only after it's committed, so a crash here means redelivery, not a lost
result. Recording is idempotent on (study_id, condition), so redelivery is harmless.
When the last result arrives, the report goes to the reports exchange for the CRM. It's marked
sent only after the broker confirms it, so a failed publish is retried on redelivery. Its
message_id is "report:<study id>", so the CRM can drop the rare duplicate.
"""
from __future__ import annotations

import json
import logging
import signal
import threading

from . import broker
from .settings import Settings
from .store import STUDY_FINAL, StudyStore

log = logging.getLogger(__name__)


class Aggregator:
    def __init__(self, s: Settings, store: StudyStore | None = None):
        self.s = s
        self.store = store or StudyStore(s.db_url)
        self._stop = threading.Event()

    def handle(self, channel, method, properties, body: bytes) -> None:
        try:
            msg = json.loads(body)
            study_id, condition, ok = msg["study_id"], msg["condition"], bool(msg["ok"])
        except (ValueError, KeyError, TypeError):
            log.error("rejecting malformed result")
            channel.basic_reject(method.delivery_tag, requeue=False)  # goes to the dead-letter queue
            return

        outcome = self.store.record_result(study_id, condition, ok, result=msg.get("result"),
                                           reason=msg.get("reason"), attempts=int(msg.get("attempts", 0)))
        if outcome.study_status == "UNKNOWN":
            log.error("result for unknown study %s; dead-lettering it", study_id)
            channel.basic_reject(method.delivery_tag, requeue=False)
            return
        if not outcome.applied:
            log.info("duplicate %s result for study %s ignored", condition, study_id)

        study = self.store.get_study(study_id) if outcome.study_status == STUDY_FINAL else None
        if study is not None and study["reported_at"] is None:
            broker.publish_json(channel, self.s.reports_exchange, "", {
                "study_id": study_id, "referral_id": study["referral_id"], "status": STUDY_FINAL,
                "tests": study["tests"], "finalised_at": study["finalised_at"],
            }, message_id=f"report:{study_id}")
            self.store.mark_reported(study_id)
            log.info("study %s FINAL: report released to the CRM", study_id)
        channel.basic_ack(method.delivery_tag)

    def _on_message(self, channel, method, properties, body) -> None:
        try:
            self.handle(channel, method, properties, body)
        except broker.PublishError as e:
            # The result is recorded but the report wasn't confirmed. Requeue: the redelivered
            # result isn't applied twice, and the unsent report is published again.
            log.error("report publish failed, requeueing: %s", e)
            channel.basic_nack(method.delivery_tag, requeue=True)

    def run(self) -> None:
        connection = broker.connect(self.s, "fundus-aggregator")
        try:
            channel = broker.open_channel(connection, self.s)
            channel.basic_qos(prefetch_count=50)
            channel.basic_consume(self.s.results_queue, self._on_message)
            log.info("aggregator consuming %s", self.s.results_queue)
            while not self._stop.is_set():
                connection.process_data_events(time_limit=0.5)
        finally:
            if connection.is_open:
                connection.close()

    def stop(self) -> None:
        self._stop.set()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    agg = Aggregator(Settings.from_env())
    signal.signal(signal.SIGTERM, lambda *_: agg.stop())
    signal.signal(signal.SIGINT, lambda *_: agg.stop())
    agg.run()


if __name__ == "__main__":
    main()
