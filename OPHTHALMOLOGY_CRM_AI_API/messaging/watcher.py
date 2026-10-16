"""Timeout watcher: alerts operations when a study has waited too long for its results.

    python -m messaging.watcher

A model that's down only delays its result, but someone should know. Every interval, studies
still not FINAL after FUNDUS_REPORT_TIMEOUT_S (default one hour) are published once to the
alerts exchange, listing the tests still missing.
"""
from __future__ import annotations

import logging
import signal
import threading
from datetime import timedelta

from . import broker
from .settings import Settings
from .store import StudyStore

log = logging.getLogger(__name__)


class TimeoutWatcher:
    def __init__(self, s: Settings, store: StudyStore | None = None):
        self.s = s
        self.store = store or StudyStore(s.db_url)
        self._stop = threading.Event()

    def check_once(self, channel) -> list[dict]:
        overdue = self.store.claim_overdue(timedelta(seconds=self.s.report_timeout_s))
        for study in overdue:
            alert = {"type": "study_overdue", **study,
                     "message": f"Study {study['study_id']} is still {study['status']}; "
                                f"waiting on {', '.join(study['missing_tests'])}"}
            broker.publish_json(channel, self.s.alerts_exchange, "", alert,
                                message_id=f"overdue:{study['study_id']}")
            log.warning(alert["message"])
        return overdue

    def run(self) -> None:
        connection = broker.connect(self.s, "fundus-timeout-watcher")
        try:
            channel = broker.open_channel(connection, self.s)
            while not self._stop.is_set():
                self.check_once(channel)
                # connection.sleep keeps the heartbeat serviced; wake every second to notice stop().
                for _ in range(self.s.watcher_interval_s):
                    if self._stop.is_set():
                        break
                    connection.sleep(1)
        finally:
            if connection.is_open:
                connection.close()

    def stop(self) -> None:
        self._stop.set()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    watcher = TimeoutWatcher(Settings.from_env())
    signal.signal(signal.SIGTERM, lambda *_: watcher.stop())
    signal.signal(signal.SIGINT, lambda *_: watcher.stop())
    watcher.run()


if __name__ == "__main__":
    main()
