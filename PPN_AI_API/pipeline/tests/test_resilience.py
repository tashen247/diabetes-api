"""The report stays pending until all eight model results are in."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pipeline.broker import InProcessBroker
from pipeline.catalog import EXPECTED_TESTS, TESTS
from pipeline.clock import ManualClock
from pipeline.errors import BadImageError, BrokerUnavailable, TransientModelError
from pipeline.flow import accept_study, drain, recover_unpublished, replay_dead_letter, scan_once
from pipeline.flow import retry_crm
from pipeline.runner import ScriptedRunner, classify_http, transient_reason
from pipeline.runtime import Deps
from pipeline.store import StudyStore
from pipeline.topology import build_topology, retry_arguments

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
BAD = b"not-an-image!!"
START = datetime(2026, 10, 16, 6, 0, tzinfo=timezone.utc)


class MemoryCrm:
    def __init__(self):
        self.posts = []
        self.fail = False

    def post(self, url, report):
        if self.fail:
            raise RuntimeError("crm down")
        self.posts.append({"url": url, "report": report})


class FlakyBroker(InProcessBroker):
    def __init__(self, topology):
        super().__init__(topology)
        self.fail = True

    def publish_fanout(self, body, headers, now):
        if self.fail:
            raise BrokerUnavailable("rabbit down")
        return super().publish_fanout(body, headers, now)


class BoomRunner:
    def run(self, test, images, body):
        raise AssertionError(f"{test.name} must not run on an unreadable image")


class ResilienceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.clock = ManualClock(START)
        self.store = StudyStore(root / "studies.db", root / "images")
        self.topology = build_topology()
        self.broker = InProcessBroker(self.topology)
        self.broker.declare()
        self.crm = MemoryCrm()
        self.runner = ScriptedRunner()
        self.deps = Deps(
            store=self.store,
            broker=self.broker,
            runner=self.runner,
            clock=self.clock,
            crm=self.crm,
            sla_seconds=3600,
        )

    def _images(self, payload: bytes = JPEG):
        return [
            ("left.jpg", "image/jpeg", payload),
            ("right.jpg", "image/jpeg", payload),
        ]

    def _accept(self, payload: bytes = JPEG, referral: str = "REF-1"):
        return accept_study(self.deps, referral, self._images(payload), "http://crm.example/reports")

    def test_topology_matches_the_diagram(self):
        self.assertEqual([test.queue for test in TESTS], [
            "q.diabetic-retinopathy",
            "q.hypertensive-retinopathy",
            "q.rvo",
            "q.cnv",
            "q.myopia",
            "q.model-6",
            "q.model-7",
            "q.model-8",
        ])
        self.assertEqual(len(self.topology.tests), EXPECTED_TESTS)
        self.assertEqual(len(self.topology.retry_queues), 24)
        cnv_30 = next(item for item in self.topology.retry_queues if item.name == "q.retry.30s.cnv")
        cnv_2m = next(item for item in self.topology.retry_queues if item.name == "q.retry.2m.cnv")
        cnv_10m = next(item for item in self.topology.retry_queues if item.name == "q.retry.10m.cnv")
        self.assertEqual(retry_arguments(cnv_30)["x-message-ttl"], 30_000)
        self.assertEqual(retry_arguments(cnv_2m)["x-message-ttl"], 120_000)
        self.assertEqual(retry_arguments(cnv_10m)["x-message-ttl"], 600_000)
        self.assertEqual(retry_arguments(cnv_30)["x-dead-letter-routing-key"], "q.cnv")
        self.assertEqual(
            [test.model_key for test in TESTS if test.name.startswith("model-")],
            ["glaucoma", "macular_degeneration", "pathological"],
        )

    def test_publish_once_and_images_stay_off_the_queue(self):
        report = self._accept()
        self.assertEqual(report["status"], "PENDING")
        self.assertFalse(report["complete"])
        self.assertEqual(report["tests_expected"], 8)
        self.assertEqual(report["tests_done"], 0)
        self.assertTrue(report["queued"])
        self.assertEqual(self.broker.fanout_count, 1)
        body = None
        for test in TESTS:
            queued = self.broker.browse(test.queue)
            self.assertEqual(len(queued), 1, test.queue)
            body = queued[0].body
            self.assertEqual(body["study_id"], report["study_id"])
            self.assertEqual(queued[0].headers["attempt"], 0)
        encoded = json.dumps(body).encode()
        self.assertNotIn(b"\xff\xd8", encoded)
        self.assertIn("image_ids", body)
        self.assertNotIn("data", body)

    def test_down_model_delays_its_result_and_report_stays_pending(self):
        self.runner.scripts["cnv"] = [
            TransientModelError("GPU busy", reason="gpu_busy"),
            {"label": "No Suspect or Negative", "test": "cnv"},
        ]
        report = self._accept()
        drain(self.deps)

        pending = self.store.report(report["study_id"])
        self.assertEqual(pending["status"], "PENDING")
        self.assertFalse(pending["complete"])
        self.assertEqual(pending["tests_done"], 7)
        self.assertEqual(self.crm.posts, [])
        cnv = next(test for test in pending["tests"] if test["name"] == "cnv")
        self.assertEqual(cnv["status"], "retrying")
        self.assertEqual(cnv["reason"], "gpu_busy")

        waiting = self.broker.browse("q.retry.30s.cnv")
        self.assertEqual(len(waiting), 1)
        self.assertEqual(waiting[0].headers["attempt"], 1)
        self.assertEqual(waiting[0].headers["model"], "cnv")
        self.assertEqual(waiting[0].headers["reason"], "gpu_busy")
        self.assertEqual(self.broker.depth("q.cnv"), 0)

        self.clock.advance(29)
        drain(self.deps)
        self.assertEqual(self.broker.depth("q.retry.30s.cnv"), 1)
        self.assertFalse(self.store.report(report["study_id"])["complete"])

        self.clock.advance(1)
        drain(self.deps)
        final = self.store.report(report["study_id"])
        self.assertTrue(final["complete"])
        self.assertEqual(final["status"], "FINAL")
        self.assertEqual(final["tests_done"], 8)
        self.assertEqual(len(self.crm.posts), 1)
        self.assertTrue(self.crm.posts[0]["report"]["complete"])
        self.assertEqual(self.crm.posts[0]["report"]["tests_done"], 8)
        self.assertTrue(final["released_to_crm"])

    def test_retries_exhausted_dead_letter_and_report_stays_pending(self):
        self.runner.scripts["myopia"] = [TransientModelError("connection refused") for _ in range(4)]
        report = self._accept()
        drain(self.deps)
        self.clock.advance(30)
        drain(self.deps)
        self.clock.advance(120)
        drain(self.deps)
        self.clock.advance(600)
        drain(self.deps)

        self.assertEqual(self.runner.calls.count("myopia"), 4)
        dead = self.broker.browse("q.dead-letter")
        self.assertEqual(len(dead), 1)
        self.assertEqual(dead[0].headers["reason"], "retries_exhausted")
        self.assertEqual(dead[0].headers["model"], "myopia")
        self.assertEqual(dead[0].headers["attempt"], 3)
        final = self.store.report(report["study_id"])
        self.assertFalse(final["complete"])
        self.assertEqual(final["status"], "PENDING")
        self.assertEqual(final["tests_done"], 7)
        self.assertTrue(final["needs_attention"])
        self.assertEqual(self.crm.posts, [])

    def test_bad_image_is_dead_lettered_without_a_retry(self):
        self.deps.runner = BoomRunner()
        report = self._accept(BAD)
        drain(self.deps)
        self.assertEqual(self.broker.depth("q.dead-letter"), 8)
        for envelope in self.broker.browse("q.dead-letter"):
            self.assertEqual(envelope.headers["reason"], "unreadable_image")
            self.assertIn("model", envelope.headers)
        stored = self.store.report(report["study_id"])
        self.assertFalse(stored["complete"])
        self.assertEqual(stored["tests_done"], 0)
        self.assertTrue(all(test["status"] == "dead" for test in stored["tests"]))
        self.assertEqual(self.broker.depth("q.retry.30s.diabetic-retinopathy"), 0)

    def test_replay_after_a_fix_can_finish_the_report(self):
        self.runner.scripts["rvo"] = [BadImageError("endpoint rejected the image")]
        report = self._accept()
        drain(self.deps)
        halfway = self.store.report(report["study_id"])
        self.assertEqual(halfway["tests_done"], 7)
        self.assertFalse(halfway["complete"])
        dead = self.broker.browse("q.dead-letter")
        self.assertEqual(dead[0].headers["model"], "rvo")
        self.assertEqual(dead[0].headers["reason"], "bad_image")

        self.runner.scripts["rvo"] = []
        replayed = replay_dead_letter(self.deps, report["study_id"], "rvo")
        self.assertFalse(replayed["complete"])
        self.assertEqual(self.broker.depth("q.rvo"), 1)
        drain(self.deps)

        final = self.store.report(report["study_id"])
        self.assertTrue(final["complete"])
        self.assertEqual(len(self.crm.posts), 1)
        self.assertTrue(all(post["report"]["complete"] for post in self.crm.posts))

    def test_overdue_alert_does_not_complete_the_report(self):
        report = self._accept()
        self.clock.advance(3600)
        alerts = scan_once(self.deps)
        self.assertEqual(len(alerts), 8)
        pending = self.store.report(report["study_id"])
        self.assertFalse(pending["complete"])
        self.assertEqual(pending["status"], "PENDING")
        self.assertTrue(pending["needs_attention"])
        self.assertTrue(all(test["status"] == "overdue" for test in pending["tests"]))

        drain(self.deps)
        final = self.store.report(report["study_id"])
        self.assertTrue(final["complete"])
        self.assertEqual(final["status"], "FINAL")

    def test_a_stored_final_flag_cannot_hide_a_missing_result(self):
        report = self._accept(referral="REF-PARTIAL")
        only = {
            "kind": "result",
            "study_id": report["study_id"],
            "test": "diabetic-retinopathy",
            "attempt": 0,
            "result": {"label": "Mild"},
        }
        # Pull the fan-out aside by applying the row directly.
        became = self.store.apply(only, self.clock.now())
        self.assertFalse(became)
        self.store.force_column_status(report["study_id"], "FINAL")
        repaired = self.store.report(report["study_id"])
        self.assertEqual(repaired["status"], "PENDING")
        self.assertFalse(repaired["complete"])
        self.assertEqual(repaired["tests_done"], 1)
        conn = sqlite3.connect(self.store.db_path)
        try:
            column = conn.execute(
                "SELECT status FROM studies WHERE study_id=?",
                (report["study_id"],),
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(column, "PENDING")

    def test_duplicate_result_does_not_release_early_or_twice(self):
        report = self._accept()
        first = {
            "kind": "result",
            "study_id": report["study_id"],
            "test": "model-6",
            "attempt": 0,
            "result": {"label": "No Suspect or Negative"},
        }
        self.assertFalse(self.store.apply(first, self.clock.now()))
        self.assertFalse(self.store.apply(first, self.clock.now()))
        self.assertEqual(self.store.report(report["study_id"])["tests_done"], 1)
        for test in TESTS:
            if test.name == "model-6":
                continue
            became = self.store.apply(
                {
                    "kind": "result",
                    "study_id": report["study_id"],
                    "test": test.name,
                    "attempt": 0,
                    "result": {"label": "No Suspect or Negative"},
                },
                self.clock.now(),
            )
        self.assertTrue(became)
        self.assertFalse(
            self.store.apply(
                {
                    "kind": "result",
                    "study_id": report["study_id"],
                    "test": "model-8",
                    "attempt": 1,
                    "result": {"label": "again"},
                },
                self.clock.now(),
            )
        )
        from pipeline.flow import deliver_final

        deliver_final(self.deps, report["study_id"])
        deliver_final(self.deps, report["study_id"])
        self.assertEqual(len(self.crm.posts), 1)

    def test_crm_outage_keeps_the_final_report_and_retries_the_release(self):
        self.crm.fail = True
        report = self._accept()
        drain(self.deps)
        stored = self.store.report(report["study_id"])
        self.assertTrue(stored["complete"])
        self.assertEqual(stored["status"], "FINAL")
        self.assertFalse(stored["released_to_crm"])
        self.assertEqual(stored["crm_status"], "failed")
        self.assertEqual(self.crm.posts, [])

        self.crm.fail = False
        retry_crm(self.deps)
        released = self.store.report(report["study_id"])
        self.assertTrue(released["released_to_crm"])
        self.assertEqual(len(self.crm.posts), 1)
        retry_crm(self.deps)
        self.assertEqual(len(self.crm.posts), 1)

    def test_broker_down_at_intake_does_not_lose_the_study(self):
        flaky = FlakyBroker(self.topology)
        flaky.declare()
        self.deps.broker = flaky
        report = self._accept()
        self.assertFalse(report["queued"])
        self.assertFalse(report["complete"])
        self.assertEqual(flaky.fanout_count, 0)
        self.assertEqual(self.store.report(report["study_id"])["published"], False)

        flaky.fail = False
        self.assertEqual(recover_unpublished(self.deps), 1)
        self.assertEqual(flaky.fanout_count, 1)
        self.assertEqual(self.broker.depth("q.cnv") if False else flaky.depth("q.cnv"), 1)
        drain(self.deps)
        self.assertTrue(self.store.report(report["study_id"])["complete"])

    def test_http_intake_stays_pending_until_workers_finish(self):
        from fastapi.testclient import TestClient

        from pipeline.api import create_app

        client = TestClient(create_app(self.deps))
        response = client.post(
            "/intake/studies",
            data={"referral_id": "REF-HTTP", "crm_callback_url": "http://crm.example/reports"},
            files=[
                ("images", ("left.jpg", JPEG, "image/jpeg")),
                ("images", ("right.jpg", JPEG, "image/jpeg")),
            ],
        )
        self.assertEqual(response.status_code, 202, response.text)
        body = response.json()
        self.assertEqual(body["status"], "PENDING")
        self.assertFalse(body["complete"])
        self.assertEqual(self.broker.fanout_count, 1)
        self.assertEqual(response.headers["cache-control"], "no-store")

        pending = client.get(f"/intake/studies/{body['study_id']}/report")
        self.assertEqual(pending.status_code, 200)
        self.assertFalse(pending.json()["complete"])

        drain(self.deps)
        final = client.get(f"/intake/studies/{body['study_id']}/report")
        self.assertTrue(final.json()["complete"])
        self.assertEqual(final.json()["status"], "FINAL")
        self.assertEqual(len(self.crm.posts), 1)

    def test_http_classifies_model_failures(self):
        self.assertIsNone(classify_http(200, "{}"))
        with self.assertRaises(TransientModelError) as busy:
            classify_http(503, "gpu memory busy")
        self.assertEqual(busy.exception.reason, "gpu_busy")
        self.assertEqual(transient_reason(500, "boom"), "model_down")
        with self.assertRaises(BadImageError):
            classify_http(422, "not an image")


if __name__ == "__main__":
    unittest.main()
