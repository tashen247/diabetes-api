"""End-to-end tests against a real RabbitMQ. Skipped when none is reachable.

    docker compose -f docker-compose.rabbitmq.yml up -d
    python -m pytest messaging/tests

Each test gets its own exchange and queue prefix and deletes everything afterwards. The models
are stand-ins whose behaviour each test controls (healthy, down, rejecting the image), so the
tests run without GPUs or trained weights.
"""
from __future__ import annotations

import collections
import json
import os
import threading
import time
import uuid

import pika
import pytest
from fastapi.testclient import TestClient

from messaging import broker, intake, ops, topology
from messaging.aggregator import Aggregator
from messaging.settings import DEFAULT_MODEL_URLS, Settings
from messaging.store import StudyStore
from messaging.watcher import TimeoutWatcher
from messaging.worker import ModelWorker, PermanentError, TransientError

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/%2F")
CONDITIONS = tuple(DEFAULT_MODEL_URLS)


def _rabbit_available() -> bool:
    try:
        pika.BlockingConnection(pika.URLParameters(RABBITMQ_URL)).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _rabbit_available(), reason="RabbitMQ is not reachable")


class Models:
    """Stand-in models. behaviour[condition] is 'ok', 'down', 'bad_image', or an int N
    meaning 'down for the first N calls, then ok'."""

    def __init__(self):
        self.behaviour: dict[str, object] = {c: "ok" for c in CONDITIONS}
        self.calls = collections.Counter()
        self._lock = threading.Lock()

    def __call__(self, condition, study, left, right):
        assert left == b"left-eye" and right == b"right-eye"
        with self._lock:
            self.calls[condition] += 1
            calls = self.calls[condition]
        b = self.behaviour[condition]
        if b == "down" or (isinstance(b, int) and calls <= b):
            raise TransientError("model unreachable: ConnectionError")
        if b == "bad_image":
            raise PermanentError("model rejected the images: HTTP 422 ungradable image")
        return {"condition": condition, "grade": "No Suspect or Negative", "confidence": 0.97}


class Pipeline:
    def __init__(self, tmp_path, **overrides):
        self.s = Settings(rabbitmq_url=RABBITMQ_URL, prefix=f"it{uuid.uuid4().hex[:8]}",
                          conditions=CONDITIONS, retry_delays_ms=(200, 400, 600),
                          image_store_dir=str(tmp_path / "images"),
                          db_url=f"sqlite:///{tmp_path / 'studies.db'}", **overrides)
        self.store = StudyStore(self.s.db_url)
        self.models = Models()
        self.conn = broker.connect(self.s, "fundus-tests")
        self.ch = broker.open_channel(self.conn, self.s)
        self._threads: list[tuple[object, threading.Thread]] = []
        self.publisher = broker.SharedPublisher(self.s, "fundus-tests-intake")
        intake.configure(self.s, self.store, self.publisher)
        self.client = TestClient(intake.app)

    def _start(self, runner):
        t = threading.Thread(target=runner.run, daemon=True)
        t.start()
        self._threads.append((runner, t))
        return runner

    def start_aggregator(self):
        return self._start(Aggregator(self.s, self.store))

    def start_worker(self, condition):
        return self._start(ModelWorker(self.s, condition, self.models))

    def start_workers(self, *skip):
        for c in CONDITIONS:
            if c not in skip:
                self.start_worker(c)

    def submit(self, referral_id="R1") -> str:
        resp = self.client.post("/studies", data={"referral_id": referral_id}, files={
            "left_eye_file": ("left.jpg", b"left-eye", "image/jpeg"),
            "right_eye_file": ("right.jpg", b"right-eye", "image/jpeg")})
        assert resp.status_code == 202, resp.text
        return resp.json()["study_id"]

    def study(self, study_id):
        return self.client.get(f"/studies/{study_id}").json()

    def wait_for(self, predicate, timeout=15.0, what="condition"):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(0.05)
        raise AssertionError(f"timed out waiting for {what}")

    def wait_status(self, study_id, status, timeout=15.0):
        return self.wait_for(lambda: self.study(study_id)["status"] == status and self.study(study_id),
                             timeout, f"study {study_id} to become {status}")

    def depth(self, queue):
        return self.ch.queue_declare(queue, passive=True).method.message_count

    def drain(self, queue):
        out = []
        while True:
            method, props, body = self.ch.basic_get(queue, auto_ack=True)
            if method is None:
                return out
            out.append((props, json.loads(body)))

    def close(self):
        for runner, _ in self._threads:
            runner.stop()
        for _, t in self._threads:
            t.join(timeout=5)
        self.publisher.close()
        topology.delete_all(self.ch, self.s)
        self.conn.close()


@pytest.fixture
def pipe(tmp_path):
    p = Pipeline(tmp_path)
    yield p
    p.close()


def test_topology_declares_durable_queues_with_dead_lettering(pipe):
    for c in CONDITIONS:
        assert pipe.depth(pipe.s.model_queue(c)) == 0
        for d in pipe.s.retry_delays_ms:
            assert pipe.depth(pipe.s.retry_queue(c, d)) == 0
    # Declaring again with the same arguments is a no-op, so every service can declare on startup.
    topology.declare(pipe.ch, pipe.s)


def test_one_study_fans_out_to_all_eight_models_and_one_final_report(pipe):
    pipe.start_aggregator()
    pipe.start_workers()
    study_id = pipe.submit("REF-100")

    study = pipe.wait_status(study_id, "FINAL")
    assert [t["status"] for t in study["tests"]] == ["DONE"] * 8
    assert all(t["result"]["grade"] == "No Suspect or Negative" for t in study["tests"])
    assert pipe.models.calls == {c: 1 for c in CONDITIONS}

    reports = pipe.wait_for(lambda: pipe.drain(pipe.s.reports_queue), what="the CRM report")
    assert len(reports) == 1
    props, report = reports[0]
    assert report["referral_id"] == "REF-100" and report["status"] == "FINAL"
    assert props.message_id == f"report:{study_id}"


def test_a_model_that_is_down_is_retried_alone_and_the_report_waits(pipe):
    pipe.models.behaviour["rvo"] = 2  # down for two calls, then back
    pipe.start_aggregator()
    pipe.start_workers()
    study_id = pipe.submit()

    # While RVO is down, the report is PENDING, never an incomplete FINAL.
    pending = pipe.wait_for(lambda: (s := pipe.study(study_id)) and
                            sum(t["status"] == "DONE" for t in s["tests"]) == 7 and s, what="seven results")
    assert pending["status"] == "PENDING"

    study = pipe.wait_status(study_id, "FINAL")
    rvo = next(t for t in study["tests"] if t["condition"] == "rvo")
    assert rvo["attempts"] == 3
    # Only RVO was retried; the other seven models ran once.
    assert pipe.models.calls["rvo"] == 3
    assert all(pipe.models.calls[c] == 1 for c in CONDITIONS if c != "rvo")
    assert pipe.depth(pipe.s.dead_letter_queue) == 0


def test_a_study_waits_in_the_queue_while_a_worker_is_offline(pipe):
    pipe.start_aggregator()
    pipe.start_workers("cnv")  # the CNV service has crashed
    study_id = pipe.submit()

    pipe.wait_for(lambda: sum(t["status"] == "DONE" for t in pipe.study(study_id)["tests"]) == 7,
                  what="the other seven results")
    assert pipe.study(study_id)["status"] == "PENDING"
    assert pipe.depth(pipe.s.model_queue("cnv")) == 1  # nothing lost, just waiting

    pipe.start_worker("cnv")  # the service comes back
    pipe.wait_status(study_id, "FINAL")
    assert pipe.depth(pipe.s.model_queue("cnv")) == 0


def test_exhausted_retries_are_dead_lettered_and_replayed_after_the_fix(pipe):
    pipe.models.behaviour["glaucoma"] = "down"
    pipe.start_aggregator()
    pipe.start_workers()
    study_id = pipe.submit()

    study = pipe.wait_status(study_id, "NEEDS_ATTENTION")
    glaucoma = next(t for t in study["tests"] if t["condition"] == "glaucoma")
    assert glaucoma["status"] == "FAILED" and glaucoma["attempts"] == 4
    assert pipe.models.calls["glaucoma"] == 4  # first try plus three retries

    [dead] = ops.list_dead_letters(pipe.s)
    assert dead["study_id"] == study_id and dead["condition"] == "glaucoma"
    assert dead["reason"] == "retries exhausted after 4 attempts: model unreachable: ConnectionError"
    assert pipe.depth(pipe.s.dead_letter_queue) == 1  # listing doesn't consume

    pipe.models.behaviour["glaucoma"] = "ok"  # operations fixes the service
    replayed = ops.replay(pipe.s, study_id=study_id, store=pipe.store)
    assert [r["condition"] for r in replayed] == ["glaucoma"]

    pipe.wait_status(study_id, "FINAL")
    assert pipe.models.calls["glaucoma"] == 5
    assert all(pipe.models.calls[c] == 1 for c in CONDITIONS if c != "glaucoma")
    assert pipe.depth(pipe.s.dead_letter_queue) == 0


def test_a_bad_image_is_dead_lettered_at_once_and_does_not_block_other_studies(pipe):
    pipe.start_aggregator()
    pipe.start_workers()
    pipe.models.behaviour["cnv"] = "bad_image"
    bad = pipe.submit("BAD")
    pipe.wait_status(bad, "NEEDS_ATTENTION")
    assert pipe.models.calls["cnv"] == 1  # never retried

    pipe.models.behaviour["cnv"] = "ok"
    good = pipe.submit("GOOD")
    pipe.wait_status(good, "FINAL")

    [dead] = ops.list_dead_letters(pipe.s)
    assert dead["study_id"] == bad and "ungradable image" in dead["reason"] and dead["attempts"] == 1
    assert [d["study_id"] for d in ops.discard(pipe.s, bad)] == [bad]
    assert pipe.depth(pipe.s.dead_letter_queue) == 0


def test_replay_only_touches_the_selected_study(pipe):
    pipe.models.behaviour["myopia"] = "bad_image"
    pipe.start_aggregator()
    pipe.start_workers()
    first, second = pipe.submit("A"), pipe.submit("B")
    pipe.wait_status(first, "NEEDS_ATTENTION")
    pipe.wait_status(second, "NEEDS_ATTENTION")

    pipe.models.behaviour["myopia"] = "ok"
    ops.replay(pipe.s, study_id=second, store=pipe.store)
    pipe.wait_status(second, "FINAL")
    assert pipe.study(first)["status"] == "NEEDS_ATTENTION"
    assert [d["study_id"] for d in ops.list_dead_letters(pipe.s)] == [first]


def test_overdue_study_raises_one_alert_naming_the_missing_test(tmp_path):
    pipe = Pipeline(tmp_path, report_timeout_s=0)
    try:
        pipe.start_aggregator()
        pipe.start_workers("rvo")
        study_id = pipe.submit()
        pipe.wait_for(lambda: sum(t["status"] == "DONE" for t in pipe.study(study_id)["tests"]) == 7,
                      what="seven results")
        time.sleep(0.01)

        watcher = TimeoutWatcher(pipe.s, pipe.store)
        assert len(watcher.check_once(pipe.ch)) == 1
        assert watcher.check_once(pipe.ch) == []  # alerted once, not every interval
        [(_, alert)] = pipe.drain(pipe.s.alerts_queue)
        assert alert["study_id"] == study_id and alert["missing_tests"] == ["rvo"]
        assert alert["type"] == "study_overdue"
    finally:
        pipe.close()


def test_duplicate_results_do_not_produce_a_second_report(pipe):
    pipe.start_aggregator()
    pipe.start_workers()
    study_id = pipe.submit()
    pipe.wait_status(study_id, "FINAL")
    pipe.wait_for(lambda: pipe.depth(pipe.s.reports_queue) == 1, what="the report")

    # Simulate a redelivered result after the study is complete.
    broker.publish_json(pipe.ch, pipe.s.results_exchange, "", {
        "study_id": study_id, "condition": "diabetes", "ok": True, "result": {"grade": "Severe"}, "attempts": 1})
    pipe.wait_for(lambda: pipe.depth(pipe.s.results_queue) == 0, what="the duplicate to be consumed")
    time.sleep(0.3)
    assert pipe.depth(pipe.s.reports_queue) == 1
    diabetes = next(t for t in pipe.study(study_id)["tests"] if t["condition"] == "diabetes")
    assert diabetes["result"]["grade"] == "No Suspect or Negative"


def test_intake_returns_503_when_rabbitmq_is_unreachable(tmp_path):
    s = Settings(rabbitmq_url="amqp://guest:guest@127.0.0.1:1/%2F", prefix="unreachable",
                 image_store_dir=str(tmp_path / "images"), db_url=f"sqlite:///{tmp_path / 'db.sqlite'}")
    store = StudyStore(s.db_url)
    intake.configure(s, store, broker.SharedPublisher(s))
    resp = TestClient(intake.app).post("/studies", data={"referral_id": "R1"}, files={
        "left_eye_file": ("l.jpg", b"l", "image/jpeg"), "right_eye_file": ("r.jpg", b"r", "image/jpeg")})
    assert resp.status_code == 503
    with store.engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM fundus_studies").scalar() == 0
