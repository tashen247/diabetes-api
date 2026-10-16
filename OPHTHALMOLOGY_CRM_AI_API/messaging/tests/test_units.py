"""Unit tests: no RabbitMQ needed. Run from OPHTHALMOLOGY_CRM_AI_API with: python -m pytest messaging/tests"""
from __future__ import annotations

import http.server
import json
import threading
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from messaging import intake
from messaging.broker import PublishError
from messaging.settings import Settings
from messaging.store import DONE, FAILED, PENDING, StudyStore
from messaging.worker import ATTEMPT_HEADER, ModelWorker, PermanentError, TransientError, http_model_call

CONDITIONS = ("diabetes", "rvo", "cnv")


@pytest.fixture
def settings(tmp_path):
    return Settings(prefix="unit", conditions=CONDITIONS,
                    model_urls={c: f"http://127.0.0.1:1/{c}" for c in CONDITIONS},
                    retry_delays_ms=(100, 200), image_store_dir=str(tmp_path / "images"),
                    db_url=f"sqlite:///{tmp_path / 'studies.db'}")


@pytest.fixture
def store(settings):
    return StudyStore(settings.db_url)


# Store ---------------------------------------------------------------------------------

def test_report_is_final_only_when_every_test_is_done(store):
    store.create_study("s1", "R1", "s1/left.jpg", "s1/right.jpg", CONDITIONS)
    assert store.record_result("s1", "diabetes", True, {"grade": "Mild"}).study_status == "PENDING"
    assert store.record_result("s1", "rvo", True, {}).study_status == "PENDING"
    last = store.record_result("s1", "cnv", True, {})
    assert last.study_status == "FINAL" and last.became_final
    study = store.get_study("s1")
    assert study["status"] == "FINAL" and study["finalised_at"]
    assert {t["status"] for t in study["tests"]} == {DONE}


def test_recording_a_result_twice_changes_nothing(store):
    store.create_study("s1", "R1", "l", "r", CONDITIONS)
    first = store.record_result("s1", "diabetes", True, {"grade": "Mild"})
    again = store.record_result("s1", "diabetes", True, {"grade": "Severe"})
    assert first.applied and not again.applied
    assert store.get_study("s1")["tests"][1]["result"] == {"grade": "Mild"}  # sorted: cnv, diabetes, rvo


def test_dead_lettered_test_flags_the_study_and_a_replay_can_still_finish_it(store):
    store.create_study("s1", "R1", "l", "r", CONDITIONS)
    store.record_result("s1", "diabetes", True, {})
    store.record_result("s1", "cnv", True, {})
    failed = store.record_result("s1", "rvo", False, reason="image unreadable")
    assert failed.study_status == "NEEDS_ATTENTION"
    assert store.get_study("s1")["tests"][2]["failure_reason"] == "image unreadable"

    store.mark_replayed("s1", "rvo")
    assert store.get_study("s1")["status"] == "PENDING"
    assert store.record_result("s1", "rvo", True, {}).study_status == "FINAL"


def test_a_failure_never_overwrites_a_result(store):
    store.create_study("s1", "R1", "l", "r", CONDITIONS)
    store.record_result("s1", "rvo", True, {})
    assert not store.record_result("s1", "rvo", False, reason="late failure").applied
    assert store.get_study("s1")["tests"][2]["status"] == DONE


def test_result_for_unknown_study(store):
    assert store.record_result("missing", "rvo", True, {}).study_status == "UNKNOWN"


def test_overdue_studies_are_alerted_once(store):
    store.create_study("s1", "R1", "l", "r", CONDITIONS)
    store.record_result("s1", "diabetes", True, {})
    store.create_study("s2", "R2", "l", "r", CONDITIONS)
    for c in CONDITIONS:
        store.record_result("s2", c, True, {})
    overdue = store.claim_overdue(timedelta(seconds=-1))
    assert [o["study_id"] for o in overdue] == ["s1"]
    assert overdue[0]["missing_tests"] == ["cnv", "rvo"]
    assert store.claim_overdue(timedelta(seconds=-1)) == []


# Worker --------------------------------------------------------------------------------

class FakeChannel:
    def __init__(self):
        self.published, self.acked, self.rejected = [], [], []

    def basic_publish(self, exchange, routing_key, body, properties, mandatory):
        self.published.append(SimpleNamespace(exchange=exchange, routing_key=routing_key,
                                              body=json.loads(body), headers=properties.headers))

    def basic_ack(self, tag):
        self.acked.append(tag)

    def basic_reject(self, tag, requeue):
        self.rejected.append((tag, requeue))


def _study(settings, tmp_path, left=b"left-bytes", right=b"right-bytes"):
    folder = tmp_path / "images" / "s1"
    folder.mkdir(parents=True, exist_ok=True)
    if left is not None:
        (folder / "left.jpg").write_bytes(left)
    (folder / "right.jpg").write_bytes(right)
    return {"study_id": "s1", "referral_id": "R1", "left_image": "s1/left.jpg", "right_image": "s1/right.jpg"}


def _deliver(worker, body, attempt=0):
    channel = FakeChannel()
    props = SimpleNamespace(headers={ATTEMPT_HEADER: attempt} if attempt else {}, message_id="s1")
    outcome = worker.handle(channel, SimpleNamespace(delivery_tag=7), props, json.dumps(body).encode())
    return outcome, channel


def test_success_publishes_the_result_then_acks(settings, tmp_path):
    seen = {}

    def model(condition, study, left, right):
        seen.update(condition=condition, left=left, right=right)
        return {"grade": "Mild"}

    outcome, ch = _deliver(ModelWorker(settings, "rvo", model), _study(settings, tmp_path))
    assert outcome.action == "done" and ch.acked == [7]
    assert seen == {"condition": "rvo", "left": b"left-bytes", "right": b"right-bytes"}
    [result] = ch.published
    assert result.exchange == "unit.results"
    assert result.body == {"study_id": "s1", "condition": "rvo", "ok": True, "result": {"grade": "Mild"}, "attempts": 1}


def test_temporary_failure_goes_to_this_models_next_retry_queue(settings, tmp_path):
    def down(*_):
        raise TransientError("model unreachable")

    worker = ModelWorker(settings, "rvo", down)
    outcome, ch = _deliver(worker, _study(settings, tmp_path))
    assert outcome.action == "retry" and ch.acked == [7]
    [retry] = ch.published
    assert (retry.exchange, retry.routing_key) == ("unit.retry", "rvo.100")
    assert retry.headers[ATTEMPT_HEADER] == 1

    outcome, ch = _deliver(worker, _study(settings, tmp_path), attempt=1)
    assert ch.published[0].routing_key == "rvo.200"


def test_retries_exhausted_dead_letters_and_tells_the_aggregator(settings, tmp_path):
    def down(*_):
        raise TransientError("model returned HTTP 503")

    outcome, ch = _deliver(ModelWorker(settings, "rvo", down), _study(settings, tmp_path), attempt=2)
    assert outcome.action == "dead" and ch.acked == [7]
    dead, failed = ch.published
    assert (dead.exchange, dead.routing_key) == ("unit.dlx", "dead")
    assert dead.headers["x-condition"] == "rvo" and dead.headers["x-attempts"] == 3
    assert "retries exhausted after 3 attempts: model returned HTTP 503" == dead.headers["x-reason"]
    assert failed.exchange == "unit.results" and failed.body["ok"] is False


def test_bad_image_is_dead_lettered_without_retrying(settings, tmp_path):
    def rejects(*_):
        raise PermanentError("model rejected the images: HTTP 422")

    outcome, ch = _deliver(ModelWorker(settings, "cnv", rejects), _study(settings, tmp_path))
    assert outcome.action == "dead"
    assert ch.published[0].exchange == "unit.dlx" and ch.published[0].headers["x-attempts"] == 1


def test_missing_image_is_dead_lettered(settings, tmp_path):
    outcome, ch = _deliver(ModelWorker(settings, "cnv", lambda *_: {}), _study(settings, tmp_path, left=None))
    assert outcome.action == "dead"
    assert ch.published[0].headers["x-reason"] == "image missing from the store: s1/left.jpg"


def test_image_path_cannot_escape_the_store(settings, tmp_path):
    body = {**_study(settings, tmp_path), "left_image": "../../etc/passwd"}
    outcome, ch = _deliver(ModelWorker(settings, "cnv", lambda *_: {}), body)
    assert outcome.action == "dead" and "escapes the image store" in ch.published[0].headers["x-reason"]


def test_malformed_message_is_rejected_to_the_dead_letter_queue(settings):
    channel = FakeChannel()
    ModelWorker(settings, "cnv", lambda *_: {}).handle(
        channel, SimpleNamespace(delivery_tag=3), SimpleNamespace(headers={}, message_id=None), b"not json")
    assert channel.rejected == [(3, False)] and channel.published == []


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        code = int(self.path.split("?")[0].strip("/"))
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b'{"grade": "Mild"}' if code == 200 else b"error"
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_http_status_codes_are_classified(settings):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    study = {"referral_id": "R1", "left_image": "s1/left.jpg", "right_image": "s1/right.jpg"}
    try:
        for code, expected in [(200, None), (503, TransientError), (500, TransientError),
                               (429, TransientError), (422, PermanentError), (400, PermanentError)]:
            s = Settings(conditions=("rvo",), model_urls={"rvo": f"{base}/{code}"})
            call = http_model_call(s)
            if expected is None:
                assert call("rvo", study, b"l", b"r") == {"grade": "Mild"}
            else:
                with pytest.raises(expected):
                    call("rvo", study, b"l", b"r")
        down = http_model_call(Settings(conditions=("rvo",), model_urls={"rvo": "http://127.0.0.1:1/x"}))
        with pytest.raises(TransientError):
            down("rvo", study, b"l", b"r")
    finally:
        server.shutdown()


# Intake --------------------------------------------------------------------------------

class FakePublisher:
    def __init__(self, fail=False):
        self.fail, self.messages = fail, []

    def publish(self, exchange, routing_key, body, **kwargs):
        if self.fail:
            raise PublishError("RabbitMQ unavailable")
        self.messages.append((exchange, body))

    def close(self):
        pass


def _files():
    return {"left_eye_file": ("l.jpg", b"left", "image/jpeg"), "right_eye_file": ("r.png", b"right", "image/png")}


def test_intake_stores_images_creates_the_study_and_publishes_once(settings, store):
    publisher = FakePublisher()
    intake.configure(settings, store, publisher)
    resp = TestClient(intake.app).post("/studies", data={"referral_id": "R1"}, files=_files())
    assert resp.status_code == 202
    study_id = resp.json()["study_id"]
    [(exchange, message)] = publisher.messages
    assert exchange == "unit.studies"
    assert message["left_image"] == f"{study_id}/left.jpg" and message["right_image"] == f"{study_id}/right.png"
    assert (Path(settings.image_store_dir) / study_id / "left.jpg").read_bytes() == b"left"

    status = TestClient(intake.app).get(f"/studies/{study_id}").json()
    assert status["status"] == "PENDING" and [t["status"] for t in status["tests"]] == [PENDING] * 3


def test_intake_returns_503_and_keeps_nothing_when_the_broker_is_down(settings, store):
    intake.configure(settings, store, FakePublisher(fail=True))
    resp = TestClient(intake.app).post("/studies", data={"referral_id": "R1"}, files=_files())
    assert resp.status_code == 503
    assert list(Path(settings.image_store_dir).glob("*")) == []
    with store.engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM fundus_studies").scalar() == 0


@pytest.mark.parametrize("files,code", [
    ({"left_eye_file": ("l.gif", b"x", "image/gif"), "right_eye_file": ("r.jpg", b"x", "image/jpeg")}, 400),
    ({"left_eye_file": ("l.jpg", b"", "image/jpeg"), "right_eye_file": ("r.jpg", b"x", "image/jpeg")}, 400),
])
def test_intake_rejects_bad_uploads(settings, store, files, code):
    publisher = FakePublisher()
    intake.configure(settings, store, publisher)
    assert TestClient(intake.app).post("/studies", data={"referral_id": "R1"}, files=files).status_code == code
    assert publisher.messages == []


def test_unknown_study_is_404(settings, store):
    intake.configure(settings, store, FakePublisher())
    assert TestClient(intake.app).get("/studies/nope").status_code == 404
