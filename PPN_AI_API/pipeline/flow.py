"""How a study moves from intake to a final report.

The worker decides retry, dead-letter, or result. The aggregator is the only
place that turns those events into test rows. FINAL is withheld until every
test is done.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from pipeline.broker import Envelope, attempt_of, model_of
from pipeline.catalog import TESTS, TestSpec, require_test
from pipeline.errors import BadImageError, MissingImage, TransientModelError
from pipeline.store import StudyStore, image_problem
from pipeline.topology import DEAD_QUEUE, RESULTS_QUEUE, build_topology

log = logging.getLogger("screening")

_TOPOLOGY = build_topology()


@dataclass(frozen=True)
class Publication:
    queue: str
    body: dict
    headers: dict


def accept_study(deps, referral_id: str, images: list[tuple[str, str, bytes]], crm_callback_url: str | None) -> dict:
    now = deps.clock.now()
    study_id, image_ids = deps.store.create_study(referral_id, images, crm_callback_url, now)
    body = {"study_id": study_id, "referral_id": referral_id, "image_ids": image_ids}
    queued = False
    try:
        deps.broker.publish_fanout(body, {"attempt": 0, "study_id": study_id}, now)
        deps.store.mark_published(study_id, now)
        queued = True
    except Exception as exc:
        log.warning("study %s stored but not published: %s", study_id, exc)
        deps.store.add_alert(study_id, None, "publish_failed", str(exc), now)
    report = deps.store.report(study_id)
    report["queued"] = queued
    return report


def recover_unpublished(deps) -> int:
    published = 0
    now = deps.clock.now()
    for study_id in deps.store.unpublished_ids():
        body = deps.store.work_body(study_id)
        try:
            deps.broker.publish_fanout(body, {"attempt": 0, "study_id": study_id}, now)
            deps.store.mark_published(study_id, now)
            published += 1
        except Exception as exc:
            log.warning("republish failed for %s: %s", study_id, exc)
            deps.store.add_alert(study_id, None, "publish_failed", str(exc), now)
    return published


def handle_delivery(test: TestSpec, envelope: Envelope, store: StudyStore, runner) -> list[Publication]:
    """Turn one queue delivery into the next messages. Does not drop the work."""
    body = envelope.body if isinstance(envelope.body, dict) else {}
    attempt = attempt_of(envelope.headers)
    if "study_id" not in body or "image_ids" not in body:
        return [
            Publication(
                DEAD_QUEUE,
                body or {"raw": str(envelope.body)},
                _headers(test, "", attempt, "unusable_message"),
            )
        ]

    study_id = body["study_id"]
    try:
        store.work_body(study_id)
    except KeyError:
        return _dead(test, body, attempt, "unknown_study", "study is not in the studies database", include_event=False)

    try:
        images = store.load_images(study_id, list(body["image_ids"]))
    except MissingImage as exc:
        return _dead(test, body, attempt, "unreadable_image", str(exc))
    except KeyError as exc:
        return _dead(test, body, attempt, "unreadable_image", f"missing image {exc}")

    for image in images:
        problem = image_problem(image.data)
        if problem:
            return _dead(test, body, attempt, "unreadable_image", problem)

    try:
        result = runner.run(test, images, body)
        if not isinstance(result, dict):
            result = {"value": result}
        json.dumps(result)
    except BadImageError as exc:
        return _dead(test, body, attempt, exc.reason, str(exc))
    except TransientModelError as exc:
        return _retry_or_dead(test, body, attempt, exc.reason, str(exc))
    except Exception as exc:
        log.exception("model %s failed on study %s", test.name, study_id)
        return _retry_or_dead(test, body, attempt, "model_down", str(exc))

    return [
        Publication(
            RESULTS_QUEUE,
            {
                "kind": "result",
                "study_id": study_id,
                "test": test.name,
                "attempt": attempt,
                "result": result,
            },
            _headers(test, study_id, attempt),
        )
    ]


def replay_dead_letter(deps, study_id: str, test_name: str) -> dict:
    test = require_test(test_name)
    now = deps.clock.now()

    def matches(envelope: Envelope) -> bool:
        body = envelope.body if isinstance(envelope.body, dict) else {}
        return body.get("study_id") == study_id and model_of(envelope.headers) == test.name

    envelope = deps.broker.pop_match(DEAD_QUEUE, matches, now)
    if envelope is None:
        row = deps.store.next_dead(study_id, test.name)
        work = (row or {}).get("work") or {}
        if not work.get("study_id"):
            raise KeyError(test_name)
        envelope = Envelope(
            message_id=f"dead-{row['dead_id']}",
            queue=DEAD_QUEUE,
            body=work,
            headers={
                "attempt": row["attempt"],
                "model": test.name,
                "reason": row["reason"],
                "study_id": study_id,
            },
        )

    work_headers = _headers(test, study_id, 0)
    work_headers["replayed"] = 1
    event = {"kind": "replayed", "study_id": study_id, "test": test.name, "attempt": 0}
    try:
        deps.broker.publish(test.queue, envelope.body, work_headers, now)
        deps.broker.publish(RESULTS_QUEUE, event, _headers(test, study_id, 0), now)
        deps.store.apply(event, now)
        deps.store.mark_dead_replayed(study_id, test.name, now)
    except Exception:
        deps.broker.publish(DEAD_QUEUE, envelope.body, envelope.headers, now)
        raise
    return {"replayed": True, "study_id": study_id, "test": test.name, "status": "PENDING", "complete": False}


def apply_result_message(deps, body: dict) -> dict:
    if not isinstance(body, dict) or "kind" not in body or "study_id" not in body:
        raise ValueError("not a result event")
    became_final = deps.store.apply(body, deps.clock.now())
    if became_final:
        deliver_final(deps, body["study_id"])
    return deps.store.report(body["study_id"])


def deliver_final(deps, study_id: str) -> None:
    report = deps.store.report(study_id)
    if not report["complete"]:
        return
    if report["crm_status"] in {"sent", "skipped"}:
        return
    url = report.get("crm_callback_url")
    if not url:
        deps.store.set_crm(study_id, "skipped")
        return
    if not deps.store.claim_crm(study_id):
        return
    outbound = dict(report)
    outbound["released_to_crm"] = True
    outbound["crm_status"] = "sent"
    try:
        deps.crm.post(url, outbound)
    except Exception as exc:
        log.warning("CRM delivery failed for %s: %s", study_id, exc)
        deps.store.set_crm(study_id, "failed")
        deps.store.add_alert(study_id, None, "crm_delivery_failed", str(exc), deps.clock.now())
        return
    deps.store.set_crm(study_id, "sent")
    log.info("final report for study %s released to CRM", study_id)


def retry_crm(deps) -> None:
    for study_id in deps.store.studies_needing_crm():
        deliver_final(deps, study_id)


def scan_once(deps) -> list[dict]:
    alerts = deps.store.mark_overdue(deps.clock.now(), deps.sla_seconds)
    retry_crm(deps)
    return alerts


def drain(deps, max_cycles: int = 1000) -> int:
    """Run workers and the aggregator until every due message has been handled."""
    processed = 0
    for _ in range(max_cycles):
        now = deps.clock.now()
        deps.broker.release_due(now)
        progressed = False
        for test in TESTS:
            envelope = deps.broker.pull(test.queue, now)
            if envelope is None:
                continue
            progressed = True
            processed += 1
            _finish_work(deps, test, envelope, now)
        while True:
            envelope = deps.broker.pull(RESULTS_QUEUE, now)
            if envelope is None:
                break
            progressed = True
            processed += 1
            _finish_result(deps, envelope)
        if not progressed:
            break
    return processed


def queue_snapshot(broker) -> list[dict]:
    topology = broker.topology
    always = {test.queue for test in topology.tests} | {topology.results_queue, topology.dead_queue}
    names = [test.queue for test in topology.tests]
    names.append(topology.results_queue)
    names.append(topology.dead_queue)
    names.extend(item.name for item in topology.retry_queues)
    rows = []
    for name in names:
        depth = broker.depth(name)
        if depth or name in always:
            rows.append({"name": name, "depth": depth})
    return rows


def _finish_work(deps, test: TestSpec, envelope: Envelope, now) -> None:
    try:
        publications = handle_delivery(test, envelope, deps.store, deps.runner)
        for publication in publications:
            deps.broker.publish(publication.queue, publication.body, publication.headers, now)
        deps.broker.ack(envelope)
    except Exception:
        log.exception("worker failed for %s; message returned to the queue", test.name)
        deps.broker.nack(envelope, requeue=True)
        raise


def _finish_result(deps, envelope: Envelope) -> None:
    try:
        apply_result_message(deps, envelope.body)
        deps.broker.ack(envelope)
    except (ValueError, KeyError) as exc:
        log.warning("unusable result moved to the dead-letter queue: %s", exc)
        deps.broker.publish(
            DEAD_QUEUE,
            envelope.body if isinstance(envelope.body, dict) else {"raw": str(envelope.body)},
            {"reason": "unusable_message", "model": "aggregator"},
            deps.clock.now(),
        )
        deps.broker.ack(envelope)
    except Exception:
        log.exception("aggregator failed; result returned to the queue")
        deps.broker.nack(envelope, requeue=True)
        raise


def _retry_or_dead(test: TestSpec, body: dict, attempt: int, reason: str, message: str) -> list[Publication]:
    retry = _TOPOLOGY.retry_queue_for(test, attempt)
    study_id = body.get("study_id") or ""
    if retry is None:
        return _dead(test, body, attempt, "retries_exhausted", message)
    next_attempt = attempt + 1
    headers = _headers(test, study_id, next_attempt, reason)
    return [
        Publication(retry.name, body, headers),
        Publication(
            RESULTS_QUEUE,
            {
                "kind": "retrying",
                "study_id": study_id,
                "test": test.name,
                "attempt": next_attempt,
                "reason": reason,
                "message": message,
                "delay_seconds": retry.ttl_seconds,
            },
            headers,
        ),
    ]


def _dead(
    test: TestSpec,
    body: dict,
    attempt: int,
    reason: str,
    message: str,
    include_event: bool = True,
) -> list[Publication]:
    study_id = body.get("study_id") or ""
    headers = _headers(test, study_id, attempt, reason)
    dead = Publication(DEAD_QUEUE, body, headers)
    if not include_event or not study_id:
        return [dead]
    event = Publication(
        RESULTS_QUEUE,
        {
            "kind": "dead",
            "study_id": study_id,
            "test": test.name,
            "attempt": attempt,
            "reason": reason,
            "message": message,
            "work": {
                "study_id": study_id,
                "referral_id": body.get("referral_id"),
                "image_ids": body.get("image_ids") or [],
            },
        },
        headers,
    )
    return [dead, event]


def _headers(test: TestSpec, study_id: str, attempt: int, reason: str | None = None) -> dict:
    headers = {"attempt": attempt, "model": test.name, "study_id": study_id}
    if reason:
        headers["reason"] = reason
    return headers
