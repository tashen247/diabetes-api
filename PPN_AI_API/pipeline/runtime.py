"""Process wiring: store, broker, clock, and the loops for each role."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

from pipeline.broker import InProcessBroker, UnavailableBroker, make_broker
from pipeline.catalog import TestSpec
from pipeline.clock import SystemClock
from pipeline.errors import BrokerUnavailable
from pipeline.flow import (
    _finish_result,
    _finish_work,
    recover_unpublished,
    retry_crm,
    scan_once,
)
from pipeline.runner import HttpModelRunner
from pipeline.store import StudyStore
from pipeline.topology import build_topology

log = logging.getLogger("screening")

_SINGLETON = None
_SINGLETON_LOCK = threading.Lock()
_EMBEDDED = False
_STOP = threading.Event()


@dataclass
class Deps:
    store: StudyStore
    broker: object
    runner: object
    clock: object
    crm: object
    sla_seconds: float = 3600.0


class RequestsCrm:
    def post(self, url: str, report: dict) -> None:
        import requests

        response = requests.post(url, json=report, timeout=float(os.getenv("CRM_TIMEOUT_SECONDS", "10")))
        response.raise_for_status()


def build_deps_from_env() -> Deps:
    store = StudyStore(
        os.getenv("STUDIES_DB", os.path.join("data", "studies.db")),
        os.getenv("IMAGE_STORE", os.path.join("data", "images")),
    )
    kind = os.getenv("SCREENING_BROKER", "rabbitmq").strip().lower()
    topology = build_topology()
    url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5673/")
    try:
        broker = make_broker(kind, topology, url)
    except Exception as exc:
        log.error("screening broker is unavailable: %s", exc)
        broker = UnavailableBroker(str(exc))
        broker.topology = topology
    return Deps(
        store=store,
        broker=broker,
        runner=HttpModelRunner(),
        clock=SystemClock(),
        crm=RequestsCrm(),
        sla_seconds=float(os.getenv("TEST_TIMEOUT_SECONDS", "3600")),
    )


def get_singleton() -> Deps:
    global _SINGLETON
    if _SINGLETON is not None:
        return _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = build_deps_from_env()
        return _SINGLETON


def should_embed(deps: Deps) -> bool:
    flag = os.getenv("SCREENING_EMBED_WORKERS")
    if flag is not None:
        return flag == "1"
    return isinstance(deps.broker, InProcessBroker)


def on_startup(app=None) -> Deps:
    deps = getattr(getattr(app, "state", None), "screening", None)
    if deps is None:
        deps = get_singleton()
        if app is not None:
            app.state.screening = deps
    try:
        deps.store.reset_stuck_crm()
        recover_unpublished(deps)
        retry_crm(deps)
    except BrokerUnavailable:
        log.warning("screening recovery skipped; broker is down and the stored studies are still pending")
    except Exception:
        log.exception("screening recovery failed")
    if should_embed(deps):
        start_embedded(deps)
    return deps


def start_embedded(deps: Deps) -> None:
    global _EMBEDDED
    if _EMBEDDED:
        return
    _EMBEDDED = True
    from pipeline.catalog import TESTS

    for test in TESTS:
        threading.Thread(
            target=run_worker,
            args=(test, deps),
            kwargs={"stop_event": _STOP},
            name=f"screening-{test.name}",
            daemon=True,
        ).start()
    threading.Thread(
        target=run_aggregator,
        args=(deps,),
        kwargs={"stop_event": _STOP},
        name="screening-aggregator",
        daemon=True,
    ).start()
    threading.Thread(
        target=run_watcher,
        args=(deps,),
        kwargs={"stop_event": _STOP},
        name="screening-watcher",
        daemon=True,
    ).start()


def run_worker(test: TestSpec, deps: Deps, stop_event: threading.Event | None = None, idle_sleep: float = 0.5) -> None:
    broker = deps.broker
    while stop_event is None or not stop_event.is_set():
        now = deps.clock.now()
        try:
            envelope = broker.pull(test.queue, now)
        except BrokerUnavailable:
            broker = _reconnect(deps) or broker
            _wait(stop_event, 5)
            continue
        if envelope is None:
            _wait(stop_event, idle_sleep)
            continue
        try:
            _finish_work(deps, test, envelope, now)
        except Exception:
            broker = _reconnect(deps) or broker
            _wait(stop_event, idle_sleep)


def run_aggregator(deps: Deps, stop_event: threading.Event | None = None, idle_sleep: float = 0.5) -> None:
    from pipeline.topology import RESULTS_QUEUE

    broker = deps.broker
    while stop_event is None or not stop_event.is_set():
        now = deps.clock.now()
        try:
            envelope = broker.pull(RESULTS_QUEUE, now)
        except BrokerUnavailable:
            broker = _reconnect(deps) or broker
            _wait(stop_event, 5)
            continue
        if envelope is None:
            _wait(stop_event, idle_sleep)
            continue
        try:
            _finish_result(deps, envelope)
        except Exception:
            broker = _reconnect(deps) or broker


def run_watcher(deps: Deps, stop_event: threading.Event | None = None, interval: float | None = None) -> None:
    interval = interval if interval is not None else float(os.getenv("WATCH_INTERVAL_SECONDS", "30"))
    while stop_event is None or not stop_event.is_set():
        try:
            scan_once(deps)
        except Exception:
            log.exception("timeout watcher failed")
        _wait(stop_event, interval)


def _reconnect(deps: Deps):
    if isinstance(deps.broker, InProcessBroker):
        return deps.broker
    try:
        kind = os.getenv("SCREENING_BROKER", "rabbitmq").strip().lower()
        broker = make_broker(kind, build_topology(), os.getenv("RABBITMQ_URL"))
        deps.broker = broker
        return broker
    except Exception:
        log.exception("broker reconnect failed")
        return None


def _wait(stop_event: threading.Event | None, seconds: float) -> None:
    if stop_event is None:
        time.sleep(seconds)
    else:
        stop_event.wait(seconds)
