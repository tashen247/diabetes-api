"""Operations tool for the dead-letter queue: inspect, replay after a fix, or discard.

    python -m messaging.ops list
    python -m messaging.ops replay --all
    python -m messaging.ops replay --study <study_id> [--condition rvo]
    python -m messaging.ops discard --study <study_id>

Replay sends a message back to its own model's queue only, with the attempt count reset, so
a fixed study isn't re-run through the other seven models. Messages that don't match the
filter are put back on the dead-letter queue untouched.
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Callable

from . import broker
from .settings import Settings
from .store import StudyStore

log = logging.getLogger(__name__)


def _condition_of(headers: dict[str, Any]) -> str | None:
    if headers.get("x-condition"):
        return str(headers["x-condition"])
    # A message rejected by a queue's DLX carries x-death instead, naming the queue it came from.
    for death in headers.get("x-death") or []:
        queue = death.get("queue", b"")
        queue = queue.decode() if isinstance(queue, bytes) else str(queue)
        if ".q." in queue:
            return queue.rsplit(".q.", 1)[1]
    return None


def _describe(body: bytes, headers: dict[str, Any]) -> dict[str, Any]:
    try:
        msg = json.loads(body)
    except ValueError:
        msg = {}
    return {
        "study_id": headers.get("x-study-id") or msg.get("study_id"),
        "referral_id": msg.get("referral_id"),
        "condition": _condition_of(headers),
        "reason": headers.get("x-reason") or "rejected by consumer (see x-death)",
        "attempts": headers.get("x-attempts"),
        "dead_lettered_at": headers.get("x-dead-lettered-at"),
    }


def _walk(s: Settings, decide: Callable[[dict[str, Any]], str | None], store: StudyStore | None = None) -> list[dict]:
    """Takes every dead letter once; decide() returns 'replay', 'discard', or None to keep it."""
    connection = broker.connect(s, "fundus-ops")
    handled: list[dict] = []
    try:
        channel = broker.open_channel(connection, s)
        pending = channel.queue_declare(s.dead_letter_queue, passive=True).method.message_count
        # Skipped messages stay unacked until the pass is over. Returned straight away, RabbitMQ
        # puts them back at the head of the queue and the next basic_get would fetch the same one.
        skipped: list[int] = []
        for _ in range(pending):
            method, props, body = channel.basic_get(s.dead_letter_queue, auto_ack=False)
            if method is None:
                break
            headers = dict(props.headers or {})
            info = _describe(body, headers)
            action = decide(info)
            if action == "replay" and info["condition"] in s.conditions:
                clean = {k: v for k, v in headers.items()
                         if k not in ("x-death", "x-reason", "x-attempts", "x-condition",
                                      "x-dead-lettered-at", "x-attempt", "x-last-error", "x-study-id")}
                broker.publish_json(channel, "", s.model_queue(info["condition"]), json.loads(body),
                                    headers={**clean, "x-replayed": True}, message_id=props.message_id)
                if store is not None and info["study_id"]:
                    store.mark_replayed(info["study_id"], info["condition"])
                channel.basic_ack(method.delivery_tag)
                handled.append({**info, "action": "replayed"})
            elif action == "discard":
                channel.basic_ack(method.delivery_tag)
                handled.append({**info, "action": "discarded"})
            else:
                skipped.append(method.delivery_tag)
                handled.append({**info, "action": "kept"})
        for tag in skipped:
            channel.basic_nack(tag, requeue=True)
        return handled
    finally:
        if connection.is_open:
            connection.close()


def list_dead_letters(s: Settings) -> list[dict]:
    return _walk(s, lambda info: None)


def replay(s: Settings, study_id: str | None = None, condition: str | None = None,
           store: StudyStore | None = None) -> list[dict]:
    def decide(info: dict) -> str | None:
        if study_id and info["study_id"] != study_id:
            return None
        if condition and info["condition"] != condition:
            return None
        return "replay"
    return [h for h in _walk(s, decide, store) if h["action"] == "replayed"]


def discard(s: Settings, study_id: str) -> list[dict]:
    return [h for h in _walk(s, lambda i: "discard" if i["study_id"] == study_id else None)
            if h["action"] == "discarded"]


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Inspect and replay dead-lettered screening messages.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    rp = sub.add_parser("replay")
    group = rp.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--study")
    rp.add_argument("--condition")
    dp = sub.add_parser("discard")
    dp.add_argument("--study", required=True)
    args = parser.parse_args()

    s = Settings.from_env()
    if args.command == "list":
        rows = list_dead_letters(s)
    elif args.command == "replay":
        rows = replay(s, study_id=None if args.all else args.study, condition=args.condition,
                      store=StudyStore(s.db_url))
    else:
        rows = discard(s, args.study)
    print(json.dumps(rows, indent=2, default=str))
    print(f"{len(rows)} message(s)")


if __name__ == "__main__":
    main()
