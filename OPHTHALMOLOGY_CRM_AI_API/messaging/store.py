"""Study and per-test status, so a report is only FINAL when every model has answered.

Works on SQLite (local runs and tests) and SQL Server (set FUNDUS_STUDY_DB_URL to an
mssql+pyodbc URL). Tables are created on first use.

Study status:
    PENDING          waiting for one or more results; never shown as complete
    NEEDS_ATTENTION  at least one test was dead-lettered (for example, an unreadable image)
    FINAL            every test has a result; the report can go back to the CRM
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import (Column, DateTime, Integer, MetaData, String, Table, Text, create_engine, func,
                        select, update)
from sqlalchemy.engine import Engine

PENDING, DONE, FAILED = "PENDING", "DONE", "FAILED"
STUDY_PENDING, STUDY_NEEDS_ATTENTION, STUDY_FINAL = "PENDING", "NEEDS_ATTENTION", "FINAL"

metadata = MetaData()

studies = Table(
    "fundus_studies", metadata,
    Column("study_id", String(36), primary_key=True),
    Column("referral_id", String(64), nullable=False, index=True),
    Column("status", String(20), nullable=False),
    Column("left_image", String(400), nullable=False),
    Column("right_image", String(400), nullable=False),
    Column("expected_tests", Integer, nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("finalised_at", DateTime),
    Column("reported_at", DateTime),   # when the FINAL report was confirmed sent to the CRM
    Column("alerted_at", DateTime),
)

study_tests = Table(
    "fundus_study_tests", metadata,
    # The (study_id, condition) key is what makes recording a result idempotent.
    Column("study_id", String(36), primary_key=True),
    Column("condition", String(40), primary_key=True),
    Column("status", String(20), nullable=False),
    Column("attempts", Integer, nullable=False, default=0),
    Column("result_json", Text),
    Column("failure_reason", String(400)),
    Column("completed_at", DateTime),
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True)
class RecordOutcome:
    applied: bool          # False when this result was already recorded (a redelivery)
    study_status: str
    became_final: bool     # True only for the one result that completed the study


class StudyStore:
    def __init__(self, db_url: str):
        kwargs: dict[str, Any] = {"future": True}
        if db_url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self.engine: Engine = create_engine(db_url, **kwargs)
        metadata.create_all(self.engine)

    def create_study(self, study_id: str, referral_id: str, left_image: str, right_image: str,
                     conditions: list[str] | tuple[str, ...]) -> None:
        now = utcnow()
        with self.engine.begin() as conn:
            conn.execute(studies.insert().values(
                study_id=study_id, referral_id=referral_id, status=STUDY_PENDING,
                left_image=left_image, right_image=right_image,
                expected_tests=len(conditions), created_at=now))
            conn.execute(study_tests.insert(), [
                {"study_id": study_id, "condition": c, "status": PENDING, "attempts": 0} for c in conditions
            ])

    def delete_study(self, study_id: str) -> None:
        """Used by intake when the publish fails, so no study is left waiting for nothing."""
        with self.engine.begin() as conn:
            conn.execute(study_tests.delete().where(study_tests.c.study_id == study_id))
            conn.execute(studies.delete().where(studies.c.study_id == study_id))

    def record_result(self, study_id: str, condition: str, ok: bool, result: Any = None,
                      reason: str | None = None, attempts: int = 0) -> RecordOutcome:
        """Records one model's outcome. Redelivered or duplicate results change nothing."""
        now = utcnow()
        with self.engine.begin() as conn:
            key = (study_tests.c.study_id == study_id) & (study_tests.c.condition == condition)
            if ok:
                # A test that failed earlier can still succeed after a replay, but DONE is final.
                applied = conn.execute(
                    update(study_tests).where(key & (study_tests.c.status != DONE)).values(
                        status=DONE, result_json=json.dumps(result), failure_reason=None,
                        attempts=attempts, completed_at=now)
                ).rowcount == 1
            else:
                applied = conn.execute(
                    update(study_tests).where(key & (study_tests.c.status == PENDING)).values(
                        status=FAILED, failure_reason=(reason or "")[:400], attempts=attempts)
                ).rowcount == 1

            counts = dict(conn.execute(
                select(study_tests.c.status, func.count()).where(study_tests.c.study_id == study_id)
                .group_by(study_tests.c.status)
            ).all())
            expected = conn.execute(
                select(studies.c.expected_tests).where(studies.c.study_id == study_id)
            ).scalar_one_or_none()
            if expected is None:
                return RecordOutcome(applied=False, study_status="UNKNOWN", became_final=False)

            if counts.get(DONE, 0) == expected:
                became_final = conn.execute(
                    update(studies).where((studies.c.study_id == study_id) & (studies.c.status != STUDY_FINAL))
                    .values(status=STUDY_FINAL, finalised_at=now)
                ).rowcount == 1
                return RecordOutcome(applied, STUDY_FINAL, became_final)

            status = STUDY_NEEDS_ATTENTION if counts.get(FAILED, 0) else STUDY_PENDING
            conn.execute(update(studies).where(studies.c.study_id == study_id).values(status=status))
            return RecordOutcome(applied, status, False)

    def mark_replayed(self, study_id: str, condition: str) -> None:
        """A dead-lettered test was sent back to its model after a fix: it's waiting again."""
        with self.engine.begin() as conn:
            conn.execute(update(study_tests).where(
                (study_tests.c.study_id == study_id) & (study_tests.c.condition == condition)
                & (study_tests.c.status == FAILED)).values(status=PENDING, failure_reason=None))
            failed = conn.execute(select(func.count()).select_from(study_tests).where(
                (study_tests.c.study_id == study_id) & (study_tests.c.status == FAILED))).scalar_one()
            if not failed:
                conn.execute(update(studies).where(
                    (studies.c.study_id == study_id) & (studies.c.status == STUDY_NEEDS_ATTENTION))
                    .values(status=STUDY_PENDING))

    def mark_reported(self, study_id: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(studies).where(studies.c.study_id == study_id).values(reported_at=utcnow()))

    def get_study(self, study_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(studies).where(studies.c.study_id == study_id)).mappings().first()
            if row is None:
                return None
            tests = conn.execute(select(study_tests).where(study_tests.c.study_id == study_id)
                                 .order_by(study_tests.c.condition)).mappings().all()
        return {
            **{k: (v.isoformat() + "Z" if isinstance(v, datetime) else v) for k, v in row.items()},
            "tests": [
                {
                    "condition": t["condition"],
                    "status": t["status"],
                    "attempts": t["attempts"],
                    "failure_reason": t["failure_reason"],
                    "result": json.loads(t["result_json"]) if t["result_json"] else None,
                }
                for t in tests
            ],
        }

    def claim_overdue(self, older_than: timedelta) -> list[dict[str, Any]]:
        """Returns studies not FINAL after the timeout, marking each so it's alerted only once."""
        cutoff = utcnow() - older_than
        overdue: list[dict[str, Any]] = []
        with self.engine.begin() as conn:
            rows = conn.execute(select(studies.c.study_id, studies.c.referral_id, studies.c.status,
                                       studies.c.created_at)
                                .where((studies.c.status != STUDY_FINAL) & (studies.c.created_at < cutoff)
                                       & studies.c.alerted_at.is_(None))).mappings().all()
            for r in rows:
                claimed = conn.execute(update(studies).where(
                    (studies.c.study_id == r["study_id"]) & studies.c.alerted_at.is_(None))
                    .values(alerted_at=utcnow())).rowcount == 1
                if not claimed:
                    continue
                missing = conn.execute(select(study_tests.c.condition).where(
                    (study_tests.c.study_id == r["study_id"]) & (study_tests.c.status != DONE))
                    .order_by(study_tests.c.condition)).scalars().all()
                overdue.append({"study_id": r["study_id"], "referral_id": r["referral_id"],
                                "status": r["status"], "created_at": r["created_at"].isoformat() + "Z",
                                "missing_tests": list(missing)})
        return overdue
