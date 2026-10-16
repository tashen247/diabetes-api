"""Studies database and fundus image store.

The rows are the source of truth. A study is FINAL only when all eight tests
are done. Reading a report repairs a column that disagrees with those rows.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pipeline.catalog import EXPECTED_TESTS, TESTS

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}


def guess_eye(filename: str) -> str | None:
    stem = os.path.splitext(os.path.basename(filename or ""))[0].lower()
    tokens = set(re.split(r"[^a-z0-9]+", stem))
    if "left" in tokens or "os" in tokens:
        return "left"
    if "right" in tokens or "od" in tokens:
        return "right"
    if tokens & {"l", "le"}:
        return "left"
    if tokens & {"r", "re"}:
        return "right"
    return None


def image_problem(data: bytes) -> str | None:
    """Return a reason when the bytes are not a fundus image we can hand to a model."""
    if not data or len(data) < 8:
        return "empty or truncated image"
    if data.startswith(b"\xff\xd8"):
        return None
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    if data.startswith((b"GIF87a", b"GIF89a")):
        return None
    if data.startswith(b"BM"):
        return None
    if data.startswith((b"II*\x00", b"MM\x00*")):
        return None
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return None
    return "unreadable image"


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass(frozen=True)
class StoredImage:
    image_id: str
    filename: str
    content_type: str
    eye: str | None
    data: bytes


class StudyStore:
    def __init__(self, db_path: str | Path, image_root: str | Path):
        self.db_path = Path(db_path)
        self.image_root = Path(image_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.image_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        finally:
            conn.close()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS studies (
                    study_id TEXT PRIMARY KEY,
                    referral_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    finalized_at TEXT,
                    crm_callback_url TEXT,
                    crm_status TEXT NOT NULL,
                    crm_attempts INTEGER NOT NULL DEFAULT 0,
                    published_at TEXT
                );
                CREATE TABLE IF NOT EXISTS study_tests (
                    study_id TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    reason TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (study_id, test_name),
                    FOREIGN KEY (study_id) REFERENCES studies(study_id)
                );
                CREATE TABLE IF NOT EXISTS images (
                    image_id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_type TEXT,
                    eye TEXT,
                    path TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    FOREIGN KEY (study_id) REFERENCES studies(study_id)
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id TEXT NOT NULL,
                    test_name TEXT,
                    kind TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dead_letters (
                    dead_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    work_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    replayed_at TEXT
                );
                """
            )
        finally:
            conn.close()

    @contextmanager
    def _tx(self):
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()

    def create_study(
        self,
        referral_id: str,
        images: list[tuple[str, str, bytes]],
        crm_callback_url: str | None,
        now: datetime,
    ) -> tuple[str, list[str]]:
        study_id = uuid.uuid4().hex
        written: list[tuple[str, str, str, str | None, Path, int]] = []
        try:
            for filename, content_type, data in images:
                image_id = uuid.uuid4().hex
                safe_name = os.path.basename(filename or "fundus.jpg") or "fundus.jpg"
                suffix = os.path.splitext(safe_name)[1].lower()
                if suffix not in _IMAGE_SUFFIXES:
                    suffix = ".img"
                path = self.image_root / study_id / f"{image_id}{suffix}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                written.append(
                    (image_id, safe_name, content_type or "application/octet-stream", guess_eye(safe_name), path, len(data))
                )
            stamp = now.isoformat()
            with self._tx() as conn:
                conn.execute(
                    """
                    INSERT INTO studies (
                        study_id, referral_id, status, created_at, crm_callback_url, crm_status, crm_attempts
                    ) VALUES (?, ?, 'PENDING', ?, ?, 'not_ready', 0)
                    """,
                    (study_id, referral_id, stamp, crm_callback_url),
                )
                for test in TESTS:
                    conn.execute(
                        """
                        INSERT INTO study_tests (study_id, test_name, status, attempt, updated_at)
                        VALUES (?, ?, 'pending', 0, ?)
                        """,
                        (study_id, test.name, stamp),
                    )
                for image_id, safe_name, content_type, eye, path, size in written:
                    conn.execute(
                        """
                        INSERT INTO images (image_id, study_id, filename, content_type, eye, path, byte_size)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (image_id, study_id, safe_name, content_type, eye, str(path), size),
                    )
        except Exception:
            for *_, path, _size in written:
                path.unlink(missing_ok=True)
            raise
        return study_id, [item[0] for item in written]

    def mark_published(self, study_id: str, now: datetime) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE studies SET published_at=? WHERE study_id=? AND published_at IS NULL",
                (now.isoformat(), study_id),
            )

    def unpublished_ids(self) -> list[str]:
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT study_id FROM studies WHERE published_at IS NULL ORDER BY created_at"
            ).fetchall()
        return [row["study_id"] for row in rows]

    def work_body(self, study_id: str) -> dict:
        with self._tx() as conn:
            study = conn.execute("SELECT referral_id FROM studies WHERE study_id=?", (study_id,)).fetchone()
            if study is None:
                raise KeyError(study_id)
            images = conn.execute(
                "SELECT image_id FROM images WHERE study_id=? ORDER BY rowid",
                (study_id,),
            ).fetchall()
        return {
            "study_id": study_id,
            "referral_id": study["referral_id"],
            "image_ids": [row["image_id"] for row in images],
        }

    def load_images(self, study_id: str, image_ids: list[str]) -> list[StoredImage]:
        loaded: list[StoredImage] = []
        with self._tx() as conn:
            for image_id in image_ids:
                row = conn.execute("SELECT * FROM images WHERE image_id=?", (image_id,)).fetchone()
                if row is None or row["study_id"] != study_id:
                    raise KeyError(image_id)
                path = Path(row["path"])
                if not path.is_file():
                    from pipeline.errors import MissingImage

                    raise MissingImage(image_id)
                loaded.append(
                    StoredImage(
                        image_id=image_id,
                        filename=row["filename"],
                        content_type=row["content_type"] or "application/octet-stream",
                        eye=row["eye"],
                        data=path.read_bytes(),
                    )
                )
        return loaded

    def apply(self, event: dict, now: datetime) -> bool:
        """Apply one aggregator event. Returns True only when this call makes the study FINAL."""
        kind = event["kind"]
        study_id = event["study_id"]
        test_name = event["test"]
        stamp = now.isoformat()
        with self._tx() as conn:
            study = conn.execute("SELECT study_id FROM studies WHERE study_id=?", (study_id,)).fetchone()
            if study is None:
                raise KeyError(study_id)
            row = conn.execute(
                "SELECT status, reason FROM study_tests WHERE study_id=? AND test_name=?",
                (study_id, test_name),
            ).fetchone()
            if row is None:
                raise KeyError(test_name)
            status = row["status"]
            reason = event.get("reason")
            attempt = int(event.get("attempt") or 0)
            if kind == "result":
                if status != "done":
                    conn.execute(
                        """
                        UPDATE study_tests
                        SET status='done', attempt=?, result_json=?, reason=NULL, updated_at=?
                        WHERE study_id=? AND test_name=? AND status!='done'
                        """,
                        (attempt, json.dumps(event.get("result") or {}), stamp, study_id, test_name),
                    )
            elif kind == "dead":
                if status != "done":
                    conn.execute(
                        """
                        UPDATE study_tests
                        SET status='dead', attempt=?, reason=?, updated_at=?
                        WHERE study_id=? AND test_name=? AND status!='done'
                        """,
                        (attempt, reason, stamp, study_id, test_name),
                    )
                already = status == "dead" and row["reason"] == reason
                if not already:
                    message = str(event.get("message") or reason or "dead letter")[:1000]
                    conn.execute(
                        """
                        INSERT INTO dead_letters (
                            study_id, test_name, reason, attempt, message, work_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            study_id,
                            test_name,
                            reason or "dead_letter",
                            attempt,
                            message,
                            json.dumps(event.get("work") or {}),
                            stamp,
                        ),
                    )
                    self._insert_alert(conn, study_id, test_name, "dead_letter", message, stamp)
            elif kind == "retrying":
                if status not in ("done", "dead"):
                    conn.execute(
                        """
                        UPDATE study_tests
                        SET status='retrying', attempt=?, reason=?, updated_at=?
                        WHERE study_id=? AND test_name=? AND status NOT IN ('done', 'dead')
                        """,
                        (attempt, reason, stamp, study_id, test_name),
                    )
            elif kind == "replayed":
                if status != "done":
                    conn.execute(
                        """
                        UPDATE study_tests
                        SET status='pending', attempt=0, reason=NULL, updated_at=?
                        WHERE study_id=? AND test_name=? AND status!='done'
                        """,
                        (stamp, study_id, test_name),
                    )
                conn.execute(
                    """
                    UPDATE dead_letters SET replayed_at=?
                    WHERE study_id=? AND test_name=? AND replayed_at IS NULL
                    """,
                    (stamp, study_id, test_name),
                )
            else:
                raise ValueError(f"unknown result kind {kind!r}")
            return self._enforce(conn, study_id, stamp)

    def _enforce(self, conn: sqlite3.Connection, study_id: str, stamp: str) -> bool:
        rows = conn.execute(
            "SELECT status FROM study_tests WHERE study_id=?",
            (study_id,),
        ).fetchall()
        complete = len(rows) == EXPECTED_TESTS and all(row["status"] == "done" for row in rows)
        study = conn.execute("SELECT status FROM studies WHERE study_id=?", (study_id,)).fetchone()
        if study is None:
            raise KeyError(study_id)
        if complete and study["status"] != "FINAL":
            updated = conn.execute(
                """
                UPDATE studies
                SET status='FINAL', finalized_at=?, crm_status='pending'
                WHERE study_id=? AND status!='FINAL'
                """,
                (stamp, study_id),
            )
            return updated.rowcount == 1
        if not complete and study["status"] == "FINAL":
            conn.execute(
                """
                UPDATE studies
                SET status='PENDING', finalized_at=NULL
                WHERE study_id=? AND status='FINAL'
                """,
                (study_id,),
            )
        return False

    def report(self, study_id: str) -> dict:
        with self._tx() as conn:
            self._enforce(conn, study_id, datetime.now(timezone.utc).isoformat())
            study = conn.execute("SELECT * FROM studies WHERE study_id=?", (study_id,)).fetchone()
            if study is None:
                raise KeyError(study_id)
            rows = conn.execute(
                "SELECT * FROM study_tests WHERE study_id=?",
                (study_id,),
            ).fetchall()
        return self._report_from(study, rows)

    def _report_from(self, study: sqlite3.Row, rows: list[sqlite3.Row]) -> dict:
        order = {test.name: index for index, test in enumerate(TESTS)}
        titles = {test.name: test for test in TESTS}
        ordered = sorted(rows, key=lambda row: order.get(row["test_name"], 99))
        tests = []
        for row in ordered:
            spec = titles.get(row["test_name"])
            result = json.loads(row["result_json"]) if row["result_json"] else None
            tests.append(
                {
                    "name": row["test_name"],
                    "title": spec.title if spec else row["test_name"],
                    "model_key": spec.model_key if spec else row["test_name"],
                    "queue": spec.queue if spec else None,
                    "status": row["status"],
                    "attempt": row["attempt"],
                    "reason": row["reason"],
                    "result": result,
                }
            )
        done = sum(1 for test in tests if test["status"] == "done")
        complete = len(tests) == EXPECTED_TESTS and done == EXPECTED_TESTS
        needs_attention = any(test["status"] in {"dead", "overdue"} for test in tests)
        return {
            "study_id": study["study_id"],
            "referral_id": study["referral_id"],
            "status": "FINAL" if complete else "PENDING",
            "complete": complete,
            "tests_expected": EXPECTED_TESTS,
            "tests_done": done,
            "needs_attention": needs_attention,
            "published": study["published_at"] is not None,
            "crm_callback_url": study["crm_callback_url"],
            "crm_status": study["crm_status"],
            "released_to_crm": bool(complete and study["crm_status"] == "sent"),
            "created_at": study["created_at"],
            "finalized_at": study["finalized_at"] if complete else None,
            "tests": tests,
        }

    def mark_overdue(self, now: datetime, sla_seconds: float) -> list[dict]:
        created: list[dict] = []
        stamp = now.isoformat()
        with self._tx() as conn:
            rows = conn.execute(
                """
                SELECT t.study_id, t.test_name, t.status, s.created_at
                FROM study_tests t
                JOIN studies s ON s.study_id = t.study_id
                WHERE t.status IN ('pending', 'retrying', 'overdue')
                """
            ).fetchall()
            for row in rows:
                age = (now - _parse_time(row["created_at"])).total_seconds()
                if age < sla_seconds:
                    continue
                if row["status"] != "overdue":
                    conn.execute(
                        """
                        UPDATE study_tests
                        SET status='overdue', updated_at=?
                        WHERE study_id=? AND test_name=? AND status IN ('pending', 'retrying')
                        """,
                        (stamp, row["study_id"], row["test_name"]),
                    )
                exists = conn.execute(
                    """
                    SELECT 1 FROM alerts
                    WHERE study_id=? AND test_name=? AND kind='overdue'
                    """,
                    (row["study_id"], row["test_name"]),
                ).fetchone()
                if exists:
                    continue
                message = f"test overdue after {int(sla_seconds)}s"
                self._insert_alert(conn, row["study_id"], row["test_name"], "overdue", message, stamp)
                created.append(
                    {"study_id": row["study_id"], "test_name": row["test_name"], "kind": "overdue", "message": message}
                )
            # An overdue test is not a finished test. Re-check so a stored FINAL cannot survive.
            study_ids = {row["study_id"] for row in rows}
            for study_id in study_ids:
                self._enforce(conn, study_id, stamp)
        return created

    def add_alert(self, study_id: str, test_name: str | None, kind: str, message: str, now: datetime) -> None:
        with self._tx() as conn:
            self._insert_alert(conn, study_id, test_name, kind, message[:1000], now.isoformat())

    def _insert_alert(self, conn, study_id, test_name, kind, message, stamp) -> None:
        conn.execute(
            """
            INSERT INTO alerts (study_id, test_name, kind, message, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (study_id, test_name, kind, message, stamp),
        )

    def alerts(self, study_id: str | None = None, limit: int = 200) -> list[dict]:
        with self._tx() as conn:
            if study_id is None:
                rows = conn.execute(
                    "SELECT * FROM alerts ORDER BY alert_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM alerts WHERE study_id=? ORDER BY alert_id DESC LIMIT ?",
                    (study_id, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def list_dead(self, replayed: bool = False) -> list[dict]:
        clause = "replayed_at IS NOT NULL" if replayed else "replayed_at IS NULL"
        with self._tx() as conn:
            rows = conn.execute(
                f"SELECT * FROM dead_letters WHERE {clause} ORDER BY dead_id"
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["work"] = json.loads(row["work_json"])
            items.append(item)
        return items

    def next_dead(self, study_id: str, test_name: str) -> dict | None:
        with self._tx() as conn:
            row = conn.execute(
                """
                SELECT * FROM dead_letters
                WHERE study_id=? AND test_name=? AND replayed_at IS NULL
                ORDER BY dead_id LIMIT 1
                """,
                (study_id, test_name),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["work"] = json.loads(row["work_json"])
        return item

    def mark_dead_replayed(self, study_id: str, test_name: str, now: datetime) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE dead_letters SET replayed_at=?
                WHERE study_id=? AND test_name=? AND replayed_at IS NULL
                """,
                (now.isoformat(), study_id, test_name),
            )

    def claim_crm(self, study_id: str) -> bool:
        with self._tx() as conn:
            updated = conn.execute(
                """
                UPDATE studies
                SET crm_status='sending', crm_attempts=crm_attempts+1
                WHERE study_id=? AND crm_status IN ('pending', 'failed')
                """,
                (study_id,),
            )
            return updated.rowcount == 1

    def set_crm(self, study_id: str, status: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE studies SET crm_status=? WHERE study_id=?",
                (status, study_id),
            )

    def studies_needing_crm(self) -> list[str]:
        with self._tx() as conn:
            rows = conn.execute(
                """
                SELECT study_id FROM studies
                WHERE status='FINAL' AND crm_status IN ('pending', 'failed')
                """
            ).fetchall()
        return [row["study_id"] for row in rows]

    def reset_stuck_crm(self) -> None:
        with self._tx() as conn:
            conn.execute("UPDATE studies SET crm_status='failed' WHERE crm_status='sending'")

    def counts(self) -> dict:
        with self._tx() as conn:
            pending = conn.execute("SELECT COUNT(*) AS n FROM studies WHERE status!='FINAL'").fetchone()["n"]
            final = conn.execute("SELECT COUNT(*) AS n FROM studies WHERE status='FINAL'").fetchone()["n"]
        return {"pending_studies": pending, "final_studies": final}

    def force_column_status(self, study_id: str, status: str) -> None:
        """Test hook: write the status column without touching test rows."""
        with self._tx() as conn:
            conn.execute("UPDATE studies SET status=? WHERE study_id=?", (status, study_id))
