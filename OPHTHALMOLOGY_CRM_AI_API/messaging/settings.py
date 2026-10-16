"""Configuration for the RabbitMQ screening pipeline, read from environment variables.

Nothing here imports config.py, so the pipeline runs without the model weights or the
CRM database connection string.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# The eight screening models, and the existing HTTP endpoint that runs each one.
# These are the same endpoints /predict-all used to call synchronously.
DEFAULT_MODEL_URLS = {
    "diabetes": "http://localhost:8000/predict-diabetes",
    "hypertension": "http://localhost:8000/predict-hypertension",
    "macular_degeneration": "http://localhost:8000/predict-macular-degeneration",
    "cnv": "http://localhost:8000/predict-cnv",
    "glaucoma": "http://localhost:8000/predict-glaucoma",
    "rvo": "http://localhost:8000/predict-rvo",
    "myopia": "http://localhost:8000/predict-myopia",
    "pathological": "http://localhost:8000/predict-pathological",
}


def _int_list(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/%2F"
    # Every exchange and queue name starts with this, so test runs can be isolated.
    prefix: str = "fundus"
    conditions: tuple[str, ...] = tuple(DEFAULT_MODEL_URLS)
    model_urls: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODEL_URLS))
    # Wait before each retry of a temporary failure; one retry per entry.
    retry_delays_ms: tuple[int, ...] = (10_000, 60_000, 600_000)
    model_timeout_s: float = 120.0
    image_store_dir: str = "image_store"
    db_url: str = "sqlite:///studies.db"
    # A study still not FINAL after this long raises an alert.
    report_timeout_s: int = 3600
    watcher_interval_s: int = 60
    bearer_token: str = ""

    # Exchange and queue names -------------------------------------------------
    @property
    def studies_exchange(self) -> str:
        return f"{self.prefix}.studies"

    def model_queue(self, condition: str) -> str:
        return f"{self.prefix}.q.{condition}"

    @property
    def retry_exchange(self) -> str:
        return f"{self.prefix}.retry"

    def retry_queue(self, condition: str, delay_ms: int) -> str:
        # The delay is part of the name: a queue's TTL can't be changed once declared, so a new
        # delay gets a new queue instead of failing with PRECONDITION_FAILED.
        return f"{self.prefix}.retry.{condition}.{delay_ms}ms"

    @staticmethod
    def retry_routing_key(condition: str, delay_ms: int) -> str:
        return f"{condition}.{delay_ms}"

    @property
    def dead_letter_exchange(self) -> str:
        return f"{self.prefix}.dlx"

    @property
    def dead_letter_queue(self) -> str:
        return f"{self.prefix}.dead"

    DEAD_LETTER_ROUTING_KEY = "dead"

    @property
    def results_exchange(self) -> str:
        return f"{self.prefix}.results"

    @property
    def results_queue(self) -> str:
        return f"{self.prefix}.results"

    @property
    def reports_exchange(self) -> str:
        return f"{self.prefix}.reports"

    @property
    def reports_queue(self) -> str:
        """Where FINAL reports wait for the CRM to collect them."""
        return f"{self.prefix}.reports.crm"

    @property
    def alerts_exchange(self) -> str:
        return f"{self.prefix}.alerts"

    @property
    def alerts_queue(self) -> str:
        return f"{self.prefix}.alerts.ops"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        conditions = tuple(
            c.strip() for c in env.get("FUNDUS_CONDITIONS", ",".join(DEFAULT_MODEL_URLS)).split(",") if c.strip()
        )
        model_urls = {
            c: env.get(f"FUNDUS_MODEL_URL_{c.upper()}", DEFAULT_MODEL_URLS.get(c, f"http://localhost:8000/predict-{c}"))
            for c in conditions
        }
        return cls(
            rabbitmq_url=env.get("RABBITMQ_URL", cls.rabbitmq_url),
            prefix=env.get("FUNDUS_MQ_PREFIX", cls.prefix),
            conditions=conditions,
            model_urls=model_urls,
            retry_delays_ms=tuple(_int_list(env.get("FUNDUS_RETRY_DELAYS_MS", "10000,60000,600000"))),
            model_timeout_s=float(env.get("FUNDUS_MODEL_TIMEOUT_S", cls.model_timeout_s)),
            image_store_dir=env.get("FUNDUS_IMAGE_STORE_DIR", cls.image_store_dir),
            db_url=env.get("FUNDUS_STUDY_DB_URL", cls.db_url),
            report_timeout_s=int(env.get("FUNDUS_REPORT_TIMEOUT_S", cls.report_timeout_s)),
            watcher_interval_s=int(env.get("FUNDUS_WATCHER_INTERVAL_S", cls.watcher_interval_s)),
            bearer_token=env.get("AI_BEARER_TOKEN", "").strip(),
        )
