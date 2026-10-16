"""Standalone intake process. Workers stay separate when the broker is RabbitMQ."""

from __future__ import annotations

import logging

from pipeline.api import create_app
from pipeline.runtime import on_startup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = create_app()


@app.on_event("startup")
def _startup() -> None:
    on_startup(app)
