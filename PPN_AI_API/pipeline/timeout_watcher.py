"""Alert when a test is still unfinished after the SLA. Do not complete the report."""

from __future__ import annotations

import logging

from pipeline.runtime import get_singleton, run_watcher


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run_watcher(get_singleton())


if __name__ == "__main__":
    main()
