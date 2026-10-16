"""Apply model results. The report becomes FINAL only when all eight are in."""

from __future__ import annotations

import logging

from pipeline.runtime import get_singleton, run_aggregator


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run_aggregator(get_singleton())


if __name__ == "__main__":
    main()
