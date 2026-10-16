"""Consume one model queue. A failure is retried or dead-lettered, never dropped."""

from __future__ import annotations

import argparse
import logging

from pipeline.catalog import require_test
from pipeline.runtime import get_singleton, run_worker


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Run one fundus model worker")
    parser.add_argument("test", help="Test name, for example diabetic-retinopathy or model-6")
    args = parser.parse_args(argv)
    test = require_test(args.test)
    run_worker(test, get_singleton())


if __name__ == "__main__":
    main()
