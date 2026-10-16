"""python -m pipeline  —  how to run the screening processes."""

from pipeline.catalog import TESTS

print("Fundus screening: one study, eight models.")
print("A model being down delays its result. The report stays PENDING until all 8 are in.")
print()
print("  docker compose up -d")
print("  python -m pipeline.asgi            # or: uvicorn pipeline.asgi:app")
for test in TESTS:
    print(f"  python -m pipeline.worker {test.name}")
print("  python -m pipeline.aggregator")
print("  python -m pipeline.timeout_watcher")
print()
print("Intake: POST /intake/studies")
print("Report: GET  /intake/studies/{study_id}/report")
