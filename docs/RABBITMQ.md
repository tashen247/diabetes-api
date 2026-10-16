# Queued screening with RabbitMQ

One study in, eight independent model results out, and a report that is only released when all
eight are in.

## Why

`POST /predict-all` called the eight model services in one request and returned whatever came
back. If one service was down, or its server had crashed, the report reached the CRM with
missing results, and the study had to be reprocessed by hand.

With the queue, a model being down only **delays** its result. The study waits in that model's
queue until the service is back, the other seven carry on, and the report stays `PENDING`, never
an incomplete `FINAL`. A failure becomes a late report instead of a wrong one.

## How it works

```mermaid
flowchart LR
    CRM[Ophthalmology CRM] -->|left + right images| IN[Intake API<br/>POST /studies]
    IN -->|save| IMG[(Image store)]
    IN -->|study expects 8 tests| DB[(Study store)]
    IN -->|publish once, confirmed| X{{fundus.studies<br/>fanout}}
    X --> Q1[q.diabetes] & Q2[q.hypertension] & Q3[q.rvo] & Q8[... 8 queues]
    Q1 --> W1[diabetes worker]
    Q3 --> W3[rvo worker]
    W1 & W3 -->|result| R{{fundus.results}} --> AG[Aggregator]
    AG -->|mark test done, idempotent| DB
    AG -->|all 8 in| REP{{fundus.reports}} --> CRM
    W3 -.->|temporary failure| RT[retry queues<br/>10s / 1m / 10m] -.->|TTL expires| Q3
    W3 -.->|bad image or retries used up| DLQ[(fundus.dead)]
    DLQ -.->|inspect, fix, replay| OPS[messaging.ops]
    OPS -.-> Q3
    WT[Timeout watcher] -->|study overdue| AL{{fundus.alerts}}
```

| Piece | Module | What it does |
|---|---|---|
| Intake API | `messaging/intake.py` | Saves both images, creates the study expecting every test, publishes **one** message to the `fundus.studies` fanout exchange and returns `202`. |
| Topology | `messaging/topology.py` | Declares everything: one durable queue per model bound to the fanout, retry queues, results, reports, alerts and the dead-letter queue. |
| Model worker | `messaging/worker.py` | One per model. Loads the images, calls that model's existing HTTP endpoint, and publishes the result. |
| Aggregator | `messaging/aggregator.py` | Records each result per study and test, and publishes the report when the last one arrives. |
| Timeout watcher | `messaging/watcher.py` | Alerts operations, once, about any study still not `FINAL` after the timeout (one hour by default). |
| Operations tool | `messaging/ops.py` | Lists dead letters with their reasons, replays them after a fix, or discards them. |
| Study store | `messaging/store.py` | Study and per-test status. SQLite locally, SQL Server in production. |

### Study status

| Status | Meaning |
|---|---|
| `PENDING` | Waiting for one or more results. Never shown as complete. |
| `NEEDS_ATTENTION` | At least one test was dead-lettered, for example an ungradable image. |
| `FINAL` | All results are in. The report has gone to `fundus.reports` for the CRM. |

### What happens when something fails

| Failure | What the worker does | Outcome |
|---|---|---|
| Model service down, busy, timing out, or returning 5xx / 429 | Sends a copy to that model's next retry queue (10 s, then 1 min, then 10 min) and acks. | Only that model retries; the other seven are untouched. The report waits. |
| Still failing after the last retry | Dead-letters it with the reason and attempt count, and tells the aggregator. | Study is `NEEDS_ATTENTION`; operations replays it after the fix. |
| Image missing, corrupt or ungradable (4xx from the model) | Dead-letters at once. Retrying a bad image never helps. | Study is `NEEDS_ATTENTION`; the clinic retakes the image. |
| Worker process crashed | Nothing: the message was never acked. | RabbitMQ keeps it in the queue and redelivers it when the worker restarts. |
| Message that can't be parsed | Rejects it without requeue; the queue's dead-letter exchange moves it. | It lands in the dead-letter queue instead of looping forever. |
| RabbitMQ down when a study is submitted | Intake gets no confirm, removes the study and images, and returns `503`. | Nothing half-stored; the CRM retries. |
| Duplicate or redelivered result | Recording is keyed on `(study_id, condition)`. | Changes nothing, and the CRM gets one report. |
| Study taking too long | The watcher publishes one alert listing the missing tests. | Someone is told, without the report being released early. |

## Design decisions

- **Publish once, fan out in the broker.** The intake API doesn't know which models exist. Adding a
  ninth model means binding one more queue to `fundus.studies` and deploying its worker; the CRM,
  the intake API and the other models don't change.
- **Publisher confirms everywhere.** A `202` means RabbitMQ has the study on disk. Workers and the
  aggregator ack a message only after the next message is confirmed, so a crash at any point means
  redelivery, never loss.
- **Retry queues per model, routed back through the default exchange.** When a retry's TTL expires
  it goes straight back to that one model's queue, not through the fanout, so the other seven never
  re-run. The delay is in the queue name (`fundus.retry.rvo.10000ms`) because a queue's TTL can't
  be changed after it's declared; a new delay gets a new queue instead of a `PRECONDITION_FAILED`.
- **Temporary versus permanent errors.** Connection errors, timeouts, 5xx and 429 are retried. 4xx
  means the model rejected the input and will reject it every time, so it goes straight to the
  dead-letter queue.
- **Prefetch 1 per model worker.** Each message is a GPU inference, so a worker takes one at a time
  and RabbitMQ shares the queue fairly across several workers for the same model.
- **The models are unchanged.** Workers call the same HTTP endpoints `/predict-all` used, so this
  adds reliability without touching the inference code. `/predict-all` still works and is marked
  deprecated.

## Running it

```bash
docker compose -f docker-compose.rabbitmq.yml up -d          # RabbitMQ; management UI on :15672
pip install -r requirements-messaging.txt
cd OPHTHALMOLOGY_CRM_AI_API

uvicorn messaging.intake:app --port 8100                       # or the main app.py, which mounts it
python -m messaging.aggregator
python -m messaging.watcher
python -m messaging.worker diabetes                            # one process per model
python -m messaging.worker rvo                                 # ...and so on for all eight
```

Run each process under a supervisor (systemd, a Windows service or Docker with a restart policy).
A worker that loses its RabbitMQ connection exits; on restart it picks up where it left off,
because anything it hadn't acked is still in its queue.

```bash
curl -F referral_id=REF-1 -F left_eye_file=@left.jpg -F right_eye_file=@right.jpg localhost:8100/studies
curl localhost:8100/studies/<study_id>

python -m messaging.ops list                                   # dead letters, with reasons
python -m messaging.ops replay --study <study_id>              # after the fix
python -m messaging.ops replay --all
python -m messaging.ops discard --study <study_id>
```

### Configuration

| Variable | Default |
|---|---|
| `RABBITMQ_URL` | `amqp://guest:guest@localhost:5672/%2F` |
| `FUNDUS_MODEL_URL_<CONDITION>` | `http://localhost:8000/predict-<condition>` |
| `FUNDUS_CONDITIONS` | all eight models |
| `FUNDUS_RETRY_DELAYS_MS` | `10000,60000,600000` |
| `FUNDUS_MODEL_TIMEOUT_S` | `120` |
| `FUNDUS_REPORT_TIMEOUT_S` | `3600` |
| `FUNDUS_IMAGE_STORE_DIR` | `image_store` |
| `FUNDUS_STUDY_DB_URL` | `sqlite:///studies.db` (use an `mssql+pyodbc://` URL for SQL Server) |
| `FUNDUS_MQ_PREFIX` | `fundus` |

## Demo without GPUs

`messaging/stub_models.py` serves the same eight endpoints with stand-in answers, and lets you
switch any model's behaviour while the pipeline runs:

```bash
uvicorn messaging.stub_models:app --port 8090
export FUNDUS_MODEL_URL_RVO=http://localhost:8090/predict-rvo     # and so on for each model

curl -X PUT localhost:8090/behaviour/rvo/down        # watch the study wait as PENDING and RVO retry
curl -X PUT localhost:8090/behaviour/rvo/ok          # the retry succeeds and the report goes FINAL
curl -X PUT localhost:8090/behaviour/cnv/bad_image   # dead-lettered at once; see `ops list`
```

Stopping a worker process (Ctrl+C) and starting it again shows the study waiting safely in that
model's queue in the RabbitMQ management UI, then completing.

## Tests

```bash
cd OPHTHALMOLOGY_CRM_AI_API
python -m pytest messaging/tests
```

`test_units.py` needs nothing running. `test_integration.py` runs the whole pipeline against a
real RabbitMQ (skipped if none is reachable): all eight models; a model down then back; a worker
offline; retries exhausted, then replayed after the fix; a bad image; replaying one study only;
the timeout alert; duplicate results; and RabbitMQ unreachable at intake. GitHub Actions runs both
on every push.

## What I'd add for production

- **A transactional outbox** in intake, so writing the study and publishing it can't disagree even
  if the process dies between the two.
- **Quorum queues** on a three-node cluster, so losing one RabbitMQ node loses nothing.
- **Metrics and alerting** on queue depth, dead-letter count and study age, alongside the watcher.
- **A CRM consumer** for `fundus.reports.crm` that writes the final report back, deduplicating on
  the message id.
