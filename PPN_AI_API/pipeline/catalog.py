"""The eight fundus tests, named the way the queues are named.

Five queues use the clinical name from the screening diagram. The remaining
three production models keep the diagram's model-6/7/8 queue names:

- model-6 is glaucoma
- model-7 is macular degeneration
- model-8 is other pathology
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TestSpec:
    name: str
    queue: str
    title: str
    model_key: str
    url_env: str
    default_url: str


TESTS: tuple[TestSpec, ...] = (
    TestSpec(
        name="diabetic-retinopathy",
        queue="q.diabetic-retinopathy",
        title="Diabetic retinopathy",
        model_key="diabetes",
        url_env="DIABETES_URL",
        default_url="http://localhost:8000/predict-diabetes",
    ),
    TestSpec(
        name="hypertensive-retinopathy",
        queue="q.hypertensive-retinopathy",
        title="Hypertensive retinopathy",
        model_key="hypertension",
        url_env="HYPERTENSION_URL",
        default_url="http://ppnai-prod:8000/predict-hypertension",
    ),
    TestSpec(
        name="rvo",
        queue="q.rvo",
        title="RVO",
        model_key="rvo",
        url_env="RVO_URL",
        default_url="http://ppnai-prod:8000/predict-rvo",
    ),
    TestSpec(
        name="cnv",
        queue="q.cnv",
        title="CNV",
        model_key="cnv",
        url_env="CNV_URL",
        default_url="http://ppnai-prod:8000/predict-cnv",
    ),
    TestSpec(
        name="myopia",
        queue="q.myopia",
        title="Myopia",
        model_key="myopia",
        url_env="MYOPIA_URL",
        default_url="http://ppnai-prod:8000/predict-myopia",
    ),
    TestSpec(
        name="model-6",
        queue="q.model-6",
        title="Glaucoma",
        model_key="glaucoma",
        url_env="GLAUCOMA_URL",
        default_url="http://ppnai-prod:8000/predict-glaucoma",
    ),
    TestSpec(
        name="model-7",
        queue="q.model-7",
        title="Macular degeneration",
        model_key="macular_degeneration",
        url_env="AMD_URL",
        default_url="http://ppnai-prod:8000/predict-macular-degeneration",
    ),
    TestSpec(
        name="model-8",
        queue="q.model-8",
        title="Pathological",
        model_key="pathological",
        url_env="PATHO_URL",
        default_url="http://localhost:8000/predict-pathological",
    ),
)

EXPECTED_TESTS = 8

if len(TESTS) != EXPECTED_TESTS:
    raise RuntimeError(f"screening requires {EXPECTED_TESTS} tests, found {len(TESTS)}")

BY_NAME = {test.name: test for test in TESTS}
BY_QUEUE = {test.queue: test for test in TESTS}


def require_test(name: str) -> TestSpec:
    try:
        return BY_NAME[name]
    except KeyError as exc:
        known = ", ".join(BY_NAME)
        raise KeyError(f"unknown test {name!r}; expected one of: {known}") from exc
