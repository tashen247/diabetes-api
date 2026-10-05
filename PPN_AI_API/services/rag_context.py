"""Retrieval for the LLaVA fundus prompts.

The specialist models still choose the stored label. This module only retrieves
the grade list, the clinical signs, the override rule, and the facts already
known about this referral, then inserts those passages into the LLaVA prompt.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


K1 = 1.5
B = 0.75

# Grade strings and ads ids used by the screener and the glaucoma decision.
# Kept here so prompt retrieval does not import the model services.
GRADES: Dict[str, Dict[str, int]] = {
    "diabetes": {
        "No Suspect or Negative": 2,
        "Mild": 3,
        "Moderate": 4,
        "Severe": 5,
        "Proliferative Diabetic Retinopathy (PDR)": 6,
    },
    "hypertension": {
        "No Suspect or Negative": 7,
        "Decreased Retinal Artery Elasticity": 8,
        "Hypertensive Retinopathy Grade 1 or 2": 9,
        "Hypertensive Retinopathy Grade 3 or 4": 10,
    },
    "macular_degeneration": {
        "No Suspect or Negative": 11,
        "Drusen": 12,
        "AMD Early or Intermediate Stage": 13,
        "AMD Advanced Stage": 14,
        "Large Drusen with Abnormal Pigmentation": 15,
    },
    "cnv": {
        "No Suspect or Negative": 22,
        "Suspect or Positive": 23,
    },
    "rvo": {
        "No Suspect or Negative": 24,
        "Suspect or Positive": 25,
    },
    "pathological": {
        "No Suspect or Negative": 18,
        "Minor Pathological": 19,
        "Obvious Pathological": 20,
    },
    "glaucoma": {
        "No Suspect or Negative": 16,
        "Suspect or Positive": 17,
    },
}

# Positive-override thresholds from fundus_screener.OVERRIDE_THRESHOLDS.
THRESHOLDS: Dict[str, float] = {
    "diabetes": 0.70,
    "hypertension": 0.72,
    "macular_degeneration": 0.72,
    "cnv": 0.75,
    "rvo": 0.72,
    "pathological": 0.65,
}

BILATERAL_THRESHOLD = 0.80


@dataclass(frozen=True)
class Passage:
    id: str
    condition: str
    text: str


def _grade_line(condition: str) -> str:
    pairs = ", ".join(f"{label}={ads_id}" for label, ads_id in GRADES[condition].items())
    return (
        f"Allowed {condition} grades and ads ids: {pairs}. "
        "The grade string must be copied from this list."
    )


def _build_corpus() -> List[Passage]:
    signs = {
        "diabetes": (
            "Diabetes clinical signs: microaneurysms, dot and blot haemorrhages, "
            "hard exudates, cotton wool spots, venous beading, neovascularisation, "
            "vitreous haemorrhage, fibrous proliferation."
        ),
        "hypertension": (
            "Hypertension clinical signs: arteriovenous nicking, arteriolar narrowing, "
            "silver or copper wiring, flame-shaped haemorrhages, cotton wool spots, "
            "macular star exudates, disc oedema."
        ),
        "macular_degeneration": (
            "Macular degeneration clinical signs: drusen, pigment epithelial changes, "
            "geographic atrophy, subretinal fluid, choroidal neovascular membrane, disciform scar."
        ),
        "cnv": (
            "CNV clinical signs: subretinal haemorrhage, grey-green membrane, "
            "subretinal fluid, pigment epithelial detachment, disciform scar."
        ),
        "rvo": (
            "RVO clinical signs: flame-shaped haemorrhages in one or more quadrants, "
            "dilated tortuous veins, disc swelling, cotton wool spots. "
            "Branch RVO stays in one sector. Central RVO involves all four quadrants."
        ),
        "pathological": (
            "Pathological clinical signs: subretinal fluid, retinal detachment, "
            "choroidal lesions, chorioretinal atrophy, pigment changes, ischaemic white or grey patches."
        ),
        "glaucoma": (
            "Glaucoma clinical signs: rim thinning in the superior, inferior, nasal, or temporal "
            "quadrant, notching, and disc haemorrhage. Name the quadrant when thinning is visible."
        ),
    }
    passages = [
        Passage("shared.override", "shared", (
            "Positive-only override: a vision-language grade may upgrade a negative result "
            "to a positive one. It must not downgrade an existing positive ads id to negative."
        )),
        Passage("shared.bilateral", "shared", (
            f"Bilateral rule: if the other eye is positive at confidence >= {BILATERAL_THRESHOLD:.2f}, "
            "this eye may be promoted to that positive grade. A negative other eye does not "
            "cancel a positive finding on this eye."
        )),
        Passage("glaucoma.measurement_rule", "glaucoma", (
            "Geometric cup-to-disc ratio is the measurement. "
            "High suspect when the ratio is >= 0.70 (ads id 17). "
            "Borderline when the ratio is >= 0.55 (ads id 17). "
            "Below 0.55 with no rim thinning stays ads id 16."
        )),
    ]
    for condition, labels in GRADES.items():
        passages.append(Passage(f"{condition}.grades", condition, _grade_line(condition)))
        passages.append(Passage(f"{condition}.signs", condition, signs[condition]))
        if condition in THRESHOLDS:
            passages.append(Passage(f"{condition}.threshold", condition, (
                f"{condition} positive override threshold is {THRESHOLDS[condition]:.2f}. "
                "Apply a positive grade only when confidence is at least this threshold."
            )))
        del labels
    return passages


CORPUS: Sequence[Passage] = tuple(_build_corpus())


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def label_for_ads(condition: str, ads_id: int) -> Optional[str]:
    for label, stored in GRADES[condition].items():
        if stored == int(ads_id):
            return label
    return None


def _bm25(passages: Sequence[Passage], query: str, k: int) -> List[dict]:
    tokenized = [tokenize(p.text) for p in passages]
    total = len(passages)
    if total == 0:
        return []
    avgdl = sum(len(tokens) for tokens in tokenized) / total or 1.0
    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    def idf(term: str) -> float:
        n = df.get(term, 0)
        return math.log(1.0 + (total - n + 0.5) / (n + 0.5))

    scored = []
    q_terms = tokenize(query)
    for passage, tokens in zip(passages, tokenized):
        tf = Counter(tokens)
        length = len(tokens) or 1
        score = 0.0
        for term in q_terms:
            freq = tf.get(term, 0)
            if not freq:
                continue
            denom = freq + K1 * (1 - B + B * length / avgdl)
            score += idf(term) * (freq * (K1 + 1)) / denom
        scored.append((score, passage))
    scored.sort(key=lambda item: (-item[0], item[1].id))
    return [
        {
            "id": passage.id,
            "condition": passage.condition,
            "score": round(score, 4),
            "text": passage.text,
        }
        for score, passage in scored[:k]
    ]


def retrieve(
    query: str,
    condition: Optional[str] = None,
    extra_passages: Optional[Iterable[Passage]] = None,
    k: int = 6,
) -> List[dict]:
    """Return the top passages for a prompt. Condition limits the corpus to that disease plus shared rules."""
    pool = list(CORPUS)
    if extra_passages:
        pool.extend(extra_passages)
    if condition is not None:
        if condition not in GRADES:
            raise ValueError(f"Unknown condition: {condition}")
        pool = [p for p in pool if p.condition in (condition, "shared")]
    return _bm25(pool, query, k)


def _ads_fact(condition: str, ads_id: Optional[int], kind: str) -> Optional[Passage]:
    if ads_id is None:
        return None
    label = label_for_ads(condition, ads_id)
    if kind == "this_eye":
        named = f"{label}={ads_id}" if label else f"unlisted ads id {ads_id}"
        text = (
            f"Existing classifier result on this eye: {named}. "
            "Do not downgrade an existing positive classifier result to negative."
        )
    else:
        named = f"{label}={ads_id}" if label else f"unlisted ads id {ads_id}"
        text = (
            f"Other eye bilateral result: {named}. "
            "Use the bilateral rule with this other-eye grade. "
            "A negative other eye does not cancel a positive finding on this eye."
        )
    return Passage(f"{condition}.{kind}", condition, text)


def _inject(base_prompt: str, hits: Sequence[dict]) -> str:
    lines = [
        "Retrieved context. Copy the grade from the allowed grades. Do not invent a label.",
    ]
    for index, hit in enumerate(hits, start=1):
        lines.append(f"{index}. ({hit['id']}) {hit['text']}")
    block = "\n".join(lines) + "\n"
    if "[/INST]" in base_prompt:
        return base_prompt.replace("[/INST]", block + "[/INST]", 1)
    return base_prompt.rstrip() + "\n" + block


def build_screen_prompt(
    condition: str,
    base_prompt: str,
    existing_ads: Optional[int] = None,
    other_eye_ads: Optional[int] = None,
    eye: Optional[str] = None,
) -> dict:
    """Retrieve grade, sign, and case passages and insert them before [/INST]."""
    if condition not in THRESHOLDS:
        raise ValueError(f"Unknown screener condition: {condition}")
    extras = []
    this_eye = _ads_fact(condition, _as_int(existing_ads), "this_eye")
    other_eye = _ads_fact(condition, _as_int(other_eye_ads), "other_eye")
    if this_eye and eye:
        this_eye = Passage(this_eye.id, this_eye.condition, f"{eye} eye. " + this_eye.text)
    if this_eye:
        extras.append(this_eye)
    if other_eye:
        extras.append(other_eye)

    query_terms = [condition, "allowed grades", "ads id", "clinical signs", "override threshold"]
    if this_eye:
        query_terms.append("existing classifier this eye")
    if other_eye:
        query_terms.append("other eye bilateral")
    hits = retrieve(" ".join(query_terms), condition=condition, extra_passages=extras, k=8)
    return {"prompt": _inject(base_prompt, hits), "retrieved": hits}


def build_glaucoma_prompt(
    base_prompt: str,
    geometric_cdr: Optional[float] = None,
    existing_ads: Optional[int] = None,
    other_eye_ads: Optional[int] = None,
) -> dict:
    """Retrieve glaucoma grades and the measured CDR, then insert them into the LLaVA prompt."""
    if geometric_cdr is None:
        measured = (
            "Geometric cup-to-disc ratio is unavailable for this image. "
            "Do not invent a precise decimal."
        )
    else:
        value = f"{float(geometric_cdr):.3f}"
        measured = (
            f"Measured geometric cup-to-disc ratio for this image is {value}. "
            f"If you state a CDR, use {value}. Do not invent a different decimal."
        )
    extras = [Passage("glaucoma.measurement", "glaucoma", measured)]
    this_eye = _ads_fact("glaucoma", _as_int(existing_ads), "this_eye")
    other_eye = _ads_fact("glaucoma", _as_int(other_eye_ads), "other_eye")
    if this_eye:
        extras.append(this_eye)
    if other_eye:
        extras.append(other_eye)

    query = "glaucoma allowed grades ads id rim thinning geometric cup-to-disc ratio"
    if this_eye:
        query += " existing classifier this eye"
    if other_eye:
        query += " other eye bilateral"
    hits = retrieve(query, condition="glaucoma", extra_passages=extras, k=8)
    return {"prompt": _inject(base_prompt, hits), "retrieved": hits}


def _as_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
