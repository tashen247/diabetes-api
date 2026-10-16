#!/usr/bin/env python3
"""
One-file RVO finish script for C:\\source_controls\\development_sanele

Usage (PowerShell, from that folder):
  python finish_rvo_validation.py

Uses:
  C:\\AI_Images\\RVO
  C:\\AI_Images\\trained_models\\rvo_model.keras
Writes:
  C:\\AI_Images\\RVO\\eval_rvo_results.json
  .\\Fundus_AI_Validation.xlsx (downloads from GitHub if missing)
"""

from __future__ import annotations

import json
import os
import random
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

DATA_ROOT = Path(r"C:\AI_Images\RVO")
MODEL_PATH = Path(r"C:\AI_Images\trained_models\rvo_model.keras")
JSON_OUT = Path(r"C:\AI_Images\RVO\eval_rvo_results.json")
BRANCH = "cursor/fundus-ai-validation-finish-a330"
XLSX_URL = (
    f"https://raw.githubusercontent.com/tashen247/diabetes-api/{BRANCH}/"
    "validation/Fundus_AI_Validation.xlsx"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
POS_NAMES = {"rvo", "positive", "pos", "suspect"}
NEG_NAMES = {"no_rvo", "norvo", "no-rvo", "normal", "negative", "neg", "no_suspect", "nosuspect", "no"}


def ensure_workbook(path: Path) -> Path:
    if path.exists():
        return path
    print(f"[RVO] downloading workbook -> {path}")
    urllib.request.urlretrieve(XLSX_URL, path)
    return path


def list_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def discover(data_root: Path) -> Tuple[Path, Path]:
    children = [p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    by = {p.name.lower(): p for p in children}
    pos = neg = None
    for name, path in by.items():
        key = name.replace(" ", "")
        if name in POS_NAMES or key in POS_NAMES or name == "rvo":
            pos = path
        if name in NEG_NAMES or key in NEG_NAMES:
            neg = path
    if not (pos and neg):
        for sub in children:
            submap = {p.name.lower(): p for p in sub.iterdir() if p.is_dir()}
            p2 = n2 = None
            for name, path in submap.items():
                if name in POS_NAMES or name == "rvo":
                    p2 = path
                if name in NEG_NAMES:
                    n2 = path
            if p2 and n2:
                return p2, n2
        raise RuntimeError(f"Need RVO + negative folders under {data_root}; found {[c.name for c in children]}")
    return pos, neg


def holdout(pos: Sequence[Path], neg: Sequence[Path], frac=0.1, seed=42):
    rng = random.Random(seed)
    pos_i, neg_i = list(pos), list(neg)
    rng.shuffle(pos_i)
    rng.shuffle(neg_i)
    n_pos_val = max(1, int(round(len(pos_i) * frac))) if pos_i else 0
    n_neg_val = max(1, int(round(len(neg_i) * frac))) if neg_i else 0
    if len(pos_i) > 1:
        n_pos_val = min(n_pos_val, len(pos_i) - 1)
    if len(neg_i) > 1:
        n_neg_val = min(n_neg_val, len(neg_i) - 1)
    val = [(p, 1) for p in pos_i[:n_pos_val]] + [(p, 0) for p in neg_i[:n_neg_val]]
    train = [(p, 1) for p in pos_i[n_pos_val:]] + [(p, 0) for p in neg_i[n_neg_val:]]
    rng.shuffle(val)
    rng.shuffle(train)
    return train, val


def load_model(model_path: Path):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    errors = []
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(str(model_path), compile=False)
    except Exception as e:
        errors.append(f"tf.keras: {e}")
    try:
        import keras
        return keras.models.load_model(str(model_path), compile=False)
    except Exception as e:
        errors.append(f"keras: {e}")
    raise RuntimeError("Model load failed:\n  - " + "\n  - ".join(errors))


def preprocess(paths: Sequence[Path], image_size: int):
    import numpy as np
    from PIL import Image
    arr = np.zeros((len(paths), image_size, image_size, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            arr[i] = np.asarray(im.convert("RGB").resize((image_size, image_size)), dtype=np.float32) / 255.0
    return arr


def predict(model, samples, batch_size, image_size):
    import numpy as np
    y_true, y_score = [], []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        paths = [p for p, _ in batch]
        labels = [y for _, y in batch]
        pred = np.asarray(model.predict(preprocess(paths, image_size), verbose=0))
        if pred.ndim == 2 and pred.shape[1] == 1:
            scores = pred[:, 0]
        elif pred.ndim == 2 and pred.shape[1] == 2:
            scores = pred[:, 1]
        elif pred.ndim == 1:
            scores = pred
        else:
            raise RuntimeError(f"Unexpected pred shape {pred.shape}")
        y_true.extend(labels)
        y_score.extend(float(s) for s in scores)
    return y_true, y_score


def metrics(y_true, y_score):
    import numpy as np
    from sklearn.metrics import confusion_matrix, roc_auc_score
    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    out = {}
    try:
        out["auroc"] = float(roc_auc_score(yt, ys))
    except ValueError:
        out["auroc"] = float("nan")
    yhat = (ys >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yhat, labels=[0, 1]).ravel()
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    out["accuracy"] = float((tp + tn) / max(len(yt), 1))
    out.update(tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn))
    return out


def apply_to_workbook(xlsx: Path, payload: dict):
    import openpyxl
    val, split = payload["val"], payload["split"]
    wb = openpyxl.load_workbook(xlsx)
    ws = wb["Summary"]
    ws["H8"] = int(split["n_train"])
    ws["I8"] = int(split["n_pos_train"])
    ws["J8"] = "post-hoc val 10% (random_state=42)"
    ws["K8"] = int(split["n_val"])
    ws["L8"] = int(split["n_pos_val"])
    ws["M8"] = round(float(val["sensitivity"]), 4)
    ws["N8"] = round(float(val["specificity"]), 4)
    ws["O8"] = round(float(val["auroc"]), 4)
    ws["P8"] = f"Train AUROC {payload['train']['auroc']:.4f}. Quote val. Root={payload['data_root']}"
    ws["Q8"] = r"AUROC on post-hoc 10% of C:\AI_Images\RVO"
    ws["R8"] = f"pos={payload['pos_dir']}; neg={payload['neg_dir']}; quote val {val['auroc']:.4f}"
    ws["E8"] = f"{payload['data_root']} ({payload['layout']})"
    ws["F8"] = payload.get("assembled_utc", "")[:10]

    ws = wb["RVO"]
    ws["B6"] = int(payload["n_pos_total"])
    ws["C6"] = int(payload["n_neg_total"])
    ws["B11"] = round(float(val["sensitivity"]), 4)
    ws["B12"] = round(float(val["specificity"]), 4)
    ws["B13"] = round(float(val["auroc"]), 4)
    ws["C13"] = f"val n={split['n_val']} n_pos={split['n_pos_val']}"

    ws = wb["Holdout_eval"]
    ws["B17"] = "post-hoc val 10%"
    ws["C17"] = int(split["n_val"])
    ws["D17"] = int(split["n_pos_val"])
    ws["E17"] = round(float(val["auroc"]), 4)
    ws["F17"] = round(float(val["sensitivity"]), 4)
    ws["G17"] = round(float(val["specificity"]), 4)
    ws["H17"] = r"from C:\AI_Images\RVO; quote AUROC"

    if "Gaps" in wb.sheetnames:
        wb["Gaps"]["B8"] = "None (AUROC filled from C:\\AI_Images\\RVO)"
        wb["Gaps"]["D8"] = f"Filled val AUROC {val['auroc']:.4f}"
    if "Finish_status" in wb.sheetnames:
        wb["Finish_status"]["B8"] = f"{val['auroc']:.4f} (post-hoc val)"
        wb["Finish_status"]["C8"] = "filled (argmax)"
        wb["Finish_status"]["D8"] = r"DONE — quote val AUROC from C:\AI_Images\RVO"

    wb.save(xlsx)


def main() -> int:
    here = Path(__file__).resolve().parent
    xlsx = ensure_workbook(here / "Fundus_AI_Validation.xlsx")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if not DATA_ROOT.exists():
        raise FileNotFoundError(DATA_ROOT)

    pos_dir, neg_dir = discover(DATA_ROOT)
    pos_imgs, neg_imgs = list_images(pos_dir), list_images(neg_dir)
    if not pos_imgs or not neg_imgs:
        raise RuntimeError(f"Empty class folders: pos={pos_dir} ({len(pos_imgs)}) neg={neg_dir} ({len(neg_imgs)})")

    train, val = holdout(pos_imgs, neg_imgs)
    print(f"[RVO] pos={pos_dir} n={len(pos_imgs)} | neg={neg_dir} n={len(neg_imgs)}")
    print(f"[RVO] train={len(train)} val={len(val)}")
    model = load_model(MODEL_PATH)
    image_size = 224
    try:
        shape = model.input_shape
        if shape and len(shape) == 4 and shape[1] and shape[2]:
            image_size = int(shape[1])
    except Exception:
        pass
    print(f"[RVO] image_size={image_size}")
    yv, sv = predict(model, val, 64, image_size)
    val_m = metrics(yv, sv)
    yt, st = predict(model, train, 64, image_size)
    train_m = metrics(yt, st)

    payload = {
        "condition": "rvo",
        "assembled_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(DATA_ROOT),
        "model": str(MODEL_PATH),
        "layout": "two-class under C:\\AI_Images\\RVO",
        "pos_dir": str(pos_dir),
        "neg_dir": str(neg_dir),
        "n_pos_total": len(pos_imgs),
        "n_neg_total": len(neg_imgs),
        "n_labelled": len(pos_imgs) + len(neg_imgs),
        "split": {
            "type": "post-hoc stratified",
            "val_frac": 0.1,
            "random_state": 42,
            "n_train": len(train),
            "n_pos_train": sum(1 for _, y in train if y == 1),
            "n_val": len(val),
            "n_pos_val": sum(1 for _, y in val if y == 1),
        },
        "image_size": image_size,
        "batch_size": 64,
        "val": val_m,
        "train": train_m,
        "quote": "Use val.auroc for the workbook.",
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    apply_to_workbook(xlsx, payload)
    print(json.dumps({"val": val_m, "train_auroc": train_m["auroc"], "xlsx": str(xlsx)}, indent=2))
    print(f"[RVO] wrote {JSON_OUT}")
    print(f"[RVO] updated {xlsx}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[RVO] FAILED: {exc}", file=sys.stderr)
        raise
