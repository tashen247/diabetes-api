#!/usr/bin/env python3
"""
Score RVO keras weights on C:\\AI_Images\\RVO (post-hoc 10% hold-out).

Designed for the Windows GPU host used for Fundus_AI_Validation.xlsx.
Does not invent metrics: writes whatever the model+folders produce.

Example:
  python eval_rvo_ai_images.py --batch-size 64
  python eval_rvo_ai_images.py --data-root C:\\AI_Images\\RVO --model C:\\AI_Images\\trained_models\\rvo_model.keras
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Positive / negative folder name hints (case-insensitive).
POS_NAMES = {"rvo", "cnv_rvo", "positive", "pos", "suspect"}
NEG_NAMES = {"no_rvo", "norvo", "no-rvo", "normal", "negative", "neg", "no_suspect", "nosuspect"}


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def _list_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if _is_image(p))


def discover_binary_roots(data_root: Path) -> Tuple[Path, Path, str]:
    """
    Find (pos_dir, neg_dir, layout_note) under data_root.
    Accepts either:
      data_root/{RVO, NO_RVO|Normal|...}
    or a single nested labelled tree with two class folders.
    """
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    children = [p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    by_lower = {p.name.lower(): p for p in children}

    pos = None
    neg = None
    for name, path in by_lower.items():
        if name in POS_NAMES or name.replace(" ", "") in POS_NAMES:
            pos = path
        if name in NEG_NAMES or name.replace(" ", "") in NEG_NAMES:
            neg = path

    # Common exact names
    for cand in ("RVO", "rvo"):
        if cand.lower() in by_lower:
            pos = by_lower[cand.lower()]
    for cand in ("NO_RVO", "No_RVO", "NoRVO", "Normal", "NO", "No"):
        if cand.lower() in by_lower:
            neg = by_lower[cand.lower()]

    if pos and neg:
        return pos, neg, f"two-class under {data_root}"

    # One level deeper (e.g. train/RVO)
    for sub in children:
        sub_children = [p for p in sub.iterdir() if p.is_dir()]
        sub_map = {p.name.lower(): p for p in sub_children}
        p2 = n2 = None
        for name, path in sub_map.items():
            if name in POS_NAMES or name == "rvo":
                p2 = path
            if name in NEG_NAMES or name in {"no_rvo", "normal"}:
                n2 = path
        if p2 and n2:
            return p2, n2, f"two-class under {sub}"

    names = [p.name for p in children]
    raise RuntimeError(
        f"Could not find RVO vs negative class folders under {data_root}. "
        f"Subdirs: {names}"
    )


def stratified_holdout(
    pos: Sequence[Path],
    neg: Sequence[Path],
    val_frac: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    rng = random.Random(seed)
    pos_i = list(pos)
    neg_i = list(neg)
    rng.shuffle(pos_i)
    rng.shuffle(neg_i)

    n_pos_val = max(1, int(round(len(pos_i) * val_frac))) if pos_i else 0
    n_neg_val = max(1, int(round(len(neg_i) * val_frac))) if neg_i else 0
    # Keep at least one in train when possible
    if len(pos_i) > 1:
        n_pos_val = min(n_pos_val, len(pos_i) - 1)
    if len(neg_i) > 1:
        n_neg_val = min(n_neg_val, len(neg_i) - 1)

    val = [(p, 1) for p in pos_i[:n_pos_val]] + [(p, 0) for p in neg_i[:n_neg_val]]
    train = [(p, 1) for p in pos_i[n_pos_val:]] + [(p, 0) for p in neg_i[n_neg_val:]]
    rng.shuffle(val)
    rng.shuffle(train)
    return train, val


def load_keras_model(model_path: Path):
    """Load .keras / SavedModel with a few compatibility fallbacks."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    errors = []

    # Prefer tf.keras (often more tolerant of older exports than standalone keras 3)
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

    # Last resort: force keras 2-style module alias if keras.src breaks unpickling
    try:
        import keras

        if "keras.src" in str(errors):
            sys.modules.setdefault("keras.src", keras)
        return keras.models.load_model(str(model_path), compile=False)
    except Exception as e:
        errors.append(f"keras-alias: {e}")

    raise RuntimeError("Failed to load model:\n  - " + "\n  - ".join(errors))


def preprocess_batch(paths: Sequence[Path], image_size: int):
    import numpy as np
    from PIL import Image

    arr = np.zeros((len(paths), image_size, image_size, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = im.convert("RGB").resize((image_size, image_size))
            arr[i] = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def predict_scores(model, samples: Sequence[Tuple[Path, int]], batch_size: int, image_size: int):
    import numpy as np

    y_true = []
    y_score = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        paths = [p for p, _ in batch]
        labels = [y for _, y in batch]
        x = preprocess_batch(paths, image_size)
        pred = model.predict(x, verbose=0)
        pred = np.asarray(pred)
        if pred.ndim == 2 and pred.shape[1] == 1:
            scores = pred[:, 0]
        elif pred.ndim == 2 and pred.shape[1] == 2:
            # Assume class 1 = RVO if softmax/sigmoid dual head
            scores = pred[:, 1]
        elif pred.ndim == 1:
            scores = pred
        else:
            raise RuntimeError(f"Unexpected prediction shape: {pred.shape}")
        y_true.extend(labels)
        y_score.extend(float(s) for s in scores)
    return y_true, y_score


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float]) -> Dict[str, float]:
    import numpy as np
    from sklearn.metrics import roc_auc_score, confusion_matrix

    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    out: Dict[str, float] = {}
    try:
        out["auroc"] = float(roc_auc_score(yt, ys))
    except ValueError:
        out["auroc"] = float("nan")

    yhat = (ys >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yhat, labels=[0, 1]).ravel()
    out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    out["accuracy"] = float((tp + tn) / max(len(yt), 1))
    out["tp"] = int(tp)
    out["tn"] = int(tn)
    out["fp"] = int(fp)
    out["fn"] = int(fn)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate RVO model on C:\\AI_Images\\RVO")
    ap.add_argument("--data-root", default=r"C:\AI_Images\RVO")
    ap.add_argument("--model", default=r"C:\AI_Images\trained_models\rvo_model.keras")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default=r"C:\AI_Images\RVO\eval_rvo_results.json",
        help="JSON results path",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    model_path = Path(args.model)
    out_path = Path(args.out)

    pos_dir, neg_dir, layout = discover_binary_roots(data_root)
    pos_imgs = _list_images(pos_dir)
    neg_imgs = _list_images(neg_dir)
    if not pos_imgs or not neg_imgs:
        raise RuntimeError(
            f"Need images in both classes. pos={pos_dir} ({len(pos_imgs)}), "
            f"neg={neg_dir} ({len(neg_imgs)})"
        )

    train, val = stratified_holdout(pos_imgs, neg_imgs, args.val_frac, args.seed)
    print(f"[RVO] layout={layout}")
    print(f"[RVO] pos_dir={pos_dir} n={len(pos_imgs)}")
    print(f"[RVO] neg_dir={neg_dir} n={len(neg_imgs)}")
    print(f"[RVO] train={len(train)} val={len(val)} (seed={args.seed}, frac={args.val_frac})")
    print(f"[RVO] loading {model_path}")

    model = load_keras_model(model_path)
    # Infer size from model if available
    image_size = args.image_size
    try:
        ishape = model.input_shape
        if ishape and len(ishape) == 4 and ishape[1] and ishape[2]:
            image_size = int(ishape[1])
            print(f"[RVO] using model input size {image_size}")
    except Exception:
        pass

    print("[RVO] scoring val…")
    yv, sv = predict_scores(model, val, args.batch_size, image_size)
    val_m = binary_metrics(yv, sv)
    print("[RVO] scoring train…")
    yt, st = predict_scores(model, train, args.batch_size, image_size)
    train_m = binary_metrics(yt, st)

    payload = {
        "condition": "rvo",
        "assembled_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "model": str(model_path),
        "layout": layout,
        "pos_dir": str(pos_dir),
        "neg_dir": str(neg_dir),
        "n_pos_total": len(pos_imgs),
        "n_neg_total": len(neg_imgs),
        "n_labelled": len(pos_imgs) + len(neg_imgs),
        "split": {
            "type": "post-hoc stratified",
            "val_frac": args.val_frac,
            "random_state": args.seed,
            "n_train": len(train),
            "n_pos_train": sum(1 for _, y in train if y == 1),
            "n_val": len(val),
            "n_pos_val": sum(1 for _, y in val if y == 1),
        },
        "image_size": image_size,
        "batch_size": args.batch_size,
        "val": val_m,
        "train": train_m,
        "quote": "Use val.auroc for Fundus_AI_Validation.xlsx (not max(train,val)).",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"[RVO] wrote {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[RVO] FAILED: {exc}", file=sys.stderr)
        raise
