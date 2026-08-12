"""Evaluate a fine-tuned BERT checkpoint on the held-out test split."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def evaluate_bert(split: str = "test") -> dict:
    setup_logging()
    settings = get_settings()
    model_dir = settings.resolve(settings.bert_model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"No fine-tuned model at {model_dir}. Run training/train_bert.py first."
        )

    split_path = settings.resolve(settings.data_splits_dir) / f"{split}.csv"
    df = pd.read_csv(split_path)
    texts = df[settings.text_column].astype(str).tolist()
    y_true = df[settings.label_column].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    preds = []
    latencies = []

    for text in texts:
        start = time.perf_counter()
        encoded = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=settings.bert_max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[0]
            pred_id = int(torch.argmax(F.softmax(logits, dim=-1)).item())
        preds.append(id2label.get(pred_id, str(pred_id)))
        latencies.append((time.perf_counter() - start) * 1000.0)

    labels = sorted(set(y_true) | set(preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="weighted", zero_division=0, labels=labels
    )
    result = {
        "model": "fine_tuned_bert",
        "split": split,
        "n_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "classification_report": classification_report(
            y_true, preds, zero_division=0, output_dict=True
        ),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, preds, labels=labels).tolist(),
        },
    }

    out_dir = settings.resolve(settings.evaluation_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bert_{split}_metrics.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    logger.info("Wrote BERT evaluation to %s", out_path)
    return result


if __name__ == "__main__":
    metrics = evaluate_bert("test")
    print(json.dumps({k: metrics[k] for k in metrics if k not in {"classification_report", "confusion_matrix"}}, indent=2))
