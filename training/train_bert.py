"""Fine-tune a Hugging Face Transformer for multi-class log classification."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.preprocessing.dataset import create_train_val_test_splits  # noqa: E402
from training.config import load_train_config  # noqa: E402

logger = get_logger(__name__)


class LogDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "f1_macro": macro_f1,
    }


def _ensure_splits(settings) -> None:
    splits_dir = settings.resolve(settings.data_splits_dir)
    if not (splits_dir / "train.csv").exists():
        logger.info("Splits missing; creating train/val/test splits")
        create_train_val_test_splits()


def main() -> None:
    setup_logging()
    settings = get_settings()
    cfg = load_train_config()
    set_seed(cfg.seed)
    _ensure_splits(settings)

    train_df = pd.read_csv(settings.resolve(cfg.train_file))
    val_df = pd.read_csv(settings.resolve(cfg.val_file))

    text_col = settings.text_column
    label_col = settings.label_column

    labels = sorted(train_df[label_col].astype(str).unique().tolist())
    # Include any val-only labels to keep mapping complete
    for lab in val_df[label_col].astype(str).unique():
        if lab not in labels:
            labels.append(lab)
    labels = sorted(labels)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def encode(df: pd.DataFrame):
        encodings = tokenizer(
            df[text_col].astype(str).tolist(),
            truncation=True,
            padding=True,
            max_length=cfg.max_length,
        )
        y = [label2id[str(x)] for x in df[label_col].tolist()]
        return LogDataset(encodings, y)

    train_ds = encode(train_df)
    val_ds = encode(val_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    output_dir = settings.resolve(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = torch.cuda.is_available()
    steps_per_epoch = max(1, len(train_ds) // max(1, cfg.batch_size))
    warmup_steps = max(1, int(steps_per_epoch * cfg.num_epochs * cfg.warmup_ratio))
    # Compatible with Transformers v5 TrainingArguments API.
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        seed=cfg.seed,
        fp16=use_fp16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    logger.info(
        "Starting fine-tuning: model=%s device=%s epochs=%s",
        cfg.model_name,
        "cuda" if torch.cuda.is_available() else "cpu",
        cfg.num_epochs,
    )
    trainer.train()
    metrics = trainer.evaluate()
    logger.info("Validation metrics: %s", metrics)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    label_map = {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    with (output_dir / "label_map.json").open("w", encoding="utf-8") as handle:
        json.dump(label_map, handle, indent=2)

    with (output_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2)

    with (output_dir / "val_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({k: float(v) for k, v in metrics.items()}, handle, indent=2)

    logger.info("Saved best model to %s", output_dir)


if __name__ == "__main__":
    main()
