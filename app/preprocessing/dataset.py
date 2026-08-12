"""Reusable dataset loading, analysis, and split utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.config import get_settings
from app.core.logging import get_logger
from app.preprocessing import normalize_log_message

logger = get_logger(__name__)


def load_raw_dataset(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the synthetic logs CSV with basic validation."""
    settings = get_settings()
    path = Path(csv_path) if csv_path else settings.resolve(settings.data_raw_csv)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    required = {settings.text_column, settings.label_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df = df.copy()
    df[settings.text_column] = df[settings.text_column].map(normalize_log_message)
    df = df[df[settings.text_column].str.len() > 0]
    df = df[df[settings.label_column].notna()]
    df[settings.label_column] = df[settings.label_column].astype(str).str.strip()
    df = df.drop_duplicates(subset=[settings.text_column, settings.label_column])
    df = df.reset_index(drop=True)
    logger.info("Loaded dataset from %s with %d rows", path, len(df))
    return df


def analyze_class_distribution(df: pd.DataFrame, label_col: Optional[str] = None) -> Dict[str, Any]:
    """Return class counts, ratios, and imbalance diagnostics."""
    settings = get_settings()
    label_col = label_col or settings.label_column
    counts = df[label_col].value_counts()
    total = int(counts.sum())
    ratios = {k: float(v) / total for k, v in counts.items()}
    max_c = int(counts.max()) if len(counts) else 0
    min_c = int(counts.min()) if len(counts) else 0
    imbalance_ratio = float(max_c / min_c) if min_c else float("inf")
    return {
        "total_samples": total,
        "num_classes": int(counts.shape[0]),
        "counts": {str(k): int(v) for k, v in counts.items()},
        "ratios": ratios,
        "imbalance_ratio_max_over_min": imbalance_ratio,
        "is_imbalanced": imbalance_ratio > 3.0,
    }


def create_train_val_test_splits(
    df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible stratified train/val/test splits without leakage.

    Strategy:
    - Drop exact duplicate log texts before splitting.
    - Stratify on label where each class has >= 2 samples.
    - Rare classes (<2) are forced into train to avoid split failures,
      and noted in the analysis report.
    """
    settings = get_settings()
    df = load_raw_dataset() if df is None else df.copy()
    text_col = settings.text_column
    label_col = settings.label_column

    # Prevent leakage: unique messages only (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset=[text_col], keep="first").reset_index(drop=True)
    logger.info("Dropped %d duplicate messages for leakage prevention", before - len(df))

    counts = df[label_col].value_counts()
    rare_labels = counts[counts < 2].index.tolist()
    rare_df = df[df[label_col].isin(rare_labels)]
    main_df = df[~df[label_col].isin(rare_labels)]

    test_size = settings.data_test_size
    val_size = settings.data_val_size
    seed = settings.random_seed

    train_val_df, test_df = train_test_split(
        main_df,
        test_size=test_size,
        random_state=seed,
        stratify=main_df[label_col],
    )
    relative_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val,
        random_state=seed,
        stratify=train_val_df[label_col],
    )

    if len(rare_df):
        logger.warning(
            "Rare labels with <2 samples forced into train: %s",
            rare_labels,
        )
        train_df = pd.concat([train_df, rare_df], ignore_index=True)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    out_dir = Path(output_dir) if output_dir else settings.resolve(settings.data_splits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    report = {
        "seed": seed,
        "test_size": test_size,
        "val_size": val_size,
        "rare_labels_forced_to_train": rare_labels,
        "overall": analyze_class_distribution(df),
        "train": analyze_class_distribution(train_df),
        "val": analyze_class_distribution(val_df),
        "test": analyze_class_distribution(test_df),
        "sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }
    report_path = out_dir / "split_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Wrote splits to %s and report to %s", out_dir, report_path)
    return train_df, val_df, test_df


def load_split(split: str) -> pd.DataFrame:
    """Load a previously written split CSV (train|val|test)."""
    settings = get_settings()
    path = settings.resolve(settings.data_splits_dir) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Split file missing: {path}. Run scripts/prepare_dataset.py first."
        )
    return pd.read_csv(path)
