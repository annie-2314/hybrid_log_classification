"""Prepare reproducible train/val/test splits and class-distribution report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.logging import setup_logging  # noqa: E402
from app.preprocessing.dataset import (  # noqa: E402
    analyze_class_distribution,
    create_train_val_test_splits,
    load_raw_dataset,
)


def main() -> None:
    setup_logging()
    df = load_raw_dataset()
    analysis = analyze_class_distribution(df)
    print(json.dumps(analysis, indent=2))
    train_df, val_df, test_df = create_train_val_test_splits(df)
    print(
        json.dumps(
            {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
