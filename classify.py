"""Backward-compatible classify helpers using the new hybrid router."""

from __future__ import annotations

from typing import Iterable, List, Tuple

from app.routing.router import get_router
from app.services.batch import classify_dataframe
import pandas as pd


def classify(logs: Iterable[Tuple[str, str]]) -> List[str]:
    """Classify (source, log_message) pairs; returns category labels."""
    router = get_router()
    labels = []
    for source, log_msg in logs:
        result = router.classify(log_msg, source=source)
        labels.append(result.category)
    return labels


def classify_log(source: str, log_msg: str) -> str:
    return get_router().classify(log_msg, source=source).category


def classify_csv(input_file: str) -> str:
    df = pd.read_csv(input_file)
    out = classify_dataframe(df)
    output_file = "resources/output.csv"
    out.to_csv(output_file, index=False)
    return output_file


if __name__ == "__main__":
    classify_csv("resources/test.csv")
