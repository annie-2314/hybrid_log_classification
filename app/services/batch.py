"""Batch classification helpers."""

from __future__ import annotations

import io
from typing import BinaryIO, Optional, Union

import pandas as pd

from app.routing.router import HybridRouter, get_router
from app.schemas import ClassificationResult


def classify_dataframe(
    df: pd.DataFrame,
    router: Optional[HybridRouter] = None,
    text_column: str = "log_message",
    source_column: str = "source",
) -> pd.DataFrame:
    """Classify all rows in a DataFrame and append result columns."""
    if text_column not in df.columns:
        raise ValueError(f"CSV must contain '{text_column}' column")

    router = router or get_router()
    results = []
    for _, row in df.iterrows():
        source = row[source_column] if source_column in df.columns else None
        result: ClassificationResult = router.classify(
            str(row[text_column]),
            source=None if pd.isna(source) else str(source),
        )
        results.append(result.model_dump())

    result_df = pd.DataFrame(results)
    out = df.copy().reset_index(drop=True)
    out["predicted_category"] = result_df["category"]
    out["severity"] = result_df["severity"]
    out["confidence"] = result_df["confidence"]
    out["classification_method"] = result_df["classification_method"]
    out["explanation"] = result_df["explanation"]
    out["latency_ms"] = result_df["latency_ms"]
    out["llm_invoked"] = result_df["llm_invoked"]
    return out


def classify_csv_bytes(
    file_obj: Union[BinaryIO, io.BytesIO],
    router: Optional[HybridRouter] = None,
) -> pd.DataFrame:
    df = pd.read_csv(file_obj)
    return classify_dataframe(df, router=router)
