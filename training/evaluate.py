"""Evaluate Regex, Legacy ST+LR, Fine-tuned BERT, LLM, and Hybrid Router.

Only writes metrics that are actually computed. LLM evaluation is skipped
when GROQ_API_KEY is not configured (recorded as skipped, not fabricated).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.classifiers.bert_classifier import BertClassifier  # noqa: E402
from app.classifiers.legacy_st_lr import LegacySTLRClassifier  # noqa: E402
from app.classifiers.llm_classifier import LLMClassifier  # noqa: E402
from app.classifiers.regex_classifier import RegexClassifier  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.routing.router import HybridRouter  # noqa: E402

logger = get_logger(__name__)


def _metrics_bundle(
    name: str,
    y_true: List[str],
    y_pred: List[str],
    latencies: List[float],
    extra: Optional[Dict] = None,
) -> Dict:
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0, labels=labels
    )
    payload = {
        "model": name,
        "n_samples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        ),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
    }
    if extra:
        payload.update(extra)
    return payload


def eval_regex(df: pd.DataFrame) -> Dict:
    clf = RegexClassifier()
    text_col = get_settings().text_column
    label_col = get_settings().label_column
    y_true, y_pred, latencies = [], [], []
    matched = 0
    for _, row in df.iterrows():
        start = time.perf_counter()
        result = clf.classify(str(row[text_col]))
        latencies.append((time.perf_counter() - start) * 1000.0)
        pred = result.category if result.category else "Unclassified"
        if result.category:
            matched += 1
        y_true.append(str(row[label_col]))
        y_pred.append(pred)
    return _metrics_bundle(
        "regex_only",
        y_true,
        y_pred,
        latencies,
        extra={"regex_match_rate": matched / len(df) if len(df) else 0.0},
    )


def eval_legacy(df: pd.DataFrame) -> Dict:
    clf = LegacySTLRClassifier()
    if not clf.is_available:
        return {"model": "sentence_transformer_lr", "status": "skipped", "reason": "model_unavailable"}
    text_col = get_settings().text_column
    label_col = get_settings().label_column
    y_true, y_pred, latencies = [], [], []
    for _, row in df.iterrows():
        start = time.perf_counter()
        result = clf.classify(str(row[text_col]))
        latencies.append((time.perf_counter() - start) * 1000.0)
        y_true.append(str(row[label_col]))
        y_pred.append(result.category)
    return _metrics_bundle("sentence_transformer_lr", y_true, y_pred, latencies)


def eval_bert(df: pd.DataFrame) -> Dict:
    clf = BertClassifier()
    if not clf.is_available:
        return {"model": "fine_tuned_bert", "status": "skipped", "reason": "model_unavailable"}
    text_col = get_settings().text_column
    label_col = get_settings().label_column
    y_true, y_pred, latencies = [], [], []
    for _, row in df.iterrows():
        start = time.perf_counter()
        result = clf.classify(str(row[text_col]))
        latencies.append((time.perf_counter() - start) * 1000.0)
        y_true.append(str(row[label_col]))
        y_pred.append(result.category)
    return _metrics_bundle("fine_tuned_bert", y_true, y_pred, latencies)


def eval_llm(df: pd.DataFrame, max_samples: int = 20) -> Dict:
    """Evaluate LLM on a small capped subset to control cost."""
    clf = LLMClassifier()
    if not clf.is_configured:
        return {
            "model": "llm_only",
            "status": "skipped",
            "reason": "GROQ_API_KEY not configured",
        }
    sample = df.head(max_samples)
    text_col = get_settings().text_column
    label_col = get_settings().label_column
    y_true, y_pred, latencies = [], [], []
    cost = 0.0
    tokens = 0
    for _, row in sample.iterrows():
        start = time.perf_counter()
        result = clf.classify(str(row[text_col]))
        latencies.append((time.perf_counter() - start) * 1000.0)
        y_true.append(str(row[label_col]))
        y_pred.append(result.category)
        cost += result.estimated_cost_usd
        tokens += result.prompt_tokens + result.completion_tokens
    bundle = _metrics_bundle("llm_only", y_true, y_pred, latencies)
    bundle["estimated_llm_cost_usd"] = cost
    bundle["total_tokens"] = tokens
    bundle["note"] = f"Evaluated on first {len(sample)} test rows only to limit API cost."
    return bundle


def eval_hybrid(df: pd.DataFrame) -> Dict:
    router = HybridRouter()
    text_col = get_settings().text_column
    label_col = get_settings().label_column
    y_true, y_pred, latencies = [], [], []
    method_counts: Dict[str, int] = {}
    llm_count = 0
    cost = 0.0
    for _, row in df.iterrows():
        source = row["source"] if "source" in df.columns else None
        start = time.perf_counter()
        result = router.classify(str(row[text_col]), source=None if pd.isna(source) else str(source))
        # Use result.latency_ms when present; also track wall time
        latencies.append(result.latency_ms or ((time.perf_counter() - start) * 1000.0))
        y_true.append(str(row[label_col]))
        y_pred.append(result.category)
        method_counts[result.classification_method] = method_counts.get(result.classification_method, 0) + 1
        if result.llm_invoked:
            llm_count += 1
    bundle = _metrics_bundle("hybrid_router", y_true, y_pred, latencies)
    bundle["method_counts"] = method_counts
    bundle["llm_invocation_count"] = llm_count
    bundle["llm_usage_percentage"] = (llm_count / len(df) * 100.0) if len(df) else 0.0
    return bundle


def build_comparison_table(results: List[Dict]) -> List[Dict]:
    rows = []
    for item in results:
        if item.get("status") == "skipped":
            rows.append(
                {
                    "Model": item["model"],
                    "Accuracy": None,
                    "Macro F1": None,
                    "Weighted F1": None,
                    "Avg Latency (ms)": None,
                    "Status": f"skipped: {item.get('reason')}",
                }
            )
            continue
        rows.append(
            {
                "Model": item["model"],
                "Accuracy": round(item["accuracy"], 4),
                "Macro F1": round(item["f1_macro"], 4),
                "Weighted F1": round(item["f1_weighted"], 4),
                "Avg Latency (ms)": round(item["avg_latency_ms"], 3),
                "Status": "ok",
            }
        )
    return rows


def main(include_llm: bool = False, llm_max_samples: int = 10) -> None:
    setup_logging()
    settings = get_settings()
    test_path = settings.resolve(settings.data_splits_dir) / "test.csv"
    if not test_path.exists():
        from app.preprocessing.dataset import create_train_val_test_splits

        create_train_val_test_splits()
    df = pd.read_csv(test_path)
    logger.info("Evaluating on test split with %d rows", len(df))

    results = [
        eval_regex(df),
        eval_legacy(df),
        eval_bert(df),
        eval_hybrid(df),
    ]
    if include_llm:
        results.insert(3, eval_llm(df, max_samples=llm_max_samples))
    else:
        results.insert(
            3,
            {
                "model": "llm_only",
                "status": "skipped",
                "reason": "Pass --include-llm to evaluate (uses API credits)",
            },
        )

    comparison = build_comparison_table(results)
    out_dir = settings.resolve(settings.evaluation_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "model_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump({"results": results, "comparison_table": comparison}, handle, indent=2)

    # Markdown report with only real values
    lines = [
        "# Model Comparison Report",
        "",
        "Generated from actual evaluation runs. Skipped models are listed explicitly.",
        "",
        "| Model | Accuracy | Macro F1 | Weighted F1 | Avg Latency (ms) | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['Model']} | {row['Accuracy']} | {row['Macro F1']} | "
            f"{row['Weighted F1']} | {row['Avg Latency (ms)']} | {row['Status']} |"
        )
    report_path = out_dir / "model_comparison.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote comparison report to %s", report_path)
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    include = "--include-llm" in sys.argv
    main(include_llm=include)
