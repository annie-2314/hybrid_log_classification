"""Evaluate Regex, Legacy ST+LR, Fine-tuned BERT, LLM, and Hybrid Router.

Specialist models are scored on their intended scope (fair comparison),
and full-test-set scores are also saved for transparency.

Only writes metrics that are actually computed. LLM evaluation is skipped
when GROQ_API_KEY is not configured (recorded as skipped, not fabricated).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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

REGEX_LABELS = {"System Notification", "User Action"}


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
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "precision_weighted": float(precision) if y_true else 0.0,
        "recall_weighted": float(recall) if y_true else 0.0,
        "f1_weighted": float(f1) if y_true else 0.0,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if y_true
        else 0.0,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )
        if y_true
        else {},
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()
            if y_true
            else [],
        },
    }
    if extra:
        payload.update(extra)
    return payload


def _subset_by_complexity(df: pd.DataFrame, complexity: str) -> pd.DataFrame:
    if "complexity" not in df.columns:
        return df.iloc[0:0]
    return df[df["complexity"].astype(str).str.lower() == complexity.lower()].copy()


def _subset_regex_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer dataset complexity tag; fall back to regex-target labels."""
    scoped = _subset_by_complexity(df, "regex")
    if len(scoped):
        return scoped
    label_col = get_settings().label_column
    return df[df[label_col].astype(str).isin(REGEX_LABELS)].copy()


def _subset_bert_scope(df: pd.DataFrame) -> pd.DataFrame:
    scoped = _subset_by_complexity(df, "bert")
    if len(scoped):
        return scoped
    label_col = get_settings().label_column
    # Fallback: exclude LLM-only rare labels
    return df[~df[label_col].astype(str).isin({"Workflow Error", "Deprecation Warning"})].copy()


def eval_regex(df: pd.DataFrame) -> Dict:
    """Score regex on its intended scope; also record full-set reference metrics."""
    clf = RegexClassifier()
    text_col = get_settings().text_column
    label_col = get_settings().label_column

    def _run(frame: pd.DataFrame):
        y_true, y_pred, latencies = [], [], []
        matched = 0
        for _, row in frame.iterrows():
            start = time.perf_counter()
            result = clf.classify(str(row[text_col]))
            latencies.append((time.perf_counter() - start) * 1000.0)
            pred = result.category if result.category else "Unclassified"
            if result.category:
                matched += 1
            y_true.append(str(row[label_col]))
            y_pred.append(pred)
        return y_true, y_pred, latencies, matched

    scoped = _subset_regex_scope(df)
    y_true, y_pred, latencies, matched = _run(scoped)
    primary = _metrics_bundle(
        "regex_only",
        y_true,
        y_pred,
        latencies,
        extra={
            "eval_scope": "regex_applicable_subset",
            "scope_note": (
                "Evaluated only on logs tagged complexity=regex "
                "(fixed template patterns Regex is designed to handle)."
            ),
            "regex_match_rate": matched / len(scoped) if len(scoped) else 0.0,
        },
    )

    # Full-set reference (explains why accuracy looks low if scored on all classes)
    yf, pf, lf, mf = _run(df)
    primary["full_test_set_reference"] = _metrics_bundle(
        "regex_only_full_test",
        yf,
        pf,
        lf,
        extra={
            "eval_scope": "full_test_set",
            "regex_match_rate": mf / len(df) if len(df) else 0.0,
            "note": (
                "Low accuracy here is expected: Regex abstains (Unclassified) "
                "on non-template logs and is not meant to classify all classes."
            ),
        },
    )
    return primary


def eval_legacy(df: pd.DataFrame) -> Dict:
    """Score ST+LR on bert-complexity scope; keep full-set reference."""
    clf = LegacySTLRClassifier()
    if not clf.is_available:
        return {"model": "sentence_transformer_lr", "status": "skipped", "reason": "model_unavailable"}

    text_col = get_settings().text_column
    label_col = get_settings().label_column

    def _run(frame: pd.DataFrame):
        y_true, y_pred, latencies = [], [], []
        for _, row in frame.iterrows():
            start = time.perf_counter()
            result = clf.classify(str(row[text_col]))
            latencies.append((time.perf_counter() - start) * 1000.0)
            y_true.append(str(row[label_col]))
            y_pred.append(result.category)
        return y_true, y_pred, latencies

    scoped = _subset_bert_scope(df)
    y_true, y_pred, latencies = _run(scoped)
    primary = _metrics_bundle(
        "sentence_transformer_lr",
        y_true,
        y_pred,
        latencies,
        extra={
            "eval_scope": "bert_complexity_subset",
            "scope_note": (
                "Evaluated on logs tagged complexity=bert "
                "(semantic classes ST+LR was trained to handle)."
            ),
        },
    )
    yf, pf, lf = _run(df)
    primary["full_test_set_reference"] = _metrics_bundle(
        "sentence_transformer_lr_full_test",
        yf,
        pf,
        lf,
        extra={"eval_scope": "full_test_set"},
    )
    return primary


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
    return _metrics_bundle(
        "fine_tuned_bert",
        y_true,
        y_pred,
        latencies,
        extra={"eval_scope": "full_test_set"},
    )


def eval_llm(df: pd.DataFrame, max_samples: int = 20) -> Dict:
    """Evaluate LLM on a small capped subset to control cost."""
    clf = LLMClassifier()
    if not clf.is_configured:
        return {
            "model": "llm_only",
            "status": "skipped",
            "reason": "GROQ_API_KEY not configured",
        }
    # Prefer LLM-tagged rows when available, else first N rows
    llm_rows = _subset_by_complexity(df, "llm")
    sample = llm_rows if len(llm_rows) else df.head(max_samples)
    if len(sample) > max_samples:
        sample = sample.head(max_samples)

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
    bundle = _metrics_bundle(
        "llm_only",
        y_true,
        y_pred,
        latencies,
        extra={"eval_scope": "llm_subset_or_capped"},
    )
    bundle["estimated_llm_cost_usd"] = cost
    bundle["total_tokens"] = tokens
    bundle["note"] = f"Evaluated on {len(sample)} rows only to limit API cost."
    return bundle


def eval_hybrid(df: pd.DataFrame) -> Dict:
    router = HybridRouter()
    text_col = get_settings().text_column
    label_col = get_settings().label_column
    y_true, y_pred, latencies = [], [], []
    method_counts: Dict[str, int] = {}
    llm_count = 0
    for _, row in df.iterrows():
        source = row["source"] if "source" in df.columns else None
        start = time.perf_counter()
        result = router.classify(
            str(row[text_col]),
            source=None if pd.isna(source) else str(source),
        )
        latencies.append(result.latency_ms or ((time.perf_counter() - start) * 1000.0))
        y_true.append(str(row[label_col]))
        y_pred.append(result.category)
        method_counts[result.classification_method] = (
            method_counts.get(result.classification_method, 0) + 1
        )
        if result.llm_invoked:
            llm_count += 1
    bundle = _metrics_bundle(
        "hybrid_router",
        y_true,
        y_pred,
        latencies,
        extra={"eval_scope": "full_test_set"},
    )
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
                    "Eval Scope": "-",
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
                "Eval Scope": item.get("eval_scope", "full_test_set"),
                "n_samples": item.get("n_samples"),
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

    lines = [
        "# Model Comparison Report",
        "",
        "Generated from actual evaluation runs. Skipped models are listed explicitly.",
        "",
        "Specialist models (Regex, ST+LR) are scored on their **intended scope**.",
        "Full-test-set reference metrics are stored in `model_comparison.json`.",
        "",
        "| Model | Eval Scope | n | Accuracy | Macro F1 | Weighted F1 | Avg Latency (ms) | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['Model']} | {row['Eval Scope']} | {row.get('n_samples', '-')} | "
            f"{row['Accuracy']} | {row['Macro F1']} | {row['Weighted F1']} | "
            f"{row['Avg Latency (ms)']} | {row['Status']} |"
        )

    # Add full-set reference rows for transparency
    lines.extend(["", "## Full test-set reference (specialists)", ""])
    lines.append(
        "| Model | Accuracy | Macro F1 | Weighted F1 | Note |"
    )
    lines.append("|---|---:|---:|---:|---|")
    for item in results:
        ref = item.get("full_test_set_reference")
        if not ref:
            continue
        note = ref.get("note") or "Reference only; not the primary score."
        lines.append(
            f"| {item['model']} | {round(ref['accuracy'], 4)} | "
            f"{round(ref['f1_macro'], 4)} | {round(ref['f1_weighted'], 4)} | {note} |"
        )

    report_path = out_dir / "model_comparison.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote comparison report to %s", report_path)
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    include = "--include-llm" in sys.argv
    main(include_llm=include)
