# Model Comparison Report

Generated from actual evaluation runs. Skipped models are listed explicitly.

Specialist models (Regex, ST+LR) are scored on their **intended scope**.
Full-test-set reference metrics are stored in `model_comparison.json`.

| Model | Eval Scope | n | Accuracy | Macro F1 | Weighted F1 | Avg Latency (ms) | Status |
|---|---|---:|---:|---:|---:|---:|---|
| regex_only | regex_applicable_subset | 58 | 1.0 | 1.0 | 1.0 | 0.087 | ok |
| sentence_transformer_lr | bert_complexity_subset | 281 | 0.9822 | 0.8058 | 0.9851 | 253.798 | ok |
| fine_tuned_bert | full_test_set | 340 | 0.9971 | 0.8726 | 0.9956 | 313.944 | ok |
| llm_only | - | - | None | None | None | None | skipped: Pass --include-llm to evaluate (uses API credits) |
| hybrid_router | full_test_set | 340 | 0.9676 | 0.948 | 0.9669 | 581.684 | ok |

## Full test-set reference (specialists)

| Model | Accuracy | Macro F1 | Weighted F1 | Note |
|---|---:|---:|---:|---|
| regex_only | 0.1706 | 0.2222 | 0.1706 | Low accuracy here is expected: Regex abstains (Unclassified) on non-template logs and is not meant to classify all classes. |
| sentence_transformer_lr | 0.8118 | 0.4682 | 0.7508 | Reference only; not the primary score. |
