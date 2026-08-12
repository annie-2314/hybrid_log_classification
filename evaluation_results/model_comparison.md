# Model Comparison Report

Generated from actual evaluation runs. Skipped models are listed explicitly.

| Model | Accuracy | Macro F1 | Weighted F1 | Avg Latency (ms) | Status |
|---|---:|---:|---:|---:|---|
| regex_only | 0.1706 | 0.2222 | 0.1706 | 0.012 | ok |
| sentence_transformer_lr | 0.8118 | 0.4682 | 0.7508 | 27.649 | ok |
| fine_tuned_bert | 0.9971 | 0.8726 | 0.9956 | 72.796 | ok |
| llm_only | None | None | None | None | skipped: Pass --include-llm to evaluate (uses API credits) |
| hybrid_router | 0.9676 | 0.9488 | 0.9666 | 144.369 | ok |
