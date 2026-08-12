# IntelliLog AI — Intelligent Hybrid Log Classification & Routing System

Production-oriented upgrade of the hybrid NLP log classification project ([codebasics/project-nlp-log-classification](https://github.com/codebasics/project-nlp-log-classification)).

IntelliLog AI classifies software/system logs using an **intelligent router**:

1. **Regex** for fixed, predictable patterns  
2. **Fine-tuned Transformer (DistilBERT)** for ML-classifiable logs  
3. **LLM fallback (Groq)** when confidence is low or patterns are rare/complex  

Each prediction returns **category, severity, confidence, method, and explanation**, with **latency/cost monitoring**.

---

## Problem statement

Operational logs arrive from many systems with mixed regularity:

- Some messages follow rigid templates (easy for rules)
- Many require semantic understanding (better for supervised ML)
- A small tail of rare/complex messages lacks enough labels for reliable ML

Sending **every** log to an LLM is slow and expensive. A hybrid router keeps accuracy high while minimizing LLM usage.

---

## Why hybrid classification?

| Approach | Strength | Weakness |
|---|---|---|
| Regex | Fast, exact, free | Brittle on novel wording |
| Fine-tuned BERT | Strong on labeled patterns | Needs data; can be uncertain |
| LLM | Handles rare/complex text | Latency + API cost |

The router uses each method where it is strongest.

---

## Architecture

```text
User Log
   ↓
Preprocessing
   ↓
Intelligent Routing Layer
   ↓
 ┌──────────────┬────────────────────┬──────────────────┐
 ↓              ↓                    ↓
Regex      Fine-tuned BERT      LLM Fallback
 ↓              ↓                    ↓
Easy logs   ML-classifiable     Complex/uncertain logs
 └──────────────┬────────────────────┘
                ↓
        Confidence Evaluation
                ↓
        Final Classification
                ↓
 Category + Severity + Explanation
                ↓
      Evaluation / Monitoring
                ↓
 Metrics + Latency + Cost
```

### Detailed workflow

1. Normalize the log text (whitespace/control chars).
2. Try **Regex**. On match → return category with confidence `1.0`.
3. Else run **fine-tuned DistilBERT** and read max softmax probability.
4. If confidence ≥ `BERT_CONFIDENCE_THRESHOLD` (default `0.90`) → accept BERT.
5. Else call **LLM** with a structured JSON prompt; validate output.
6. Map category → severity; attach explanation + method + latency.
7. Record metrics (method counts, latency, LLM tokens/cost when available).

Legacy **SentenceTransformer + Logistic Regression** remains available as a fallback when the fine-tuned BERT checkpoint is not present (`ENABLE_LEGACY_ST_LR=true`).

---

## Dataset

Primary dataset (domain-specific logs):

- `training/data/synthetic_logs.csv` (from the original project)
- Columns: `timestamp`, `source`, `log_message`, `target_label`, `complexity`
- ~2.4k labeled logs across 9 categories (imbalanced; rare LLM-oriented classes exist)

If you need a larger generic text-classification corpus, you can swap in a [Kaggle text classification dataset](https://www.kaggle.com/search?q=text+classification), but labels/schema must be adapted. This repo’s default pipeline is built for the log dataset above.

Prepare reproducible splits:

```bash
python scripts/prepare_dataset.py
```

This writes `training/data/splits/{train,val,test}.csv` and `split_report.json` (class distribution + imbalance diagnostics, leakage-aware dedupe).

---

## Classifiers

### Regex

Patterns for predictable messages such as login/logout, backups, uploads, account creation (`app/classifiers/regex_classifier.py`).

### Fine-tuned BERT / DistilBERT

True Hugging Face sequence classification fine-tuning (not embeddings + logistic regression):

```bash
python training/train_bert.py
python training/evaluate_bert.py
```

Artifacts land in `models/bert_classifier/` (weights, tokenizer, `label_map.json`, metrics).

### LLM fallback

- Provider: Groq (`GROQ_API_KEY`)
- Prompt: `app/prompts/llm_classify.txt`
- Structured JSON validated by Pydantic
- Retries, timeout, and cost estimation from returned token usage

---

## Confidence-based routing

Configurable via env / `configs/settings.yaml`:

```bash
BERT_CONFIDENCE_THRESHOLD=0.90
```

- `confidence >= threshold` → accept BERT  
- `confidence < threshold` → LLM fallback  

---

## Evaluation methodology

```bash
python training/evaluate.py
# optional (uses API credits):
python training/evaluate.py --include-llm
```

Compares (when available):

- Regex only  
- SentenceTransformer + Logistic Regression  
- Fine-tuned BERT  
- LLM only (opt-in)  
- Hybrid router  

Metrics: Accuracy, Precision/Recall, Macro F1, Weighted F1, confusion matrix, average latency.  
Results: `evaluation_results/model_comparison.json` and `.md`.

**No fabricated metrics** — skipped models are marked `skipped` with a reason.

### Latest measured comparison (test split, n=340)

| Model | Accuracy | Macro F1 | Weighted F1 | Avg Latency (ms) |
|---|---:|---:|---:|---:|
| Regex only | 0.1706 | 0.2222 | 0.1706 | 0.012 |
| SentenceTransformer + LR | 0.8118 | 0.4682 | 0.7508 | 27.649 |
| Fine-tuned DistilBERT | 0.9971 | 0.8726 | 0.9956 | 72.796 |
| LLM only | skipped (opt-in with `--include-llm`) | | | |
| Hybrid router | 0.9676 | 0.9488 | 0.9666 | 144.369 |

Hybrid routing breakdown on the same test set: **regex 58**, **BERT 242**, **LLM fallback 40** (~11.8% LLM usage). Hybrid Macro F1 is higher than BERT-alone because rare classes benefit from LLM fallback, while average latency rises when LLM is invoked.

---

## FastAPI

```bash
uvicorn app.main:app --reload
# or legacy alias:
uvicorn server:app --reload
```

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Single log JSON classification |
| POST | `/batch-predict` | JSON list of logs |
| POST | `/batch-predict/csv` | CSV upload → CSV download |
| GET | `/health` | Health + model readiness |
| GET | `/metrics` | Runtime counts, latency, LLM cost |

### Example

Request:

```json
{
  "log_message": "Unauthorized access to data was attempted"
}
```

Response shape:

```json
{
  "category": "Security Alert",
  "severity": "HIGH",
  "confidence": 0.94,
  "classification_method": "fine_tuned_bert",
  "explanation": "Fine-tuned Transformer predicted 'Security Alert' with softmax confidence 0.940.",
  "request_id": "...",
  "latency_ms": 12.3,
  "llm_invoked": false,
  "routing_path": ["regex", "fine_tuned_bert"]
}
```

Interactive docs: `http://127.0.0.1:8000/docs`

---

## Batch processing

CSV input must include `log_message` (optional `source`).  
Output adds: `predicted_category`, `severity`, `confidence`, `classification_method`, `explanation`, `latency_ms`, `llm_invoked`.

---

## Cost & latency tracking

`GET /metrics` exposes:

- Total / Regex / BERT / Legacy / LLM request counts  
- Average latency  
- LLM fallback percentage  
- Estimated LLM cost (from provider token usage × configured rates)  

Cost is **only** computed when the LLM returns usage metadata.

---

## Project structure

```text
app/
  api/              # FastAPI routes
  classifiers/      # Regex, BERT, legacy ST+LR, LLM
  core/             # Config + logging
  monitoring/       # Metrics + structured events
  preprocessing/    # Normalize + dataset pipeline
  prompts/          # LLM prompt templates
  routing/          # Hybrid router
  schemas/          # Pydantic models
  services/         # Batch + severity
configs/settings.yaml
training/
  train_bert.py
  evaluate_bert.py
  evaluate.py
  config.py
  data/
models/
  log_classifier.joblib      # legacy ST+LR
  bert_classifier/           # fine-tuned transformer
tests/
scripts/prepare_dataset.py
Dockerfile
docker-compose.yml
```

---

## Installation

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# set GROQ_API_KEY in .env for LLM fallback
python scripts/prepare_dataset.py
python training/train_bert.py
```

---

## Docker

```bash
docker compose up --build
```

Provide secrets via `.env` / compose environment — they are **not** baked into the image.

---

## Tests

```bash
pytest -q
```

Covers regex, routing confidence thresholds, LLM validation paths, API validation, batch CSV, and dataset loading.

---

## Model comparison (interview talking point)

The hybrid design is useful because:

1. Regex absorbs high-volume template logs at near-zero cost.  
2. Fine-tuned BERT handles the bulk of semantic classes with measurable confidence.  
3. LLM is reserved for low-confidence / rare cases — reducing spend and latency vs LLM-only.  
4. Observability proves the savings via method counts + estimated cost.

Run evaluation after training to populate real numbers in `evaluation_results/`.

---

## Future improvements

- Active learning loop for low-confidence samples  
- Per-tenant routing thresholds  
- Streaming log ingestion  
- Stronger calibration (temperature scaling) for BERT confidence  
- Optional Langfuse tracing  

---

## Disclaimer

Original educational project copyright: Codebasics Inc / LearnerX Pvt Ltd.  
This upgrade is for learning and portfolio demonstration; validate independently before production use.
