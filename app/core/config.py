"""Centralized configuration loaded from YAML + environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "settings.yaml"


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None and value != "" else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None and value != "" else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class Settings(BaseModel):
    """Application settings with env overrides for secrets and tunables."""

    app_name: str = "IntelliLog AI"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    bert_confidence_threshold: float = 0.90
    enable_legacy_st_lr: bool = True
    legacy_st_lr_threshold: float = 0.50

    bert_model_name: str = "distilbert-base-uncased"
    bert_max_length: int = 128
    bert_batch_size: int = 16
    bert_learning_rate: float = 2.0e-5
    bert_num_epochs: int = 3
    bert_weight_decay: float = 0.01
    bert_warmup_ratio: float = 0.1
    bert_early_stopping_patience: int = 2
    bert_seed: int = 42
    bert_model_dir: str = "models/bert_classifier"

    data_raw_csv: str = "training/data/synthetic_logs.csv"
    data_splits_dir: str = "training/data/splits"
    data_test_size: float = 0.15
    data_val_size: float = 0.15
    random_seed: int = 42
    text_column: str = "log_message"
    label_column: str = "target_label"

    groq_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.1
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 30
    llm_input_cost_per_1m: float = 0.59
    llm_output_cost_per_1m: float = 0.79

    legacy_st_lr_model: str = "models/log_classifier.joblib"
    evaluation_dir: str = "evaluation_results"
    metrics_store: str = "evaluation_results/runtime_metrics.json"

    severity_map: Dict[str, str] = Field(default_factory=dict)

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else self.project_root / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once from YAML, then apply environment overrides."""
    raw = _load_yaml(DEFAULT_CONFIG_PATH)

    app = raw.get("app", {})
    routing = raw.get("routing", {})
    bert = raw.get("bert", {})
    data = raw.get("data", {})
    llm = raw.get("llm", {})
    paths = raw.get("paths", {})
    severity = raw.get("severity", {})

    return Settings(
        app_name=app.get("name", "IntelliLog AI"),
        app_version=app.get("version", "1.0.0"),
        log_level=os.getenv("LOG_LEVEL", app.get("log_level", "INFO")),
        bert_confidence_threshold=_env_float(
            "BERT_CONFIDENCE_THRESHOLD",
            float(routing.get("bert_confidence_threshold", 0.90)),
        ),
        enable_legacy_st_lr=_env_bool(
            "ENABLE_LEGACY_ST_LR",
            bool(routing.get("enable_legacy_st_lr", True)),
        ),
        legacy_st_lr_threshold=float(routing.get("legacy_st_lr_threshold", 0.50)),
        bert_model_name=os.getenv("BERT_MODEL_NAME", bert.get("model_name", "distilbert-base-uncased")),
        bert_max_length=int(bert.get("max_length", 128)),
        bert_batch_size=int(bert.get("batch_size", 16)),
        bert_learning_rate=float(bert.get("learning_rate", 2.0e-5)),
        bert_num_epochs=int(bert.get("num_epochs", 3)),
        bert_weight_decay=float(bert.get("weight_decay", 0.01)),
        bert_warmup_ratio=float(bert.get("warmup_ratio", 0.1)),
        bert_early_stopping_patience=int(bert.get("early_stopping_patience", 2)),
        bert_seed=_env_int("RANDOM_SEED", int(bert.get("seed", 42))),
        bert_model_dir=os.getenv("BERT_MODEL_DIR", bert.get("output_dir", "models/bert_classifier")),
        data_raw_csv=data.get("raw_csv", "training/data/synthetic_logs.csv"),
        data_splits_dir=data.get("splits_dir", "training/data/splits"),
        data_test_size=float(data.get("test_size", 0.15)),
        data_val_size=float(data.get("val_size", 0.15)),
        random_seed=_env_int("RANDOM_SEED", int(data.get("random_seed", 42))),
        text_column=data.get("text_column", "log_message"),
        label_column=data.get("label_column", "target_label"),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        llm_model=os.getenv("LLM_MODEL", llm.get("model", "llama-3.3-70b-versatile")),
        llm_temperature=float(llm.get("temperature", 0.1)),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", int(llm.get("max_retries", 3))),
        llm_timeout_seconds=_env_int(
            "LLM_TIMEOUT_SECONDS", int(llm.get("timeout_seconds", 30))
        ),
        llm_input_cost_per_1m=float(llm.get("input_cost_per_1m_tokens", 0.59)),
        llm_output_cost_per_1m=float(llm.get("output_cost_per_1m_tokens", 0.79)),
        legacy_st_lr_model=paths.get("legacy_st_lr_model", "models/log_classifier.joblib"),
        evaluation_dir=paths.get("evaluation_dir", "evaluation_results"),
        metrics_store=paths.get("metrics_store", "evaluation_results/runtime_metrics.json"),
        severity_map={str(k): str(v) for k, v in severity.items()},
    )
