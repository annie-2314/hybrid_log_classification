"""Training hyperparameters for fine-tuned BERT/DistilBERT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

from app.core.config import get_settings


@dataclass
class BertTrainConfig:
    model_name: str
    max_length: int
    batch_size: int
    learning_rate: float
    num_epochs: int
    weight_decay: float
    warmup_ratio: float
    early_stopping_patience: int
    seed: int
    output_dir: str
    train_file: str
    val_file: str
    test_file: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_train_config() -> BertTrainConfig:
    settings = get_settings()
    return BertTrainConfig(
        model_name=settings.bert_model_name,
        max_length=settings.bert_max_length,
        batch_size=settings.bert_batch_size,
        learning_rate=settings.bert_learning_rate,
        num_epochs=settings.bert_num_epochs,
        weight_decay=settings.bert_weight_decay,
        warmup_ratio=settings.bert_warmup_ratio,
        early_stopping_patience=settings.bert_early_stopping_patience,
        seed=settings.bert_seed,
        output_dir=settings.bert_model_dir,
        train_file=f"{settings.data_splits_dir}/train.csv",
        val_file=f"{settings.data_splits_dir}/val.csv",
        test_file=f"{settings.data_splits_dir}/test.csv",
    )
