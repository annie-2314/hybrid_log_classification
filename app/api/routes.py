"""API routes for IntelliLog AI."""

from __future__ import annotations

import io
from typing import List

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.monitoring.metrics import get_metrics_collector
from app.routing.router import get_router
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ClassificationResult,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
)
from app.services.batch import classify_dataframe

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    settings = get_settings()
    hybrid = get_router()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        bert_loaded=hybrid.bert.is_available,
        legacy_st_lr_loaded=hybrid.legacy.is_available,
        llm_configured=hybrid.llm.is_configured,
    )


@router.get("/metrics", response_model=MetricsResponse, tags=["system"])
def metrics() -> MetricsResponse:
    snapshot = get_metrics_collector().snapshot()
    return MetricsResponse(**snapshot)


@router.post("/predict", response_model=ClassificationResult, tags=["classification"])
def predict(payload: PredictRequest) -> ClassificationResult:
    return get_router().classify(payload.log_message, source=payload.source)


@router.post("/batch-predict", response_model=BatchPredictResponse, tags=["classification"])
def batch_predict(payload: BatchPredictRequest) -> BatchPredictResponse:
    hybrid = get_router()
    results: List[ClassificationResult] = []
    for item in payload.logs:
        results.append(hybrid.classify(item.log_message, source=item.source))
    avg_latency = (
        sum(r.latency_ms or 0.0 for r in results) / len(results) if results else 0.0
    )
    return BatchPredictResponse(
        results=results,
        total=len(results),
        avg_latency_ms=round(avg_latency, 3),
    )


@router.post("/batch-predict/csv", tags=["classification"])
async def batch_predict_csv(file: UploadFile = File(...)) -> StreamingResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        if "log_message" not in df.columns:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain a 'log_message' column.",
            )
        out_df = classify_dataframe(df, router=get_router())
        buffer = io.StringIO()
        out_df.to_csv(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=classified_logs.csv"},
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await file.close()


# Backward-compatible alias for the original /classify/ CSV upload endpoint.
@router.post("/classify/", tags=["classification", "legacy"])
async def classify_legacy(file: UploadFile = File(...)) -> StreamingResponse:
    return await batch_predict_csv(file)
