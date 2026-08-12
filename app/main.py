"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    setup_logging()
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Intelligent Hybrid Log Classification & Routing System. "
            "Routes logs through Regex → Fine-tuned BERT → LLM fallback."
        ),
    )
    app.include_router(api_router)
    return app


app = create_app()
