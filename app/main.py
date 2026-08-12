"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm up classifiers once so /health does not race first imports."""
    logger = get_logger(__name__)
    from app.routing.router import get_router

    router = get_router()
    logger.info(
        "Startup ready | bert_loaded=%s legacy_st_lr=%s llm_configured=%s",
        router.bert.is_available,
        router.legacy.is_available,
        router.llm.is_configured,
    )
    yield


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
        lifespan=lifespan,
    )
    app.include_router(api_router)
    return app


app = create_app()
