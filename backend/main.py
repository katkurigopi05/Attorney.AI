"""
Attorney.AI Backend — FastAPI Application Entry Point
"""
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config import settings
from api.routes import health, research, search, contract


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    logger.info(f"Starting Attorney.AI backend [{settings.app_env}]")
    logger.info(f"Qdrant: {settings.qdrant_url} → {settings.qdrant_collection}")
    logger.info(f"LLM model: {settings.openai_chat_model}")
    logger.info(f"Embedding model: {settings.openai_embedding_model}")
    yield
    logger.info("Attorney.AI backend shutting down.")


app = FastAPI(
    title="Attorney.AI API",
    description=(
        "Citation-first U.S. legal research RAG assistant. "
        "Every answer is grounded in authoritative legal sources. "
        "This is NOT legal advice."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(research.router, prefix="/api", tags=["Legal Research"])
app.include_router(search.router, prefix="/api", tags=["Search"])
app.include_router(contract.router, prefix="/api", tags=["Contract Review"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )
