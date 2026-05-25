"""
med-pipeline-backend · src/app.py
FastAPI entry point — health, versioning, router registration.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as pipeline_router
from src.db.session import init_db
from src.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting med-pipeline-backend")
    await init_db()
    yield
    logger.info("Shutting down med-pipeline-backend")


app = FastAPI(
    title="Med Pipeline API",
    description="Unstructured clinical text → validated FHIR R4 resources",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline_router, prefix="/api/v1")


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "version": "0.1.0"}
