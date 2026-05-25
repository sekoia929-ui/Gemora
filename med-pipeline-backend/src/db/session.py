"""
src/db/session.py
Async PostgreSQL session management using asyncpg.
"""

from __future__ import annotations
import os
import asyncpg
from src.utils.logging import get_logger

logger = get_logger(__name__)

_pool: asyncpg.Pool | None = None


async def init_db() -> None:
    global _pool
    dsn = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/medpipeline")
    _pool = await asyncpg.create_pool(dsn=dsn, min_size=2, max_size=10)
    logger.info("DB pool initialised")

    await _pool.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_audit (
            id           UUID PRIMARY KEY,
            request_id   UUID NOT NULL,
            patient_id   TEXT NOT NULL,
            entity_count INTEGER,
            confidence   NUMERIC(5,4),
            latency_ms   NUMERIC(10,2),
            fhir_bundle  JSONB,
            created_at   TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_audit_patient ON pipeline_audit(patient_id);
        CREATE INDEX IF NOT EXISTS idx_audit_created ON pipeline_audit(created_at);
    """)
    logger.info("Schema ensured")


async def get_db_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB pool not initialised. Call init_db() at startup.")
    return _pool


async def close_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("DB pool closed")
