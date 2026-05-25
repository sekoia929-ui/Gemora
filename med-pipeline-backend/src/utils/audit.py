"""
src/utils/audit.py
Immutable pipeline audit trail — writes to PostgreSQL.

All writes are fire-and-forget (asyncio.create_task) so they never
block the API response path.
"""

from __future__ import annotations
import uuid
from src.db.session import get_db_pool
from src.utils.logging import get_logger

logger = get_logger(__name__)


async def log_pipeline_run(
    *,
    request_id: str,
    patient_id: str,
    entity_count: int,
    confidence: float,
    latency_ms: float,
    audit_id: str,
    fhir_bundle: dict | None = None,
) -> None:
    """
    Persist an immutable audit record for a completed pipeline run.

    The fhir_bundle is stored as JSONB for traceability.
    In high-volume production, consider writing to a separate audit
    service or append-only S3 bucket instead of inline PostgreSQL writes.
    """
    try:
        pool = await get_db_pool()
        await pool.execute(
            """
            INSERT INTO pipeline_audit
                (id, request_id, patient_id, entity_count, confidence, latency_ms, fhir_bundle)
            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
            """,
            uuid.UUID(audit_id),
            uuid.UUID(request_id),
            patient_id,
            entity_count,
            confidence,
            latency_ms,
            str(fhir_bundle) if fhir_bundle else None,
        )
        logger.debug("Audit record written | audit_id=%s", audit_id)
    except Exception as exc:
        # Audit failures must never crash the pipeline
        logger.error("Audit write failed | audit_id=%s error=%s", audit_id, exc)
