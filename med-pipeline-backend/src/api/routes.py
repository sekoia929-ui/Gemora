"""
src/api/routes.py
FastAPI router — exposes the pipeline as REST endpoints.
"""

from __future__ import annotations
import asyncio
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from src.models.clinical import PipelineRequest, PipelineResponse
from src.pipeline.processor import run_pipeline
from src.db.session import get_db_pool
from src.utils.audit import log_pipeline_run
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["pipeline"])


# ── Text input ────────────────────────────────────────────────────────────────

@router.post(
    "/process",
    response_model=PipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Process clinical text → FHIR Bundle",
)
async def process_text(request: PipelineRequest) -> PipelineResponse:
    """
    Accepts raw clinical text and returns a validated FHIR R4 transaction Bundle.

    - **text**: clinical note, discharge summary, or any free-text block
    - **patient_id**: your system's patient reference
    - **source_system**: optional provenance tag (e.g. "EMR-EPIC", "PDF-SCAN")
    """
    try:
        result = await asyncio.to_thread(run_pipeline, request)
    except Exception as exc:
        logger.exception("Pipeline error for patient=%s: %s", request.patient_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pipeline execution failed. Check logs for details.",
        )

    # Persist audit record (non-blocking)
    asyncio.create_task(log_pipeline_run(
        request_id=result.request_id,
        patient_id=result.patient_id,
        entity_count=len(result.entities),
        confidence=result.pipeline_confidence,
        latency_ms=result.latency_ms,
        audit_id=result.audit_id,
    ))

    return PipelineResponse(
        request_id=result.request_id,
        patient_id=result.patient_id,
        fhir_bundle=result.fhir_bundle,
        entity_count=len(result.entities),
        pipeline_confidence=result.pipeline_confidence,
        audit_id=result.audit_id,
    )


# ── PDF / file upload ─────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=PipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload PDF or text file → FHIR Bundle",
)
async def upload_document(
    patient_id: str,
    source_system: str = "FILE-UPLOAD",
    encounter_id: str | None = None,
    file: UploadFile = File(...),
) -> PipelineResponse:
    """
    Upload a PDF or plain-text clinical document.

    - PDF text extraction requires `pdfplumber` to be installed.
    - Max file size: 10 MB (enforced by infrastructure; add middleware for prod).
    """
    MAX_BYTES = 10 * 1024 * 1024  # 10 MB
    raw = await file.read()

    if len(raw) > MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds 10 MB limit.",
        )

    content_type = file.content_type or ""

    if "pdf" in content_type or file.filename.endswith(".pdf"):
        text = _extract_pdf_text(raw)
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not decode file as UTF-8 text.",
            )

    request = PipelineRequest(
        text=text,
        patient_id=patient_id,
        source_system=source_system,
        encounter_id=encounter_id,
    )
    return await process_text(request)


def _extract_pdf_text(raw: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    try:
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages).strip()
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="PDF support requires pdfplumber. Install it with: pip install pdfplumber",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"PDF extraction failed: {exc}",
        )
