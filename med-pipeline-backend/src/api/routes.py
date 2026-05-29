"""
src/api/routes.py
FastAPI router — now includes Claude AI reasoning in response.
"""

from __future__ import annotations
import asyncio
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from src.models.clinical import PipelineRequest, PipelineResponse, ClinicalReasoningSummary
from src.pipeline.processor import run_pipeline_async
from src.utils.audit import log_pipeline_run
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["pipeline"])


@router.post(
    "/process",
    response_model=PipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Process clinical text → FHIR Bundle + AI Reasoning",
)
async def process_text(request: PipelineRequest) -> PipelineResponse:
    try:
        result = await run_pipeline_async(request)
    except Exception as exc:
        logger.exception("Pipeline error for patient=%s: %s", request.patient_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pipeline execution failed. Check logs for details.",
        )

    asyncio.create_task(log_pipeline_run(
        request_id=result.request_id,
        patient_id=result.patient_id,
        entity_count=len(result.entities),
        confidence=result.pipeline_confidence,
        latency_ms=result.latency_ms,
        audit_id=result.audit_id,
    ))

    # Build reasoning summary for response
    reasoning_summary = None
    if result.reasoning:
        reasoning_summary = ClinicalReasoningSummary(
            summary=result.reasoning.summary,
            urgent_flags=result.reasoning.urgent_flags,
            differentials=result.reasoning.differentials,
            overall_severity=result.reasoning.overall_severity,
            confidence=result.reasoning.confidence,
        )

    return PipelineResponse(
        request_id=result.request_id,
        patient_id=result.patient_id,
        fhir_bundle=result.fhir_bundle,
        entity_count=len(result.entities),
        pipeline_confidence=result.pipeline_confidence,
        audit_id=result.audit_id,
        reasoning=reasoning_summary,
    )


@router.post(
    "/upload",
    response_model=PipelineResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload PDF or text file → FHIR Bundle + AI Reasoning",
)
async def upload_document(
    patient_id: str,
    source_system: str = "FILE-UPLOAD",
    encounter_id: str | None = None,
    file: UploadFile = File(...),
) -> PipelineResponse:
    MAX_BYTES = 10 * 1024 * 1024
    raw = await file.read()

    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File exceeds 10 MB limit.")

    content_type = file.content_type or ""
    if "pdf" in content_type or file.filename.endswith(".pdf"):
        text = _extract_pdf_text(raw)
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Could not decode file as UTF-8.")

    return await process_text(PipelineRequest(
        text=text, patient_id=patient_id,
        source_system=source_system, encounter_id=encounter_id,
    ))


def _extract_pdf_text(raw: bytes) -> str:
    try:
        import pdfplumber, io
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages).strip()
    except ImportError:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Install pdfplumber for PDF support.")
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"PDF extraction failed: {exc}")
