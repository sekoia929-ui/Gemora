"""
src/pipeline/processor.py
Orchestrates: NLP extraction → context enrichment → terminology mapping
              → FHIR serialization → confidence scoring.

This is the single composable entry point.  All stages are pure functions —
no side effects here; persistence and audit happen in the API layer.
"""

from __future__ import annotations
import uuid
import time
from dataclasses import dataclass, field
from src.models.clinical import PipelineRequest, PipelineResponse, ClinicalEntity
from src.nlp.extractor import extract_entities
from src.nlp.context import enrich_with_context, ContextualNote
from src.fhir.mapper import map_entities
from src.fhir.serializers import build_fhir_bundle
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Confidence scoring ────────────────────────────────────────────────────────

def _score_pipeline(
    entities: list[ClinicalEntity],
    extraction_confidence: float,
) -> float:
    """
    Aggregate confidence across pipeline stages.

    Formula (simple weighted mean — adjust weights per clinical context):
      pipeline_score = 0.4 * extraction_confidence
                     + 0.4 * mean(entity.confidence for mapped entities)
                     + 0.2 * mapped_ratio

    Returns 0.0 if no entities extracted.
    """
    if not entities:
        return 0.0

    mapped = [e for e in entities if e.code is not None]
    mapped_ratio = len(mapped) / len(entities)
    mean_entity_confidence = sum(e.confidence for e in entities) / len(entities)

    score = (
        0.4 * extraction_confidence
        + 0.4 * mean_entity_confidence
        + 0.2 * mapped_ratio
    )
    return round(min(score, 1.0), 4)


# ── Pipeline result container ─────────────────────────────────────────────────

@dataclass
class PipelineResult:
    request_id:          str
    patient_id:          str
    fhir_bundle:         dict
    entities:            list[ClinicalEntity]
    pipeline_confidence: float
    latency_ms:          float
    audit_id:            str = field(default_factory=lambda: str(uuid.uuid4()))


# ── Main processor ────────────────────────────────────────────────────────────

def run_pipeline(request: PipelineRequest) -> PipelineResult:
    """
    Synchronous pipeline execution.  Wrap in asyncio.to_thread if called
    from an async FastAPI handler.
    """
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())

    logger.info(
        "Pipeline start | request=%s patient=%s source=%s",
        request_id, request.patient_id, request.source_system
    )

    # ── Stage 1: NLP extraction ───────────────────────────────────────────────
    extraction = extract_entities(request.text)
    logger.debug("Stage 1 done | entities=%d confidence=%.3f",
                 len(extraction.entities), extraction.extraction_confidence)

    # ── Stage 2: Context enrichment ───────────────────────────────────────────
    notes: list[ContextualNote] = enrich_with_context(extraction)
    logger.debug("Stage 2 done | sections=%d", len([n for n in notes if n.section]))

    # ── Stage 3: Terminology mapping ──────────────────────────────────────────
    mapped_entities = map_entities(extraction.entities)
    mapped_count = sum(1 for e in mapped_entities if e.code)
    logger.debug("Stage 3 done | mapped=%d/%d", mapped_count, len(mapped_entities))

    # ── Stage 4: FHIR serialization ───────────────────────────────────────────
    fhir_bundle = build_fhir_bundle(
        entities=mapped_entities,
        notes=notes,
        patient_id=request.patient_id,
        encounter_id=request.encounter_id,
    )
    logger.debug("Stage 4 done | bundle_entries=%d", fhir_bundle["total"])

    # ── Stage 5: Confidence scoring ───────────────────────────────────────────
    pipeline_confidence = _score_pipeline(mapped_entities, extraction.extraction_confidence)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info(
        "Pipeline complete | request=%s confidence=%.3f latency=%.1fms",
        request_id, pipeline_confidence, latency_ms
    )

    return PipelineResult(
        request_id=request_id,
        patient_id=request.patient_id,
        fhir_bundle=fhir_bundle,
        entities=mapped_entities,
        pipeline_confidence=pipeline_confidence,
        latency_ms=latency_ms,
    )
