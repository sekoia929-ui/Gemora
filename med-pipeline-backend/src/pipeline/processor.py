"""
src/pipeline/processor.py
Orchestrates all pipeline stages including Claude AI reasoning.
"""

from __future__ import annotations
import uuid
import time
import asyncio
from dataclasses import dataclass, field
from src.models.clinical import PipelineRequest, PipelineResponse, ClinicalEntity, ClinicalReasoningSummary
from src.nlp.extractor import extract_entities
from src.nlp.context import enrich_with_context, ContextualNote
from src.nlp.reasoner import get_clinical_reasoning, ClinicalReasoning
from src.fhir.mapper import map_entities
from src.fhir.serializers import build_fhir_bundle
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _score_pipeline(
    entities: list[ClinicalEntity],
    extraction_confidence: float,
    reasoning: ClinicalReasoning | None = None,
) -> float:
    if not entities:
        return 0.0

    mapped            = [e for e in entities if e.code is not None]
    mapped_ratio      = len(mapped) / len(entities)
    mean_entity_conf  = sum(e.confidence for e in entities) / len(entities)

    score = (
        0.35 * extraction_confidence
        + 0.35 * mean_entity_conf
        + 0.20 * mapped_ratio
        + 0.10 * (reasoning.confidence if reasoning else 0.0)
    )
    return round(min(score, 1.0), 4)


@dataclass
class PipelineResult:
    request_id:          str
    patient_id:          str
    fhir_bundle:         dict
    entities:            list[ClinicalEntity]
    pipeline_confidence: float
    latency_ms:          float
    reasoning:           ClinicalReasoning | None = None
    audit_id:            str = field(default_factory=lambda: str(uuid.uuid4()))


async def run_pipeline_async(request: PipelineRequest) -> PipelineResult:
    """
    Async pipeline — runs NLP synchronously in a thread,
    then calls Claude AI reasoning concurrently with FHIR build.
    """
    t0         = time.perf_counter()
    request_id = str(uuid.uuid4())

    logger.info("Pipeline start | request=%s patient=%s", request_id, request.patient_id)

    # Stage 1: NLP extraction (CPU-bound → thread)
    extraction = await asyncio.to_thread(extract_entities, request.text)
    logger.debug("Stage 1 | entities=%d confidence=%.3f", len(extraction.entities), extraction.extraction_confidence)

    # Stage 2: Context enrichment
    notes: list[ContextualNote] = await asyncio.to_thread(enrich_with_context, extraction)
    logger.debug("Stage 2 | context enriched")

    # Stage 3: Terminology mapping
    mapped_entities = await asyncio.to_thread(map_entities, extraction.entities)
    logger.debug("Stage 3 | mapped=%d/%d", sum(1 for e in mapped_entities if e.code), len(mapped_entities))

    # Stage 4 + 5: FHIR build and Claude reasoning — run concurrently
    fhir_task      = asyncio.to_thread(build_fhir_bundle, mapped_entities, notes, request.patient_id, request.encounter_id)
    reasoning_task = get_clinical_reasoning(request.text, mapped_entities)

    fhir_bundle, reasoning = await asyncio.gather(fhir_task, reasoning_task)
    logger.debug("Stage 4+5 | fhir_entries=%d reasoning=%s", fhir_bundle["total"], "ok" if reasoning else "skipped")

    # Stage 6: Confidence scoring
    pipeline_confidence = _score_pipeline(mapped_entities, extraction.extraction_confidence, reasoning)

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    logger.info("Pipeline complete | request=%s confidence=%.3f latency=%.1fms", request_id, pipeline_confidence, latency_ms)

    return PipelineResult(
        request_id=request_id,
        patient_id=request.patient_id,
        fhir_bundle=fhir_bundle,
        entities=mapped_entities,
        pipeline_confidence=pipeline_confidence,
        latency_ms=latency_ms,
        reasoning=reasoning,
    )


def run_pipeline(request: PipelineRequest) -> PipelineResult:
    """Sync wrapper for backwards compatibility with tests."""
    return asyncio.get_event_loop().run_until_complete(run_pipeline_async(request))
