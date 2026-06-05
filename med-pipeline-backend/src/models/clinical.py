"""
src/models/clinical.py
Domain models — typed contracts between pipeline stages.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────

class ClinicalEntityType(str, Enum):
    CONDITION   = "CONDITION"
    MEDICATION  = "MEDICATION"
    PROCEDURE   = "PROCEDURE"
    OBSERVATION = "OBSERVATION"
    ALLERGY     = "ALLERGY"


class Certainty(str, Enum):
    CONFIRMED  = "confirmed"
    SUSPECTED  = "suspected"
    NEGATED    = "negated"
    HISTORICAL = "historical"


class TerminologySystem(str, Enum):
    SNOMED = "http://snomed.info/sct"
    ICD10  = "http://hl7.org/fhir/sid/icd-10"
    RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
    LOINC  = "http://loinc.org"


# ── Core entity ───────────────────────────────────────────────────────────────

class ClinicalEntity(BaseModel):
    """A single extracted clinical concept before FHIR mapping."""

    raw_text: str                = Field(..., description="Original text span")
    entity_type: ClinicalEntityType
    certainty: Certainty         = Certainty.CONFIRMED
    confidence: float            = Field(..., ge=0.0, le=1.0)
    start_char: Optional[int]    = None
    end_char: Optional[int]      = None

    # populated after terminology mapping
    code: Optional[str]          = None
    code_system: Optional[TerminologySystem] = None
    display: Optional[str]       = None


class ExtractionResult(BaseModel):
    """Full output of the NLP extraction stage."""

    source_text: str
    entities: list[ClinicalEntity]
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str           = "spacy-med-ner-v1"


# ── Pipeline request / response ───────────────────────────────────────────────

class PipelineRequest(BaseModel):
    text: str          = Field(..., min_length=10, max_length=50_000)
    patient_id: str    = Field(..., description="External patient reference")
    source_system: str = Field(default="UNKNOWN")
    encounter_id: Optional[str] = None


class PipelineResponse(BaseModel):
    request_id: str
    patient_id: str
    fhir_bundle: dict          # validated FHIR R4 Bundle JSON
    entity_count: int
    pipeline_confidence: float
    audit_id: str
