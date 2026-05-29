"""
src/models/clinical.py
Domain models — typed contracts between pipeline stages.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


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


class ClinicalEntity(BaseModel):
    raw_text:    str                = Field(..., description="Original text span")
    entity_type: ClinicalEntityType
    certainty:   Certainty          = Certainty.CONFIRMED
    confidence:  float              = Field(..., ge=0.0, le=1.0)
    start_char:  Optional[int]      = None
    end_char:    Optional[int]      = None
    code:        Optional[str]      = None
    code_system: Optional[TerminologySystem] = None
    display:     Optional[str]      = None


class ExtractionResult(BaseModel):
    source_text:           str
    entities:              list[ClinicalEntity]
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
    model_version:         str   = "spacy-med-ner-v1"


class PipelineRequest(BaseModel):
    text:          str           = Field(..., min_length=10, max_length=50_000)
    patient_id:    str           = Field(..., description="External patient reference")
    source_system: str           = Field(default="UNKNOWN")
    encounter_id:  Optional[str] = None


class ClinicalReasoningSummary(BaseModel):
    """Subset of ClinicalReasoning safe to expose in API response."""
    summary:          str
    urgent_flags:     list[str]
    differentials:    list[str]
    overall_severity: str
    confidence:       float


class PipelineResponse(BaseModel):
    request_id:          str
    patient_id:          str
    fhir_bundle:         dict
    entity_count:        int
    pipeline_confidence: float
    audit_id:            str
    reasoning:           Optional[ClinicalReasoningSummary] = None
