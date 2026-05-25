"""
src/fhir/serializers.py
FHIR R4 resource builders.

Each serializer returns a valid FHIR R4 dict.
Validation is handled downstream by fhir.resources or fhirclient.
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from src.models.clinical import (
    ClinicalEntity, ClinicalEntityType, Certainty
)
from src.nlp.context import ContextualNote
from src.utils.logging import get_logger

logger = get_logger(__name__)

_NOW = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _coding(entity: ClinicalEntity) -> list[dict]:
    if entity.code:
        return [{
            "system":  entity.code_system.value if entity.code_system else "unknown",
            "code":    entity.code,
            "display": entity.display or entity.raw_text,
        }]
    return [{"display": entity.raw_text}]


def _patient_ref(patient_id: str) -> dict:
    return {"reference": f"Patient/{patient_id}"}


def _certainty_to_verification_status(certainty: Certainty) -> dict:
    status_map = {
        Certainty.CONFIRMED:  ("confirmed",  "http://terminology.hl7.org/CodeSystem/condition-ver-status"),
        Certainty.SUSPECTED:  ("provisional","http://terminology.hl7.org/CodeSystem/condition-ver-status"),
        Certainty.NEGATED:    ("refuted",    "http://terminology.hl7.org/CodeSystem/condition-ver-status"),
        Certainty.HISTORICAL: ("confirmed",  "http://terminology.hl7.org/CodeSystem/condition-ver-status"),
    }
    code, system = status_map.get(certainty, ("unknown", ""))
    return {"coding": [{"system": system, "code": code, "display": code.capitalize()}]}


# ── Resource builders ─────────────────────────────────────────────────────────

def build_condition(
    entity: ClinicalEntity,
    patient_id: str,
    note: ContextualNote | None = None,
    encounter_id: str | None = None,
) -> dict:
    resource: dict = {
        "resourceType": "Condition",
        "id":           str(uuid.uuid4()),
        "meta":         {"lastUpdated": _NOW()},
        "clinicalStatus": {
            "coding": [{
                "system":  "http://terminology.hl7.org/CodeSystem/condition-clinical",
                "code":    "active" if entity.certainty != Certainty.HISTORICAL else "resolved",
                "display": "Active" if entity.certainty != Certainty.HISTORICAL else "Resolved",
            }]
        },
        "verificationStatus": _certainty_to_verification_status(entity.certainty),
        "code":    {"coding": _coding(entity), "text": entity.raw_text},
        "subject": _patient_ref(patient_id),
        "recordedDate": _NOW(),
        "extension": [{
            "url":   "http://example.org/fhir/StructureDefinition/nlp-confidence",
            "valueDecimal": round(entity.confidence, 4),
        }],
    }
    if encounter_id:
        resource["encounter"] = {"reference": f"Encounter/{encounter_id}"}
    if note and note.temporal_hint:
        resource["note"] = [{"text": f"temporal-context: {note.temporal_hint}"}]
    return resource


def build_medication_statement(
    entity: ClinicalEntity,
    patient_id: str,
    encounter_id: str | None = None,
) -> dict:
    return {
        "resourceType":  "MedicationStatement",
        "id":            str(uuid.uuid4()),
        "meta":          {"lastUpdated": _NOW()},
        "status":        "recorded",
        "medicationCodeableConcept": {
            "coding": _coding(entity),
            "text":   entity.raw_text,
        },
        "subject":       _patient_ref(patient_id),
        "dateAsserted":  _NOW(),
        "extension": [{
            "url":          "http://example.org/fhir/StructureDefinition/nlp-confidence",
            "valueDecimal": round(entity.confidence, 4),
        }],
    }


def build_procedure(
    entity: ClinicalEntity,
    patient_id: str,
    encounter_id: str | None = None,
) -> dict:
    return {
        "resourceType": "Procedure",
        "id":           str(uuid.uuid4()),
        "meta":         {"lastUpdated": _NOW()},
        "status":       "completed",
        "code":         {"coding": _coding(entity), "text": entity.raw_text},
        "subject":      _patient_ref(patient_id),
        "performedDateTime": _NOW(),
        "extension": [{
            "url":          "http://example.org/fhir/StructureDefinition/nlp-confidence",
            "valueDecimal": round(entity.confidence, 4),
        }],
    }


def build_observation(
    entity: ClinicalEntity,
    patient_id: str,
    note: ContextualNote | None = None,
    encounter_id: str | None = None,
) -> dict:
    obs: dict = {
        "resourceType": "Observation",
        "id":           str(uuid.uuid4()),
        "meta":         {"lastUpdated": _NOW()},
        "status":       "preliminary",
        "code":         {"coding": _coding(entity), "text": entity.raw_text},
        "subject":      _patient_ref(patient_id),
        "effectiveDateTime": _NOW(),
        "extension": [{
            "url":          "http://example.org/fhir/StructureDefinition/nlp-confidence",
            "valueDecimal": round(entity.confidence, 4),
        }],
    }
    if note and note.severity_score is not None:
        obs["interpretation"] = [{
            "coding": [{
                "system":  "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                "code":    "A",
                "display": f"Severity {note.severity_score}",
            }]
        }]
    return obs


# ── Bundle ────────────────────────────────────────────────────────────────────

_SERIALIZER_MAP = {
    ClinicalEntityType.CONDITION:   build_condition,
    ClinicalEntityType.MEDICATION:  build_medication_statement,
    ClinicalEntityType.PROCEDURE:   build_procedure,
    ClinicalEntityType.OBSERVATION: build_observation,
    ClinicalEntityType.ALLERGY:     build_observation,   # simplification — use AllergyIntolerance in v2
}


def build_fhir_bundle(
    entities: list[ClinicalEntity],
    notes: list[ContextualNote | None],
    patient_id: str,
    encounter_id: str | None = None,
) -> dict:
    """
    Assemble a FHIR R4 transaction Bundle from entity list.
    Unknown entity types are skipped with a warning.
    """
    entries = []

    for entity, note in zip(entities, notes):
        builder = _SERIALIZER_MAP.get(entity.entity_type)
        if not builder:
            logger.warning("No serializer for entity type: %s", entity.entity_type)
            continue

        try:
            resource = builder(entity, patient_id, note, encounter_id)  # type: ignore[call-arg]
        except TypeError:
            resource = builder(entity, patient_id)  # fallback for builders without note

        entries.append({
            "fullUrl":  f"urn:uuid:{resource['id']}",
            "resource": resource,
            "request":  {"method": "POST", "url": resource["resourceType"]},
        })

    return {
        "resourceType": "Bundle",
        "id":           str(uuid.uuid4()),
        "type":         "transaction",
        "timestamp":    _NOW(),
        "total":        len(entries),
        "entry":        entries,
    }
