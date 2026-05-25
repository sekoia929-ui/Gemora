"""
src/fhir/mapper.py
Terminology mapping layer — resolves raw NER text to coded concepts.

Production note: This stub uses a local lookup table.
Replace with UMLS API, BioPortal, or a locally hosted SNOMED/ICD-10
terminological server (e.g., Ontoserver, HAPI FHIR Terminology Server).
"""

from __future__ import annotations
from src.models.clinical import (
    ClinicalEntity, ClinicalEntityType, TerminologySystem
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Stub vocabulary ───────────────────────────────────────────────────────────
# Format: normalized_text → (code, system, display)
# In production this hits a terminology server, not a dict.

_CONDITION_MAP: dict[str, tuple[str, TerminologySystem, str]] = {
    "diabetes mellitus":    ("44054006",   TerminologySystem.SNOMED, "Diabetes mellitus"),
    "hypertension":         ("38341003",   TerminologySystem.SNOMED, "Hypertensive disorder"),
    "chest pain":           ("29857009",   TerminologySystem.SNOMED, "Chest pain"),
    "pneumonia":            ("233604007",  TerminologySystem.SNOMED, "Pneumonia"),
    "myocardial infarction":("22298006",   TerminologySystem.SNOMED, "Myocardial infarction"),
    "asthma":               ("195967001",  TerminologySystem.SNOMED, "Asthma"),
    "anemia":               ("271737000",  TerminologySystem.SNOMED, "Anemia"),
}

_MEDICATION_MAP: dict[str, tuple[str, TerminologySystem, str]] = {
    "metformin":    ("860975",  TerminologySystem.RXNORM, "Metformin"),
    "lisinopril":   ("29046",   TerminologySystem.RXNORM, "Lisinopril"),
    "atorvastatin": ("83367",   TerminologySystem.RXNORM, "Atorvastatin"),
    "aspirin":      ("1191",    TerminologySystem.RXNORM, "Aspirin"),
    "amoxicillin":  ("723",     TerminologySystem.RXNORM, "Amoxicillin"),
}

_PROCEDURE_MAP: dict[str, tuple[str, TerminologySystem, str]] = {
    "ecg":             ("29303009",  TerminologySystem.SNOMED, "Electrocardiogram"),
    "electrocardiogram":("29303009", TerminologySystem.SNOMED, "Electrocardiogram"),
    "chest x-ray":     ("399208008", TerminologySystem.SNOMED, "Chest X-ray"),
    "blood culture":   ("30088009",  TerminologySystem.SNOMED, "Blood culture"),
}

_TYPE_MAPS = {
    ClinicalEntityType.CONDITION:  _CONDITION_MAP,
    ClinicalEntityType.MEDICATION: _MEDICATION_MAP,
    ClinicalEntityType.PROCEDURE:  _PROCEDURE_MAP,
}


def _normalize(text: str) -> str:
    return text.lower().strip()


def map_entity(entity: ClinicalEntity) -> ClinicalEntity:
    """
    Attempt to resolve entity to a standard code.
    Sets entity.code, entity.code_system, entity.display in-place.
    Reduces confidence if no mapping found.
    """
    lookup = _TYPE_MAPS.get(entity.entity_type)
    if lookup is None:
        return entity

    key = _normalize(entity.raw_text)
    match = lookup.get(key)

    if match:
        entity.code, entity.code_system, entity.display = match
        logger.debug("Mapped '%s' → %s:%s", entity.raw_text, entity.code_system, entity.code)
    else:
        # Partial match: check if any key is a substring
        for candidate_key, candidate_val in lookup.items():
            if candidate_key in key or key in candidate_key:
                entity.code, entity.code_system, entity.display = candidate_val
                entity.confidence *= 0.75  # penalise fuzzy match
                logger.debug(
                    "Fuzzy mapped '%s' → %s:%s (confidence reduced)",
                    entity.raw_text, entity.code_system, entity.code
                )
                return entity

        logger.warning("No terminology mapping for '%s' (%s)", entity.raw_text, entity.entity_type)
        entity.confidence *= 0.50  # unmapped entities are less reliable

    return entity


def map_entities(entities: list[ClinicalEntity]) -> list[ClinicalEntity]:
    return [map_entity(e) for e in entities]
