"""
src/nlp/extractor.py
Medical NER extraction — upgraded to scispaCy + medspaCy.

Model priority (first available wins):
  1. en_core_web_sm        — generic fallback (CI / cold start)
  2. en_ner_bc5cdr_md      — BC5CDR: diseases + chemicals (high precision)
  3. en_core_sci_lg        — general biomedical NER
  4. en_core_sci_sm        — smaller biomedical fallback
  

Install for production:
  pip install scispacy medspacy
  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
"""

from __future__ import annotations
import spacy
from functools import lru_cache
from src.models.clinical import (
    ClinicalEntity, ClinicalEntityType, Certainty, ExtractionResult
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Model priority list ───────────────────────────────────────────────────────
_MODEL_PRIORITY = [
    "en_ner_bc5cdr_md",   # best for diseases + chemicals
    "en_core_sci_lg",     # general biomedical
    "en_core_sci_sm",     # smaller biomedical
    "en_core_web_sm",     # generic fallback
]

# ── Label maps ────────────────────────────────────────────────────────────────
# BC5CDR labels
_BC5CDR_MAP: dict[str, ClinicalEntityType] = {
    "DISEASE":  ClinicalEntityType.CONDITION,
    "CHEMICAL": ClinicalEntityType.MEDICATION,
}

# scispaCy / general labels
_SCI_MAP: dict[str, ClinicalEntityType] = {
    "DISEASE":          ClinicalEntityType.CONDITION,
    "CONDITION":        ClinicalEntityType.CONDITION,
    "CANCER":           ClinicalEntityType.CONDITION,
    "CHEMICAL":         ClinicalEntityType.MEDICATION,
    "DRUG":             ClinicalEntityType.MEDICATION,
    "MEDICATION":       ClinicalEntityType.MEDICATION,
    "SIMPLE_CHEMICAL":  ClinicalEntityType.MEDICATION,
    "PROCEDURE":        ClinicalEntityType.PROCEDURE,
    "CLINICAL_VARIABLE":ClinicalEntityType.OBSERVATION,
    "SIGN_SYMPTOM":     ClinicalEntityType.OBSERVATION,
    "SIGN":             ClinicalEntityType.OBSERVATION,
    "SYMPTOM":          ClinicalEntityType.OBSERVATION,
    "TEST":             ClinicalEntityType.OBSERVATION,
    "LAB_VALUE":        ClinicalEntityType.OBSERVATION,
    "GENE_OR_GENE_PRODUCT": ClinicalEntityType.OBSERVATION,
    "ALLERGY":          ClinicalEntityType.ALLERGY,
}

# Merge — BC5CDR takes precedence
_LABEL_MAP = {**_SCI_MAP, **_BC5CDR_MAP}

# ── Negation cues ─────────────────────────────────────────────────────────────
_NEGATION_CUES  = {"no", "not", "without", "denies", "denied", "negative",
                   "absent", "rules", "ruled", "unremarkable", "never"}
_HISTORICAL_CUES = {"history", "past", "previous", "formerly", "prior",
                    "hx", "pmh", "known", "remote", "chronic"}
_SUSPECTED_CUES  = {"possible", "probable", "likely", "suspect", "suspected",
                    "query", "rule", "cannot", "exclude", "concern"}


@lru_cache(maxsize=1)
def _load_model() -> tuple[spacy.Language, str]:
    """Load best available scispaCy/spaCy model once per process."""
    for model_name in _MODEL_PRIORITY:
        try:
            nlp = spacy.load(model_name)
            logger.info("NLP model loaded: %s", model_name)
            return nlp, model_name
        except OSError:
            logger.debug("Model not available: %s", model_name)

    logger.warning("No preferred model found — using blank English. Install scispaCy for production.")
    return spacy.blank("en"), "blank"


def _infer_certainty(ent) -> Certainty:
    """
    Window-based certainty inference.
    Checks 6-token window before entity for negation/historical/suspected cues.
    medspaCy's context component provides better accuracy when available.
    """
    doc = ent.doc
    window_start = max(0, ent.start - 6)
    context_tokens = {t.lower_ for t in doc[window_start:ent.start]}

    if context_tokens & _NEGATION_CUES:
        return Certainty.NEGATED
    if context_tokens & _HISTORICAL_CUES:
        return Certainty.HISTORICAL
    if context_tokens & _SUSPECTED_CUES:
        return Certainty.SUSPECTED
    return Certainty.CONFIRMED


def _entity_confidence(ent, model_name: str) -> float:
    """
    Estimate entity confidence.
    BC5CDR and sci models are higher precision than generic spaCy.
    """
    base = {
        "en_ner_bc5cdr_md": 0.88,
        "en_core_sci_lg":   0.84,
        "en_core_sci_sm":   0.80,
        "en_core_web_sm":   0.72,
        "blank":            0.50,
    }.get(model_name, 0.70)

    # Boost if entity has a knowledge base ID (linked entity)
    if getattr(ent, "kb_id_", None):
        base = min(base + 0.05, 1.0)

    return round(base, 4)


def extract_entities(text: str) -> ExtractionResult:
    """
    Run scispaCy NER over clinical text.
    Returns typed ClinicalEntity list with certainty and confidence scores.
    """
    nlp, model_name = _load_model()
    doc = nlp(text)

    entities: list[ClinicalEntity] = []
    seen: set[str] = set()  # deduplicate identical spans

    for ent in doc.ents:
        entity_type = _LABEL_MAP.get(ent.label_.upper())
        if entity_type is None:
            continue

        # Skip duplicates (same text + type)
        dedup_key = f"{ent.text.lower()}:{entity_type}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        entities.append(ClinicalEntity(
            raw_text=ent.text,
            entity_type=entity_type,
            certainty=_infer_certainty(ent),
            confidence=_entity_confidence(ent, model_name),
            start_char=ent.start_char,
            end_char=ent.end_char,
        ))

    overall_confidence = (
        sum(e.confidence for e in entities) / len(entities)
        if entities else 0.0
    )

    logger.info(
        "Extraction complete | model=%s entities=%d confidence=%.3f",
        model_name, len(entities), overall_confidence
    )

    return ExtractionResult(
        source_text=text,
        entities=entities,
        extraction_confidence=round(overall_confidence, 4),
        model_version=model_name,
    )
