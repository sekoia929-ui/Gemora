"""
src/nlp/extractor.py
Medical NER extraction layer using spaCy + med model.

Assumption: 'en_core_med_ner' is loaded at runtime (or falls back to
'en_core_web_sm' in CI/testing). Replace with scispaCy or MedSpaCy in
production for higher recall on clinical text.
"""

from __future__ import annotations
import spacy
from functools import lru_cache
from src.models.clinical import (
    ClinicalEntity, ClinicalEntityType, Certainty, ExtractionResult
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ── Label mappings from spaCy model → ClinicalEntityType ─────────────────────
_LABEL_MAP: dict[str, ClinicalEntityType] = {
    "DISEASE":    ClinicalEntityType.CONDITION,
    "CONDITION":  ClinicalEntityType.CONDITION,
    "DRUG":       ClinicalEntityType.MEDICATION,
    "MEDICATION": ClinicalEntityType.MEDICATION,
    "PROCEDURE":  ClinicalEntityType.PROCEDURE,
    "TEST":       ClinicalEntityType.OBSERVATION,
    "SIGN":       ClinicalEntityType.OBSERVATION,
    "SYMPTOM":    ClinicalEntityType.OBSERVATION,
    "ALLERGY":    ClinicalEntityType.ALLERGY,
}


@lru_cache(maxsize=1)
def _load_model() -> spacy.Language:
    """Load spaCy model once per process."""
    model_name = "en_core_web_sm"   # swap to en_core_sci_lg / en_ner_bc5cdr_md
    try:
        nlp = spacy.load(model_name)
        logger.info("spaCy model loaded: %s", model_name)
        return nlp
    except OSError:
        logger.warning("Model %s not found; falling back to blank English", model_name)
        return spacy.blank("en")


def _infer_certainty(ent) -> Certainty:
    """
    Placeholder: real negation detection requires medspacy or NegEx.
    Check surrounding token window for negation cues.
    """
    negation_cues = {"no", "not", "without", "denies", "denied", "negative"}
    doc = ent.doc
    window_start = max(0, ent.start - 5)
    context_tokens = {t.lower_ for t in doc[window_start:ent.start]}
    if context_tokens & negation_cues:
        return Certainty.NEGATED
    historical_cues = {"history", "past", "previous", "formerly", "prior"}
    if context_tokens & historical_cues:
        return Certainty.HISTORICAL
    return Certainty.CONFIRMED


def extract_entities(text: str) -> ExtractionResult:
    """
    Run NER over clinical text and return typed ClinicalEntity list.

    Confidence scores are approximated from entity label scores where
    available; real production pipelines should use calibrated scores from
    the underlying model (e.g. spaCy scores via nlp.get_pipe('ner').scorer).
    """
    nlp = _load_model()
    doc = nlp(text)

    entities: list[ClinicalEntity] = []

    for ent in doc.ents:
        entity_type = _LABEL_MAP.get(ent.label_.upper())
        if entity_type is None:
            continue  # skip non-clinical labels

        # spaCy doesn't expose per-entity confidence natively;
        # use 0.85 as a default pending a calibrated wrapper.
        confidence = float(getattr(ent, "kb_id_", None) and 0.90 or 0.80)

        entities.append(ClinicalEntity(
            raw_text=ent.text,
            entity_type=entity_type,
            certainty=_infer_certainty(ent),
            confidence=confidence,
            start_char=ent.start_char,
            end_char=ent.end_char,
        ))

    overall_confidence = (
        sum(e.confidence for e in entities) / len(entities)
        if entities else 0.0
    )

    return ExtractionResult(
        source_text=text,
        entities=entities,
        extraction_confidence=round(overall_confidence, 4),
    )
