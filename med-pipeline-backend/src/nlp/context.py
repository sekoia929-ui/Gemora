"""
src/nlp/context.py
Clinical context enrichment — section detection, temporal reasoning,
severity inference.

Limitation: Section detection here uses simple regex heuristics.
Production upgrade path: medspacy section detector or a fine-tuned
section-classification model.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from src.models.clinical import ExtractionResult, ClinicalEntity, Certainty
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Section definitions ───────────────────────────────────────────────────────

_SECTION_PATTERNS: dict[str, re.Pattern] = {
    "chief_complaint":       re.compile(r"chief\s+complaint", re.I),
    "history_present_illness": re.compile(r"history\s+of\s+present\s+illness|hpi", re.I),
    "past_medical_history":  re.compile(r"past\s+medical\s+history|pmh", re.I),
    "medications":           re.compile(r"medications?|current\s+meds", re.I),
    "allergies":             re.compile(r"allergies|allergy", re.I),
    "assessment_plan":       re.compile(r"assessment\s+(?:and\s+)?plan|a/p", re.I),
    "review_of_systems":     re.compile(r"review\s+of\s+systems|ros", re.I),
}

_SEVERITY_TERMS = {
    "mild":     0.3,
    "moderate": 0.6,
    "severe":   0.9,
    "critical": 1.0,
}


@dataclass
class DocumentSection:
    label: str
    start: int
    end: int


@dataclass
class ContextualNote:
    section: str | None
    severity_score: float | None
    temporal_hint: str | None   # "acute" | "chronic" | "historical"


def detect_sections(text: str) -> list[DocumentSection]:
    """Return detected section boundaries (best-effort)."""
    sections = []
    lines = text.split("\n")
    pos = 0
    current_section = None
    current_start = 0

    for line in lines:
        for label, pattern in _SECTION_PATTERNS.items():
            if pattern.search(line):
                if current_section:
                    sections.append(DocumentSection(
                        label=current_section, start=current_start, end=pos
                    ))
                current_section = label
                current_start = pos
                break
        pos += len(line) + 1

    if current_section:
        sections.append(DocumentSection(label=current_section, start=current_start, end=pos))

    return sections


def _get_section_for_entity(
    entity: ClinicalEntity, sections: list[DocumentSection]
) -> str | None:
    if entity.start_char is None:
        return None
    for sec in sections:
        if sec.start <= entity.start_char < sec.end:
            return sec.label
    return None


def _infer_severity(text: str) -> float | None:
    lower = text.lower()
    for term, score in sorted(_SEVERITY_TERMS.items(), key=lambda x: -x[1]):
        if term in lower:
            return score
    return None


def _infer_temporal(section: str | None) -> str | None:
    if section == "past_medical_history":
        return "historical"
    if section in ("chief_complaint", "history_present_illness", "assessment_plan"):
        return "acute"
    return None


def enrich_with_context(result: ExtractionResult) -> list[ContextualNote]:
    """
    Attach section + severity + temporal context to each entity.
    Returns a parallel list of ContextualNote (same order as result.entities).
    """
    sections = detect_sections(result.source_text)
    notes: list[ContextualNote] = []

    for entity in result.entities:
        section_label = _get_section_for_entity(entity, sections)

        # If entity is in PMH and wasn't negated → mark historical
        if (
            section_label == "past_medical_history"
            and entity.certainty == Certainty.CONFIRMED
        ):
            entity.certainty = Certainty.HISTORICAL

        severity = _infer_severity(entity.raw_text)
        temporal = _infer_temporal(section_label)

        notes.append(ContextualNote(
            section=section_label,
            severity_score=severity,
            temporal_hint=temporal,
        ))

    return notes
