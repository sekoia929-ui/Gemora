"""
src/tests/test_pipeline.py
Core pipeline tests — NLP extraction, FHIR serialization, end-to-end.
"""

import pytest
from src.models.clinical import (
    ClinicalEntity, ClinicalEntityType, Certainty,
    PipelineRequest, TerminologySystem,
)
from src.nlp.extractor import _infer_certainty
from src.nlp.context import detect_sections, enrich_with_context, ContextualNote
from src.fhir.mapper import map_entity, map_entities
from src.fhir.serializers import (
    build_condition, build_medication_statement,
    build_procedure, build_fhir_bundle,
)
from src.pipeline.processor import run_pipeline, _score_pipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def condition_entity():
    return ClinicalEntity(
        raw_text="diabetes mellitus",
        entity_type=ClinicalEntityType.CONDITION,
        certainty=Certainty.CONFIRMED,
        confidence=0.90,
    )


@pytest.fixture
def medication_entity():
    return ClinicalEntity(
        raw_text="metformin",
        entity_type=ClinicalEntityType.MEDICATION,
        certainty=Certainty.CONFIRMED,
        confidence=0.85,
    )


@pytest.fixture
def clinical_text():
    return (
        "Chief Complaint: chest pain.\n"
        "History of Present Illness: Patient presents with severe chest pain "
        "and shortness of breath. No fever.\n"
        "Medications: aspirin 81mg daily, metformin 500mg.\n"
        "Past Medical History: hypertension, diabetes mellitus.\n"
        "Assessment and Plan: Rule out myocardial infarction. ECG ordered."
    )


# ── Terminology mapping ───────────────────────────────────────────────────────

class TestTerminologyMapper:
    def test_condition_maps_to_snomed(self, condition_entity):
        result = map_entity(condition_entity)
        assert result.code == "44054006"
        assert result.code_system == TerminologySystem.SNOMED
        assert result.display == "Diabetes mellitus"

    def test_medication_maps_to_rxnorm(self, medication_entity):
        result = map_entity(medication_entity)
        assert result.code == "860975"
        assert result.code_system == TerminologySystem.RXNORM

    def test_unknown_entity_reduces_confidence(self):
        entity = ClinicalEntity(
            raw_text="xylophagic syndrome",
            entity_type=ClinicalEntityType.CONDITION,
            certainty=Certainty.SUSPECTED,
            confidence=0.80,
        )
        result = map_entity(entity)
        assert result.code is None
        assert result.confidence < 0.80  # penalised

    def test_fuzzy_match_reduces_confidence(self):
        entity = ClinicalEntity(
            raw_text="diabetic mellitus type 2",
            entity_type=ClinicalEntityType.CONDITION,
            certainty=Certainty.CONFIRMED,
            confidence=0.80,
        )
        result = map_entity(entity)
        # Fuzzy match should reduce confidence
        assert result.confidence <= 0.80


# ── FHIR serializers ──────────────────────────────────────────────────────────

class TestFhirSerializers:
    def test_condition_resource_type(self, condition_entity):
        mapped = map_entity(condition_entity)
        resource = build_condition(mapped, patient_id="patient-123")
        assert resource["resourceType"] == "Condition"
        assert "id" in resource
        assert resource["subject"]["reference"] == "Patient/patient-123"

    def test_condition_has_coding(self, condition_entity):
        mapped = map_entity(condition_entity)
        resource = build_condition(mapped, patient_id="patient-123")
        coding = resource["code"]["coding"]
        assert len(coding) >= 1
        assert coding[0]["code"] == "44054006"

    def test_condition_negated_maps_to_refuted(self):
        entity = ClinicalEntity(
            raw_text="pneumonia",
            entity_type=ClinicalEntityType.CONDITION,
            certainty=Certainty.NEGATED,
            confidence=0.80,
        )
        map_entity(entity)
        resource = build_condition(entity, patient_id="p1")
        ver_status = resource["verificationStatus"]["coding"][0]["code"]
        assert ver_status == "refuted"

    def test_medication_statement(self, medication_entity):
        mapped = map_entity(medication_entity)
        resource = build_medication_statement(mapped, patient_id="p1")
        assert resource["resourceType"] == "MedicationStatement"
        assert resource["medicationCodeableConcept"]["coding"][0]["code"] == "860975"

    def test_nlp_confidence_extension(self, condition_entity):
        mapped = map_entity(condition_entity)
        resource = build_condition(mapped, patient_id="p1")
        ext = resource["extension"][0]
        assert "nlp-confidence" in ext["url"]
        assert 0.0 <= ext["valueDecimal"] <= 1.0

    def test_bundle_structure(self, condition_entity, medication_entity):
        entities = [map_entity(condition_entity), map_entity(medication_entity)]
        notes = [None, None]
        bundle = build_fhir_bundle(entities, notes, patient_id="p1")
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "transaction"
        assert bundle["total"] == 2
        assert len(bundle["entry"]) == 2

    def test_bundle_entry_has_request(self, condition_entity):
        entities = [map_entity(condition_entity)]
        bundle = build_fhir_bundle(entities, [None], patient_id="p1")
        entry = bundle["entry"][0]
        assert "request" in entry
        assert entry["request"]["method"] == "POST"


# ── Context enrichment ────────────────────────────────────────────────────────

class TestContextEnrichment:
    def test_section_detection(self, clinical_text):
        sections = detect_sections(clinical_text)
        labels = {s.label for s in sections}
        assert "medications" in labels or "chief_complaint" in labels

    def test_pmh_entity_becomes_historical(self, clinical_text):
        from src.nlp.extractor import ExtractionResult
        entity = ClinicalEntity(
            raw_text="hypertension",
            entity_type=ClinicalEntityType.CONDITION,
            certainty=Certainty.CONFIRMED,
            confidence=0.85,
            start_char=clinical_text.find("hypertension"),
            end_char=clinical_text.find("hypertension") + len("hypertension"),
        )
        extraction = ExtractionResult(
            source_text=clinical_text,
            entities=[entity],
            extraction_confidence=0.85,
        )
        notes = enrich_with_context(extraction)
        # entity certainty should now be HISTORICAL (in PMH section)
        assert entity.certainty == Certainty.HISTORICAL


# ── Confidence scoring ────────────────────────────────────────────────────────

class TestConfidenceScoring:
    def test_zero_entities_returns_zero(self):
        assert _score_pipeline([], 0.9) == 0.0

    def test_fully_mapped_entities_score_high(self, condition_entity, medication_entity):
        entities = [map_entity(condition_entity), map_entity(medication_entity)]
        score = _score_pipeline(entities, extraction_confidence=0.9)
        assert score >= 0.70

    def test_score_bounded(self):
        entities = [
            ClinicalEntity(
                raw_text="x", entity_type=ClinicalEntityType.CONDITION,
                certainty=Certainty.CONFIRMED, confidence=1.0, code="12345",
                code_system=TerminologySystem.SNOMED, display="X",
            )
        ]
        score = _score_pipeline(entities, extraction_confidence=1.0)
        assert 0.0 <= score <= 1.0


# ── End-to-end (no DB) ────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_pipeline_runs_without_crash(self, clinical_text):
        req = PipelineRequest(
            text=clinical_text,
            patient_id="test-patient-001",
            source_system="PYTEST",
        )
        result = run_pipeline(req)
        assert result.patient_id == "test-patient-001"
        assert result.fhir_bundle["resourceType"] == "Bundle"
        assert 0.0 <= result.pipeline_confidence <= 1.0
        assert result.latency_ms > 0

    def test_pipeline_response_has_audit_id(self, clinical_text):
        req = PipelineRequest(
            text=clinical_text,
            patient_id="test-patient-002",
            source_system="PYTEST",
        )
        result = run_pipeline(req)
        assert result.audit_id is not None
        assert len(result.audit_id) == 36  # UUID format
