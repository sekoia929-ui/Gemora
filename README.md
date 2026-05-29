# med-pipeline-backend

AI-native healthcare interoperability pipeline — transforms unstructured clinical text into validated **FHIR R4** resources using Medical NLP.

## What it does

```
Clinical text / PDF  →  NLP extraction  →  Context enrichment
  →  Terminology mapping (SNOMED / ICD-10 / RxNorm)
  →  FHIR R4 Bundle  →  PostgreSQL (JSONB)  →  REST API
```

## Stack

| Layer | Tech |
|---|---|
| API | FastAPI + Pydantic v2 |
| NLP | spaCy (upgrade path: scispaCy / medspaCy) |
| FHIR | R4 — Condition, MedicationStatement, Procedure, Observation |
| Terminology | SNOMED-CT · ICD-10 · RxNorm |
| Database | PostgreSQL + asyncpg (JSONB audit store) |
| CI/CD | GitHub Actions → Docker |

## Quick start

```bash
cp .env.example .env
docker compose up --build
# API docs → http://localhost:8000/docs
```

## Test the pipeline

```bash
curl -X POST http://localhost:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient has diabetes mellitus. Taking metformin 500mg daily. No hypertension.",
    "patient_id": "pt-001",
    "source_system": "TEST"
  }'
```

## Project structure

```
src/
├── app.py                  # FastAPI entry point
├── models/clinical.py      # Domain types (ClinicalEntity, PipelineRequest/Response)
├── nlp/
│   ├── extractor.py        # spaCy NER → ClinicalEntity list
│   └── context.py          # Section detection · negation · temporal hints
├── fhir/
│   ├── mapper.py           # Raw text → SNOMED / ICD-10 / RxNorm
│   └── serializers.py      # Typed entities → FHIR R4 JSON
├── pipeline/
│   └── processor.py        # 5-stage orchestrator + confidence scoring
├── api/routes.py           # /process (text) · /upload (PDF)
├── db/session.py           # asyncpg pool + audit table schema
└── utils/
    ├── audit.py            # Immutable pipeline audit trail
    ├── logging.py          # Structured stdout logger
    └── security.py         # Input sanitisation · PHI masking helper
```

## Run tests

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
pytest src/tests/ -v
```

## Upgrade path

| When ready | What to swap |
|---|---|
| Better NER | `en_core_web_sm` → `en_ner_bc5cdr_md` or `en_core_sci_lg` |
| Real terminology | `fhir/mapper.py` → UMLS / Ontoserver API calls |
| FHIR validation | Uncomment `fhir.resources` in `requirements.txt` |
| PDF support | Uncomment `pdfplumber` in `requirements.txt` |

---

Part of **Gemora** — AI-native healthcare infrastructure.
