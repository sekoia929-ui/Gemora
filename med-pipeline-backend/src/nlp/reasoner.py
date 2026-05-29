"""
src/nlp/reasoner.py
Claude AI clinical reasoning layer.

Takes extracted entities + original text and returns:
- Plain English clinical summary
- Urgent findings flags
- Differential diagnosis suggestions
- Reasoning explanation for each FHIR resource
"""

from __future__ import annotations
import os
import json
import httpx
from dataclasses import dataclass
from src.models.clinical import ClinicalEntity, ClinicalEntityType, Certainty
from src.utils.logging import get_logger

logger = get_logger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = "claude-sonnet-4-20250514"


@dataclass
class ClinicalReasoning:
    summary:          str
    urgent_flags:     list[str]
    differentials:    list[str]
    entity_reasoning: dict[str, str]   # raw_text → why it was extracted
    overall_severity: str              # "low" | "moderate" | "high" | "critical"
    confidence:       float


def _build_prompt(text: str, entities: list[ClinicalEntity]) -> str:
    entity_lines = "\n".join(
        f"- {e.raw_text} ({e.entity_type.value}, {e.certainty.value}, "
        f"confidence={e.confidence:.2f}, code={e.code or 'unmapped'})"
        for e in entities
    )

    return f"""You are a clinical AI assistant analyzing a medical note.

CLINICAL NOTE:
{text}

EXTRACTED ENTITIES:
{entity_lines}

Respond ONLY with a JSON object in exactly this format (no markdown, no preamble):
{{
  "summary": "2-3 sentence plain English summary of this clinical note",
  "urgent_flags": ["list any urgent/critical findings, empty array if none"],
  "differentials": ["top 3 differential diagnoses based on the note, empty if insufficient info"],
  "entity_reasoning": {{
    "entity_raw_text": "brief clinical reason why this entity is significant"
  }},
  "overall_severity": "low|moderate|high|critical",
  "confidence": 0.0
}}

Rules:
- overall_severity: low=routine, moderate=needs attention, high=urgent, critical=emergency
- confidence: your confidence in the clinical analysis (0.0-1.0)
- entity_reasoning keys must exactly match the raw_text values from EXTRACTED ENTITIES
- Be concise and clinically accurate
- Never invent findings not present in the note"""


async def get_clinical_reasoning(
    text: str,
    entities: list[ClinicalEntity],
) -> ClinicalReasoning | None:
    """
    Call Claude API for clinical reasoning.
    Returns None if API key not configured or call fails —
    pipeline continues without reasoning rather than crashing.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping clinical reasoning")
        return None

    prompt = _build_prompt(text, entities)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key":         api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json={
                    "model":      CLAUDE_MODEL,
                    "max_tokens": 1024,
                    "messages":   [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            data = response.json()

            raw_text = data["content"][0]["text"].strip()
            # Strip markdown fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]

            parsed = json.loads(raw_text)

            return ClinicalReasoning(
                summary=parsed.get("summary", ""),
                urgent_flags=parsed.get("urgent_flags", []),
                differentials=parsed.get("differentials", []),
                entity_reasoning=parsed.get("entity_reasoning", {}),
                overall_severity=parsed.get("overall_severity", "low"),
                confidence=float(parsed.get("confidence", 0.0)),
            )

    except httpx.HTTPStatusError as e:
        logger.error("Claude API HTTP error: %s %s", e.response.status_code, e.response.text)
    except json.JSONDecodeError as e:
        logger.error("Claude response JSON parse error: %s", e)
    except Exception as e:
        logger.error("Clinical reasoning failed: %s", e)

    return None
