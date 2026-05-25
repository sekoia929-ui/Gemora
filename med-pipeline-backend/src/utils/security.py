"""
src/utils/security.py
Input sanitisation helpers.

Production checklist (not implemented here — each needs domain decisions):
  - [ ] mTLS between services
  - [ ] JWT/API-key middleware (add FastAPI dependency)
  - [ ] Rate limiting (slowapi or nginx upstream)
  - [ ] PHI field masking in logs
  - [ ] HIPAA audit controls
"""

from __future__ import annotations
import re

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_TEXT_LEN  = 50_000


def sanitise_clinical_text(text: str) -> str:
    """
    Remove control characters and enforce length limit.
    Does NOT strip clinical punctuation (slashes, hyphens, etc.).
    """
    text = _CONTROL_CHARS.sub(" ", text)
    text = text[:_MAX_TEXT_LEN]
    return text.strip()


def mask_phi(text: str) -> str:
    """
    Partial PHI masking for log output — NOT a substitute for full de-identification.
    Masks patterns that look like MRNs, SSNs, dates of birth.
    
    ⚠️  Assumption: this is a logging helper only.
    Production de-identification requires a dedicated NLP de-id pipeline
    (e.g. AWS Comprehend Medical Detect PHI, Google DLP, or philter).
    """
    # SSN-like
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
    # MRN-like (6-10 digits)
    text = re.sub(r"\bMRN[:\s]?\d{6,10}\b", "[MRN]", text, flags=re.I)
    # DOB patterns
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\b", "[DOB]", text)
    return text
