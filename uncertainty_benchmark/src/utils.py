"""Shared utilities: env loading, exceptions, response wrapping, JSON parsing,
context cleaning."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Exceptions ─────────────────────────────────────────────────────────────

class SafetyBlockError(Exception):
    """Raised when a model response is blocked by safety filters.
    Not retried — safety blocks are deterministic for a given input."""


class LLMProviderError(Exception):
    """Raised when an LLM provider encounters an unrecoverable error."""


class PromptLoadError(Exception):
    """Raised when an instruction prompt file cannot be read."""


# ── Environment ────────────────────────────────────────────────────────────

def load_dotenv(path: Path = Path(".env")) -> None:
    """Load key=value pairs from a .env file into os.environ (no-op if absent)."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ── Text helpers ───────────────────────────────────────────────────────────

def clean_text(text) -> str:
    return " ".join(str(text).strip().split())


# ── Gemini response wrapping ───────────────────────────────────────────────

class ModelResponse:
    def __init__(self, text: str, finish_reason: str) -> None:
        self.text = text
        self.finish_reason = finish_reason

    @property
    def was_blocked(self) -> bool:
        return self.finish_reason in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT")

    @property
    def is_ok(self) -> bool:
        return self.finish_reason in ("STOP", "MAX_TOKENS", "")


def extract_non_thinking_text(response) -> ModelResponse:
    """Extract final output text and finish_reason from a Gemini response."""
    finish_reason = ""
    try:
        candidate = response.candidates[0]
        raw_reason = getattr(candidate, "finish_reason", None)
        if raw_reason is not None:
            finish_reason = str(raw_reason.name) if hasattr(raw_reason, "name") else str(raw_reason)
    except (IndexError, AttributeError):
        pass

    parts_text: list[str] = []
    try:
        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", "")
            if text:
                parts_text.append(text)
    except (IndexError, AttributeError, TypeError):
        pass

    text = clean_text("\n".join(parts_text)) if parts_text else clean_text(getattr(response, "text", ""))
    return ModelResponse(text=text, finish_reason=finish_reason)


# ── JSON parsing ──────────────────────────────────────────────────────────
def parse_json_response(raw: str) -> Optional[dict]:
    """Strip markdown fences and parse JSON. Returns None on failure."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ── Simulator context cleaner (medqa) ─────────────────────────────────────
_CONCLUSION_HEADER_RE = re.compile(
    r"^\s*(?:#{1,4}\s*)?(?:\*\*)?"
    r"(?:Summary(?:\s*(?:Statement|Table|and\s+Diagnostic\s+Reasoning"
    r"|/\s*(?:Impression|Diagnosis|Clinical\s+Reasoning|Assessment)))?"
    r"|(?:Final\s+)?(?:Assessment|Diagnosis|Impression|Interpretation|Conclusion)"
    r"(?:\s*\([^)]*\))?)"
    r"(?:\*\*)?[:\s*]*$",
    re.IGNORECASE,
)

_DIAG_INLINE_RES = [
    re.compile(r"^[^\S\r\n]*(?:[-*]\s*)?\*\*Diagnosis[^*\n]*(?:\*\*)?[:\s].*$",
               re.IGNORECASE | re.MULTILINE),
    re.compile(r"^[^\S\r\n]*(?:[-*]\s*)?Diagnosis\s*:.*$",
               re.IGNORECASE | re.MULTILINE),
    re.compile(r"^[^\S\r\n]*(?:[-*]\s*)?\*\*(?:Best next step|Treatment|Management)[^*\n]*\*\*[:\s].*$",
               re.IGNORECASE | re.MULTILINE),
    re.compile(
        r"[^.\n]*\b(?:most consistent with|consistent with a diagnosis of"
        r"|The findings are consistent with|most likely (?:represents?|diagnosis is)"
        r"|findings are most consistent)[^.\n]*\.",
        re.IGNORECASE,
    ),
]


def clean_simulator_context(text: str) -> str:
    """Strip diagnostic conclusions and summary sections from a context string."""
    if not text:
        return text
    parts = re.split(r"\n(?:-{3,})\n", text)
    kept_parts: list[str] = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        first_line = stripped.splitlines()[0].strip()
        if _CONCLUSION_HEADER_RE.match(first_line):
            continue
        kept_parts.append(part)
    text = "\n---\n".join(kept_parts)
    for pattern in _DIAG_INLINE_RES:
        text = pattern.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
