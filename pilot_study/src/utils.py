"""Shared utilities: env loading, text helpers, JSON parsing, response wrapping."""

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


class RateLimitError(LLMProviderError):
    """Raised specifically on rate-limit (429) responses."""


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
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed: %s | raw: %.200s", exc, raw)
        return None


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


# ── Domain helpers ─────────────────────────────────────────────────────────

def is_assessment_correct(assessment: str, correct_text: str) -> bool:
    """Case-insensitive substring check: does assessment mention correct answer?"""
    if not assessment or not correct_text:
        return False
    return correct_text.lower() in assessment.lower()


def format_answer_choices(choices: dict) -> str:
    return "\n".join(f"{k}. {v}" for k, v in choices.items())
