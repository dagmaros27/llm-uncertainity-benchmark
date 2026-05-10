"""Simulator: answers a clarifying question using a Layer 1 context essay.

One simulator class per dataset (different system prompt + framing). All share
the same JSON output schema {"answer": "<str>"}.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from ..providers import LLMProvider
from ..utils import SafetyBlockError, parse_json_response

logger = logging.getLogger(__name__)


HEDGE_STRING = "That information is not available."

# Regex used to detect hedged answers even with minor wording variations.
HEDGE_PATTERNS = [
    re.compile(r"that information is not available", re.IGNORECASE),
    re.compile(r"information is not (?:explicitly )?available", re.IGNORECASE),
    re.compile(r"\bnot (?:explicitly )?(?:provided|documented|stated|specified|mentioned)\b", re.IGNORECASE),
    re.compile(r"\bI (?:do not|don't) have (?:that|this)\b", re.IGNORECASE),
    re.compile(r"\bcannot (?:determine|tell|provide|answer)\b", re.IGNORECASE),
]


def is_hedged(answer: str) -> bool:
    if not answer:
        return False
    return any(p.search(answer) for p in HEDGE_PATTERNS)


class Simulator:
    """Loads a dataset-specific simulator prompt; answers a CQ from Layer 1 context."""

    def __init__(
        self,
        provider: LLMProvider,
        instructions_path: str | Path,
        context_label: str = "Situation summary",
    ) -> None:
        self._provider = provider
        self._instructions_path = Path(instructions_path)
        self._instructions = self._instructions_path.read_text(encoding="utf-8").strip()
        self._context_label = context_label
        logger.info(
            "Simulator ready — provider=%s model=%s prompt=%s",
            provider.provider_name, provider.model_name, self._instructions_path.name,
        )

    def answer(
        self,
        clarifying_question: str,
        simulator_context: str,
        temperature: float = 0.0,
    ) -> str:
        user_message = (
            f"{self._context_label}:\n{simulator_context.strip()}\n\n"
            f"Question:\n{clarifying_question.strip()}"
        )
        try:
            raw = self._provider.call(
                system_instruction=self._instructions,
                user_message=user_message,
                temperature=temperature,
                max_tokens=4000,
            )
        except SafetyBlockError:
            logger.warning("Simulator blocked by safety filter — returning hedge.")
            return HEDGE_STRING

        parsed = parse_json_response(raw)
        if parsed and "answer" in parsed:
            return str(parsed["answer"]).strip()

        # Fallback regex extraction
        match = re.search(r'"answer"\s*:\s*"([^"]+)"', raw)
        if match:
            return match.group(1).strip()
        logger.warning("Simulator JSON parse failed; returning raw: %.150s", raw)
        return raw.strip() or HEDGE_STRING


# Per-dataset context labels (cosmetic — affects the user-message framing).
DATASET_CONTEXT_LABELS = {
    "medqa":    "Clinical details",
    "msdialog": "Situation summary",
    "sharc":    "Situation summary",
}
