"""LLM-as-a-Judge classifier for clarifying questions.

Loads instruction prompt + few-shot examples, calls the provider, returns label.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from ..providers import LLMProvider
from ..utils import LLMProviderError, PromptLoadError
from .few_shot_examples import FewShotExample

logger = logging.getLogger(__name__)


class EvaluationResult:
    def __init__(
        self,
        input_text: str,
        label: str,
        raw_response: str,
        provider: str,
        model: str,
        latency_seconds: float,
        temperature: float,
        timestamp: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        self.input_text = input_text
        self.label = label
        self.raw_response = raw_response
        self.provider = provider
        self.model = model
        self.latency_seconds = latency_seconds
        self.temperature = temperature
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.error = error

    def to_dict(self) -> dict:
        return {
            "input_text": self.input_text,
            "label": self.label,
            "raw_response": self.raw_response,
            "provider": self.provider,
            "model": self.model,
            "latency_seconds": self.latency_seconds,
            "temperature": self.temperature,
            "timestamp": self.timestamp,
            "error": self.error or "",
        }


class LLMJudge:
    """Evaluates a clarifying question using an LLMProvider + externalised prompt.

    The instruction file is loaded once at construction; few-shot examples are
    formatted into a fixed block and prepended to each user message.
    """

    def __init__(
        self,
        provider: LLMProvider,
        instructions_path: str | Path,
        few_shot_examples: Optional[List[FewShotExample]] = None,
        label_parser: Optional[Any] = None,
    ) -> None:
        self._provider = provider
        self._instructions_path = Path(instructions_path)
        self._few_shot_examples: List[FewShotExample] = few_shot_examples or []
        self._label_parser = label_parser or (lambda text: text.strip().upper())
        self._instructions: str = ""
        self._load_instructions()
        logger.info(
            "LLMJudge ready — provider=%s model=%s few_shot=%d",
            provider.provider_name, provider.model_name, len(self._few_shot_examples),
        )

    def _load_instructions(self) -> None:
        try:
            self._instructions = self._instructions_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise PromptLoadError(f"Instructions file not found: {self._instructions_path}") from exc
        except OSError as exc:
            raise PromptLoadError(f"Could not read instructions file: {exc}") from exc

    def _format_few_shot_block(self) -> str:
        if not self._few_shot_examples:
            return ""
        lines: List[str] = ["--- FEW-SHOT EXAMPLES ---"]
        for idx, ex in enumerate(self._few_shot_examples, start=1):
            lines.append(f"\nExample {idx}:")
            lines.append(f"  Input:           {ex.input}")
            if ex.explanation:
                lines.append(f"  Reasoning:       {ex.explanation}")
            lines.append(f"  Expected Output: {ex.expected_output}")
        lines.append("\n--- END OF EXAMPLES ---")
        return "\n".join(lines)

    def _build_user_message(self, text_to_evaluate: str) -> str:
        parts: List[str] = []
        few_shot_block = self._format_few_shot_block()
        if few_shot_block:
            parts.append(few_shot_block)
        parts.append(
            "\n--- TEXT TO EVALUATE ---\n"
            f"{text_to_evaluate}\n"
            "--- END OF TEXT ---\n\n"
            "Your classification:"
        )
        return "\n\n".join(parts)

    def evaluate(
        self,
        text_to_evaluate: str,
        temperature: float = 0.0,
    ) -> EvaluationResult:
        user_message = self._build_user_message(text_to_evaluate)
        start = time.monotonic()
        raw_response = ""
        error_msg: Optional[str] = None

        try:
            raw_response = self._provider.call(
                system_instruction=self._instructions,
                user_message=user_message,
                temperature=temperature,
                max_tokens=4000,
            )
        except LLMProviderError as exc:
            error_msg = str(exc)
            logger.error("Evaluation failed: %s", error_msg)
        except Exception as exc:
            error_msg = str(exc)
            logger.error("Evaluation crashed: %s", error_msg)

        latency = time.monotonic() - start
        label = self._label_parser(raw_response) if raw_response else "ERROR"

        return EvaluationResult(
            input_text=text_to_evaluate,
            label=label,
            raw_response=raw_response,
            provider=self._provider.provider_name,
            model=self._provider.model_name,
            latency_seconds=round(latency, 3),
            temperature=temperature,
            error=error_msg,
        )
