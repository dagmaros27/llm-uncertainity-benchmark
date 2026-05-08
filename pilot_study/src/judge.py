"""LLM-as-a-Judge: classifies clarifying questions as EPISTEMIC or ALEATORIC.

Architecture:
  - LLMJudge: loads instruction prompt, formats few-shot examples, calls provider
  - CSVBatchClassifier: resume-capable batch runner over a CSV of CQs
  - FewShotExample / EvaluationResult: Pydantic models for structured I/O
"""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .providers import LLMProvider
from .utils import LLMProviderError, PromptLoadError

logger = logging.getLogger(__name__)


# ── Pydantic models ────────────────────────────────────────────────────────

class FewShotExample(BaseModel):
    input: str = Field(..., description="Example clarifying question.")
    expected_output: str = Field(..., description="Expected label (EPISTEMIC or ALEATORIC).")
    explanation: Optional[str] = Field(None, description="Optional chain-of-thought.")


class EvaluationResult(BaseModel):
    input_text: str
    label: str
    raw_response: str
    provider: str
    model: str
    latency_seconds: float
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None


# ── LLM Judge ─────────────────────────────────────────────────────────────

class LLMJudge:
    """Evaluates a single piece of text using an LLMProvider + externalized prompt.

    Args:
        provider:           An LLMProvider instance (use GeminiProvider).
        instructions_path:  Path to the base instruction .txt file.
        few_shot_examples:  Optional list of FewShotExample objects.
        label_parser:       Optional callable to extract a clean label from the raw response.
    """

    def __init__(
        self,
        provider: LLMProvider,
        instructions_path: str | Path,
        few_shot_examples: Optional[list[FewShotExample]] = None,
        label_parser: Optional[Any] = None,
    ) -> None:
        self._provider = provider
        self._instructions_path = Path(instructions_path)
        self._few_shot_examples: list[FewShotExample] = few_shot_examples or []
        self._label_parser = label_parser or (lambda text: text.strip())
        self._instructions: str = ""
        self._load_instructions()
        logger.info(
            "LLMJudge ready — provider=%s model=%s few_shot_count=%d",
            provider.provider_name, provider.model_name, len(self._few_shot_examples),
        )

    def _load_instructions(self) -> None:
        try:
            self._instructions = self._instructions_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise PromptLoadError(f"Instructions file not found: {self._instructions_path}") from exc
        except OSError as exc:
            raise PromptLoadError(f"Could not read instructions file: {exc}") from exc

    def reload_instructions(self) -> None:
        """Hot-reload instructions from disk without re-instantiating."""
        self._load_instructions()

    def set_few_shot_examples(self, examples: list[FewShotExample]) -> None:
        self._few_shot_examples = examples

    def _format_few_shot_block(self) -> str:
        if not self._few_shot_examples:
            return ""
        lines: list[str] = ["--- FEW-SHOT EXAMPLES ---"]
        for idx, ex in enumerate(self._few_shot_examples, start=1):
            lines.append(f"\nExample {idx}:")
            lines.append(f"  Input:           {ex.input}")
            if ex.explanation:
                lines.append(f"  Reasoning:       {ex.explanation}")
            lines.append(f"  Expected Output: {ex.expected_output}")
        lines.append("\n--- END OF EXAMPLES ---")
        return "\n".join(lines)

    def _build_user_message(self, text_to_evaluate: str) -> str:
        parts: list[str] = []
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

    def evaluate(self, text_to_evaluate: str) -> EvaluationResult:
        logger.info("Evaluating: '%.60s...'", text_to_evaluate)
        user_message = self._build_user_message(text_to_evaluate)
        start = time.monotonic()
        raw_response = ""
        error_msg: Optional[str] = None

        try:
            raw_response = self._provider.call(
                system_instruction=self._instructions,
                user_message=user_message,
                temperature=0.0,
                max_tokens=4000,
            )
        except LLMProviderError as exc:
            error_msg = str(exc)
            logger.error("Evaluation failed: %s", error_msg)

        latency = time.monotonic() - start
        label = self._label_parser(raw_response) if raw_response else "ERROR"
        logger.info("label='%s' latency=%.3fs", label, latency)

        return EvaluationResult(
            input_text=text_to_evaluate,
            label=label,
            raw_response=raw_response,
            provider=self._provider.provider_name,
            model=self._provider.model_name,
            latency_seconds=round(latency, 3),
            error=error_msg,
        )


# ── CSV Batch Classifier ───────────────────────────────────────────────────

class CSVBatchClassifier:
    """Resume-capable batch classifier: reads CQs from a CSV, writes labels to another.

    Args:
        judge:               Configured LLMJudge instance.
        input_csv:           Source CSV with clarifying questions.
        output_csv:          Output CSV (appended to if it already exists).
        question_column:     Column name holding the clarifying questions.
        id_column:           Stable row identifier column. Falls back to row index.
        delay_between_calls: Seconds to sleep between API calls.
    """

    OUTPUT_COLUMNS = [
        "id", "question", "label", "raw_response",
        "provider", "model", "latency_seconds", "timestamp", "error",
    ]

    def __init__(
        self,
        judge: LLMJudge,
        input_csv: str | Path,
        output_csv: str | Path,
        question_column: str = "question",
        id_column: Optional[str] = None,
        delay_between_calls: float = 1.0,
    ) -> None:
        self._judge = judge
        self._input_csv = Path(input_csv)
        self._output_csv = Path(output_csv)
        self._question_column = question_column
        self._id_column = id_column
        self._delay = delay_between_calls

    def _load_processed_ids(self) -> set[str]:
        processed: set[str] = set()
        if not self._output_csv.exists():
            return processed
        with self._output_csv.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("id"):
                    processed.add(row["id"])
        logger.info("Resuming: %d rows already processed.", len(processed))
        return processed

    def run(self) -> None:
        if not self._input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {self._input_csv}")

        processed_ids = self._load_processed_ids()
        output_exists = self._output_csv.exists()

        out_fh = self._output_csv.open(mode="a", newline="", encoding="utf-8")
        writer = csv.DictWriter(out_fh, fieldnames=self.OUTPUT_COLUMNS)
        if not output_exists:
            writer.writeheader()
            out_fh.flush()

        with self._input_csv.open(newline="", encoding="utf-8") as in_fh:
            reader = csv.DictReader(in_fh)
            if self._question_column not in (reader.fieldnames or []):
                out_fh.close()
                raise ValueError(
                    f"Column '{self._question_column}' not found. "
                    f"Available: {reader.fieldnames}"
                )

            total = skipped = classified = errors = 0
            for row_index, row in enumerate(reader):
                total += 1
                row_id = str(row.get(self._id_column, row_index) if self._id_column else row_index)
                question = row.get(self._question_column, "").strip()

                if row_id in processed_ids:
                    skipped += 1
                    continue
                if not question:
                    skipped += 1
                    continue

                result = self._judge.evaluate(question)
                errors += bool(result.error)
                classified += not bool(result.error)

                writer.writerow({
                    "id": row_id, "question": question, "label": result.label,
                    "raw_response": result.raw_response, "provider": result.provider,
                    "model": result.model, "latency_seconds": result.latency_seconds,
                    "timestamp": result.timestamp, "error": result.error or "",
                })
                out_fh.flush()

                if self._delay > 0:
                    time.sleep(self._delay)

        out_fh.close()
        logger.info(
            "Batch complete — total=%d classified=%d skipped=%d errors=%d",
            total, classified, skipped, errors,
        )
