"""
LLM-as-a-Judge Framework
========================
A robust, scalable, and flexible framework for evaluating text using
multiple LLM providers with few-shot prompting and externalized prompts.

Architecture:
    - Strategy Pattern for LLM providers (LLMProvider ABC)
    - Dependency Injection in LLMJudge
    - Pydantic models for structured I/O
    - Tenacity for exponential backoff retries
    - CSV-based batch classification with resume support
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import anthropic
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models – Structured I/O
# ---------------------------------------------------------------------------


class FewShotExample(BaseModel):
    """Represents a single few-shot example with input and expected output."""

    input: str = Field(..., description="The example input text.")
    expected_output: str = Field(..., description="The expected label/output.")
    explanation: Optional[str] = Field(
        None, description="Optional chain-of-thought explanation."
    )


class EvaluationResult(BaseModel):
    """Structured result returned from a single evaluation."""

    input_text: str = Field(..., description="The original text that was evaluated.")
    label: str = Field(..., description="The classification label returned by the LLM.")
    raw_response: str = Field(..., description="The full raw response from the LLM.")
    provider: str = Field(..., description="Name of the LLM provider used.")
    model: str = Field(..., description="Model identifier used for evaluation.")
    latency_seconds: float = Field(..., description="API round-trip time in seconds.")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of the evaluation.",
    )
    error: Optional[str] = Field(None, description="Error message, if any.")


# ---------------------------------------------------------------------------
# Abstract LLM Provider (Strategy Interface)
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """
    Abstract base class defining the Strategy interface for LLM providers.

    All concrete providers MUST implement `generate_response()`.
    This decouples the LLMJudge from any specific API, making provider
    substitution trivial.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The specific model identifier/version."""
        ...

    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        """
        Send *prompt* to the LLM and return the text response.

        Args:
            prompt: The fully assembled prompt string.

        Returns:
            The model's text completion.

        Raises:
            LLMProviderError: On unrecoverable API failures.
        """
        ...


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class LLMProviderError(Exception):
    """Raised when an LLM provider encounters an unrecoverable error."""


class RateLimitError(LLMProviderError):
    """Raised specifically on rate-limit (429) responses."""


class PromptLoadError(Exception):
    """Raised when the instruction prompt file cannot be read."""


# ---------------------------------------------------------------------------
# Concrete Provider: Anthropic (Claude)
# ---------------------------------------------------------------------------


class AnthropicProvider(LLMProvider):
    """
    Concrete LLM provider implementation for Anthropic's Claude models.

    Handles API key injection, request payload construction, and maps
    Anthropic-specific exceptions to the framework's exception hierarchy.

    Usage:
        provider = AnthropicProvider(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model="claude-sonnet-4-20250514",
        )
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = anthropic.Anthropic(api_key=api_key)
        logger.info(
            "AnthropicProvider initialised | model=%s | max_tokens=%d",
            model,
            max_tokens,
        )

    @property
    def provider_name(self) -> str:
        return "Anthropic"

    @property
    def model_name(self) -> str:
        return self._model

    @retry(
        retry=retry_if_exception_type((RateLimitError, anthropic.APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def generate_response(self, prompt: str) -> str:
        """
        Call the Anthropic Messages API with retry/backoff logic.

        The entire assembled prompt is sent as a single user message.
        Temperature 0 is recommended for deterministic classification.
        """
        logger.debug("Sending request to Anthropic | model=%s", self._model)
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text: str = message.content[0].text
            logger.debug("Anthropic response received | chars=%d", len(response_text))
            return response_text

        except anthropic.RateLimitError as exc:
            logger.warning("Anthropic rate limit hit – will retry. %s", exc)
            raise RateLimitError(str(exc)) from exc

        except anthropic.APITimeoutError as exc:
            logger.warning("Anthropic API timeout – will retry. %s", exc)
            raise  # tenacity retries on APITimeoutError directly

        except anthropic.APIError as exc:
            logger.error("Anthropic API error (unrecoverable): %s", exc)
            raise LLMProviderError(f"Anthropic API error: {exc}") from exc


# ---------------------------------------------------------------------------
# Concrete Provider: Google Vertex AI (Gemini)
# ---------------------------------------------------------------------------


class VertexAIProvider(LLMProvider):
    """
    Concrete LLM provider implementation for Google Vertex AI (Gemini models).

    Requires the `google-cloud-aiplatform` package and Application Default
    Credentials (ADC) or an explicit service-account key.

    Usage:
        provider = VertexAIProvider(
            project="my-gcp-project",
            location="us-central1",
            model="gemini-1.5-pro-002",
        )
    """

    def __init__(
        self,
        project: str,
        location: str = "us-central1",
        model: str = "gemini-1.5-pro-002",
        max_output_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        # Lazy import – keeps framework usable even if vertexai is not installed.
        try:
            import vertexai  # type: ignore[import-untyped]
            from vertexai.generative_models import (  # type: ignore[import-untyped]
                GenerativeModel,
                GenerationConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "google-cloud-aiplatform is required for VertexAIProvider. "
                "Install it with: pip install google-cloud-aiplatform"
            ) from exc

        vertexai.init(project=project, location=location)
        self._model_id = model
        self._generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        self._generative_model = GenerativeModel(model)
        logger.info(
            "VertexAIProvider initialised | project=%s | model=%s",
            project,
            model,
        )

    @property
    def provider_name(self) -> str:
        return "VertexAI"

    @property
    def model_name(self) -> str:
        return self._model_id

    @retry(
        retry=retry_if_exception_type(Exception),  # broad – tighten per SDK exceptions
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def generate_response(self, prompt: str) -> str:
        """Call the Vertex AI Generative Model API."""
        logger.debug("Sending request to VertexAI | model=%s", self._model_id)
        try:
            response = self._generative_model.generate_content(
                prompt,
                generation_config=self._generation_config,
            )
            return response.text
        except Exception as exc:
            logger.error("VertexAI error: %s", exc)
            raise LLMProviderError(f"VertexAI error: {exc}") from exc


# ---------------------------------------------------------------------------
# LLM Judge – Core Orchestrator
# ---------------------------------------------------------------------------


class LLMJudge:
    """
    Core evaluation engine that combines:
      - Externalized instruction prompts (loaded from .txt)
      - Dynamic few-shot example injection
      - Provider-agnostic LLM calls via the Strategy pattern

    The Judge owns NO business logic about *which* model to use – that
    concern is entirely delegated to the injected LLMProvider instance.

    Args:
        provider:           An LLMProvider instance (Anthropic, Vertex, etc.)
        instructions_path:  Path to the base instruction .txt file.
        few_shot_examples:  Optional list of FewShotExample objects.
        label_parser:       Optional callable to extract a clean label from
                            the raw LLM response. Defaults to strip().
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

        logger.info(
            "LLMJudge initialised | provider=%s | model=%s | few_shot_count=%d",
            provider.provider_name,
            provider.model_name,
            len(self._few_shot_examples),
        )
        self._load_instructions()

    # ------------------------------------------------------------------
    # Prompt Management
    # ------------------------------------------------------------------

    def _load_instructions(self) -> None:
        """
        Load the base instruction prompt from the configured .txt file.

        Raises:
            PromptLoadError: If the file is missing or unreadable.
        """
        logger.info("Loading instructions from '%s'", self._instructions_path)
        try:
            self._instructions = self._instructions_path.read_text(encoding="utf-8").strip()
            logger.info(
                "Instructions loaded successfully | chars=%d", len(self._instructions)
            )
        except FileNotFoundError as exc:
            raise PromptLoadError(
                f"Instructions file not found: {self._instructions_path}"
            ) from exc
        except OSError as exc:
            raise PromptLoadError(
                f"Could not read instructions file: {exc}"
            ) from exc

    def reload_instructions(self) -> None:
        """Hot-reload instructions from disk without re-instantiating the Judge."""
        logger.info("Reloading instructions from disk.")
        self._load_instructions()

    # ------------------------------------------------------------------
    # Few-Shot Management
    # ------------------------------------------------------------------

    def set_few_shot_examples(self, examples: list[FewShotExample]) -> None:
        """Replace the current few-shot examples at runtime."""
        self._few_shot_examples = examples
        logger.info("Few-shot examples updated | count=%d", len(examples))

    def _format_few_shot_block(self) -> str:
        """
        Render the few-shot examples into a structured text block.

        Each example is formatted as a numbered Q/A pair so the LLM
        clearly understands the pattern.

        Returns:
            A formatted string block, or an empty string if no examples exist.
        """
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

    # ------------------------------------------------------------------
    # Prompt Assembly
    # ------------------------------------------------------------------

    def _build_prompt(self, text_to_evaluate: str) -> str:
        """
        Assemble the final prompt from its three components:
          1. Base instructions (from .txt file)
          2. Few-shot examples block
          3. The target text to evaluate

        Returns:
            The fully assembled prompt string.
        """
        parts: list[str] = [self._instructions]

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

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, text_to_evaluate: str) -> EvaluationResult:
        """
        Evaluate a single piece of text and return a structured result.

        Args:
            text_to_evaluate: The input text to classify/evaluate.

        Returns:
            An EvaluationResult containing the label and metadata.
        """
        logger.info(
            "Evaluating text | provider=%s | text_preview='%s...'",
            self._provider.provider_name,
            text_to_evaluate[:60],
        )
        prompt = self._build_prompt(text_to_evaluate)
        start_time = time.monotonic()
        raw_response: str = ""
        error_msg: Optional[str] = None

        try:
            raw_response = self._provider.generate_response(prompt)
        except LLMProviderError as exc:
            error_msg = str(exc)
            logger.error("Evaluation failed: %s", error_msg)
            raw_response = ""

        latency = time.monotonic() - start_time
        label = self._label_parser(raw_response) if raw_response else "ERROR"

        result = EvaluationResult(
            input_text=text_to_evaluate,
            label=label,
            raw_response=raw_response,
            provider=self._provider.provider_name,
            model=self._provider.model_name,
            latency_seconds=round(latency, 3),
            error=error_msg,
        )
        logger.info(
            "Evaluation complete | label='%s' | latency=%.3fs",
            result.label,
            result.latency_seconds,
        )
        return result


# ---------------------------------------------------------------------------
# CSV Batch Classifier – Resume-Capable
# ---------------------------------------------------------------------------


class CSVBatchClassifier:
    """
    Reads a CSV containing clarifying questions, classifies each one using
    an LLMJudge, and writes results to an output CSV.

    Key features:
    - **Resume support**: Skips rows already present in the output CSV,
      so processing can continue after a rate-limit interruption.
    - **Incremental writes**: Each result is flushed immediately so partial
      progress is never lost.
    - **Per-row logging**: Every classification is logged with its status.

    Args:
        judge:              Configured LLMJudge instance.
        input_csv:          Path to the source CSV file.
        output_csv:         Path to the (possibly existing) output CSV.
        question_column:    Name of the column holding the clarifying questions.
        id_column:          Optional column to use as a stable row identifier.
                            If None, the 0-based row index is used.
        delay_between_calls: Seconds to sleep between API calls (rate limiting).
    """

    # Columns written to the output CSV
    OUTPUT_COLUMNS = [
        "id",
        "question",
        "label",
        "raw_response",
        "provider",
        "model",
        "latency_seconds",
        "timestamp",
        "error",
    ]

    def __init__(
        self,
        judge: LLMJudge,
        input_csv: str | Path,
        output_csv: str | Path,
        question_column: str = "question",
        id_column: Optional[str] = None,
        delay_between_calls: float = 0.5,
    ) -> None:
        self._judge = judge
        self._input_csv = Path(input_csv)
        self._output_csv = Path(output_csv)
        self._question_column = question_column
        self._id_column = id_column
        self._delay = delay_between_calls

        logger.info(
            "CSVBatchClassifier initialised | input=%s | output=%s | question_col='%s'",
            self._input_csv,
            self._output_csv,
            self._question_column,
        )

    # ------------------------------------------------------------------
    # Resume Logic
    # ------------------------------------------------------------------

    def _load_already_processed_ids(self) -> set[str]:
        """
        Read the output CSV (if it exists) and return the set of IDs
        that have already been classified.
        """
        processed: set[str] = set()
        if not self._output_csv.exists():
            return processed

        with self._output_csv.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("id"):
                    processed.add(row["id"])

        logger.info(
            "Resuming: found %d already-processed rows in '%s'",
            len(processed),
            self._output_csv,
        )
        return processed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Iterate over the input CSV, classify each unprocessed question,
        and append results to the output CSV.

        The output file is opened in append mode so partial progress is
        preserved across restarts.
        """
        if not self._input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {self._input_csv}")

        processed_ids = self._load_already_processed_ids()
        output_file_exists = self._output_csv.exists()

        # Open output in append mode; write header only on first creation.
        out_fh = self._output_csv.open(
            mode="a", newline="", encoding="utf-8"
        )
        writer = csv.DictWriter(out_fh, fieldnames=self.OUTPUT_COLUMNS)
        if not output_file_exists:
            writer.writeheader()
            out_fh.flush()
            logger.info("Created output CSV with headers: %s", self._output_csv)

        # Read and process the input CSV
        with self._input_csv.open(newline="", encoding="utf-8") as in_fh:
            reader = csv.DictReader(in_fh)

            if self._question_column not in (reader.fieldnames or []):
                out_fh.close()
                raise ValueError(
                    f"Column '{self._question_column}' not found in input CSV. "
                    f"Available columns: {reader.fieldnames}"
                )

            total = skipped = classified = errors = 0

            for row_index, row in enumerate(reader):
                total += 1
                row_id = str(
                    row.get(self._id_column, row_index)
                    if self._id_column
                    else row_index
                )
                question = row.get(self._question_column, "").strip()

                # --- Resume: skip already-processed rows ---
                if row_id in processed_ids:
                    skipped += 1
                    logger.debug("Skipping already-processed row | id=%s", row_id)
                    continue

                if not question:
                    logger.warning("Empty question at row id=%s – skipping.", row_id)
                    skipped += 1
                    continue

                logger.info(
                    "Classifying row %s | question='%s...'",
                    row_id,
                    question[:60],
                )

                result = self._judge.evaluate(question)

                # Track errors but keep going
                if result.error:
                    errors += 1
                else:
                    classified += 1

                # Write result immediately (incremental flush for safety)
                writer.writerow(
                    {
                        "id": row_id,
                        "question": question,
                        "label": result.label,
                        "raw_response": result.raw_response,
                        "provider": result.provider,
                        "model": result.model,
                        "latency_seconds": result.latency_seconds,
                        "timestamp": result.timestamp,
                        "error": result.error or "",
                    }
                )
                out_fh.flush()

                # Polite delay between API calls
                if self._delay > 0:
                    time.sleep(self._delay)

        out_fh.close()
        logger.info(
            "Batch complete | total=%d | classified=%d | skipped=%d | errors=%d",
            total,
            classified,
            skipped,
            errors,
        )
