"""
Phase 1 — Medical Clarifying Question Generation Pipeline
==========================================================
Loads the MediQ-AskDocs dataset from HuggingFace, selects a pilot subset,
and uses an LLM to generate a clarifying question based on the patient's
initial complaint context.

Architecture:
  - Strategy pattern for LLM providers (GeminiProvider, AnthropicProvider)
  - QuestionGeneratorPipeline orchestrates loading, generation, and saving
  - Incremental CSV writing with resumability
  - Tenacity-based retry logic for API rate limits

instruction.txt (example content — save this as instruction.txt):
----------------------------------------------------------------------
You are a medical assistant helping a clinician gather the most important
missing information before making a diagnostic decision.

You will be given an initial patient complaint. Your task is to generate
exactly ONE focused clarifying question that targets the single most
critical piece of missing clinical information needed to proceed safely.

Rules:
- Output only the question itself. No preamble, no explanation.
- The question must be clinically meaningful and specific.
- Do not attempt a diagnosis.
- Do not ask multiple questions.
----------------------------------------------------------------------
"""

from __future__ import annotations

import abc
import csv
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Optional

import anthropic
import tenacity
from datasets import load_dataset
from google import genai
from google.genai import types

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phase1_pipeline")

# ── Constants ──────────────────────────────────────────────────────────────
DEFAULT_PILOT_SIZE: int = 20
DEFAULT_OUTPUT_CSV: Path = Path("outputs/phase1_results.csv")
INSTRUCTION_FILE: Path = Path("instruction.txt")
CSV_FIELDNAMES: list[str] = [
    "id",
    "context",
    "ground_truth_question",
    "generated_question",
    "provider",
    "model_id",
]


def _mask_secret(secret: Optional[str], keep: int = 4) -> str:
    """Mask a secret for logs while still showing enough for diagnostics."""
    if not secret:
        return "<missing>"
    if len(secret) <= keep:
        return "*" * len(secret)
    return f"{'*' * (len(secret) - keep)}{secret[-keep:]}"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _print_startup_diagnostics() -> None:
    """
    Log environment and runtime details to simplify auth/rate-limit debugging.

    By default API keys are masked. Set DEBUG_SHOW_FULL_API_KEY=true to show
    full key values in logs (use with caution).
    """
    show_full = _env_flag("DEBUG_SHOW_FULL_API_KEY", default=False)
    vertex_key = os.environ.get("VERTEX_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    logger.info("========== Startup Diagnostics ==========")
    logger.info("Working directory: %s", Path.cwd())
    logger.info("Instruction file exists: %s", INSTRUCTION_FILE.exists())
    logger.info("Output CSV target: %s", DEFAULT_OUTPUT_CSV)
    logger.info("DEBUG_SHOW_FULL_API_KEY: %s", show_full)
    logger.info(
        "VERTEX_API_KEY: %s (len=%d)",
        vertex_key if show_full else _mask_secret(vertex_key),
        len(vertex_key) if vertex_key else 0,
    )
    logger.info(
        "ANTHROPIC_API_KEY: %s (len=%d)",
        anthropic_key if show_full else _mask_secret(anthropic_key),
        len(anthropic_key) if anthropic_key else 0,
    )
    logger.info("========================================")


def _extract_status_code(exc: BaseException) -> Optional[int]:
    """Best-effort extraction of HTTP status code from SDK exceptions."""
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value

    return None


def _is_retryable_exception(exc: BaseException) -> bool:
    """Retry only transient errors (rate-limit, timeout, temporary server issues)."""
    status = _extract_status_code(exc)
    if status is not None:
        return status in {408, 409, 429, 500, 502, 503, 504}

    text = str(exc).lower()
    transient_signals = (
        "rate limit",
        "too many requests",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "connection reset",
        "service unavailable",
    )
    return any(signal in text for signal in transient_signals)


def _log_retry(provider: str, rs: tenacity.RetryCallState) -> None:
    """Log retry details with status code and root exception text."""
    exc = rs.outcome.exception() if rs.outcome else None
    if exc is None:
        logger.warning(
            "%s transient error — retrying in %.0fs (attempt %d)",
            provider,
            rs.next_action.sleep,  # type: ignore[attr-defined]
            rs.attempt_number,
        )
        return

    status = _extract_status_code(exc)
    logger.warning(
        "%s transient API error (status=%s): %s — retrying in %.0fs (attempt %d)",
        provider,
        status if status is not None else "unknown",
        exc,
        rs.next_action.sleep,  # type: ignore[attr-defined]
        rs.attempt_number,
    )


# ══════════════════════════════════════════════════════════════════════════
# Strategy — Abstract LLM Provider
# ══════════════════════════════════════════════════════════════════════════

class LLMProvider(abc.ABC):
    """
    Abstract base class for LLM providers.
    All concrete providers must implement generate_question.
    """

    @abc.abstractmethod
    def generate_question(self, instruction: str, context: str) -> str:
        """
        Given an instruction prompt and a patient context string,
        return exactly one clarifying question as a plain string.

        Args:
            instruction: The system/instruction prompt loaded from instruction.txt.
            context:     The patient's initial complaint text.

        Returns:
            A single clarifying question string.
        """
        ...

    @abc.abstractmethod
    def generate_question_once(self, instruction: str, context: str) -> str:
        """Generate once without retry logic (used for smoke tests)."""
        ...

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier for CSV logging."""
        ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        """Model identifier string for CSV logging."""
        ...


# ══════════════════════════════════════════════════════════════════════════
# Concrete Provider — Gemini (via google-genai SDK)
# ══════════════════════════════════════════════════════════════════════════

class GeminiProvider(LLMProvider):
    """
    LLM provider backed by Google Gemini via the google-genai SDK.
    Reads VERTEX_API_KEY from the environment.
    """

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_output_tokens: int = 256,
    ) -> None:
        api_key = os.environ.get("VERTEX_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "VERTEX_API_KEY is not set. Add it to your .env file."
            )
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1"),
        )
        self._model_id = model_id
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        logger.info("GeminiProvider initialised with model=%s", model_id)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_id

    def _generate_question_impl(self, instruction: str, context: str) -> str:
        full_prompt = f"{instruction.strip()}\n\nPatient complaint:\n{context.strip()}"
        response = self._client.models.generate_content(
            model=self._model_id,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=self._temperature,
                max_output_tokens=self._max_output_tokens,
                top_p=0.95,
            ),
        )
        # Extract non-empty generated text from response parts when available.
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text.strip():
                    return text.strip()

        fallback_text = getattr(response, "text", None)
        if isinstance(fallback_text, str) and fallback_text.strip():
            return fallback_text.strip()

        raise RuntimeError("Gemini returned an empty response with no text content.")

    def generate_question_once(self, instruction: str, context: str) -> str:
        return self._generate_question_impl(instruction=instruction, context=context)

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=4, max=60),
        stop=tenacity.stop_after_attempt(6),
        retry=tenacity.retry_if_exception(_is_retryable_exception),
        before_sleep=lambda rs: _log_retry("Gemini", rs),
        reraise=True,
    )
    def generate_question(self, instruction: str, context: str) -> str:
        return self._generate_question_impl(instruction=instruction, context=context)


# ══════════════════════════════════════════════════════════════════════════
# Concrete Provider — Anthropic Claude
# ══════════════════════════════════════════════════════════════════════════

class AnthropicProvider(LLMProvider):
    """
    LLM provider backed by Anthropic Claude via the anthropic SDK.
    Reads ANTHROPIC_API_KEY from the environment.
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model_id = model_id
        self._temperature = temperature
        self._max_tokens = max_tokens
        logger.info("AnthropicProvider initialised with model=%s", model_id)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model_id

    def _generate_question_impl(self, instruction: str, context: str) -> str:
        message = self._client.messages.create(
            model=self._model_id,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=instruction.strip(),
            messages=[
                {
                    "role": "user",
                    "content": f"Patient complaint:\n{context.strip()}",
                }
            ],
        )

        parts = getattr(message, "content", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()

        raise RuntimeError("Anthropic returned an empty response with no text content.")

    def generate_question_once(self, instruction: str, context: str) -> str:
        return self._generate_question_impl(instruction=instruction, context=context)

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=4, max=60),
        stop=tenacity.stop_after_attempt(6),
        retry=tenacity.retry_if_exception(_is_retryable_exception),
        before_sleep=lambda rs: _log_retry("Anthropic", rs),
        reraise=True,
    )
    def generate_question(self, instruction: str, context: str) -> str:
        return self._generate_question_impl(instruction=instruction, context=context)


# ══════════════════════════════════════════════════════════════════════════
# Data Parsing Helpers
# ══════════════════════════════════════════════════════════════════════════

def _extract_text_from_block(block: list[dict]) -> str:
    """
    Extracts the plain text from a MediQ-AskDocs context or question block.
    The block is a list of dicts, each with a 'content' key that is either
    a string or a list of content dicts. Concatenates all text found.

    Args:
        block: The raw list from the dataset's 'context' or 'question' field.

    Returns:
        A single concatenated string of all text content found in the block.
    """
    parts: list[str] = []
    for item in block:
        content = item.get("content", "")
        if isinstance(content, str):
            parts.append(content.strip())
        elif isinstance(content, list):
            for sub in content:
                if isinstance(sub, dict) and sub.get("type") == "text":
                    parts.append(sub.get("text", "").strip())
                elif isinstance(sub, str):
                    parts.append(sub.strip())
    return " ".join(p for p in parts if p)


def _parse_record(record: dict) -> Optional[dict[str, str]]:
    """
    Parses a single raw MediQ-AskDocs record into a flat dict with keys:
    id, context, ground_truth_question.

    Returns None if the record cannot be parsed (missing fields etc.).

    Args:
        record: Raw dict from the HuggingFace dataset.

    Returns:
        Parsed flat dict or None on parse failure.
    """
    try:
        record_id: str = str(record["id"])
        context_text: str = _extract_text_from_block(record.get("context", []))
        gt_question: str = _extract_text_from_block(record.get("question", []))

        if not context_text:
            logger.warning("Record %s has empty context — skipping.", record_id)
            return None

        return {
            "id": record_id,
            "context": context_text,
            "ground_truth_question": gt_question,
        }
    except (KeyError, TypeError) as exc:
        logger.error("Failed to parse record: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════

class QuestionGeneratorPipeline:
    """
    Orchestrates the full Phase 1 workflow:
      1. Load MediQ-AskDocs from HuggingFace.
      2. Slice the pilot subset.
      3. Load the instruction prompt from file.
      4. For each record: check resumability, call the LLM, write to CSV.

    Accepts an LLMProvider via dependency injection so the provider can
    be swapped without changing pipeline logic (Strategy pattern).
    """

    def __init__(
        self,
        provider: LLMProvider,
        output_csv: Path = DEFAULT_OUTPUT_CSV,
        instruction_file: Path = INSTRUCTION_FILE,
        pilot_size: int = DEFAULT_PILOT_SIZE,
        request_interval: float = 3.0,
        random_seed: int = 42,
        fresh_start: bool = False,
    ) -> None:
        """
        Args:
            provider:          Concrete LLMProvider instance (injected).
            output_csv:        Path to the output CSV file.
            instruction_file:  Path to the instruction.txt prompt file.
            pilot_size:        Number of records to process.
            request_interval:  Seconds to wait between API calls.
        """
        self._provider = provider
        self._output_csv = output_csv
        self._instruction_file = instruction_file
        self._pilot_size = pilot_size
        self._request_interval = request_interval
        self._random_seed = random_seed
        self._fresh_start = fresh_start

        self._output_csv.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Pipeline initialised — provider=%s, model=%s, pilot_size=%d, output=%s, fresh_start=%s",
            provider.provider_name,
            provider.model_name,
            pilot_size,
            output_csv,
            fresh_start,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_instruction(self) -> str:
        """Load and return the instruction prompt from disk."""
        if not self._instruction_file.exists():
            raise FileNotFoundError(
                f"Instruction file not found: {self._instruction_file}. "
                "Create it with your system prompt."
            )
        instruction = self._instruction_file.read_text(encoding="utf-8").strip()
        logger.info("Instruction prompt loaded (%d chars).", len(instruction))
        return instruction

    def _load_processed_ids(self) -> set[str]:
        """
        Read the output CSV (if it exists) and return the set of already
        processed record IDs. Used for resumability.
        """
        processed: set[str] = set()
        if not self._output_csv.exists():
            return processed
        with self._output_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if "id" in row and row["id"]:
                    processed.add(row["id"])
        logger.info(
            "Resumability check: %d records already in %s.",
            len(processed),
            self._output_csv,
        )
        return processed

    def _write_header_if_needed(self) -> None:
        """Write the CSV header row if the file does not yet exist."""
        if not self._output_csv.exists():
            with self._output_csv.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()
            logger.info("CSV created with header: %s", self._output_csv)

    def _reset_output_if_needed(self) -> None:
        """Delete existing output CSV when fresh_start is enabled."""
        if not self._fresh_start:
            return
        if self._output_csv.exists():
            self._output_csv.unlink()
            logger.warning(
                "Fresh start enabled — deleted existing output CSV: %s",
                self._output_csv,
            )
        else:
            logger.info(
                "Fresh start enabled — no existing output CSV found at: %s",
                self._output_csv,
            )

    def _append_row(self, row: dict[str, str]) -> None:
        """
        Append a single result row to the CSV immediately after generation.
        Opens in append mode — never buffers multiple rows in memory.

        Args:
            row: Dict matching CSV_FIELDNAMES.
        """
        with self._output_csv.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
            writer.writerow(row)

    def _load_dataset(self) -> list[dict]:
        """
        Load the MediQ-AskDocs dataset from HuggingFace and return
        the first pilot_size records as plain dicts.
        """
        logger.info(
            "Loading MediQ-AskDocs from HuggingFace (split=train, first %d records)...",
            self._pilot_size,
        )
        import random

        ds = load_dataset("stellalisy/MediQ_AskDocs", split="train")
        total_available = len(ds)
        random.seed(self._random_seed)
        indices = random.sample(range(total_available), min(self._pilot_size, total_available))
        subset = ds.select(sorted(indices))
        records = [dict(row) for row in subset]
        logger.info("Dataset loaded — %d records selected.", len(records))
        return records

    def _run_smoke_test(self, instruction: str) -> None:
        """
        Run one minimal API call before the main loop to fail fast on
        auth/rate-limit/model issues.
        """
        logger.info("STEP 4/5: Running one-shot API smoke test before loop...")
        smoke_context = "I have had severe headache and fever for two days."
        started = time.perf_counter()
        try:
            smoke_output = self._provider.generate_question_once(
                instruction=instruction,
                context=smoke_context,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.error("Smoke test FAILED after %.2fs.", elapsed)
            logger.error(
                "Smoke test provider/model: %s / %s",
                self._provider.provider_name,
                self._provider.model_name,
            )
            logger.error(
                "Smoke test exception (status=%s): %s",
                _extract_status_code(exc) or "unknown",
                exc,
            )
            logger.debug("Smoke test traceback:\n%s", traceback.format_exc())
            raise RuntimeError(
                "Pre-loop smoke test failed (single attempt, no retries). "
                "Fix the API/config issue and rerun."
            ) from exc

        elapsed = time.perf_counter() - started
        logger.info("Smoke test PASSED in %.2fs.", elapsed)
        logger.info("Smoke test output preview: %s", smoke_output[:160])

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Execute the full pipeline:
          1. Load dataset.
          2. Load instruction.
          3. Initialise CSV.
          4. For each record: skip if done, generate, write immediately.
        """
        logger.info("STEP 1/5: Loading instruction prompt...")
        instruction = self._load_instruction()

        logger.info("STEP 2/5: Loading dataset...")
        records = self._load_dataset()

        logger.info("STEP 3/5: Preparing resumability state...")
        self._reset_output_if_needed()
        processed_ids = self._load_processed_ids()

        logger.info("STEP 3/5: Preparing output CSV...")
        self._write_header_if_needed()

        self._run_smoke_test(instruction)

        total = len(records)
        skipped = 0
        succeeded = 0
        failed = 0

        logger.info("STEP 5/5: Entering generation loop for %d records...", total)

        for i, raw_record in enumerate(records, start=1):
            parsed = _parse_record(raw_record)
            if parsed is None:
                failed += 1
                continue

            record_id = parsed["id"]

            # ── Resumability check ────────────────────────────────────────
            if record_id in processed_ids:
                logger.info(
                    "[%d/%d] SKIP — %s already processed.", i, total, record_id
                )
                skipped += 1
                continue

            logger.info(
                "[%d/%d] Processing — %s", i, total, record_id
            )
            logger.debug("Context preview: %s", parsed["context"][:120])

            # ── LLM call ──────────────────────────────────────────────────
            try:
                generated_question = self._provider.generate_question(
                    instruction=instruction,
                    context=parsed["context"],
                )
            except Exception as exc:
                logger.error(
                    "[%d/%d] FAILED after retries — %s: %s",
                    i, total, record_id, exc,
                )
                failed += 1
                continue

            logger.info(
                "[%d/%d] Generated: %s", i, total, generated_question[:100]
            )

            # ── Immediate incremental write ───────────────────────────────
            row: dict[str, str] = {
                "id": record_id,
                "context": parsed["context"],
                "ground_truth_question": parsed["ground_truth_question"],
                "generated_question": generated_question,
                "provider": self._provider.provider_name,
                "model_id": self._provider.model_name,
            }
            self._append_row(row)
            processed_ids.add(record_id)
            succeeded += 1

            # ── Rate limit courtesy sleep ─────────────────────────────────
            if i < total:
                time.sleep(self._request_interval)

        logger.info(
            "Pipeline complete — total=%d, succeeded=%d, skipped=%d, failed=%d",
            total, succeeded, skipped, failed,
        )
        logger.info("Results saved to: %s", self._output_csv)


# ══════════════════════════════════════════════════════════════════════════
# Entrypoint
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Load .env file if present ─────────────────────────────────────────
    dotenv_path = Path(".env")
    loaded_env_keys: list[str] = []
    if dotenv_path.exists():
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(
                key.strip(),
                value.strip().strip('"').strip("'"),
            )
            loaded_env_keys.append(key.strip())
        logger.info(
            "Loaded %d env vars from .env: %s",
            len(loaded_env_keys),
            ", ".join(loaded_env_keys) if loaded_env_keys else "<none>",
        )
    else:
        logger.info("No .env file found in current directory (%s).", Path.cwd())

    _print_startup_diagnostics()

    fresh_start = _env_flag("FRESH_START", default=False)
    logger.info("FRESH_START: %s", fresh_start)

    # ── Choose your provider here ─────────────────────────────────────────
    # Option A: Gemini (requires VERTEX_API_KEY in .env)
    provider = GeminiProvider(
        model_id="gemini-2.5-flash",
        temperature=0.0,
    )

    # Option B: Anthropic Claude (requires ANTHROPIC_API_KEY in .env)
    # provider = AnthropicProvider(
    #     model_id="claude-sonnet-4-6",
    #     temperature=0.0,
    # )

    # ── Initialise and run the pipeline ──────────────────────────────────
    pipeline = QuestionGeneratorPipeline(
        provider=provider,
        output_csv=Path("outputs/phase1_results.csv"),
        instruction_file=Path("instruction.txt"),
        pilot_size=20,
        request_interval=3.0,
        fresh_start=fresh_start,
    )

    pipeline.run()
