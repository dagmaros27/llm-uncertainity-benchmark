"""
phase1_core.py
==============
Core classes and functions for Phase 1 of the medical uncertainty experiment.
Import this module from the experiment notebook.

Pipeline flow per record:
  1. Show model only layer_0 (chief complaint)
  2. Model returns: clarifying_question, preliminary_assessment, confidence
  3. Simulate patient response using layer_1 (hidden clinical details)
  4. Show model layer_0 + clarifying exchange
  5. Model returns: updated_assessment, updated_confidence
  6. Evaluate: was the final assessment correct? did confidence change?
"""

from __future__ import annotations

import abc
import csv
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import tenacity
from google import genai
from google.genai import types

# ── Logging ────────────────────────────────────────────────────────────────
logger = logging.getLogger("phase1_core")


class SafetyBlockError(Exception):
    """Raised when a model response is blocked by safety filters.
    Not retried — safety blocks are deterministic for a given input."""
    pass

# ── CSV output fields ──────────────────────────────────────────────────────
PHASE1_FIELDS: list[str] = [
    "id",
    "layer_0",
    "layer_1",
    "clarifying_question",
    "cq_type",                  # to be filled by LM judge later
    "patient_response",         # simulated from layer_1
    "preliminary_assessment",
    "preliminary_confidence",
    "updated_assessment",
    "updated_confidence",
    "correct_answer_text",
    "correct_answer_idx",
    "is_correct_preliminary",   # bool: preliminary_assessment matches correct
    "is_correct_updated",       # bool: updated_assessment matches correct
    "confidence_delta",         # updated_confidence - preliminary_confidence
    "provider",
    "model_id",
    "meta_info",
    "finish_reason",
    "was_blocked",
]

# ── Structured output schemas ──────────────────────────────────────────────
# These tell Gemini exactly what JSON structure to return.
# Using native response_schema forces valid structured output every time.

TURN_0_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "clarifying_question": types.Schema(
            type=types.Type.STRING,
            description="One focused clarifying question targeting the most critical missing clinical information.",
        ),
        "preliminary_assessment": types.Schema(
            type=types.Type.STRING,
            description="Best current clinical assessment or most likely diagnosis given only the complaint.",
        ),
        "confidence": types.Schema(
            type=types.Type.INTEGER,
            description="Confidence in the preliminary assessment from 0 (no idea) to 100 (certain).",
        ),
    },
    required=["clarifying_question", "preliminary_assessment", "confidence"],
)

TURN_1_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "updated_assessment": types.Schema(
            type=types.Type.STRING,
            description="Updated clinical assessment or most likely diagnosis after receiving clarification.",
        ),
        "updated_confidence": types.Schema(
            type=types.Type.INTEGER,
            description="Updated confidence from 0 to 100.",
        ),
    },
    required=["updated_assessment", "updated_confidence"],
)

SIMULATOR_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "answer": types.Schema(
            type=types.Type.STRING,
            description="The patient's answer to the clinician's question, based only on the clinical details provided.",
        ),
    },
    required=["answer"],
)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def load_dotenv(path: Path = Path(".env")) -> None:
    """Load key=value pairs from a .env file into os.environ."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(
            key.strip(),
            value.strip().strip('"').strip("'"),
        )


def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


class ModelResponse:
    """Wraps a model response with text and finish_reason."""
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
    """
    Extract final output text and finish_reason from a Gemini response.
    Returns a ModelResponse so callers can distinguish safety blocks
    from legitimate completions.
    """
    # Extract finish_reason from the candidate
    finish_reason = ""
    try:
        candidate = response.candidates[0]
        raw_reason = getattr(candidate, "finish_reason", None)
        if raw_reason is not None:
            finish_reason = str(raw_reason.name) if hasattr(raw_reason, "name") else str(raw_reason)
    except (IndexError, AttributeError):
        pass

    # Extract text from non-thinking parts
    parts_text: list[str] = []
    try:
        for part in response.candidates[0].content.parts:
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", "")
            if text:
                parts_text.append(text)
    except (IndexError, AttributeError):
        pass

    if parts_text:
        text = clean_text("\n".join(parts_text))
    else:
        text = clean_text(getattr(response, "text", ""))

    if finish_reason not in ("STOP", "MAX_TOKENS", "") and not parts_text:
        logger.warning("Model response blocked or incomplete — finish_reason=%s", finish_reason)

    return ModelResponse(text=text, finish_reason=finish_reason)


def parse_json_response(raw: str) -> Optional[dict]:
    """
    Strip markdown fences and parse JSON from a model response.
    Returns None if parsing fails.
    """
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # If wrapper text appears, attempt to parse the largest JSON-object span.
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed: %s | raw: %s", exc, raw[:200])
        return None


def is_assessment_correct(assessment: str, correct_text: str) -> bool:
    """
    Heuristic check: does the model's assessment mention the correct answer?
    Case-insensitive substring match — good enough for pilot analysis.
    """
    if not assessment or not correct_text:
        return False
    return correct_text.lower() in assessment.lower()


def format_answer_choices(choices: dict) -> str:
    """Format the answer choices dict as a readable string for post-clarification prompt."""
    return "\n".join(f"{k}. {v}" for k, v in choices.items())


# ══════════════════════════════════════════════════════════════════════════
# Strategy — Abstract LLM Provider
# ══════════════════════════════════════════════════════════════════════════

class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    @abc.abstractmethod
    def call(
        self,
        system_instruction: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        expect_json: bool | types.Schema = False,
    ) -> str:
        """Make a single API call and return the response text."""
        ...

    @property
    @abc.abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str: ...


# ══════════════════════════════════════════════════════════════════════════
# Concrete Provider — Gemini
# ══════════════════════════════════════════════════════════════════════════

class GeminiProvider(LLMProvider):
    """Google Gemini via google-genai SDK. Reads VERTEX_API_KEY from env."""

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        default_temperature: float = 0.0,
        api_version: str = "v1beta",
    ) -> None:
        api_key = os.environ.get("VERTEX_API_KEY")
        if not api_key:
            raise EnvironmentError("VERTEX_API_KEY not set.")
        self._api_key = api_key
        self._api_version = api_version
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version=api_version),
        )
        self._model_id = model_id
        self._default_temperature = default_temperature
        logger.info("GeminiProvider ready — model=%s api_version=%s", model_id, api_version)

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_id

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=4, max=60),
        stop=tenacity.stop_after_attempt(6),
        retry=tenacity.retry_if_not_exception_type(SafetyBlockError),
        before_sleep=lambda rs: logger.warning(
            "Gemini retry — sleeping %.0fs (attempt %d)",
            rs.next_action.sleep if rs.next_action else 0,
            rs.attempt_number,
        ),
        reraise=True,
    )
    def call(
        self,
        system_instruction: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        expect_json: bool | types.Schema = False,
    ) -> str:
        full_prompt = f"{system_instruction.strip()}\n\n{user_message.strip()}"
        call_attempts: list[tuple[str, str]] = [(self._model_id, self._api_version)]

        # Common fallback when a model is not enabled for a given project/key.
        if self._model_id.startswith("gemini-2.5"):
            call_attempts.append(("gemini-2.0-flash", self._api_version))

        # Some environments still require v1beta for certain model behaviors.
        if self._api_version != "v1beta":
            call_attempts.append((self._model_id, "v1beta"))
            if self._model_id.startswith("gemini-2.5"):
                call_attempts.append(("gemini-2.0-flash", "v1beta"))

        # Deduplicate while preserving order.
        seen: set[tuple[str, str]] = set()
        unique_attempts: list[tuple[str, str]] = []
        for attempt in call_attempts:
            if attempt not in seen:
                seen.add(attempt)
                unique_attempts.append(attempt)

        last_error: Optional[Exception] = None
        for model_id, api_version in unique_attempts:
            try:
                client = self._client
                if api_version != self._api_version:
                    client = genai.Client(
                        api_key=self._api_key,
                        http_options=types.HttpOptions(api_version=api_version),
                    )

                config_kwargs: dict = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": 0.95,
                }
                if expect_json is not False:
                    config_kwargs["response_mime_type"] = "application/json"
                    if isinstance(expect_json, types.Schema):
                        config_kwargs["response_schema"] = expect_json

                response = client.models.generate_content(
                    model=model_id,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )

                if model_id != self._model_id:
                    logger.warning(
                        "GeminiProvider fallback model used: requested=%s actual=%s",
                        self._model_id,
                        model_id,
                    )
                if api_version != self._api_version:
                    logger.warning(
                        "GeminiProvider fallback api_version used: requested=%s actual=%s",
                        self._api_version,
                        api_version,
                    )

                model_response = extract_non_thinking_text(response)
                if model_response.was_blocked:
                    logger.warning(
                        "Response blocked by safety filters — finish_reason=%s model=%s",
                        model_response.finish_reason, model_id,
                    )
                    # Raise so tenacity does NOT retry — safety blocks are deterministic
                    raise SafetyBlockError(
                        f"Response blocked: finish_reason={model_response.finish_reason}"
                    )
                return model_response.text
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Gemini call failed (model=%s api_version=%s): %s",
                    model_id,
                    api_version,
                    exc,
                )

        raise RuntimeError(
            f"All Gemini call attempts failed. Last error: {last_error}"
        ) from last_error


# ══════════════════════════════════════════════════════════════════════════
# Patient Simulator
# ══════════════════════════════════════════════════════════════════════════

SIMULATOR_INSTRUCTION = """You are a patient or a patient's chart. You have been given the full clinical details of a case.
A clinician has asked you a question. Answer it using ONLY information that is present in the clinical details provided.
If the question asks about something not mentioned in the clinical details, say "That information is not available."
Be concise. Answer in one or two sentences. Do not volunteer extra information beyond what was asked.

Return ONLY a JSON object with a single key called \"answer\" containing your response string. No other text."""


class PatientSimulator:
    """
    Simulates a patient answering a clarifying question using layer_1
    as the hidden ground truth clinical details.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def answer(self, clarifying_question: str, layer_1: str) -> str:
        """
        Given a clarifying question and the hidden clinical details,
        simulate the patient's answer.

        Args:
            clarifying_question: The question the model asked.
            layer_1:             The hidden clinical details to answer from.

        Returns:
            A short simulated patient response string.
        """
        user_message = (
            f"Clinical details:\n{layer_1.strip()}\n\n"
            f"Clinician's question:\n{clarifying_question.strip()}"
        )
        try:
            raw = self._provider.call(
                system_instruction=SIMULATOR_INSTRUCTION,
                user_message=user_message,
                temperature=0.0,
                max_tokens=256,
                expect_json=SIMULATOR_SCHEMA,
            )
        except SafetyBlockError:
            logger.warning("Patient simulator blocked by safety filter.")
            return "That information is not available."

        # Try structured parse first
        parsed = parse_json_response(raw)
        if parsed and "answer" in parsed:
            answer = str(parsed["answer"]).strip()
        else:
            # Fallback — strip any JSON artifacts and use raw text
            answer = raw.strip().strip("```json").strip("```").strip()
            # If it still looks like a partial JSON, extract value after colon
            if answer.startswith("{") or "answer" in answer.lower():
                import re as _re
                match = _re.search(r'"answer"\s*:\s*"([^"]+)"', answer)
                answer = match.group(1) if match else "That information is not available."

        logger.debug("Simulated patient response: %s", answer[:100])
        return answer


# ══════════════════════════════════════════════════════════════════════════
# Phase 1 Pipeline
# ══════════════════════════════════════════════════════════════════════════

class Phase1Pipeline:
    """
    Runs the full two-turn uncertainty experiment for each record:

    Turn 0: Model sees layer_0 only → returns clarifying_question,
            preliminary_assessment, preliminary_confidence

    Turn 1: Patient simulator answers the clarifying question using layer_1
            → Model sees layer_0 + clarifying exchange + answer choices
            → returns updated_assessment, updated_confidence

    Evaluation: compare both assessments to correct_answer_text
    """

    POST_CLARIFICATION_INSTRUCTION = """You are an experienced clinician. You have just received an answer to your clarifying question.
You now have the patient's initial complaint and one additional piece of clinical information.

Based on all the information available to you, provide:
1. Your updated clinical assessment — the most likely diagnosis or condition
2. Your updated confidence in this assessment (0 to 100)

The answer choices for the clinical question are provided below. Your updated assessment should correspond to one of them.

Return ONLY a valid JSON object with no other text:
{
  "updated_assessment": "<your updated diagnosis or clinical assessment>",
  "updated_confidence": <integer between 0 and 100>
}"""

    def __init__(
        self,
        provider: LLMProvider,
        instruction_file: Path,
        output_csv: Path,
        request_interval: float = 3.0,
    ) -> None:
        self._provider = provider
        self._simulator = PatientSimulator(provider)
        self._output_csv = output_csv
        self._request_interval = request_interval

        if not instruction_file.exists():
            raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
        self._instruction = instruction_file.read_text(encoding="utf-8").strip()
        logger.info("Instruction loaded (%d chars)", len(self._instruction))

        self._output_csv.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Phase1Pipeline ready — provider=%s model=%s output=%s",
            provider.provider_name, provider.model_name, output_csv,
        )

    # ── Resumability ───────────────────────────────────────────────────────

    def _load_processed_ids(self) -> set[str]:
        processed: set[str] = set()
        if not self._output_csv.exists():
            return processed
        with self._output_csv.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("id"):
                    processed.add(row["id"])
        logger.info("Resumability: %d records already processed.", len(processed))
        return processed

    def _write_header_if_needed(self) -> None:
        if not self._output_csv.exists():
            with self._output_csv.open("w", encoding="utf-8", newline="") as fh:
                csv.DictWriter(fh, fieldnames=PHASE1_FIELDS).writeheader()

    def _append_row(self, row: dict) -> None:
        with self._output_csv.open("a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=PHASE1_FIELDS).writerow(row)

    # ── Turn 0 — Initial assessment ────────────────────────────────────────

    def _turn_0(self, layer_0: str) -> Optional[dict]:
        """
        Show the model only the chief complaint.
        Returns dict with clarifying_question, preliminary_assessment, confidence
        or None on failure.
        """
        user_message = f"Patient complaint:\n{layer_0.strip()}"
        try:
            raw = self._provider.call(
                system_instruction=self._instruction,
                user_message=user_message,
                temperature=0.0,
                max_tokens=1024,
                expect_json=TURN_0_SCHEMA,
            )
        except SafetyBlockError as exc:
            logger.warning("Turn 0 safety block: %s", exc)
            return {"_blocked": True, "_reason": str(exc)}
        parsed = parse_json_response(raw)
        if parsed is None:
            logger.error("Turn 0 JSON parse failed. Raw: %s", raw[:300])
            return None
        required = {"clarifying_question", "preliminary_assessment", "confidence"}
        if not required.issubset(parsed.keys()):
            logger.error("Turn 0 missing keys. Got: %s", list(parsed.keys()))
            return None
        return parsed

    # ── Turn 1 — Updated assessment after clarification ────────────────────

    def _turn_1(
        self,
        layer_0: str,
        clarifying_question: str,
        patient_response: str,
        answer_choices: dict,
    ) -> Optional[dict]:
        """
        Show the model the original complaint + clarifying exchange + answer choices.
        Returns dict with updated_assessment and updated_confidence or None on failure.
        """
        choices_text = format_answer_choices(answer_choices)
        user_message = (
            f"Patient complaint:\n{layer_0.strip()}\n\n"
            f"Your clarifying question:\n{clarifying_question.strip()}\n\n"
            f"Patient's answer:\n{patient_response.strip()}\n\n"
            f"Answer choices:\n{choices_text}"
        )
        try:
            raw = self._provider.call(
                system_instruction=self.POST_CLARIFICATION_INSTRUCTION,
                user_message=user_message,
                temperature=0.0,
                max_tokens=512,
                expect_json=TURN_1_SCHEMA,
            )
        except SafetyBlockError as exc:
            logger.warning("Turn 1 safety block: %s", exc)
            return {"_blocked": True, "_reason": str(exc)}
        parsed = parse_json_response(raw)
        if parsed is None:
            logger.error("Turn 1 JSON parse failed. Raw: %s", raw[:300])
            return None
        required = {"updated_assessment", "updated_confidence"}
        if not required.issubset(parsed.keys()):
            logger.error("Turn 1 missing keys. Got: %s", list(parsed.keys()))
            return None
        return parsed

    # ── Main run loop ──────────────────────────────────────────────────────

    def run(self, records: list[dict]) -> None:
        """
        Process a list of prepared MedQA records end to end.

        Args:
            records: List of dicts from medqa_prepared.json
        """
        processed_ids = self._load_processed_ids()
        self._write_header_if_needed()

        total = len(records)
        succeeded = skipped = failed = 0

        for i, record in enumerate(records, start=1):
            record_id = record["id"]

            if record_id in processed_ids:
                logger.info("[%d/%d] SKIP — %s already done.", i, total, record_id)
                skipped += 1
                continue

            logger.info("[%d/%d] Processing %s", i, total, record_id)
            layer_0 = record["layer_0"]
            layer_1 = record["layer_1"]
            correct_text = record["correct_answer_text"]
            correct_idx = record["correct_answer_idx"]
            answer_choices = record["answer_choices"]

            # ── Turn 0 ────────────────────────────────────────────────────
            turn0 = self._turn_0(layer_0)
            if turn0 is None:
                failed += 1
                continue

            if turn0.get("_blocked"):
                logger.warning("  Record %s blocked at Turn 0 — skipping.", record_id)
                self._append_row({
                    "id": record_id, "layer_0": layer_0, "layer_1": layer_1,
                    "clarifying_question": "BLOCKED", "cq_type": "",
                    "patient_response": "", "preliminary_assessment": "BLOCKED",
                    "preliminary_confidence": -1, "updated_assessment": "BLOCKED",
                    "updated_confidence": -1, "correct_answer_text": correct_text,
                    "correct_answer_idx": correct_idx,
                    "is_correct_preliminary": False, "is_correct_updated": False,
                    "confidence_delta": 0, "provider": self._provider.provider_name,
                    "model_id": self._provider.model_name,
                    "meta_info": record.get("meta_info", ""),
                    "finish_reason": turn0.get("_reason", "SAFETY"),
                    "was_blocked": True,
                })
                processed_ids.add(record_id)
                failed += 1
                continue

            cq = str(turn0.get("clarifying_question", "")).strip()
            prelim = str(turn0.get("preliminary_assessment", "")).strip()
            prelim_conf = int(turn0.get("confidence", 0))

            logger.info("  CQ: %s", cq[:100])
            logger.info("  Prelim: %s (conf=%d)", prelim[:80], prelim_conf)
            time.sleep(self._request_interval)

            # ── Patient simulation ────────────────────────────────────────
            patient_response = self._simulator.answer(cq, layer_1)
            logger.info("  Patient: %s", patient_response[:100])
            time.sleep(self._request_interval)

            # ── Turn 1 ────────────────────────────────────────────────────
            turn1 = self._turn_1(layer_0, cq, patient_response, answer_choices)
            if turn1 is None:
                failed += 1
                continue

            updated = str(turn1.get("updated_assessment", "")).strip()
            updated_conf = int(turn1.get("updated_confidence", 0))

            logger.info("  Updated: %s (conf=%d)", updated[:80], updated_conf)

            # ── Evaluation ────────────────────────────────────────────────
            is_correct_prelim = is_assessment_correct(prelim, correct_text)
            is_correct_updated = is_assessment_correct(updated, correct_text)
            conf_delta = updated_conf - prelim_conf

            # ── Write immediately ─────────────────────────────────────────
            self._append_row({
                "id": record_id,
                "layer_0": layer_0,
                "layer_1": layer_1,
                "clarifying_question": cq,
                "cq_type": "",               # filled by LM judge in Phase 2
                "patient_response": patient_response,
                "preliminary_assessment": prelim,
                "preliminary_confidence": prelim_conf,
                "updated_assessment": updated,
                "updated_confidence": updated_conf,
                "correct_answer_text": correct_text,
                "correct_answer_idx": correct_idx,
                "is_correct_preliminary": is_correct_prelim,
                "is_correct_updated": is_correct_updated,
                "confidence_delta": conf_delta,
                "provider": self._provider.provider_name,
                "model_id": self._provider.model_name,
                "meta_info": record.get("meta_info", ""),
                "finish_reason": "STOP",
                "was_blocked": False,
            })

            processed_ids.add(record_id)
            succeeded += 1
            time.sleep(self._request_interval)

        logger.info(
            "Phase 1 complete — total=%d succeeded=%d skipped=%d failed=%d",
            total, succeeded, skipped, failed,
        )
        logger.info("Results saved to: %s", self._output_csv)
