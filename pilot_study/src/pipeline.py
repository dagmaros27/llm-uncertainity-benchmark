"""Phase 1 pipeline: PatientSimulator + Phase1Pipeline.

Pipeline flow per record:
  1. Show model ehr_summary + question (no answer options)
  2. Model returns: clarifying_question, preliminary_assessment, confidence
  3. PatientSimulator answers the CQ using combined patient/nurse/specialist context
  4. Show model ehr_summary + question + answer options + clarifying exchange
  5. Model returns: updated_assessment, updated_confidence
  6. Evaluate: correctness + confidence delta

Expected record keys:
  id, ehr_summary, question, options (dict A-D),
  correct_option, correct_answer, simulator_context, difficulty
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Optional

from google.genai import types

from .providers import LLMProvider
from .utils import (
    SafetyBlockError,
    format_answer_choices,
    is_assessment_correct,
    parse_json_response,
)
from config import PHASE1_FIELDS, REQUEST_INTERVAL

logger = logging.getLogger(__name__)


# ── Structured output schemas ──────────────────────────────────────────────

TURN_0_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "clarifying_question": types.Schema(
            type=types.Type.STRING,
            description="One clarifying question that would most reduce uncertainty about the assessment.",
        ),
        "preliminary_assessment": types.Schema(
            type=types.Type.STRING,
            description="Best current clinical assessment or most likely diagnosis/management given the available information.",
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
            description="Updated clinical assessment after receiving clarification — must match one of the answer choices.",
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
            description="The patient's answer to the clinician's question.",
        ),
    },
    required=["answer"],
)

_SIMULATOR_INSTRUCTION = """You are a clinical information source for a patient case. \
You have been given the complete clinical details available for this case.
A clinician has asked you a question. Answer it using ONLY information present in the \
clinical details provided. If the question asks about something not mentioned, say \
"That information is not available." Be concise — one or two sentences. \
Do not volunteer extra information.

Return ONLY a JSON object: {"answer": "<your response>"}"""

_POST_CLARIFICATION_INSTRUCTION = """You are an experienced clinician. \
You have received an answer to your clarifying question.

Based on all available information, select the most appropriate answer to the clinical \
question from the choices provided and state your updated confidence.

Your updated assessment must correspond to one of the answer choices.

Return ONLY a valid JSON object:
{
  "updated_assessment": "<exact text of your chosen answer>",
  "updated_confidence": <integer 0-100>
}"""


# ── Patient Simulator ──────────────────────────────────────────────────────

class PatientSimulator:
    """Simulates a clinical information source answering a CQ from partitioned context."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def answer(self, clarifying_question: str, simulator_context: str) -> str:
        user_message = (
            f"Clinical details:\n{simulator_context.strip()}\n\n"
            f"Clinician's question:\n{clarifying_question.strip()}"
        )
        try:
            raw = self._provider.call(
                system_instruction=_SIMULATOR_INSTRUCTION,
                user_message=user_message,
                temperature=0.0,
                max_tokens=256,
                expect_json=SIMULATOR_SCHEMA,
            )
        except SafetyBlockError:
            logger.warning("Patient simulator blocked by safety filter.")
            return "That information is not available."

        parsed = parse_json_response(raw)
        if parsed and "answer" in parsed:
            return str(parsed["answer"]).strip()

        import re as _re
        match = _re.search(r'"answer"\s*:\s*"([^"]+)"', raw)
        return match.group(1) if match else "That information is not available."


# ── Phase 1 Pipeline ───────────────────────────────────────────────────────

class Phase1Pipeline:
    """Runs the full two-turn uncertainty experiment for each MedQA record."""

    def __init__(
        self,
        provider: LLMProvider,
        instruction_file: Path,
        output_csv: Path,
        request_interval: float = REQUEST_INTERVAL,
    ) -> None:
        if not instruction_file.exists():
            raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
        self._instruction = instruction_file.read_text(encoding="utf-8").strip()
        self._provider = provider
        self._simulator = PatientSimulator(provider)
        self._output_csv = output_csv
        self._request_interval = request_interval
        self._output_csv.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Phase1Pipeline ready — provider=%s model=%s output=%s",
            provider.provider_name, provider.model_name, output_csv,
        )

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

    def _turn_0(self, ehr_summary: str, question: str) -> Optional[dict]:
        user_message = (
            f"Patient presentation:\n{ehr_summary.strip()}\n\n"
            f"Clinical question:\n{question.strip()}"
        )
        try:
            raw = self._provider.call(
                system_instruction=self._instruction,
                user_message=user_message,
                temperature=0.0,
                max_tokens=1024,
                expect_json=TURN_0_SCHEMA,
            )
        except SafetyBlockError as exc:
            return {"_blocked": True, "_reason": str(exc)}
        parsed = parse_json_response(raw)
        if parsed is None:
            logger.error("Turn 0 JSON parse failed. Raw: %.300s", raw)
            return None
        required = {"clarifying_question", "preliminary_assessment", "confidence"}
        if not required.issubset(parsed.keys()):
            logger.error("Turn 0 missing keys. Got: %s", list(parsed.keys()))
            return None
        return parsed

    def _turn_1(
        self,
        ehr_summary: str,
        question: str,
        clarifying_question: str,
        patient_response: str,
        options: dict,
    ) -> Optional[dict]:
        user_message = (
            f"Patient presentation:\n{ehr_summary.strip()}\n\n"
            f"Your clarifying question:\n{clarifying_question.strip()}\n\n"
            f"Patient's answer:\n{patient_response.strip()}\n\n"
            f"Clinical question:\n{question.strip()}\n\n"
            f"Answer choices:\n{format_answer_choices(options)}"
        )
        try:
            raw = self._provider.call(
                system_instruction=_POST_CLARIFICATION_INSTRUCTION,
                user_message=user_message,
                temperature=0.0,
                max_tokens=512,
                expect_json=TURN_1_SCHEMA,
            )
        except SafetyBlockError as exc:
            return {"_blocked": True, "_reason": str(exc)}
        parsed = parse_json_response(raw)
        if parsed is None:
            logger.error("Turn 1 JSON parse failed. Raw: %.300s", raw)
            return None
        required = {"updated_assessment", "updated_confidence"}
        if not required.issubset(parsed.keys()):
            logger.error("Turn 1 missing keys. Got: %s", list(parsed.keys()))
            return None
        return parsed

    def run(self, records: list[dict]) -> None:
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
            ehr_summary       = record["ehr_summary"]
            question          = record["question"]
            options           = record["options"]
            correct_option    = record["correct_option"]
            correct_answer    = record["correct_answer"]
            simulator_context = record["simulator_context"]
            difficulty        = record.get("difficulty", "")

            turn0 = self._turn_0(ehr_summary, question)
            if turn0 is None:
                failed += 1
                continue

            if turn0.get("_blocked"):
                self._append_row({
                    "id": record_id,
                    "ehr_summary": ehr_summary,
                    "question": question,
                    "clarifying_question": "BLOCKED",
                    "cq_type": "",
                    "patient_response": "",
                    "preliminary_assessment": "BLOCKED",
                    "preliminary_confidence": -1,
                    "updated_assessment": "BLOCKED",
                    "updated_confidence": -1,
                    "correct_option": correct_option,
                    "correct_answer": correct_answer,
                    "is_correct_preliminary": False,
                    "is_correct_updated": False,
                    "confidence_delta": 0,
                    "provider": self._provider.provider_name,
                    "model_id": self._provider.model_name,
                    "difficulty": difficulty,
                    "finish_reason": turn0.get("_reason", "SAFETY"),
                    "was_blocked": True,
                })
                processed_ids.add(record_id)
                failed += 1
                continue

            cq          = str(turn0["clarifying_question"]).strip()
            prelim      = str(turn0["preliminary_assessment"]).strip()
            prelim_conf = int(turn0["confidence"])
            logger.info("  CQ: %s", cq[:100])
            logger.info("  Prelim: %s (conf=%d)", prelim[:80], prelim_conf)
            time.sleep(self._request_interval)

            patient_response = self._simulator.answer(cq, simulator_context)
            logger.info("  Patient: %s", patient_response[:100])
            time.sleep(self._request_interval)

            turn1 = self._turn_1(ehr_summary, question, cq, patient_response, options)
            if turn1 is None:
                failed += 1
                continue

            if turn1.get("_blocked"):
                self._append_row({
                    "id": record_id,
                    "ehr_summary": ehr_summary,
                    "question": question,
                    "clarifying_question": cq,
                    "cq_type": "",
                    "patient_response": patient_response,
                    "preliminary_assessment": prelim,
                    "preliminary_confidence": prelim_conf,
                    "updated_assessment": "BLOCKED",
                    "updated_confidence": -1,
                    "correct_option": correct_option,
                    "correct_answer": correct_answer,
                    "is_correct_preliminary": is_assessment_correct(prelim, correct_answer),
                    "is_correct_updated": False,
                    "confidence_delta": 0,
                    "provider": self._provider.provider_name,
                    "model_id": self._provider.model_name,
                    "difficulty": difficulty,
                    "finish_reason": turn1.get("_reason", "SAFETY"),
                    "was_blocked": True,
                })
                processed_ids.add(record_id)
                failed += 1
                continue

            updated      = str(turn1["updated_assessment"]).strip()
            updated_conf = int(turn1["updated_confidence"])
            logger.info("  Updated: %s (conf=%d)", updated[:80], updated_conf)

            self._append_row({
                "id": record_id,
                "ehr_summary": ehr_summary,
                "question": question,
                "clarifying_question": cq,
                "cq_type": "",
                "patient_response": patient_response,
                "preliminary_assessment": prelim,
                "preliminary_confidence": prelim_conf,
                "updated_assessment": updated,
                "updated_confidence": updated_conf,
                "correct_option": correct_option,
                "correct_answer": correct_answer,
                "is_correct_preliminary": is_assessment_correct(prelim, correct_answer),
                "is_correct_updated": is_assessment_correct(updated, correct_answer),
                "confidence_delta": updated_conf - prelim_conf,
                "provider": self._provider.provider_name,
                "model_id": self._provider.model_name,
                "difficulty": difficulty,
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
