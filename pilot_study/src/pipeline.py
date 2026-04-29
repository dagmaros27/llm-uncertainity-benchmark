"""Phase 1 pipeline: PatientSimulator + Phase1Pipeline.

Pipeline flow per record:
  1. Show model ehr_summary + question + answer options (A-D)
  2. Model returns: clarifying_question, preliminary_assessment (letter A/B/C/D), confidence
  3. PatientSimulator answers the CQ using combined patient/nurse/specialist context
  4. Show model full context + clarifying exchange + answer options again
  5. Model returns: updated_assessment (letter A/B/C/D), updated_confidence
  6. Evaluate: correctness (exact letter match) + confidence delta

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
    parse_json_response,
)
from config import PHASE1_FIELDS, PHASE1_MULTITURN_FIELDS, N_CQ_TURNS, REQUEST_INTERVAL

logger = logging.getLogger(__name__)


# ── Structured output schemas ──────────────────────────────────────────────

TURN_0_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "clarifying_question": types.Schema(
            type=types.Type.STRING,
            description="One clarifying question that would most help discriminate between the answer options.",
        ),
        "preliminary_assessment": types.Schema(
            type=types.Type.STRING,
            enum=["A", "B", "C", "D"],
            description="Best current answer — exactly one letter: A, B, C, or D.",
        ),
        "confidence": types.Schema(
            type=types.Type.INTEGER,
            description="Confidence in the preliminary answer from 0 (no idea) to 100 (certain).",
        ),
    },
    required=["clarifying_question", "preliminary_assessment", "confidence"],
)

TURN_1_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "updated_assessment": types.Schema(
            type=types.Type.STRING,
            enum=["A", "B", "C", "D"],
            description="Updated answer after receiving clarification — exactly one letter: A, B, C, or D.",
        ),
        "updated_confidence": types.Schema(
            type=types.Type.INTEGER,
            description="Updated confidence from 0 to 100.",
        ),
    },
    required=["updated_assessment", "updated_confidence"],
)

TURN_CONTINUATION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "updated_assessment": types.Schema(
            type=types.Type.STRING,
            enum=["A", "B", "C", "D"],
            description="Updated answer so far — exactly one letter: A, B, C, or D.",
        ),
        "confidence": types.Schema(
            type=types.Type.INTEGER,
            description="Confidence in the updated assessment from 0 to 100.",
        ),
        "clarifying_question": types.Schema(
            type=types.Type.STRING,
            description="Next clarifying question — must differ from all previous questions.",
        ),
    },
    required=["updated_assessment", "confidence", "clarifying_question"],
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

Return ONLY a valid JSON object:
{
  "updated_assessment": "<A, B, C, or D>",
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
                max_tokens=2048,
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

    def _turn_0(self, ehr_summary: str, question: str, options: dict) -> Optional[dict]:
        user_message = (
            f"Patient presentation:\n{ehr_summary.strip()}\n\n"
            f"Clinical question:\n{question.strip()}\n\n"
            f"Answer options:\n{format_answer_choices(options)}"
        )
        try:
            raw = self._provider.call(
                system_instruction=self._instruction,
                user_message=user_message,
                temperature=0.0,
                max_tokens=4096,
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
                max_tokens=4096,
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

            turn0 = self._turn_0(ehr_summary, question, options)
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
                    "is_correct_preliminary": prelim.upper() == correct_option.upper(),
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
                "is_correct_preliminary": prelim.upper() == correct_option.upper(),
                "is_correct_updated": updated.upper() == correct_option.upper(),
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


# ── Multi-Turn Phase 1 Pipeline ────────────────────────────────────────────

class MultiTurnPhase1Pipeline:
    """Three-round clarifying-question pipeline.

    Turn 0: model sees ehr_summary + question + options → preliminary A/B/C/D + CQ1 + confidence
    Turn k (1..n_turns-1): model sees full history → updated A/B/C/D + CQ_{k+1} + confidence
    Turn n_turns: model sees full history → final A/B/C/D + final_confidence (no more CQs)
    """

    def __init__(
        self,
        provider: LLMProvider,
        instruction_file: Path,
        continuation_instruction_file: Path,
        output_csv: Path,
        n_turns: int = N_CQ_TURNS,
        request_interval: float = REQUEST_INTERVAL,
    ) -> None:
        if not instruction_file.exists():
            raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
        if not continuation_instruction_file.exists():
            raise FileNotFoundError(f"Continuation instruction not found: {continuation_instruction_file}")
        self._instruction = instruction_file.read_text(encoding="utf-8").strip()
        self._continuation_instruction = continuation_instruction_file.read_text(encoding="utf-8").strip()
        self._provider = provider
        self._simulator = PatientSimulator(provider)
        self._output_csv = output_csv
        self._n_turns = n_turns
        self._request_interval = request_interval
        self._output_csv.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "MultiTurnPhase1Pipeline ready — provider=%s model=%s n_turns=%d output=%s",
            provider.provider_name, provider.model_name, n_turns, output_csv,
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
                csv.DictWriter(fh, fieldnames=PHASE1_MULTITURN_FIELDS).writeheader()

    def _append_row(self, row: dict) -> None:
        with self._output_csv.open("a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=PHASE1_MULTITURN_FIELDS).writerow(row)

    def _build_history_text(self, history: list[tuple[str, str]]) -> str:
        """Format completed (cq, sim_response) pairs as a readable block."""
        parts = []
        for i, (cq, resp) in enumerate(history, start=1):
            parts.append(f"Clarifying question {i}: {cq}\nPatient's answer: {resp}")
        return "\n\n".join(parts)

    def _turn_0(self, ehr_summary: str, question: str, options: dict) -> Optional[dict]:
        user_message = (
            f"Patient presentation:\n{ehr_summary.strip()}\n\n"
            f"Clinical question:\n{question.strip()}\n\n"
            f"Answer options:\n{format_answer_choices(options)}"
        )
        try:
            raw = self._provider.call(
                system_instruction=self._instruction,
                user_message=user_message,
                temperature=0.0,
                max_tokens=4096,
                expect_json=TURN_0_SCHEMA,
            )
        except SafetyBlockError as exc:
            return {"_blocked": True, "_reason": str(exc)}
        parsed = parse_json_response(raw)
        if parsed is None:
            logger.error("Turn 0 JSON parse failed. Raw: %.300s", raw)
            return None
        if not {"clarifying_question", "preliminary_assessment", "confidence"}.issubset(parsed.keys()):
            logger.error("Turn 0 missing keys. Got: %s", list(parsed.keys()))
            return None
        return parsed

    def _continuation_turn(
        self,
        ehr_summary: str,
        question: str,
        options: dict,
        history: list[tuple[str, str]],
    ) -> Optional[dict]:
        history_text = self._build_history_text(history)
        user_message = (
            f"Patient presentation:\n{ehr_summary.strip()}\n\n"
            f"Clinical question:\n{question.strip()}\n\n"
            f"Answer options:\n{format_answer_choices(options)}\n\n"
            f"{history_text}"
        )
        try:
            raw = self._provider.call(
                system_instruction=self._continuation_instruction,
                user_message=user_message,
                temperature=0.0,
                max_tokens=4096,
                expect_json=TURN_CONTINUATION_SCHEMA,
            )
        except SafetyBlockError as exc:
            return {"_blocked": True, "_reason": str(exc)}
        parsed = parse_json_response(raw)
        if parsed is None:
            logger.error("Continuation turn JSON parse failed. Raw: %.300s", raw)
            return None
        if not {"updated_assessment", "confidence", "clarifying_question"}.issubset(parsed.keys()):
            logger.error("Continuation turn missing keys. Got: %s", list(parsed.keys()))
            return None
        return parsed

    def _final_turn(
        self,
        ehr_summary: str,
        question: str,
        options: dict,
        history: list[tuple[str, str]],
    ) -> Optional[dict]:
        history_text = self._build_history_text(history)
        user_message = (
            f"Patient presentation:\n{ehr_summary.strip()}\n\n"
            f"Clinical question:\n{question.strip()}\n\n"
            f"Answer options:\n{format_answer_choices(options)}\n\n"
            f"{history_text}"
        )
        try:
            raw = self._provider.call(
                system_instruction=_POST_CLARIFICATION_INSTRUCTION,
                user_message=user_message,
                temperature=0.0,
                max_tokens=4096,
                expect_json=TURN_1_SCHEMA,
            )
        except SafetyBlockError as exc:
            return {"_blocked": True, "_reason": str(exc)}
        parsed = parse_json_response(raw)
        if parsed is None:
            logger.error("Final turn JSON parse failed. Raw: %.300s", raw)
            return None
        if not {"updated_assessment", "updated_confidence"}.issubset(parsed.keys()):
            logger.error("Final turn missing keys. Got: %s", list(parsed.keys()))
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
                logger.info("[%d/%d] SKIP — %s", i, total, record_id)
                skipped += 1
                continue

            logger.info("[%d/%d] Processing %s (difficulty=%s)", i, total, record_id, record.get("difficulty",""))
            ehr_summary       = record["ehr_summary"]
            question          = record["question"]
            options           = record["options"]
            correct_option    = record["correct_option"]
            correct_answer    = record["correct_answer"]
            simulator_context = record["simulator_context"]
            difficulty        = record.get("difficulty", "")

            # ── Turn 0 ────────────────────────────────────────────────────
            turn0 = self._turn_0(ehr_summary, question, options)
            if turn0 is None or turn0.get("_blocked"):
                self._append_row({f: "" for f in PHASE1_MULTITURN_FIELDS} | {
                    "id": record_id, "ehr_summary": ehr_summary, "question": question,
                    "difficulty": difficulty, "correct_option": correct_option,
                    "correct_answer": correct_answer,
                    "preliminary_assessment": "BLOCKED",
                    "was_blocked": True,
                    "finish_reason": (turn0 or {}).get("_reason", "PARSE_ERROR"),
                    "provider": self._provider.provider_name,
                    "model_id": self._provider.model_name,
                })
                failed += 1
                continue

            prelim      = str(turn0["preliminary_assessment"]).strip().upper()
            prelim_conf = int(turn0["confidence"])
            cqs         = [str(turn0["clarifying_question"]).strip()]
            assessments = [prelim]
            confidences = [prelim_conf]
            sim_responses: list[str] = []

            logger.info("  Prelim=%s(conf=%d) CQ1=%s", prelim, prelim_conf, cqs[0][:80])
            time.sleep(self._request_interval)

            # ── Clarification rounds ───────────────────────────────────────
            failed_mid = False
            for turn_idx in range(1, self._n_turns + 1):
                sim_resp = self._simulator.answer(cqs[turn_idx - 1], simulator_context)
                sim_responses.append(sim_resp)
                logger.info("  Sim[%d]: %s", turn_idx, sim_resp[:80])
                time.sleep(self._request_interval)

                history = list(zip(cqs, sim_responses))

                if turn_idx < self._n_turns:
                    result = self._continuation_turn(ehr_summary, question, options, history)
                    if result is None or result.get("_blocked"):
                        failed_mid = True
                        break
                    upd = str(result["updated_assessment"]).strip().upper()
                    conf = int(result["confidence"])
                    next_cq = str(result["clarifying_question"]).strip()
                    assessments.append(upd)
                    confidences.append(conf)
                    cqs.append(next_cq)
                    logger.info("  Turn%d=%s(conf=%d) CQ%d=%s", turn_idx, upd, conf, turn_idx + 1, next_cq[:80])
                else:
                    result = self._final_turn(ehr_summary, question, options, history)
                    if result is None or result.get("_blocked"):
                        failed_mid = True
                        break
                    final = str(result["updated_assessment"]).strip().upper()
                    final_conf = int(result["updated_confidence"])
                    assessments.append(final)
                    confidences.append(final_conf)
                    logger.info("  Final=%s(conf=%d)", final, final_conf)

                time.sleep(self._request_interval)

            if failed_mid:
                failed += 1
                continue

            # ── Build and write row ────────────────────────────────────────
            def is_correct(letter: str) -> bool:
                return letter == correct_option.upper()

            row: dict = {
                "id": record_id,
                "ehr_summary": ehr_summary,
                "question": question,
                "difficulty": difficulty,
                "correct_option": correct_option,
                "correct_answer": correct_answer,
                "preliminary_assessment":  assessments[0],
                "preliminary_confidence":  confidences[0],
                "is_correct_preliminary":  is_correct(assessments[0]),
                "cq_1": cqs[0],
                "patient_response_1": sim_responses[0],
                "assessment_1":  assessments[1],
                "confidence_1":  confidences[1],
                "is_correct_1":  is_correct(assessments[1]),
                "cq_2": cqs[1],
                "patient_response_2": sim_responses[1],
                "assessment_2":  assessments[2],
                "confidence_2":  confidences[2],
                "is_correct_2":  is_correct(assessments[2]),
                "cq_3": cqs[2],
                "patient_response_3": sim_responses[2],
                "final_assessment":  assessments[3],
                "final_confidence":  confidences[3],
                "is_correct_final":  is_correct(assessments[3]),
                "provider":     self._provider.provider_name,
                "model_id":     self._provider.model_name,
                "finish_reason": "STOP",
                "was_blocked":  False,
            }
            self._append_row(row)
            processed_ids.add(record_id)
            succeeded += 1

        logger.info(
            "MultiTurn Phase 1 complete — total=%d succeeded=%d skipped=%d failed=%d",
            total, succeeded, skipped, failed,
        )
        logger.info("Results saved to: %s", self._output_csv)
