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
from config import (
    PHASE1_FIELDS, PHASE1_MULTITURN_FIELDS,
    MSDIALOG_PHASE1_FIELDS, MSDIALOG_PHASE1_MULTITURN_FIELDS,
    MSDIALOG_FLEX_FIELDS,
    N_CQ_TURNS, REQUEST_INTERVAL,
)

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

_SIMULATOR_INSTRUCTION = """You are a factual retrieval system for a patient case. \
Your role is strictly to report clinical facts that are explicitly documented in the \
provided clinical details — nothing more.

Rules you must follow:
1. Answer ONLY using information that is explicitly stated in the clinical details.
2. Report facts verbatim or as brief factual statements (e.g. "The patient reports chest pain for 3 days.", "SpO2 is 94% on room air.", "CT shows a left pleural effusion.").
3. Do NOT interpret, synthesize, or draw conclusions from the findings.
4. Do NOT suggest, name, or hint at any diagnosis, differential diagnosis, or clinical impression.
5. Do NOT combine multiple facts to imply a conclusion.
6. If the answer to the question is not explicitly documented, respond with exactly: "That information is not available."
7. Be concise — one or two sentences maximum.

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
        simulator_provider: Optional[LLMProvider] = None,
    ) -> None:
        if not instruction_file.exists():
            raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
        self._instruction = instruction_file.read_text(encoding="utf-8").strip()
        self._provider = provider
        self._simulator = PatientSimulator(simulator_provider or provider)
        self._output_csv = output_csv
        self._request_interval = request_interval
        self._output_csv.parent.mkdir(parents=True, exist_ok=True)
        sim_prov = simulator_provider or provider
        logger.info(
            "Phase1Pipeline ready — clinician=%s/%s simulator=%s/%s output=%s",
            provider.provider_name, provider.model_name,
            sim_prov.provider_name, sim_prov.model_name, output_csv,
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
        simulator_provider: Optional[LLMProvider] = None,
    ) -> None:
        if not instruction_file.exists():
            raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
        if not continuation_instruction_file.exists():
            raise FileNotFoundError(f"Continuation instruction not found: {continuation_instruction_file}")
        self._instruction = instruction_file.read_text(encoding="utf-8").strip()
        self._continuation_instruction = continuation_instruction_file.read_text(encoding="utf-8").strip()
        self._provider = provider
        self._simulator = PatientSimulator(simulator_provider or provider)
        self._output_csv = output_csv
        self._n_turns = n_turns
        self._request_interval = request_interval
        self._output_csv.parent.mkdir(parents=True, exist_ok=True)
        sim_prov = simulator_provider or provider
        logger.info(
            "MultiTurnPhase1Pipeline ready — clinician=%s/%s simulator=%s/%s n_turns=%d output=%s",
            provider.provider_name, provider.model_name,
            sim_prov.provider_name, sim_prov.model_name, n_turns, output_csv,
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

    def _run_conversation(
        self,
        ehr_summary: str,
        question: str,
        options: dict,
        simulator_context: str,
    ) -> Optional[dict]:
        """Run the full multi-turn conversation using proper role-alternating history.

        The Gemini ``contents`` list grows with each turn:
          Turn 0 :  [user(EHR+options)]
          After sim1: [user, model(CQ1+prelim), user(sim1)]
          After sim2: [..., model(CQ2+upd1), user(sim2)]
          ...
          Final:    [..., model(CQn+updn), user(simn)]  → final answer

        The ``system_instruction`` changes per API call (Turn-0 instruction for
        Turn 0, continuation instruction for intermediate turns, post-clarification
        for the final turn) but is never inserted into ``contents`` — it is passed
        via ``GenerateContentConfig`` so the conversation history stays clean.

        Returns a dict with keys: cqs, assessments, confidences, sim_responses.
        Returns None on parse failure, {"_blocked": True} on safety block.
        """
        formatted_options = format_answer_choices(options)

        # ── Turn 0 ────────────────────────────────────────────────────────
        turn0_user_text = (
            f"Patient presentation:\n{ehr_summary.strip()}\n\n"
            f"Clinical question:\n{question.strip()}\n\n"
            f"Answer options:\n{formatted_options}"
        )
        contents: list[dict] = [{"role": "user", "text": turn0_user_text}]

        try:
            raw0 = self._provider.call_multiturn(
                system_instruction=self._instruction,
                contents=contents,
                temperature=0.0,
                max_tokens=4096,
                expect_json=TURN_0_SCHEMA,
            )
        except SafetyBlockError:
            return {"_blocked": True}

        parsed0 = parse_json_response(raw0)
        if not parsed0 or not {"clarifying_question", "preliminary_assessment", "confidence"}.issubset(parsed0.keys()):
            logger.error("Turn 0 parse failed. Raw: %.300s", raw0)
            return None

        # Append model reply — raw JSON text becomes the model turn in history
        contents.append({"role": "model", "text": raw0})

        prelim      = str(parsed0["preliminary_assessment"]).strip().upper()
        prelim_conf = int(parsed0["confidence"])
        cqs         = [str(parsed0["clarifying_question"]).strip()]
        assessments = [prelim]
        confidences = [prelim_conf]
        sim_responses: list[str] = []

        logger.info("  Prelim=%s(conf=%d) CQ1=%s", prelim, prelim_conf, cqs[0][:80])
        time.sleep(self._request_interval)

        # ── Clarification rounds ───────────────────────────────────────────
        for turn_idx in range(1, self._n_turns + 1):
            sim_resp = self._simulator.answer(cqs[turn_idx - 1], simulator_context)
            sim_responses.append(sim_resp)
            logger.info("  Sim[%d]: %s", turn_idx, sim_resp[:80])
            time.sleep(self._request_interval)

            # Simulator answer is the next user turn in the conversation
            contents.append({"role": "user", "text": f"Patient's answer: {sim_resp}"})

            if turn_idx < self._n_turns:
                # ── Continuation turn (ask another CQ) ────────────────────
                try:
                    raw_cont = self._provider.call_multiturn(
                        system_instruction=self._continuation_instruction,
                        contents=contents,
                        temperature=0.0,
                        max_tokens=4096,
                        expect_json=TURN_CONTINUATION_SCHEMA,
                    )
                except SafetyBlockError:
                    return {"_blocked": True}

                parsed_cont = parse_json_response(raw_cont)
                if not parsed_cont or not {"updated_assessment", "confidence", "clarifying_question"}.issubset(parsed_cont.keys()):
                    logger.error("Continuation turn %d parse failed. Raw: %.300s", turn_idx, raw_cont)
                    return None

                contents.append({"role": "model", "text": raw_cont})

                upd      = str(parsed_cont["updated_assessment"]).strip().upper()
                conf     = int(parsed_cont["confidence"])
                next_cq  = str(parsed_cont["clarifying_question"]).strip()
                assessments.append(upd)
                confidences.append(conf)
                cqs.append(next_cq)
                logger.info("  Turn%d=%s(conf=%d) CQ%d=%s", turn_idx, upd, conf, turn_idx + 1, next_cq[:80])

            else:
                # ── Final turn (commit to answer, no more CQs) ─────────────
                try:
                    raw_final = self._provider.call_multiturn(
                        system_instruction=_POST_CLARIFICATION_INSTRUCTION,
                        contents=contents,
                        temperature=0.0,
                        max_tokens=4096,
                        expect_json=TURN_1_SCHEMA,
                    )
                except SafetyBlockError:
                    return {"_blocked": True}

                parsed_final = parse_json_response(raw_final)
                if not parsed_final or not {"updated_assessment", "updated_confidence"}.issubset(parsed_final.keys()):
                    logger.error("Final turn parse failed. Raw: %.300s", raw_final)
                    return None

                final      = str(parsed_final["updated_assessment"]).strip().upper()
                final_conf = int(parsed_final["updated_confidence"])
                assessments.append(final)
                confidences.append(final_conf)
                logger.info("  Final=%s(conf=%d)", final, final_conf)

            time.sleep(self._request_interval)

        return {
            "cqs": cqs,
            "assessments": assessments,
            "confidences": confidences,
            "sim_responses": sim_responses,
        }

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

            logger.info("[%d/%d] Processing %s (difficulty=%s)", i, total, record_id, record.get("difficulty", ""))
            ehr_summary       = record["ehr_summary"]
            question          = record["question"]
            options           = record["options"]
            correct_option    = record["correct_option"]
            correct_answer    = record["correct_answer"]
            simulator_context = record["simulator_context"]
            difficulty        = record.get("difficulty", "")

            result = self._run_conversation(ehr_summary, question, options, simulator_context)

            if result is None or result.get("_blocked"):
                self._append_row({f: "" for f in PHASE1_MULTITURN_FIELDS} | {
                    "id": record_id, "ehr_summary": ehr_summary, "question": question,
                    "difficulty": difficulty, "correct_option": correct_option,
                    "correct_answer": correct_answer,
                    "preliminary_assessment": "BLOCKED" if result and result.get("_blocked") else "PARSE_ERROR",
                    "was_blocked": bool(result and result.get("_blocked")),
                    "finish_reason": "SAFETY" if result and result.get("_blocked") else "PARSE_ERROR",
                    "provider": self._provider.provider_name,
                    "model_id": self._provider.model_name,
                })
                failed += 1
                continue

            cqs           = result["cqs"]
            assessments   = result["assessments"]
            confidences   = result["confidences"]
            sim_responses = result["sim_responses"]

            def is_correct(letter: str) -> bool:
                return letter == correct_option.upper()

            row: dict = {
                "id": record_id,
                "ehr_summary": ehr_summary,
                "question": question,
                "difficulty": difficulty,
                "correct_option": correct_option,
                "correct_answer": correct_answer,
                "preliminary_assessment": assessments[0],
                "preliminary_confidence": confidences[0],
                "is_correct_preliminary": is_correct(assessments[0]),
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
                "provider":      self._provider.provider_name,
                "model_id":      self._provider.model_name,
                "finish_reason": "STOP",
                "was_blocked":   False,
            }
            self._append_row(row)
            processed_ids.add(record_id)
            succeeded += 1

        logger.info(
            "MultiTurn Phase 1 complete — total=%d succeeded=%d skipped=%d failed=%d",
            total, succeeded, skipped, failed,
        )
        logger.info("Results saved to: %s", self._output_csv)


# ══════════════════════════════════════════════════════════════════════════════
# MS-Dialog Pipeline
# ══════════════════════════════════════════════════════════════════════════════
#
# Key differences from MedQA:
#   • No MCQ options — solutions are free-text
#   • Ground truth is accepted_answer text (semantic eval done separately)
#   • No difficulty field; records have title + category
#   • UserSimulator plays a non-technical user, not a clinical information source
# ─────────────────────────────────────────────────────────────────────────────

# ── MS-Dialog JSON schemas ─────────────────────────────────────────────────

MSDIALOG_TURN_0_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "clarifying_question": types.Schema(
            type=types.Type.STRING,
            description="One clarifying question that would most help diagnose the user's issue.",
        ),
        "preliminary_solution": types.Schema(
            type=types.Type.STRING,
            description="Best current troubleshooting approach or solution given available information.",
        ),
        "confidence": types.Schema(
            type=types.Type.INTEGER,
            description="Confidence in the preliminary solution from 0 (no idea) to 100 (certain).",
        ),
    },
    required=["clarifying_question", "preliminary_solution", "confidence"],
)

MSDIALOG_CONTINUATION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "updated_solution": types.Schema(
            type=types.Type.STRING,
            description="Updated troubleshooting approach incorporating the new information.",
        ),
        "confidence": types.Schema(
            type=types.Type.INTEGER,
            description="Updated confidence from 0 to 100.",
        ),
        "clarifying_question": types.Schema(
            type=types.Type.STRING,
            description="Next clarifying question — must differ from all previous questions.",
        ),
    },
    required=["updated_solution", "confidence", "clarifying_question"],
)

MSDIALOG_FINAL_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "final_solution": types.Schema(
            type=types.Type.STRING,
            description="Complete, actionable final solution based on all gathered information.",
        ),
        "confidence": types.Schema(
            type=types.Type.INTEGER,
            description="Final confidence from 0 to 100.",
        ),
    },
    required=["final_solution", "confidence"],
)

# ── MS-Dialog Flex schemas (optional-CQ pipeline) ─────────────────────────

MSDIALOG_FLEX_TURN_0_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "needed_clarification": types.Schema(
            type=types.Type.BOOLEAN,
            description="True if asking a clarifying question would meaningfully help; False if you can already provide a good solution.",
        ),
        "clarifying_question": types.Schema(
            type=types.Type.STRING,
            description="Your single clarifying question if needed_clarification is true; empty string otherwise.",
        ),
        "preliminary_solution": types.Schema(
            type=types.Type.STRING,
            description="Your best current solution based on available information.",
        ),
        "confidence": types.Schema(
            type=types.Type.INTEGER,
            description="Confidence in the preliminary solution from 0 to 100.",
        ),
    },
    required=["needed_clarification", "clarifying_question", "preliminary_solution", "confidence"],
)

MSDIALOG_FLEX_CONTINUATION_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "needed_clarification": types.Schema(
            type=types.Type.BOOLEAN,
            description="True if another clarifying question would meaningfully help; False if you are ready to commit to a final solution.",
        ),
        "clarifying_question": types.Schema(
            type=types.Type.STRING,
            description="Your next clarifying question if needed_clarification is true; empty string otherwise.",
        ),
        "updated_solution": types.Schema(
            type=types.Type.STRING,
            description="Your updated solution incorporating all information gathered so far.",
        ),
        "confidence": types.Schema(
            type=types.Type.INTEGER,
            description="Confidence in the updated solution from 0 to 100.",
        ),
    },
    required=["needed_clarification", "clarifying_question", "updated_solution", "confidence"],
)

_USER_SIMULATOR_INSTRUCTION = """You are a factual retrieval system for a tech support case. \
Your role is strictly to report facts that are explicitly documented in the provided situation summary — nothing more.

Rules you must follow:
1. Answer ONLY using information that is explicitly stated in the situation summary.
2. Report facts as brief, plain statements in everyday language (e.g. "The error started after the Windows update.", "I am using the desktop app.").
3. Do NOT interpret, synthesize, or draw conclusions from the facts.
4. Do NOT suggest or hint at any solution or diagnosis.
5. If the answer to the question is not explicitly documented, respond with exactly: "That information is not available."
6. Be concise — one sentence maximum.

Return ONLY a JSON object: {"answer": "<your response>"}"""

_MSDIALOG_FINAL_INSTRUCTION = """You are an experienced tech support specialist. \
You have now gathered all the information you need from the user.

Provide your final, definitive solution. Be concise and direct: 2–4 sentences or a short \
numbered list (no more than 5 steps), targeting ~50–100 words — matching the length of a \
typical forum support reply. Do not pad with preamble, explanations of root cause, or disclaimers.

Return ONLY a valid JSON object:
{
  "final_solution": "<your concise, actionable solution, ~50-100 words>",
  "confidence": <integer 0-100>
}"""


# ── User Simulator ─────────────────────────────────────────────────────────

class UserSimulator:
    """Simulates a tech-support user answering a CQ from the synthesised situation summary."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def answer(self, clarifying_question: str, simulator_context: str) -> str:
        user_message = (
            f"Your situation summary:\n{simulator_context.strip()}\n\n"
            f"Support specialist's question:\n{clarifying_question.strip()}"
        )
        try:
            raw = self._provider.call(
                system_instruction=_USER_SIMULATOR_INSTRUCTION,
                user_message=user_message,
                temperature=0.0,
                max_tokens=1000,
                expect_json=SIMULATOR_SCHEMA,
            )
        except SafetyBlockError:
            logger.warning("User simulator blocked by safety filter.")
            return "That information is not available."

        parsed = parse_json_response(raw)
        if parsed and "answer" in parsed:
            return str(parsed["answer"]).strip()

        import re as _re
        match = _re.search(r'"answer"\s*:\s*"([^"]+)"', raw)
        return match.group(1) if match else "I'm not sure about that."


def _format_problem(title: str, category: str, original_question: str) -> str:
    """Format the user's problem as the model's input message."""
    return (
        f"Product category: {category}\n"
        f"Issue title: {title}\n\n"
        f"User's problem description:\n{original_question.strip()}"
    )


# ── MS-Dialog Single-Turn Pipeline ────────────────────────────────────────

class MsDialogPhase1Pipeline:
    """Single-turn pipeline for MS-Dialog: one CQ round then updated solution."""

    def __init__(
        self,
        provider: LLMProvider,
        instruction_file: Path,
        output_csv: Path,
        request_interval: float = REQUEST_INTERVAL,
        simulator_provider: Optional[LLMProvider] = None,
    ) -> None:
        if not instruction_file.exists():
            raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
        self._instruction = instruction_file.read_text(encoding="utf-8").strip()
        self._provider = provider
        self._simulator = UserSimulator(simulator_provider or provider)
        self._output_csv = output_csv
        self._request_interval = request_interval
        self._output_csv.parent.mkdir(parents=True, exist_ok=True)
        sim_prov = simulator_provider or provider
        logger.info(
            "MsDialogPhase1Pipeline ready — specialist=%s/%s simulator=%s/%s output=%s",
            provider.provider_name, provider.model_name,
            sim_prov.provider_name, sim_prov.model_name, output_csv,
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
                csv.DictWriter(fh, fieldnames=MSDIALOG_PHASE1_FIELDS).writeheader()

    def _append_row(self, row: dict) -> None:
        with self._output_csv.open("a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=MSDIALOG_PHASE1_FIELDS).writerow(row)

    def _turn_0(self, title: str, category: str, original_question: str) -> Optional[dict]:
        try:
            raw = self._provider.call(
                system_instruction=self._instruction,
                user_message=_format_problem(title, category, original_question),
                temperature=0.0,
                max_tokens=3000,
                expect_json=MSDIALOG_TURN_0_SCHEMA,
            )
        except SafetyBlockError as exc:
            return {"_blocked": True, "_reason": str(exc)}
        parsed = parse_json_response(raw)
        if parsed is None:
            logger.error("Turn 0 JSON parse failed. Raw: %.300s", raw)
            return None
        if not {"clarifying_question", "preliminary_solution", "confidence"}.issubset(parsed.keys()):
            logger.error("Turn 0 missing keys. Got: %s", list(parsed.keys()))
            return None
        return parsed

    def _turn_1(
        self,
        title: str,
        category: str,
        original_question: str,
        clarifying_question: str,
        user_response: str,
    ) -> Optional[dict]:
        user_message = (
            f"{_format_problem(title, category, original_question)}\n\n"
            f"Your clarifying question:\n{clarifying_question.strip()}\n\n"
            f"User's answer:\n{user_response.strip()}\n\n"
            f"Based on this additional information, provide your updated solution."
        )
        try:
            raw = self._provider.call(
                system_instruction=_MSDIALOG_FINAL_INSTRUCTION,
                user_message=user_message,
                temperature=0.0,
                max_tokens=3000,
                expect_json=MSDIALOG_FINAL_SCHEMA,
            )
        except SafetyBlockError as exc:
            return {"_blocked": True, "_reason": str(exc)}
        parsed = parse_json_response(raw)
        if parsed is None:
            logger.error("Turn 1 JSON parse failed. Raw: %.300s", raw)
            return None
        if not {"final_solution", "confidence"}.issubset(parsed.keys()):
            logger.error("Turn 1 missing keys. Got: %s", list(parsed.keys()))
            return None
        return parsed

    def run(self, records: list[dict]) -> None:
        processed_ids = self._load_processed_ids()
        self._write_header_if_needed()
        total = len(records)
        succeeded = skipped = failed = 0

        for i, record in enumerate(records, start=1):
            record_id = record.get("id") or record.get("case_id")
            if record_id in processed_ids:
                logger.info("[%d/%d] SKIP — %s already done.", i, total, record_id)
                skipped += 1
                continue

            title             = record["title"]
            category          = record["category"]
            original_question = record["original_question"]
            simulator_context = record["simulator_context"]
            accepted_answer   = record["accepted_answer"]

            logger.info("[%d/%d] Processing %s (%s)", i, total, record_id, category)

            turn0 = self._turn_0(title, category, original_question)
            if turn0 is None:
                failed += 1
                continue

            if turn0.get("_blocked"):
                self._append_row({
                    "id": record_id, "title": title, "category": category,
                    "original_question": original_question,
                    "clarifying_question": "BLOCKED", "cq_type": "",
                    "user_response": "", "preliminary_solution": "BLOCKED",
                    "preliminary_confidence": -1, "updated_solution": "BLOCKED",
                    "updated_confidence": -1, "accepted_answer": accepted_answer,
                    "provider": self._provider.provider_name,
                    "model_id": self._provider.model_name,
                    "finish_reason": turn0.get("_reason", "SAFETY"),
                    "was_blocked": True,
                })
                processed_ids.add(record_id)
                failed += 1
                continue

            cq          = str(turn0["clarifying_question"]).strip()
            prelim_sol  = str(turn0["preliminary_solution"]).strip()
            prelim_conf = int(turn0["confidence"])
            logger.info("  CQ: %s", cq[:100])
            logger.info("  Prelim conf=%d", prelim_conf)
            time.sleep(self._request_interval)

            user_response = self._simulator.answer(cq, simulator_context)
            logger.info("  User: %s", user_response[:100])
            time.sleep(self._request_interval)

            turn1 = self._turn_1(title, category, original_question, cq, user_response)
            if turn1 is None:
                failed += 1
                continue

            if turn1.get("_blocked"):
                self._append_row({
                    "id": record_id, "title": title, "category": category,
                    "original_question": original_question,
                    "clarifying_question": cq, "cq_type": "",
                    "user_response": user_response,
                    "preliminary_solution": prelim_sol,
                    "preliminary_confidence": prelim_conf,
                    "updated_solution": "BLOCKED", "updated_confidence": -1,
                    "accepted_answer": accepted_answer,
                    "provider": self._provider.provider_name,
                    "model_id": self._provider.model_name,
                    "finish_reason": turn1.get("_reason", "SAFETY"),
                    "was_blocked": True,
                })
                processed_ids.add(record_id)
                failed += 1
                continue

            updated_sol  = str(turn1["final_solution"]).strip()
            updated_conf = int(turn1["confidence"])
            logger.info("  Updated conf=%d", updated_conf)

            self._append_row({
                "id": record_id, "title": title, "category": category,
                "original_question": original_question,
                "clarifying_question": cq, "cq_type": "",
                "user_response": user_response,
                "preliminary_solution": prelim_sol,
                "preliminary_confidence": prelim_conf,
                "updated_solution": updated_sol,
                "updated_confidence": updated_conf,
                "accepted_answer": accepted_answer,
                "provider": self._provider.provider_name,
                "model_id": self._provider.model_name,
                "finish_reason": "STOP",
                "was_blocked": False,
            })
            processed_ids.add(record_id)
            succeeded += 1
            time.sleep(self._request_interval)

        logger.info(
            "MsDialog Phase1 complete — total=%d succeeded=%d skipped=%d failed=%d",
            total, succeeded, skipped, failed,
        )


# ── MS-Dialog Multi-Turn Pipeline ─────────────────────────────────────────

class MsDialogMultiTurnPhase1Pipeline:
    """Three-round clarifying-question pipeline for MS-Dialog.

    Turn 0 : model sees problem → preliminary_solution + CQ1 + confidence
    Turn 1–2: model sees history → updated_solution + CQ + confidence
    Turn 3 : model sees full history → final_solution + confidence (no more CQs)
    """

    def __init__(
        self,
        provider: LLMProvider,
        instruction_file: Path,
        continuation_instruction_file: Path,
        output_csv: Path,
        n_turns: int = N_CQ_TURNS,
        request_interval: float = REQUEST_INTERVAL,
        simulator_provider: Optional[LLMProvider] = None,
    ) -> None:
        if not instruction_file.exists():
            raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
        if not continuation_instruction_file.exists():
            raise FileNotFoundError(f"Continuation instruction not found: {continuation_instruction_file}")
        self._instruction = instruction_file.read_text(encoding="utf-8").strip()
        self._continuation_instruction = continuation_instruction_file.read_text(encoding="utf-8").strip()
        self._provider = provider
        self._simulator = UserSimulator(simulator_provider or provider)
        self._output_csv = output_csv
        self._n_turns = n_turns
        self._request_interval = request_interval
        self._output_csv.parent.mkdir(parents=True, exist_ok=True)
        sim_prov = simulator_provider or provider
        logger.info(
            "MsDialogMultiTurnPhase1Pipeline ready — specialist=%s/%s simulator=%s/%s n_turns=%d",
            provider.provider_name, provider.model_name,
            sim_prov.provider_name, sim_prov.model_name, n_turns,
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
                csv.DictWriter(fh, fieldnames=MSDIALOG_PHASE1_MULTITURN_FIELDS).writeheader()

    def _append_row(self, row: dict) -> None:
        with self._output_csv.open("a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=MSDIALOG_PHASE1_MULTITURN_FIELDS).writerow(row)

    def _run_conversation(
        self,
        title: str,
        category: str,
        original_question: str,
        simulator_context: str,
    ) -> Optional[dict]:
        """Full multi-turn conversation with role-alternating history."""
        problem_text = _format_problem(title, category, original_question)
        contents: list[dict] = [{"role": "user", "text": problem_text}]

        # ── Turn 0 ────────────────────────────────────────────────────────
        try:
            raw0 = self._provider.call_multiturn(
                system_instruction=self._instruction,
                contents=contents,
                temperature=0.0,
                max_tokens=3000,
                expect_json=MSDIALOG_TURN_0_SCHEMA,
            )
        except SafetyBlockError:
            return {"_blocked": True}

        parsed0 = parse_json_response(raw0)
        if not parsed0 or not {"clarifying_question", "preliminary_solution", "confidence"}.issubset(parsed0.keys()):
            logger.error("Turn 0 parse failed. Raw: %.300s", raw0)
            return None

        contents.append({"role": "model", "text": raw0})

        prelim_sol  = str(parsed0["preliminary_solution"]).strip()
        prelim_conf = int(parsed0["confidence"])
        cqs         = [str(parsed0["clarifying_question"]).strip()]
        solutions   = [prelim_sol]
        confidences = [prelim_conf]
        sim_responses: list[str] = []

        logger.info("  Prelim conf=%d | CQ1: %s", prelim_conf, cqs[0][:80])
        time.sleep(self._request_interval)

        # ── Clarification rounds ───────────────────────────────────────────
        for turn_idx in range(1, self._n_turns + 1):
            sim_resp = self._simulator.answer(cqs[turn_idx - 1], simulator_context)
            sim_responses.append(sim_resp)
            logger.info("  User[%d]: %s", turn_idx, sim_resp[:80])
            time.sleep(self._request_interval)

            contents.append({"role": "user", "text": f"User's answer: {sim_resp}"})

            if turn_idx < self._n_turns:
                # ── Continuation turn ──────────────────────────────────────
                try:
                    raw_cont = self._provider.call_multiturn(
                        system_instruction=self._continuation_instruction,
                        contents=contents,
                        temperature=0.0,
                        max_tokens=3000,
                        expect_json=MSDIALOG_CONTINUATION_SCHEMA,
                    )
                except SafetyBlockError:
                    return {"_blocked": True}

                parsed_cont = parse_json_response(raw_cont)
                if not parsed_cont or not {"updated_solution", "confidence", "clarifying_question"}.issubset(parsed_cont.keys()):
                    logger.error("Continuation turn %d parse failed. Raw: %.300s", turn_idx, raw_cont)
                    return None

                contents.append({"role": "model", "text": raw_cont})

                upd_sol  = str(parsed_cont["updated_solution"]).strip()
                conf     = int(parsed_cont["confidence"])
                next_cq  = str(parsed_cont["clarifying_question"]).strip()
                solutions.append(upd_sol)
                confidences.append(conf)
                cqs.append(next_cq)
                logger.info("  Turn%d conf=%d | CQ%d: %s", turn_idx, conf, turn_idx + 1, next_cq[:80])

            else:
                # ── Final turn — commit to solution, no more CQs ───────────
                try:
                    raw_final = self._provider.call_multiturn(
                        system_instruction=_MSDIALOG_FINAL_INSTRUCTION,
                        contents=contents,
                        temperature=0.0,
                        max_tokens=3000,
                        expect_json=MSDIALOG_FINAL_SCHEMA,
                    )
                except SafetyBlockError:
                    return {"_blocked": True}

                parsed_final = parse_json_response(raw_final)
                if not parsed_final or not {"final_solution", "confidence"}.issubset(parsed_final.keys()):
                    logger.error("Final turn parse failed. Raw: %.300s", raw_final)
                    return None

                final_sol  = str(parsed_final["final_solution"]).strip()
                final_conf = int(parsed_final["confidence"])
                solutions.append(final_sol)
                confidences.append(final_conf)
                logger.info("  Final conf=%d", final_conf)

            time.sleep(self._request_interval)

        return {
            "cqs": cqs,
            "solutions": solutions,
            "confidences": confidences,
            "sim_responses": sim_responses,
        }

    def run(self, records: list[dict]) -> None:
        processed_ids = self._load_processed_ids()
        self._write_header_if_needed()
        total = len(records)
        succeeded = skipped = failed = 0

        for i, record in enumerate(records, start=1):
            record_id = record.get("id") or record.get("case_id")
            if record_id in processed_ids:
                logger.info("[%d/%d] SKIP — %s", i, total, record_id)
                skipped += 1
                continue

            title             = record["title"]
            category          = record["category"]
            original_question = record["original_question"]
            simulator_context = record["simulator_context"]
            accepted_answer   = record["accepted_answer"]

            logger.info("[%d/%d] Processing %s (%s)", i, total, record_id, category)

            result = self._run_conversation(title, category, original_question, simulator_context)

            if result is None or result.get("_blocked"):
                self._append_row({f: "" for f in MSDIALOG_PHASE1_MULTITURN_FIELDS} | {
                    "id": record_id, "title": title, "category": category,
                    "original_question": original_question,
                    "accepted_answer": accepted_answer,
                    "preliminary_solution": "BLOCKED" if result and result.get("_blocked") else "PARSE_ERROR",
                    "was_blocked": bool(result and result.get("_blocked")),
                    "finish_reason": "SAFETY" if result and result.get("_blocked") else "PARSE_ERROR",
                    "provider": self._provider.provider_name,
                    "model_id": self._provider.model_name,
                })
                failed += 1
                continue

            cqs           = result["cqs"]
            solutions     = result["solutions"]
            confidences   = result["confidences"]
            sim_responses = result["sim_responses"]

            row: dict = {
                "id": record_id, "title": title, "category": category,
                "original_question": original_question,
                "preliminary_solution": solutions[0],
                "preliminary_confidence": confidences[0],
                "cq_1": cqs[0],
                "user_response_1": sim_responses[0],
                "solution_1": solutions[1],
                "confidence_1": confidences[1],
                "cq_2": cqs[1],
                "user_response_2": sim_responses[1],
                "solution_2": solutions[2],
                "confidence_2": confidences[2],
                "cq_3": cqs[2],
                "user_response_3": sim_responses[2],
                "final_solution": solutions[3],
                "final_confidence": confidences[3],
                "accepted_answer": accepted_answer,
                "provider": self._provider.provider_name,
                "model_id": self._provider.model_name,
                "finish_reason": "STOP",
                "was_blocked": False,
            }
            self._append_row(row)
            processed_ids.add(record_id)
            succeeded += 1

        logger.info(
            "MsDialog MultiTurn Phase1 complete — total=%d succeeded=%d skipped=%d failed=%d",
            total, succeeded, skipped, failed,
        )


# ── MS-Dialog Flex Pipeline (optional clarifying questions, 0–3 turns) ─────

class MsDialogFlexPipeline:
    """MS-Dialog pipeline where the model decides at each turn whether to ask
    a clarifying question or commit to a final solution.

    Turn 0 : model sees problem → needed_clarification + preliminary_solution + confidence
             (+ clarifying_question if needed_clarification=True)
    Turn 1–2: model sees history → needed_clarification + updated_solution + confidence
             (+ clarifying_question if needed_clarification=True)
    Turn 3 : forced final — model must commit regardless of needed_clarification.

    The pipeline stops as soon as the model sets needed_clarification=False or
    after the maximum number of CQ turns is exhausted.
    """

    MAX_CQ_TURNS: int = 3

    def __init__(
        self,
        provider: LLMProvider,
        instruction_file: Path,
        continuation_instruction_file: Path,
        output_csv: Path,
        request_interval: float = REQUEST_INTERVAL,
        simulator_provider: Optional[LLMProvider] = None,
    ) -> None:
        if not instruction_file.exists():
            raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
        if not continuation_instruction_file.exists():
            raise FileNotFoundError(f"Continuation instruction not found: {continuation_instruction_file}")
        self._instruction = instruction_file.read_text(encoding="utf-8").strip()
        self._continuation_instruction = continuation_instruction_file.read_text(encoding="utf-8").strip()
        self._provider = provider
        self._simulator = UserSimulator(simulator_provider or provider)
        self._output_csv = output_csv
        self._request_interval = request_interval
        self._output_csv.parent.mkdir(parents=True, exist_ok=True)
        sim_prov = simulator_provider or provider
        logger.info(
            "MsDialogFlexPipeline ready — specialist=%s/%s simulator=%s/%s max_cq=%d output=%s",
            provider.provider_name, provider.model_name,
            sim_prov.provider_name, sim_prov.model_name,
            self.MAX_CQ_TURNS, output_csv,
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
                csv.DictWriter(fh, fieldnames=MSDIALOG_FLEX_FIELDS).writeheader()

    def _append_row(self, row: dict) -> None:
        with self._output_csv.open("a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=MSDIALOG_FLEX_FIELDS).writerow(row)

    def _run_conversation(
        self,
        title: str,
        category: str,
        original_question: str,
        simulator_context: str,
    ) -> Optional[dict]:
        """Run the flexible conversation and return accumulated results."""
        problem_text = _format_problem(title, category, original_question)
        contents: list[dict] = [{"role": "user", "text": problem_text}]

        solutions: list[str] = []
        confidences: list[int] = []
        cqs: list[str] = []
        sim_responses: list[str] = []
        nc_flags: list[bool] = []   # needed_clarification decision at each non-final turn

        # ── Turn 0 ────────────────────────────────────────────────────────
        try:
            raw0 = self._provider.call_multiturn(
                system_instruction=self._instruction,
                contents=contents,
                temperature=0.0,
                max_tokens=4000,
                expect_json=MSDIALOG_FLEX_TURN_0_SCHEMA,
            )
        except SafetyBlockError:
            return {"_blocked": True}

        parsed0 = parse_json_response(raw0)
        if not parsed0 or not {"needed_clarification", "preliminary_solution", "confidence"}.issubset(parsed0.keys()):
            logger.error("Flex turn 0 parse failed. Raw: %.300s", raw0)
            return None

        nc0       = bool(parsed0["needed_clarification"])
        prelim    = str(parsed0["preliminary_solution"]).strip()
        prelim_c  = int(parsed0["confidence"])
        cq0       = str(parsed0.get("clarifying_question", "")).strip()

        solutions.append(prelim)
        confidences.append(prelim_c)
        nc_flags.append(nc0)
        contents.append({"role": "model", "text": raw0})

        if not nc0 or not cq0:
            if nc0 and not cq0:
                logger.warning("Turn 0: needed_clarification=True but clarifying_question empty — treating as False.")
                nc_flags[-1] = False
            logger.info("  Turn0 conf=%d nc=False — no CQs asked.", prelim_c)
            # Return immediately: final_solution = preliminary_solution
            return {
                "solutions": solutions,
                "confidences": confidences,
                "cqs": cqs,
                "sim_responses": sim_responses,
                "nc_flags": nc_flags,
            }

        cqs.append(cq0)
        logger.info("  Turn0 conf=%d nc=True | CQ1: %s", prelim_c, cq0[:80])
        time.sleep(self._request_interval)

        # ── Clarification rounds ───────────────────────────────────────────
        for turn_idx in range(1, self.MAX_CQ_TURNS + 1):
            # Simulate user answering the last CQ
            sim_resp = self._simulator.answer(cqs[-1], simulator_context)
            sim_responses.append(sim_resp)
            logger.info("  User[%d]: %s", turn_idx, sim_resp[:80])
            time.sleep(self._request_interval)
            contents.append({"role": "user", "text": f"User's answer: {sim_resp}"})

            if turn_idx == self.MAX_CQ_TURNS:
                # ── Forced final: no more CQs regardless of model preference ──
                try:
                    raw_final = self._provider.call_multiturn(
                        system_instruction=_MSDIALOG_FINAL_INSTRUCTION,
                        contents=contents,
                        temperature=0.0,
                        max_tokens=4000,
                        expect_json=MSDIALOG_FINAL_SCHEMA,
                    )
                except SafetyBlockError:
                    return {"_blocked": True}

                parsed_final = parse_json_response(raw_final)
                if not parsed_final or not {"final_solution", "confidence"}.issubset(parsed_final.keys()):
                    logger.error("Flex forced-final parse failed. Raw: %.300s", raw_final)
                    return None

                solutions.append(str(parsed_final["final_solution"]).strip())
                confidences.append(int(parsed_final["confidence"]))
                logger.info("  ForcedFinal conf=%d", confidences[-1])

            else:
                # ── Optional continuation ──────────────────────────────────
                try:
                    raw_cont = self._provider.call_multiturn(
                        system_instruction=self._continuation_instruction,
                        contents=contents,
                        temperature=0.0,
                        max_tokens=4000,
                        expect_json=MSDIALOG_FLEX_CONTINUATION_SCHEMA,
                    )
                except SafetyBlockError:
                    return {"_blocked": True}

                parsed_cont = parse_json_response(raw_cont)
                if not parsed_cont or not {"needed_clarification", "updated_solution", "confidence"}.issubset(parsed_cont.keys()):
                    logger.error("Flex continuation turn %d parse failed. Raw: %.300s", turn_idx, raw_cont)
                    return None

                nc       = bool(parsed_cont["needed_clarification"])
                upd_sol  = str(parsed_cont["updated_solution"]).strip()
                conf     = int(parsed_cont["confidence"])
                next_cq  = str(parsed_cont.get("clarifying_question", "")).strip()

                solutions.append(upd_sol)
                confidences.append(conf)
                nc_flags.append(nc)
                contents.append({"role": "model", "text": raw_cont})

                if not nc or not next_cq:
                    if nc and not next_cq:
                        logger.warning(
                            "Turn%d: needed_clarification=True but clarifying_question empty — treating as False.",
                            turn_idx,
                        )
                        nc_flags[-1] = False
                    logger.info("  Turn%d conf=%d nc=False — stopping.", turn_idx, conf)
                    break

                cqs.append(next_cq)
                logger.info("  Turn%d conf=%d nc=True | CQ%d: %s", turn_idx, conf, turn_idx + 1, next_cq[:80])

            time.sleep(self._request_interval)

        return {
            "solutions": solutions,
            "confidences": confidences,
            "cqs": cqs,
            "sim_responses": sim_responses,
            "nc_flags": nc_flags,
        }

    def run(self, records: list[dict]) -> None:
        processed_ids = self._load_processed_ids()
        self._write_header_if_needed()
        total = len(records)
        succeeded = skipped = failed = 0

        def _get(lst: list, idx: int, default=""):
            return lst[idx] if idx < len(lst) else default

        for i, record in enumerate(records, start=1):
            record_id = record.get("id") or record.get("case_id")
            if record_id in processed_ids:
                logger.info("[%d/%d] SKIP — %s", i, total, record_id)
                skipped += 1
                continue

            title             = record["title"]
            category          = record["category"]
            original_question = record["original_question"]
            simulator_context = record["simulator_context"]
            accepted_answer   = record["accepted_answer"]

            logger.info("[%d/%d] Processing %s (%s)", i, total, record_id, category)

            result = self._run_conversation(title, category, original_question, simulator_context)

            if result is None or result.get("_blocked"):
                self._append_row({f: "" for f in MSDIALOG_FLEX_FIELDS} | {
                    "id": record_id, "title": title, "category": category,
                    "original_question": original_question,
                    "accepted_answer": accepted_answer,
                    "preliminary_solution": "BLOCKED" if result and result.get("_blocked") else "PARSE_ERROR",
                    "n_cqs_asked": -1,
                    "was_blocked": bool(result and result.get("_blocked")),
                    "finish_reason": "SAFETY" if result and result.get("_blocked") else "PARSE_ERROR",
                    "provider": self._provider.provider_name,
                    "model_id": self._provider.model_name,
                })
                failed += 1
                continue

            solutions     = result["solutions"]
            confidences   = result["confidences"]
            cqs           = result["cqs"]
            sim_responses = result["sim_responses"]
            nc_flags      = result["nc_flags"]
            n_cqs         = len(cqs)

            row: dict = {
                "id": record_id,
                "title": title,
                "category": category,
                "original_question": original_question,
                # Turn 0
                "preliminary_solution":   solutions[0],
                "preliminary_confidence": confidences[0],
                "needed_clarification_0": nc_flags[0],
                # After CQ1
                "cq_1":                   _get(cqs, 0),
                "user_response_1":        _get(sim_responses, 0),
                "solution_1":             _get(solutions, 1),
                "confidence_1":           _get(confidences, 1),
                "needed_clarification_1": _get(nc_flags, 1),
                # After CQ2
                "cq_2":                   _get(cqs, 1),
                "user_response_2":        _get(sim_responses, 1),
                "solution_2":             _get(solutions, 2),
                "confidence_2":           _get(confidences, 2),
                "needed_clarification_2": _get(nc_flags, 2),
                # After CQ3 (forced final)
                "cq_3":                   _get(cqs, 2),
                "user_response_3":        _get(sim_responses, 2),
                "final_solution":         solutions[-1],
                "final_confidence":       confidences[-1],
                # Summary
                "n_cqs_asked":    n_cqs,
                "accepted_answer": accepted_answer,
                "provider":        self._provider.provider_name,
                "model_id":        self._provider.model_name,
                "finish_reason":   "STOP",
                "was_blocked":     False,
            }
            self._append_row(row)
            processed_ids.add(record_id)
            succeeded += 1
            logger.info(
                "  [%d/%d] Done — n_cqs=%d final_conf=%d",
                i, total, n_cqs, confidences[-1],
            )

        logger.info(
            "MsDialog Flex complete — total=%d succeeded=%d skipped=%d failed=%d",
            total, succeeded, skipped, failed,
        )
