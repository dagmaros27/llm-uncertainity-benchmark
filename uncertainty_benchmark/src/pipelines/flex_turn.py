"""Flex-turn pipeline: model decides whether to ask a CQ (0 or 1 CQs).

Flow per record:
  Turn 0 — model sees input context → JSON with ask_cq (bool) + preliminary_answer + confidence
            (+ clarifying_question if ask_cq=True)
  [If ask_cq=True]:
    Simulator — answers the CQ from the Layer 1 context
    Turn 1 — model sees CQ+answer → JSON with final_answer + updated_confidence
  [If ask_cq=False]:
    final_answer = preliminary_answer, final_confidence = preliminary_confidence

Works for all three datasets via per-dataset dispatch.
Output CSV is a superset of SingleTurnPipeline with "cq_asked" added.

Resumable: skips case_ids already present in the output CSV.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Optional

from ..providers import LLMProvider
from ..pipelines.simulator import Simulator
from ..parsing import parse_with_schema
from ..uq import extract_confidence, response_entropy_stats
from ..utils import SafetyBlockError

logger = logging.getLogger(__name__)

# ── CSV schema ────────────────────────────────────────────────────────────────

FIELDS = [
    "case_id", "dataset", "model_id", "provider", "method",
    "cq_asked",
    "preliminary_answer", "preliminary_confidence",
    "clarifying_question", "simulator_response",
    "final_answer", "final_confidence", "confidence_delta",
    "is_correct_preliminary", "is_correct_final",
    "logprob_mean_entropy_t0", "logprob_max_entropy_t0", "logprob_n_tokens_t0", "logprob_lnpe_t0",
    "logprob_mean_entropy_t1", "logprob_max_entropy_t1", "logprob_n_tokens_t1", "logprob_lnpe_t1",
    "finish_reason", "was_blocked", "latency_t0_s", "latency_t1_s",
]

# ── Per-dataset required keys ─────────────────────────────────────────────────

_T0_KEYS = {
    "medqa":    ["ask_cq", "clarifying_question", "preliminary_assessment", "confidence"],
    "msdialog": ["ask_cq", "clarifying_question", "preliminary_solution",   "confidence"],
    "sharc":    ["ask_cq", "clarifying_question", "preliminary_answer", "preliminary_reasoning", "confidence"],
}

_T1_KEYS = {
    "medqa":    ["updated_assessment", "updated_confidence"],
    "msdialog": ["final_solution",     "confidence"],
    "sharc":    ["final_answer",       "final_reasoning", "confidence"],
}

# ── Post-clarification system instructions (same as single_turn) ───────────────

_FINAL_INSTRUCTIONS = {
    "medqa": (
        "You are an experienced clinician. You have received an answer to your clarifying question.\n\n"
        "Based on all information gathered so far, select the most appropriate answer "
        "to the clinical question and state your updated confidence.\n\n"
        "Return ONLY a valid JSON object:\n"
        "{\n  \"updated_assessment\": \"<A, B, C, or D>\",\n  \"updated_confidence\": <integer 0-100>\n}"
    ),
    "msdialog": (
        "You are an experienced tech support specialist. You have received an answer to your clarifying question.\n\n"
        "Provide your final, definitive solution. Be concise and direct: 2–4 sentences or a short "
        "numbered list (no more than 5 steps), targeting ~50–100 words.\n\n"
        "Return ONLY a valid JSON object:\n"
        "{\n  \"final_solution\": \"<your concise final solution>\",\n  \"confidence\": <integer 0-100>\n}"
    ),
    "sharc": (
        "You are an experienced eligibility specialist. You have received an answer to your clarifying question.\n\n"
        "Provide your final, definitive determination. Pick exactly one: \"Yes\" or \"No\".\n\n"
        "Return ONLY a valid JSON object:\n"
        "{\n  \"final_answer\": \"Yes\" or \"No\",\n  \"final_reasoning\": \"<one sentence>\",\n  \"confidence\": <integer 0-100>\n}"
    ),
}


# ── Per-dataset helpers (identical to single_turn.py) ────────────────────────

def _format_input(dataset: str, record: dict) -> str:
    if dataset == "medqa":
        options_str = "\n".join(f"{k}. {v}" for k, v in record["options"].items())
        return (
            f"Patient presentation:\n{record['ehr_summary'].strip()}\n\n"
            f"Clinical question:\n{record['question'].strip()}\n\n"
            f"Answer options:\n{options_str}"
        )
    elif dataset == "msdialog":
        return (
            f"Product category: {record.get('category', '')}\n"
            f"Issue title: {record.get('title', '')}\n\n"
            f"User's problem description:\n{record['original_question'].strip()}"
        )
    elif dataset == "sharc":
        return (
            f"ELIGIBILITY RULE:\n{record['snippet'].strip()}\n\n"
            f"USER'S QUESTION:\n{record['question'].strip()}"
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def _format_t1_input(dataset: str, record: dict, cq: str, sim_response: str) -> str:
    base = _format_input(dataset, record)
    return (
        f"{base}\n\n"
        f"Your clarifying question:\n{cq.strip()}\n\n"
        f"Response:\n{sim_response.strip()}"
    )


def _extract_prelim_answer(dataset: str, parsed: dict) -> str:
    if dataset == "medqa":
        return str(parsed.get("preliminary_assessment", "")).strip().upper()
    elif dataset == "msdialog":
        return str(parsed.get("preliminary_solution", "")).strip()
    elif dataset == "sharc":
        return str(parsed.get("preliminary_answer", "")).strip()
    return ""


def _extract_final_answer(dataset: str, parsed: dict) -> str:
    if dataset == "medqa":
        return str(parsed.get("updated_assessment", "")).strip().upper()
    elif dataset == "msdialog":
        return str(parsed.get("final_solution", "")).strip()
    elif dataset == "sharc":
        return str(parsed.get("final_answer", "")).strip()
    return ""


def _get_sim_context(dataset: str, record: dict) -> str:
    if dataset == "sharc":
        return record.get("context_essay", "")
    return record.get("simulator_context", "")


def _evaluate_correct(dataset: str, answer: str, record: dict) -> Optional[bool]:
    if dataset == "medqa":
        correct = str(record.get("correct_option", "")).strip().upper()
        return answer.upper() == correct if answer and correct else None
    elif dataset == "sharc":
        gold = str(record.get("answer", "")).strip().lower()
        return answer.lower() == gold if answer and gold else None
    return None


def _record_id(dataset: str, record: dict) -> str:
    return str(record.get("case_id") or record.get("id", ""))


# ── Pipeline class ────────────────────────────────────────────────────────────

class FlexTurnPipeline:
    """Optional-CQ flex pipeline for all three datasets."""

    def __init__(
        self,
        provider: LLMProvider,
        dataset: str,
        instruction_path: Path,
        simulator: Simulator,
        output_csv: Path,
        tracker=None,
        request_interval: float = 0.5,
    ) -> None:
        if dataset not in ("medqa", "msdialog", "sharc"):
            raise ValueError(f"Unknown dataset: {dataset}")
        if not instruction_path.exists():
            raise FileNotFoundError(f"Instruction file not found: {instruction_path}")

        self._provider    = provider
        self._dataset     = dataset
        self._instruction = instruction_path.read_text(encoding="utf-8").strip()
        self._final_instr = _FINAL_INSTRUCTIONS[dataset]
        self._simulator   = simulator
        self._output_csv  = output_csv
        self._tracker     = tracker
        self._interval    = request_interval

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "FlexTurnPipeline ready — dataset=%s model=%s/%s output=%s",
            dataset, provider.provider_name, provider.model_name, output_csv.name,
        )

    # ── Resume support ─────────────────────────────────────────────────────

    def _load_processed(self) -> set[str]:
        processed: set[str] = set()
        if not self._output_csv.exists():
            return processed
        with self._output_csv.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("case_id"):
                    processed.add(row["case_id"])
        logger.info("Resumability: %d already processed", len(processed))
        return processed

    def _write_header(self):
        if not self._output_csv.exists():
            with self._output_csv.open("w", encoding="utf-8", newline="") as fh:
                csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    def _append(self, row: dict):
        with self._output_csv.open("a", encoding="utf-8", newline="") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writerow(row)
        if self._tracker:
            self._tracker.log(row)

    # ── Core turns ────────────────────────────────────────────────────────

    def _turn_0(self, record: dict) -> tuple[Optional[dict], Optional[list], float]:
        user_msg = _format_input(self._dataset, record)
        t0 = time.monotonic()
        try:
            if self._provider.supports_logprobs:
                text, logprobs = self._provider.call_with_logprobs(
                    system_instruction=self._instruction,
                    user_message=user_msg,
                    temperature=0.0,
                )
            else:
                text = self._provider.call(
                    system_instruction=self._instruction,
                    user_message=user_msg,
                    temperature=0.0,
                )
                logprobs = None
        except SafetyBlockError:
            return None, None, time.monotonic() - t0
        except Exception as exc:
            logger.error("Turn 0 provider error: %s", exc)
            return None, None, time.monotonic() - t0

        latency = time.monotonic() - t0
        parsed = parse_with_schema(
            text,
            required_keys=_T0_KEYS[self._dataset],
            provider=self._provider,
        )
        return parsed, logprobs, latency

    def _turn_1(self, record: dict, cq: str, sim_resp: str) -> tuple[Optional[dict], Optional[list], float]:
        user_msg = _format_t1_input(self._dataset, record, cq, sim_resp)
        t0 = time.monotonic()
        try:
            if self._provider.supports_logprobs:
                text, logprobs = self._provider.call_with_logprobs(
                    system_instruction=self._final_instr,
                    user_message=user_msg,
                    temperature=0.0,
                )
            else:
                text = self._provider.call(
                    system_instruction=self._final_instr,
                    user_message=user_msg,
                    temperature=0.0,
                )
                logprobs = None
        except SafetyBlockError:
            return None, None, time.monotonic() - t0
        except Exception as exc:
            logger.error("Turn 1 provider error: %s", exc)
            return None, None, time.monotonic() - t0

        latency = time.monotonic() - t0
        parsed = parse_with_schema(
            text,
            required_keys=_T1_KEYS[self._dataset],
            provider=self._provider,
        )
        return parsed, logprobs, latency

    # ── Run ───────────────────────────────────────────────────────────────

    def run(self, records: list[dict]) -> None:
        processed = self._load_processed()
        self._write_header()
        total = len(records)
        n_ok = n_skip = n_fail = 0
        n_cq_asked = 0

        for i, record in enumerate(records, 1):
            case_id = _record_id(self._dataset, record)

            if case_id in processed:
                logger.info("[%d/%d] SKIP %s", i, total, case_id)
                n_skip += 1
                continue

            logger.info("[%d/%d] %s", i, total, case_id)

            # ── Turn 0 ────────────────────────────────────────────────────
            parsed0, lp0, lat0 = self._turn_0(record)

            if parsed0 is None:
                self._append({
                    **{f: "" for f in FIELDS},
                    "case_id": case_id, "dataset": self._dataset,
                    "model_id": self._provider.model_name,
                    "provider": self._provider.provider_name,
                    "method": "flex",
                    "finish_reason": "PARSE_ERROR_T0", "was_blocked": False,
                    "latency_t0_s": round(lat0, 2),
                })
                n_fail += 1
                continue

            ask_cq   = bool(parsed0.get("ask_cq", False))
            cq       = str(parsed0.get("clarifying_question", "")).strip()
            prelim   = _extract_prelim_answer(self._dataset, parsed0)
            prelim_c = extract_confidence(parsed0) or 0.0
            lp0_stats = response_entropy_stats(lp0)

            # Guard: if model set ask_cq=True but left CQ empty, treat as False
            if ask_cq and not cq:
                logger.warning("  ask_cq=True but clarifying_question empty — treating as False")
                ask_cq = False

            logger.info("  ask_cq=%s prelim=%s (conf=%.0f%%)", ask_cq, prelim[:60], prelim_c * 100)

            if not ask_cq:
                # No CQ — final = preliminary
                is_c_prelim = _evaluate_correct(self._dataset, prelim, record)
                self._append({
                    "case_id":               case_id,
                    "dataset":               self._dataset,
                    "model_id":              self._provider.model_name,
                    "provider":              self._provider.provider_name,
                    "method":                "flex",
                    "cq_asked":              False,
                    "preliminary_answer":    prelim,
                    "preliminary_confidence": round(prelim_c, 4),
                    "clarifying_question":   "",
                    "simulator_response":    "",
                    "final_answer":          prelim,
                    "final_confidence":      round(prelim_c, 4),
                    "confidence_delta":      0.0,
                    "is_correct_preliminary": is_c_prelim,
                    "is_correct_final":       is_c_prelim,
                    "logprob_mean_entropy_t0": lp0_stats["logprob_mean_entropy"],
                    "logprob_max_entropy_t0":  lp0_stats["logprob_max_entropy"],
                    "logprob_n_tokens_t0":     lp0_stats["logprob_n_tokens"],
                    "logprob_lnpe_t0":         lp0_stats["logprob_lnpe"],
                    "logprob_mean_entropy_t1": "",
                    "logprob_max_entropy_t1":  "",
                    "logprob_n_tokens_t1":     "",
                    "logprob_lnpe_t1":         "",
                    "finish_reason":           "STOP_NO_CQ",
                    "was_blocked":           False,
                    "latency_t0_s":          round(lat0, 2),
                    "latency_t1_s":          "",
                })
                processed.add(case_id)
                n_ok += 1
                time.sleep(self._interval)
                continue

            # ── CQ was asked ───────────────────────────────────────────────
            n_cq_asked += 1
            logger.info("  CQ: %s", cq[:100])
            time.sleep(self._interval)

            sim_context = _get_sim_context(self._dataset, record)
            sim_resp = self._simulator.answer(cq, sim_context)
            logger.info("  Sim: %s", sim_resp[:100])
            time.sleep(self._interval)

            parsed1, lp1, lat1 = self._turn_1(record, cq, sim_resp)

            if parsed1 is None:
                self._append({
                    **{f: "" for f in FIELDS},
                    "case_id": case_id, "dataset": self._dataset,
                    "model_id": self._provider.model_name,
                    "provider": self._provider.provider_name,
                    "method": "flex", "cq_asked": True,
                    "preliminary_answer": prelim,
                    "preliminary_confidence": round(prelim_c, 4),
                    "clarifying_question": cq,
                    "simulator_response": sim_resp,
                    "is_correct_preliminary": _evaluate_correct(self._dataset, prelim, record),
                    "logprob_mean_entropy_t0": lp0_stats["logprob_mean_entropy"],
                    "logprob_max_entropy_t0":  lp0_stats["logprob_max_entropy"],
                    "logprob_n_tokens_t0":     lp0_stats["logprob_n_tokens"],
                    "logprob_lnpe_t0":         lp0_stats["logprob_lnpe"],
                    "finish_reason": "PARSE_ERROR_T1", "was_blocked": False,
                    "latency_t0_s": round(lat0, 2), "latency_t1_s": round(lat1, 2),
                })
                n_fail += 1
                continue

            final    = _extract_final_answer(self._dataset, parsed1)
            final_c  = extract_confidence(parsed1) or 0.0
            lp1_stats = response_entropy_stats(lp1)

            is_c_prelim = _evaluate_correct(self._dataset, prelim, record)
            is_c_final  = _evaluate_correct(self._dataset, final,  record)
            conf_delta  = round(final_c - prelim_c, 4)

            logger.info("  Final: %s (conf=%.0f%%) correct=%s", final[:60], final_c * 100, is_c_final)

            self._append({
                "case_id":               case_id,
                "dataset":               self._dataset,
                "model_id":              self._provider.model_name,
                "provider":              self._provider.provider_name,
                "method":                "flex",
                "cq_asked":              True,
                "preliminary_answer":    prelim,
                "preliminary_confidence": round(prelim_c, 4),
                "clarifying_question":   cq,
                "simulator_response":    sim_resp,
                "final_answer":          final,
                "final_confidence":      round(final_c, 4),
                "confidence_delta":      conf_delta,
                "is_correct_preliminary": is_c_prelim,
                "is_correct_final":       is_c_final,
                "logprob_mean_entropy_t0": lp0_stats["logprob_mean_entropy"],
                "logprob_max_entropy_t0":  lp0_stats["logprob_max_entropy"],
                "logprob_n_tokens_t0":     lp0_stats["logprob_n_tokens"],
                "logprob_lnpe_t0":         lp0_stats["logprob_lnpe"],
                "logprob_mean_entropy_t1": lp1_stats["logprob_mean_entropy"],
                "logprob_max_entropy_t1":  lp1_stats["logprob_max_entropy"],
                "logprob_n_tokens_t1":     lp1_stats["logprob_n_tokens"],
                "logprob_lnpe_t1":         lp1_stats["logprob_lnpe"],
                "finish_reason":           "STOP",
                "was_blocked":           False,
                "latency_t0_s":          round(lat0, 2),
                "latency_t1_s":          round(lat1, 2),
            })
            processed.add(case_id)
            n_ok += 1
            time.sleep(self._interval)

        cq_rate = n_cq_asked / max(n_ok, 1)
        logger.info(
            "FlexTurnPipeline done — total=%d ok=%d skip=%d fail=%d cq_rate=%.1f%%",
            total, n_ok, n_skip, n_fail, cq_rate * 100,
        )

        # WandB run summary
        if self._tracker and n_ok > 0:
            rows = []
            with self._output_csv.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            finished = [r for r in rows if r.get("finish_reason") in ("STOP", "STOP_NO_CQ")]
            if finished:
                correct_final = [r for r in finished if str(r.get("is_correct_final")) == "True"]
                cq_rows = [r for r in finished if str(r.get("cq_asked")) == "True"]
                self._tracker.summary({
                    "n_records":       len(finished),
                    "accuracy_final":  len(correct_final) / len(finished),
                    "cq_rate":         len(cq_rows) / len(finished),
                    "mean_conf_final": sum(float(r["final_confidence"]) for r in finished if r.get("final_confidence")) / len(finished),
                })
        if self._tracker:
            self._tracker.finish()
