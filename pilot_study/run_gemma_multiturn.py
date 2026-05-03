"""Run Phase 1 multi-turn experiment with Gemma clinician + Gemini simulator.

Clinician model  : gemma-3-12b-it  (GemmaProvider)
Simulator model  : see config.SIMULATOR_MODEL_ID (GeminiProvider)
Judge model      : gemini-3.1-pro-preview (run separately via run_gemma_judge.py)

Usage:
    python run_gemma_multiturn.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# ── Config ────────────────────────────────────────────────────────────────────
from config import SIMULATOR_MODEL_ID, N_CQ_TURNS, REQUEST_INTERVAL as _CFG_INTERVAL

DATASET             = "medqa"
CLINICIAN_MODEL_ID  = "gemma-3-12b-it"

DATASETS_DIR        = ROOT / "datasets" / DATASET
PROMPTS_DIR         = ROOT / "prompts"  / DATASET
OUTPUTS_DIR         = ROOT / "outputs"  / DATASET / CLINICIAN_MODEL_ID

CASES_PATH          = DATASETS_DIR / "multiturn_100.jsonl"
INSTRUCTION_FILE    = PROMPTS_DIR  / "phase1_instruction.txt"
CONTINUATION_FILE   = PROMPTS_DIR  / "phase1_continuation_instruction.txt"
OUTPUT_CSV          = OUTPUTS_DIR  / "phase1_multiturn_results.csv"

REQUEST_INTERVAL    = _CFG_INTERVAL

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / f"gemma_multiturn.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    from src.utils import load_dotenv, clean_simulator_context
    from src.providers import GemmaProvider, GeminiProvider
    from src.pipeline import MultiTurnPhase1Pipeline

    load_dotenv(ROOT / ".env")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Gemma Multi-Turn Experiment")
    logger.info("  Clinician : %s (GemmaProvider)", CLINICIAN_MODEL_ID)
    logger.info("  Simulator : %s (GeminiProvider)", SIMULATOR_MODEL_ID)
    logger.info("  N_CQ_TURNS: %d", N_CQ_TURNS)
    logger.info("  Output    : %s", OUTPUT_CSV)
    logger.info("=" * 70)

    # Load records
    with open(CASES_PATH, encoding="utf-8") as f:
        raw_cases = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d cases from %s", len(raw_cases), CASES_PATH.name)

    records = []
    for c in raw_cases:
        simulator_context = "\n\n---\n\n".join(
            ctx for ctx in [
                clean_simulator_context(c.get("patient_context", "")),
                clean_simulator_context(c.get("nurse_context", "")),
                clean_simulator_context(c.get("specialist_context", "")),
            ] if ctx and ctx.strip()
        )
        records.append({
            "id":               c["case_id"],
            "ehr_summary":      c["ehr_summary"],
            "question":         c["question"],
            "options":          c["options"],
            "correct_option":   c["correct_option"],
            "correct_answer":   c["correct_answer"],
            "simulator_context": simulator_context,
            "difficulty":       c.get("difficulty", ""),
        })
    logger.info("Records prepared: %d", len(records))

    # Leakage check
    leaks = sum(1 for r in records if re.search(
        r"most consistent with|Diagnosis:", r["simulator_context"], re.IGNORECASE))
    logger.info("Diagnostic leakage check: %d/%d contexts contain conclusion language", leaks, len(records))
    assert leaks == 0, f"Leakage detected in {leaks} records — fix clean_simulator_context()"

    # Providers
    clinician_provider  = GemmaProvider(model_id=CLINICIAN_MODEL_ID)
    simulator_provider  = GeminiProvider(model_id=SIMULATOR_MODEL_ID)

    # Smoke tests
    logger.info("Running smoke tests...")
    resp = clinician_provider.call(
        system_instruction="You are a helpful assistant.",
        user_message="Reply with exactly: SMOKE TEST PASSED",
        temperature=0.0, max_tokens=64,
    )
    assert "SMOKE" in resp.upper(), f"Clinician smoke test failed: {resp}"
    logger.info("Clinician smoke test PASSED: %s", resp.strip()[:60])

    resp2 = simulator_provider.call(
        system_instruction="You are a helpful assistant.",
        user_message="Reply with exactly: SMOKE TEST PASSED",
        temperature=0.0, max_tokens=64,
    )
    assert "SMOKE" in resp2.upper(), f"Simulator smoke test failed: {resp2}"
    logger.info("Simulator smoke test PASSED: %s", resp2.strip()[:60])

    # Pipeline
    pipeline = MultiTurnPhase1Pipeline(
        provider=clinician_provider,
        instruction_file=INSTRUCTION_FILE,
        continuation_instruction_file=CONTINUATION_FILE,
        output_csv=OUTPUT_CSV,
        n_turns=N_CQ_TURNS,
        request_interval=REQUEST_INTERVAL,
        simulator_provider=simulator_provider,
    )

    pipeline.run(records)
    logger.info("Multi-turn experiment complete. Results: %s", OUTPUT_CSV)


if __name__ == "__main__":
    main()
