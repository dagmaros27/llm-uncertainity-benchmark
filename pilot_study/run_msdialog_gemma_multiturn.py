"""Run MS-Dialog Phase 1 multi-turn experiment — Gemma 3-12B specialist + Gemini simulator.

Specialist model : gemma-3-12b-it  (GemmaProvider)
Simulator model  : config.SIMULATOR_MODEL_ID (GeminiProvider)
Judge            : run separately via run_msdialog_judge.py

Usage:
    python run_msdialog_gemma_multiturn.py
"""

from __future__ import annotations

import json
import logging
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

DATASET             = "ms-dialog"
CLINICIAN_MODEL_ID  = "gemma-3-12b-it"

DATASETS_DIR        = ROOT / "datasets" / DATASET
PROMPTS_DIR         = ROOT / "prompts"  / DATASET
OUTPUTS_DIR         = ROOT / "outputs"  / DATASET / CLINICIAN_MODEL_ID

CASES_PATH          = DATASETS_DIR / "msdialog_100.jsonl"
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
        logging.FileHandler(ROOT / "logs" / "msdialog_gemma_multiturn.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    from src.utils import load_dotenv
    from src.providers import GemmaProvider, GeminiProvider
    from src.pipeline import MsDialogMultiTurnPhase1Pipeline

    load_dotenv(ROOT / ".env")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MS-Dialog Gemma Multi-Turn Experiment")
    logger.info("  Specialist : %s (GemmaProvider)", CLINICIAN_MODEL_ID)
    logger.info("  Simulator  : %s (GeminiProvider)", SIMULATOR_MODEL_ID)
    logger.info("  N_CQ_TURNS : %d", N_CQ_TURNS)
    logger.info("  Dataset    : %s", CASES_PATH)
    logger.info("  Output     : %s", OUTPUT_CSV)
    logger.info("=" * 70)

    # Load records
    with open(CASES_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d records from %s", len(records), CASES_PATH.name)

    # Providers
    specialist_provider = GemmaProvider(model_id=CLINICIAN_MODEL_ID)
    simulator_provider  = GeminiProvider(model_id=SIMULATOR_MODEL_ID)

    # Smoke tests
    logger.info("Running smoke tests...")
    resp = specialist_provider.call(
        system_instruction="You are a helpful assistant.",
        user_message="Reply with exactly: SMOKE TEST PASSED",
        temperature=0.0, max_tokens=64,
    )
    assert "SMOKE" in resp.upper(), f"Specialist smoke test failed: {resp}"
    logger.info("Specialist smoke test PASSED: %s", resp.strip()[:60])

    resp2 = simulator_provider.call(
        system_instruction="You are a helpful assistant.",
        user_message="Reply with exactly: SMOKE TEST PASSED",
        temperature=0.0, max_tokens=5000,
    )
    assert "SMOKE" in resp2.upper(), f"Simulator smoke test failed: {resp2}"
    logger.info("Simulator smoke test PASSED: %s", resp2.strip()[:60])

    # Pipeline
    pipeline = MsDialogMultiTurnPhase1Pipeline(
        provider=specialist_provider,
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
