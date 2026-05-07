"""Run MS-Dialog flexible-turn experiment — Gemini specialist + Gemini simulator.

The model decides at each turn whether to ask a clarifying question (needed_clarification=True)
or commit to a final solution (needed_clarification=False). Maximum 3 CQs; the pipeline
stops earlier whenever the model signals it has enough information.

Specialist model : config.MSDIALOG_GEMINI_MODEL_ID (GeminiProvider)
Simulator model  : config.SIMULATOR_MODEL_ID (GeminiProvider)
Judge            : run separately via run_msdialog_judge.py (reuse same judge)

Usage:
    python run_msdialog_gemini_flex.py
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
from config import MSDIALOG_GEMINI_MODEL_ID, SIMULATOR_MODEL_ID, REQUEST_INTERVAL as _CFG_INTERVAL

DATASET = "ms-dialog"

DATASETS_DIR  = ROOT / "datasets" / DATASET
PROMPTS_DIR   = ROOT / "prompts"  / DATASET
OUTPUTS_DIR   = ROOT / "outputs"  / DATASET / MSDIALOG_GEMINI_MODEL_ID

CASES_PATH        = DATASETS_DIR / "msdialog_100.jsonl"
INSTRUCTION_FILE  = PROMPTS_DIR  / "flex_phase1_instruction.txt"
CONTINUATION_FILE = PROMPTS_DIR  / "flex_continuation_instruction.txt"
OUTPUT_CSV        = OUTPUTS_DIR  / "phase1_flex_results.csv"

REQUEST_INTERVAL = _CFG_INTERVAL

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "msdialog_gemini_flex.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    from src.utils import load_dotenv
    from src.providers import GeminiProvider
    from src.pipeline import MsDialogFlexPipeline

    load_dotenv(ROOT / ".env")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MS-Dialog Gemini Flex Experiment (optional CQs, 0–3 turns)")
    logger.info("  Specialist : %s (GeminiProvider)", MSDIALOG_GEMINI_MODEL_ID)
    logger.info("  Simulator  : %s (GeminiProvider)", SIMULATOR_MODEL_ID)
    logger.info("  Max CQs    : %d", MsDialogFlexPipeline.MAX_CQ_TURNS)
    logger.info("  Dataset    : %s", CASES_PATH)
    logger.info("  Output     : %s", OUTPUT_CSV)
    logger.info("=" * 70)

    # Load records
    with open(CASES_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d records from %s", len(records), CASES_PATH.name)

    # Providers
    specialist_provider = GeminiProvider(model_id=MSDIALOG_GEMINI_MODEL_ID)
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
    pipeline = MsDialogFlexPipeline(
        provider=specialist_provider,
        instruction_file=INSTRUCTION_FILE,
        continuation_instruction_file=CONTINUATION_FILE,
        output_csv=OUTPUT_CSV,
        request_interval=REQUEST_INTERVAL,
        simulator_provider=simulator_provider,
    )

    pipeline.run(records)
    logger.info("Flex experiment complete. Results: %s", OUTPUT_CSV)


if __name__ == "__main__":
    main()
