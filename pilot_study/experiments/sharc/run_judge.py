"""Run the LM judge on ShARC flex results — classifies each CQ as EPISTEMIC or ALEATORIC.

Reads  : outputs/sharc/<model>/phase1_flex_results.csv
Writes : outputs/sharc/<model>/phase1_flex_classified.csv
        (long-format: one row per CQ with id, turn, question, label)

Usage:
    python run_sharc_judge.py
    python run_sharc_judge.py --model gemini-2.5-flash
"""

from __future__ import annotations

import argparse
import logging
import sys
import io
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from config import JUDGE_MODEL_ID

DATASET     = "sharc"
OUTPUTS_DIR = ROOT / "outputs" / DATASET
PROMPTS_DIR = ROOT / "prompts" / DATASET
JUDGE_INSTRUCTION = PROMPTS_DIR / "judge_instruction.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "sharc_judge.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Few-shot examples (eligibility / regulatory domain) ─────────────────────
FEW_SHOT_DEFS = [
    (
        "Are you a UK resident?",
        "EPISTEMIC",
        "There is a single correct factual answer about the user's residency status. "
        "Knowing this fact definitively determines whether the residency requirement is met.",
    ),
    (
        "Do you have a high school diploma or equivalent qualification?",
        "EPISTEMIC",
        "The user either holds the qualification or does not. Once stated, the eligibility "
        "condition about education is fully resolved.",
    ),
    (
        "Did you serve in the armed forces?",
        "EPISTEMIC",
        "Military service is a concrete factual matter with one correct answer. The fact "
        "directly determines whether the veteran-related rule applies.",
    ),
    (
        "Is your child diagnosed with a birth defect?",
        "EPISTEMIC",
        "Whether the child has a diagnosed condition is an objective medical fact with one "
        "correct answer. This factual answer fully closes the knowledge gap.",
    ),
    (
        "When you say you 'live in the UK', do you mean you have legal residency status, or "
        "that you currently reside there on a temporary basis?",
        "ALEATORIC",
        "Two valid interpretations of 'living in the UK' exist. Only the user can clarify "
        "which framing they intended — no objective fact resolves the ambiguity.",
    ),
    (
        "By 'family member', do you mean an immediate relative such as a spouse or child, or "
        "any extended family member?",
        "ALEATORIC",
        "The user's word 'family member' covers multiple equally valid scopes. Only the user's "
        "intent can resolve which they meant.",
    ),
    (
        "Are you asking about eligibility for yourself, or on behalf of someone else?",
        "ALEATORIC",
        "This is a question about the user's framing of their own request. Only the user "
        "can specify whose situation they are asking about.",
    ),
    (
        "Is the equipment medical, veterinary, or scientific equipment?",
        "EPISTEMIC",
        "The classification of the equipment is a concrete factual matter. Once specified, "
        "the eligibility condition is fully resolved.",
    ),
]


def run_judge_for_model(model_id: str, judge, FewShotExample, CSVBatchClassifier) -> None:
    model_dir = OUTPUTS_DIR / model_id

    flex_results = model_dir / "phase1_flex_results.csv"
    flex_input   = model_dir / "phase1_flex_cq_input.csv"
    flex_output  = model_dir / "phase1_flex_classified.csv"

    if not flex_results.exists():
        logger.warning("Flex results not found for %s — skipping.", model_id)
        return

    logger.info("Loading flex results for %s", model_id)
    df = pd.read_csv(flex_results)
    valid = df[~df["was_blocked"].astype(bool)].copy()

    # Build long-format CQ table: one row per non-empty CQ
    rows = []
    for _, r in valid.iterrows():
        for turn in range(1, 4):
            cq = r.get(f"cq_{turn}")
            if pd.notna(cq) and str(cq).strip():
                rows.append({
                    "id": r["id"],
                    "turn": turn,
                    "clarifying_question": str(cq).strip(),
                })
    long_df = pd.DataFrame(rows)
    long_df.to_csv(flex_input, index=False)
    logger.info("Built %d CQs across %d cases", len(long_df),
                long_df["id"].nunique() if len(long_df) else 0)

    if len(long_df) == 0:
        logger.warning("No CQs to classify — skipping.")
        return

    CSVBatchClassifier(
        judge=judge,
        input_csv=flex_input,
        output_csv=flex_output,
        question_column="clarifying_question",
        id_column="id",
        delay_between_calls=1.0,
    ).run()

    # Re-attach turn column
    clf = pd.read_csv(flex_output)
    q_col = "question" if "question" in clf.columns else "clarifying_question"
    clf = clf.merge(
        long_df[["id", "turn", "clarifying_question"]].rename(columns={"clarifying_question": q_col}),
        on=["id", q_col], how="left",
    )
    clf.to_csv(flex_output, index=False)
    logger.info("Flex judge complete → %s", flex_output)
    logger.info("Label distribution:\n%s", clf["label"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Model subdirectory (e.g. gemini-2.5-flash). Omit to run all.")
    args = parser.parse_args()

    from src.utils import load_dotenv
    from src.providers import GeminiProvider
    from src.judge import LLMJudge, CSVBatchClassifier, FewShotExample

    load_dotenv(ROOT / ".env")
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)

    few_shot = [FewShotExample(input=q, expected_output=l, explanation=e)
                for q, l, e in FEW_SHOT_DEFS]

    provider = GeminiProvider(model_id=JUDGE_MODEL_ID)
    judge = LLMJudge(
        provider=provider,
        instructions_path=JUDGE_INSTRUCTION,
        few_shot_examples=few_shot,
        label_parser=lambda text: text.strip().upper(),
    )

    smoke = judge.evaluate("Are you a UK resident?")
    assert smoke.label == "EPISTEMIC", f"Smoke test failed: {smoke.label}"
    logger.info("Smoke test PASSED: %s", smoke.label)

    if args.model:
        model_ids = [args.model]
    else:
        model_ids = [p.name for p in OUTPUTS_DIR.iterdir() if p.is_dir()]
        logger.info("Found %d model directories: %s", len(model_ids), model_ids)

    for model_id in model_ids:
        logger.info("=" * 60)
        logger.info("Judging model: %s", model_id)
        run_judge_for_model(model_id, judge, FewShotExample, CSVBatchClassifier)

    logger.info("All judge runs complete.")


if __name__ == "__main__":
    main()
