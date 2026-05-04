"""Run the LM judge on MS-Dialog single-turn and multi-turn results.

Judge model : config.JUDGE_MODEL_ID
Reads  : outputs/ms-dialog/<model>/phase1_singleturn_results.csv
         outputs/ms-dialog/<model>/phase1_multiturn_results.csv
Writes : outputs/ms-dialog/<model>/phase1_singleturn_classified.csv
         outputs/ms-dialog/<model>/phase1_multiturn_classified.csv

Run for a specific model:
    python run_msdialog_judge.py --model gemma-3-12b-it
    python run_msdialog_judge.py --model gemini-2.5-flash

Or run for all models in the outputs directory (default):
    python run_msdialog_judge.py
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

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import JUDGE_MODEL_ID

DATASET     = "ms-dialog"
OUTPUTS_DIR = ROOT / "outputs" / DATASET
PROMPTS_DIR = ROOT / "prompts" / DATASET
JUDGE_INSTRUCTION = PROMPTS_DIR / "judge_instruction.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "msdialog_judge.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Few-shot examples (tech-support domain) ────────────────────────────────
FEW_SHOT_DEFS = [
    (
        "Which version of Windows are you running — Windows 10 or Windows 11?",
        "EPISTEMIC",
        "There is a single correct factual answer about the user's system. Knowing the OS version "
        "definitively resolves which troubleshooting path applies.",
    ),
    (
        "What is the exact error message or error code you see when the application crashes?",
        "EPISTEMIC",
        "The exact error message is a concrete fact with one correct value. Once provided, "
        "the knowledge gap about the crash cause is fully resolved.",
    ),
    (
        "Are you using the desktop version of Skype or the web browser version?",
        "EPISTEMIC",
        "There is one correct factual answer about which client the user is running. "
        "This fact directly determines which fix applies.",
    ),
    (
        "When you say the app is 'not working', do you mean it doesn't open at all, "
        "or it opens but a specific feature doesn't function?",
        "ALEATORIC",
        "Two clinically distinct failure modes are both valid readings of 'not working'. "
        "Only the user knows which pattern they actually experienced — no system fact resolves this.",
    ),
    (
        "Are you looking for a quick workaround to get back to work today, or do you want "
        "to find and fix the underlying root cause?",
        "ALEATORIC",
        "The answer depends entirely on the user's personal priorities and time constraints. "
        "No external fact can determine what they prefer — it is irreducibly user-specific.",
    ),
    (
        "When you say the file 'disappeared', do you mean it was deleted, it's hidden, "
        "or you simply can't find it in the expected location?",
        "ALEATORIC",
        "Three equally valid interpretations of 'disappeared' exist. Only the user knows "
        "which situation they actually experienced — no system fact resolves this ambiguity.",
    ),
    (
        "Did this problem start after you installed a Windows Update or after installing "
        "a third-party application?",
        "EPISTEMIC",
        "There is a factual timeline of what happened on the user's machine. Once confirmed, "
        "this fact permanently resolves which event triggered the issue.",
    ),
    (
        "Is this happening on all your devices or only on this specific computer?",
        "EPISTEMIC",
        "Whether the issue is device-specific or account-wide is a concrete observable fact. "
        "The answer definitively narrows the root cause — no ambiguity remains once stated.",
    ),
]


def run_judge_for_model(model_id: str, judge, FewShotExample, CSVBatchClassifier) -> None:
    model_dir = OUTPUTS_DIR / model_id

    # ── Single-turn ────────────────────────────────────────────────────────
    st_results = model_dir / "phase1_singleturn_results.csv"
    st_input   = model_dir / "phase1_singleturn_cq_input.csv"
    st_output  = model_dir / "phase1_singleturn_classified.csv"

    if not st_results.exists():
        logger.warning("Single-turn results not found for %s — skipping.", model_id)
    else:
        logger.info("Running single-turn judge for %s", model_id)
        st_df = pd.read_csv(st_results)
        work = st_df[
            (~st_df["was_blocked"])
            & (st_df["clarifying_question"].notna())
            & (st_df["clarifying_question"].str.strip() != "")
            & (st_df["clarifying_question"] != "BLOCKED")
        ].copy()
        work[["id", "clarifying_question"]].to_csv(st_input, index=False)
        logger.info("Single-turn: %d CQs to classify", len(work))

        CSVBatchClassifier(
            judge=judge,
            input_csv=st_input,
            output_csv=st_output,
            question_column="clarifying_question",
            id_column="id",
            delay_between_calls=1.0,
        ).run()
        logger.info("Single-turn judge complete → %s", st_output)

        clf = pd.read_csv(st_output)
        logger.info("Single-turn label distribution:\n%s", clf["label"].value_counts().to_string())

    # ── Multi-turn ────────────────────────────────────────────────────────
    mt_results = model_dir / "phase1_multiturn_results.csv"
    mt_input   = model_dir / "phase1_multiturn_cq_input.csv"
    mt_output  = model_dir / "phase1_multiturn_classified.csv"

    if not mt_results.exists():
        logger.warning("Multi-turn results not found for %s — skipping.", model_id)
        return

    logger.info("Running multi-turn judge for %s", model_id)
    mt_df = pd.read_csv(mt_results)
    valid = mt_df[~mt_df["was_blocked"]].copy()
    logger.info("Multi-turn: %d cases, building long-format CQ table", len(valid))

    rows = []
    for _, r in valid.iterrows():
        for turn in range(1, 4):
            cq = r[f"cq_{turn}"]
            if pd.notna(cq) and str(cq).strip():
                rows.append({"id": r["id"], "turn": turn, "clarifying_question": cq})
    long_df = pd.DataFrame(rows)
    long_df.to_csv(mt_input, index=False)
    logger.info("Multi-turn long-format: %d CQs", len(long_df))

    CSVBatchClassifier(
        judge=judge,
        input_csv=mt_input,
        output_csv=mt_output,
        question_column="clarifying_question",
        id_column="id",
        delay_between_calls=1.0,
    ).run()
    logger.info("Multi-turn judge complete → %s", mt_output)

    # Re-join turn number from input CSV so analysis can group by turn
    clf_mt = pd.read_csv(mt_output)
    turn_map = pd.read_csv(mt_input)[["id", "turn", "clarifying_question"]]
    q_col = "question" if "question" in clf_mt.columns else "clarifying_question"
    clf_mt = clf_mt.merge(
        turn_map.rename(columns={"clarifying_question": q_col}),
        on=["id", q_col], how="left",
    )
    clf_mt.to_csv(mt_output, index=False)
    logger.info("Multi-turn label distribution:\n%s", clf_mt["label"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Model subdirectory to judge (e.g. gemma-3-12b-it). "
                             "Omit to run for all models found in outputs/ms-dialog/.")
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

    # Smoke test
    smoke = judge.evaluate("What version of Windows are you running?")
    assert smoke.label == "EPISTEMIC", f"Smoke test failed: {smoke.label}"
    logger.info("Smoke test PASSED: %s", smoke.label)

    # Determine which models to judge
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
