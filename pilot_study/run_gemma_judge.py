"""Run the LM judge on Gemma single-turn and multi-turn results.

Judge model : gemini-3.1-pro-preview
Reads  : outputs/medqa/gemma-3-12b-it/phase1_singleturn_results.csv
         outputs/medqa/gemma-3-12b-it/phase1_multiturn_results.csv
Writes : outputs/medqa/gemma-3-12b-it/phase1_singleturn_classified.csv
         outputs/medqa/gemma-3-12b-it/phase1_multiturn_classified.csv

Usage:
    python run_gemma_judge.py
"""

from __future__ import annotations

import logging
import sys
import io
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

DATASET            = "medqa"
CLINICIAN_MODEL_ID = "gemma-3-12b-it"
JUDGE_MODEL_ID     = "gemini-3.1-pro-preview"

OUTPUTS_DIR        = ROOT / "outputs" / DATASET / CLINICIAN_MODEL_ID
PROMPTS_DIR        = ROOT / "prompts" / DATASET
JUDGE_INSTRUCTION  = PROMPTS_DIR / "judge_instruction.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "gemma_judge.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

FEW_SHOT_DEFS = [
    ("You mentioned having 'MCAS' — could you describe what symptoms it causes for you or what your doctor told you about it?",
     "EPISTEMIC",
     "The model doesn't have enough context about this entity to reason clinically. There is a definite answer — once the patient explains it, the knowledge gap is fully and permanently resolved."),
    ("You described the rash as both 'spreading' and 'fading' — is it currently getting larger or is it clearing up?",
     "EPISTEMIC",
     "The two descriptions contradict each other, making clinical categorisation impossible. There is one correct factual state right now — the model is resolving a factual contradiction, not a preference."),
    ("When you say you feel 'weak', do you mean you have no energy and feel fatigued, or that you have actual muscle weakness and difficulty lifting things?",
     "ALEATORIC",
     "'Weak' has two clinically distinct meanings (fatigue vs true motor weakness) that point to completely different differentials. No external fact can resolve which meaning the patient intends — only the patient can."),
    ("When you said 'it started after the procedure', are you referring to the chest pain or the shortness of breath?",
     "ALEATORIC",
     "The pronoun 'it' could validly refer to either symptom. No external fact can determine which one the patient meant — only the patient's own context resolves this."),
    ("Which aspect of your recovery matters most to you — getting back to work quickly, minimising pain, or avoiding surgery?",
     "ALEATORIC",
     "The answer depends entirely on this individual patient's values and priorities. No clinical fact or external knowledge can determine their personal preference — it is irreducibly patient-specific."),
    ("When you ask about treatment options, are you looking for information about medications, surgical approaches, or lifestyle changes?",
     "ALEATORIC",
     "The request is underspecified — multiple valid interpretations exist and the correct path depends entirely on what the patient wants, not on any clinical fact."),
    ("When you say your symptoms are 'intermittent', do you mean they come and go throughout the day, or that you have symptom-free periods lasting weeks?",
     "ALEATORIC",
     "Two valid temporal interpretations exist, each with different clinical significance. Only the patient knows which pattern applies — no external fact resolves this."),
    ("When you say the pain is 'everywhere', do you mean it is diffuse throughout your abdomen, or that it shifts between different locations?",
     "ALEATORIC",
     "Two spatially distinct clinical patterns (diffuse vs migratory pain) are both plausible readings. Only the patient can clarify which pattern they actually experience."),
]


def main() -> None:
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
    smoke = judge.evaluate("Have you been diagnosed with hypertension or any other heart condition before?")
    assert smoke.label == "EPISTEMIC", f"Smoke test failed: {smoke.label}"
    logger.info("Smoke test PASSED: %s", smoke.label)

    # ── Single-turn judge ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running single-turn judge")
    st_results  = OUTPUTS_DIR / "phase1_singleturn_results.csv"
    st_input    = OUTPUTS_DIR / "phase1_singleturn_cq_input.csv"
    st_output   = OUTPUTS_DIR / "phase1_singleturn_classified.csv"

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

    # Quick summary
    clf = pd.read_csv(st_output)
    logger.info("Single-turn label distribution:\n%s", clf["label"].value_counts().to_string())

    # ── Multi-turn judge ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Running multi-turn judge (300 CQs)")
    mt_results  = OUTPUTS_DIR / "phase1_multiturn_results.csv"
    mt_input    = OUTPUTS_DIR / "phase1_multiturn_cq_input.csv"
    mt_output   = OUTPUTS_DIR / "phase1_multiturn_classified.csv"

    mt_df   = pd.read_csv(mt_results)
    valid   = mt_df[~mt_df["was_blocked"]].copy()
    logger.info("Multi-turn: %d cases, building long-format CQ table", len(valid))

    rows = []
    for _, r in valid.iterrows():
        for turn in range(1, 4):
            cq = r[f"cq_{turn}"]
            if pd.notna(cq) and str(cq).strip():
                rows.append({"id": r["id"], "turn": turn, "clarifying_question": cq})
    long_df = pd.DataFrame(rows)
    long_df.to_csv(mt_input, index=False)
    logger.info("Multi-turn long-format: %d rows", len(long_df))

    CSVBatchClassifier(
        judge=judge,
        input_csv=mt_input,
        output_csv=mt_output,
        question_column="clarifying_question",
        id_column="id",
        delay_between_calls=1.0,
    ).run()
    logger.info("Multi-turn judge complete → %s", mt_output)

    clf_mt = pd.read_csv(mt_output)
    logger.info("Multi-turn label distribution:\n%s", clf_mt["label"].value_counts().to_string())
    logger.info("All judge runs complete.")


if __name__ == "__main__":
    main()
