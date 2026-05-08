"""Run the CLAMBER judge accuracy evaluation for a given model.

Reads  : outputs/clamber/clamber_eval_input.csv  (200 labelled CQs, fixed seed)
Writes : outputs/clamber/clamber_judge_eval_fewshot_{model_id}.csv
Prints : accuracy vs gemini-2.5-flash baseline for direct comparison

Usage:
    python run_clamber_judge_eval.py --model gemini-2.5-pro-preview
    python run_clamber_judge_eval.py --model gemini-2.5-flash   # reproduce baseline
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

OUTPUTS_DIR     = ROOT / "outputs" / "clamber"
PROMPTS_DIR     = ROOT / "prompts" / "medqa"
JUDGE_INSTR     = PROMPTS_DIR / "judge_instruction.txt"
EVAL_INPUT      = OUTPUTS_DIR / "clamber_eval_input.csv"
FLASH_RESULTS   = OUTPUTS_DIR / "clamber_judge_eval_fewshot.csv"   # existing baseline
REQUEST_INTERVAL = 1.5   # slightly longer for pro to avoid rate limits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Few-shot examples (identical to clamber_judge_eval.ipynb) ─────────────
FEW_SHOT_DEFS = [
    (
        "What is the context or field in which 'Comallcium' is typically used or referenced?",
        "EPISTEMIC",
        "The model has never encountered this entity and is asking for information to fill a "
        "specific knowledge gap. A definitive answer exists — once the user provides context "
        "the uncertainty is completely resolved.",
    ),
    (
        "Is the category either animal or outdoor location?",
        "EPISTEMIC",
        "The query contains examples that contradict each other under any single category rule "
        "the model knows. The binary phrasing does NOT mean multiple valid answers exist — it "
        "means the model cannot reconcile the contradiction with its knowledge. There is one "
        "correct categorisation; the model simply cannot determine it. This is a "
        "knowledge/reasoning gap, not a preference gap.",
    ),
    (
        "Is the category either religious leader or male pronoun?",
        "EPISTEMIC",
        "Same ICL pattern: the model is exposing an internal contradiction it cannot resolve. "
        "The two options are not equally valid user preferences — one is the correct answer "
        "according to the underlying rule, and the model lacks the knowledge to determine which.",
    ),
    (
        "What does she refer to? the sister-in-law or Amanda?",
        "ALEATORIC",
        "The pronoun 'she' could validly refer to either person. This is NOT a knowledge gap — "
        "no additional fact or training data can determine the correct referent. Only the user's "
        "original conversational context can resolve it, making this irreducibly ambiguous.",
    ),
    (
        "Which year: 1984, 1985, 1998, 2004, or 2009?",
        "ALEATORIC",
        "The query is underspecified across multiple valid years. The model is asking the user "
        "to select which version or instance they mean — all options are equally valid. No "
        "external fact can determine which year the user intends; only the user can resolve this.",
    ),
    (
        "What factors are most important to you in a job, such as salary, location, career "
        "growth, or company culture?",
        "ALEATORIC",
        "The correct answer depends entirely on the individual user's preferences. No fact, "
        "no training data, and no amount of knowledge can resolve this — it is inherently "
        "personal and has many equally valid answers.",
    ),
]


def compute_accuracy(eval_df: pd.DataFrame, results_csv: Path) -> dict:
    res = pd.read_csv(results_csv)
    merged = eval_df.merge(
        res[["id", "label"]].rename(columns={"id": "eval_id"}),
        on="eval_id", how="left",
    )
    merged["pred"] = merged["label"].str.strip().str.upper()
    merged["valid"] = merged["pred"].isin({"EPISTEMIC", "ALEATORIC"})
    merged["correct"] = merged["true_label"] == merged["pred"]
    valid = merged[merged["valid"]]
    per_class = valid.groupby("true_label")["correct"].mean().to_dict()
    per_sub   = valid.groupby(["true_label", "subclass"])["correct"].mean().to_dict()
    return {
        "total":     len(merged),
        "valid":     len(valid),
        "correct":   int(valid["correct"].sum()),
        "accuracy":  valid["correct"].mean(),
        "per_class": per_class,
        "per_sub":   per_sub,
        "merged":    merged,
    }


def print_comparison(flash: dict, pro: dict, pro_model: str) -> None:
    print("\n" + "=" * 65)
    print("CLAMBER JUDGE ACCURACY COMPARISON")
    print("=" * 65)
    print(f"{'Metric':<30} {'gemini-2.5-flash':>18} {pro_model:>20}")
    print("-" * 65)
    print(f"{'Overall accuracy':<30} {flash['accuracy']:>17.1%} {pro['accuracy']:>19.1%}  "
          f"({'+'if pro['accuracy']>flash['accuracy'] else ''}"
          f"{pro['accuracy']-flash['accuracy']:+.1%})")
    for cls in ["EPISTEMIC", "ALEATORIC"]:
        f_acc = flash["per_class"].get(cls, float("nan"))
        p_acc = pro["per_class"].get(cls, float("nan"))
        delta = p_acc - f_acc
        print(f"  {cls:<28} {f_acc:>17.1%} {p_acc:>19.1%}  ({delta:+.1%})")
    print("-" * 65)
    print("\nPer-subclass (Flash -> Pro):")
    all_keys = sorted(set(flash["per_sub"]) | set(pro["per_sub"]))
    for key in all_keys:
        f_a = flash["per_sub"].get(key, float("nan"))
        p_a = pro["per_sub"].get(key, float("nan"))
        label, sub = key
        delta = p_a - f_a
        print(f"  {label:12} {sub:14} {f_a:6.1%} -> {p_a:6.1%}  ({delta:+.1%})")
    print("=" * 65)

    verdict = "UPGRADE" if pro["accuracy"] > flash["accuracy"] else (
              "TIE"     if pro["accuracy"] == flash["accuracy"] else "NO IMPROVEMENT")
    print(f"\nVerdict: {verdict}")
    if verdict == "UPGRADE":
        print(f"  Pro model improves overall accuracy by "
              f"{pro['accuracy']-flash['accuracy']:+.1%} -> REPLACE flash judge")
    elif verdict == "TIE":
        print("  Same accuracy -> keep flash (faster + cheaper)")
    else:
        print("  Flash is better or equal -> keep flash judge")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3.1-pro-preview",
                        help="Gemini model ID to test as judge")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output file already exists")
    args = parser.parse_args()

    model_id = args.model
    # Sanitise model ID for use in filename
    safe_name = model_id.replace("/", "_").replace(":", "_")
    output_csv = OUTPUTS_DIR / f"clamber_judge_eval_fewshot_{safe_name}.csv"

    from src.utils import load_dotenv
    from src.providers import GeminiProvider
    from src.judge import LLMJudge, CSVBatchClassifier, FewShotExample

    load_dotenv(ROOT / ".env")
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_csv(EVAL_INPUT)
    logger.info("Eval set: %d rows (%s)",
                len(eval_df), eval_df["true_label"].value_counts().to_dict())

    # ── Run pro judge ─────────────────────────────────────────────────────
    if output_csv.exists() and not args.force:
        logger.info("Output already exists: %s  (use --force to rerun)", output_csv)
    else:
        if output_csv.exists():
            output_csv.unlink()
            logger.info("Deleted existing output — regenerating.")

        few_shot = [
            FewShotExample(input=q, expected_output=lbl, explanation=exp)
            for q, lbl, exp in FEW_SHOT_DEFS
        ]

        provider = GeminiProvider(model_id=model_id)

        # Smoke test
        from src.judge import LLMJudge
        judge = LLMJudge(
            provider=provider,
            instructions_path=JUDGE_INSTR,
            few_shot_examples=few_shot,
            label_parser=lambda text: text.strip().upper(),
        )
        smoke = judge.evaluate(
            "Have you been diagnosed with hypertension or any other heart condition before?"
        )
        logger.info("Smoke test -> %s (expected EPISTEMIC)", smoke.label)
        if smoke.label != "EPISTEMIC":
            logger.warning("Smoke test unexpected result: %s", smoke.label)

        classifier = CSVBatchClassifier(
            judge=judge,
            input_csv=EVAL_INPUT,
            output_csv=output_csv,
            question_column="clarifying_question",
            id_column="eval_id",
            delay_between_calls=REQUEST_INTERVAL,
        )
        logger.info("Running judge with model: %s", model_id)
        classifier.run()
        logger.info("Done -> %s", output_csv)

    # ── Compare accuracy ──────────────────────────────────────────────────
    logger.info("Computing accuracy comparison...")
    flash_stats = compute_accuracy(eval_df, FLASH_RESULTS)
    pro_stats   = compute_accuracy(eval_df, output_csv)
    print_comparison(flash_stats, pro_stats, model_id)

    # Print misclassified by pro
    pro_merged = pro_stats["merged"]
    errors = pro_merged[pro_merged["valid"] & ~pro_merged["correct"]]
    if len(errors):
        print(f"\nMisclassified by {model_id} ({len(errors)} errors):")
        for _, row in errors.iterrows():
            print(f"  [{row['true_label']:>9} -> {row['pred']:>9}] "
                  f"[{row['subclass']:>12}] {str(row['clarifying_question'])[:90]}")


if __name__ == "__main__":
    main()
