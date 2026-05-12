"""Phase 0.2 — Validate the LLM judge on CLAMBER.

Runs the few-shot judge on the 200-record CLAMBER eval set across multiple
temperatures, computes accuracy / F1 / κ / per-subclass breakdown / self-
consistency across temperatures, and saves PDF figures plus a markdown report.

Usage (from project root):
    python uncertainty_benchmark/scripts/validate_judge_clamber.py

Outputs:
    outputs/judge_validation/<run_id>/predictions.csv
    outputs/judge_validation/<run_id>/metrics.json
    outputs/judge_validation/<run_id>/judge_validation_report.md
    outputs/judge_validation/<run_id>/figures/*.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make `uncertainty_benchmark` importable when run from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from uncertainty_benchmark.src.judge import (  # noqa: E402
    LLMJudge,
    FEW_SHOT_EXAMPLES,
    FEW_SHOT_EXCLUSION_SET,
    few_shot_summary,
)
from uncertainty_benchmark.src.providers import GeminiProvider  # noqa: E402
from uncertainty_benchmark.src.utils import load_dotenv  # noqa: E402

logger = logging.getLogger("validate_judge")
VALID_LABELS = {"EPISTEMIC", "ALEATORIC"}


# ── Config ──────────────────────────────────────────────────────────────────

DEFAULT_INSTRUCTION = PROJECT_ROOT / "uncertainty_benchmark" / "prompts" / "judge.txt"
DEFAULT_EVAL_CSV   = PROJECT_ROOT / "uncertainty_benchmark" / "datasets" / "clamber" / "clamber_eval_200.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "uncertainty_benchmark" / "outputs" / "judge_validation"

DEFAULT_TEMPERATURES = [0.0, 0.3, 0.5]
DEFAULT_MODEL_ID = "gemini-3.1-pro-preview"


# ── Data loading ────────────────────────────────────────────────────────────

def load_eval_set(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"eval_id", "subclass", "true_label", "clarifying_question"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eval CSV missing columns: {missing}")

    # Filter out any rows whose CQ matches a few-shot example (leakage check)
    pre = len(df)
    df = df[~df["clarifying_question"].isin(FEW_SHOT_EXCLUSION_SET)].copy()
    leaked = pre - len(df)
    if leaked:
        logger.warning("Filtered %d eval rows that matched few-shot examples", leaked)

    df["true_label"] = df["true_label"].str.strip().str.upper()
    return df.reset_index(drop=True)


# ── Run judge ───────────────────────────────────────────────────────────────

def run_judge_at_temperature(
    judge: LLMJudge,
    df: pd.DataFrame,
    temperature: float,
    delay_s: float = 0.5,
) -> pd.DataFrame:
    """Run the judge over every eval row at a given temperature."""
    rows: List[Dict] = []
    n = len(df)
    logger.info("Running judge at T=%.1f over %d examples...", temperature, n)
    t0 = time.monotonic()
    for i, row in df.iterrows():
        result = judge.evaluate(row["clarifying_question"], temperature=temperature)
        rows.append({
            "eval_id":        row["eval_id"],
            "subclass":       row["subclass"],
            "true_label":     row["true_label"],
            "predicted":      result.label,
            "raw_response":   result.raw_response,
            "temperature":    temperature,
            "latency_s":      result.latency_seconds,
            "error":          result.error or "",
            "is_valid_label": result.label in VALID_LABELS,
        })
        if (i + 1) % 25 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info("  %d/%d (%.1f calls/s)", i + 1, n, rate)
        if delay_s > 0:
            time.sleep(delay_s)
    logger.info("T=%.1f done in %.1fs", temperature, time.monotonic() - t0)
    return pd.DataFrame(rows)


# ── Metrics ─────────────────────────────────────────────────────────────────

def cohens_kappa(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    if n == 0:
        return 0.0
    po = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n
    pe = sum(
        (y_true.count(lbl) / n) * (y_pred.count(lbl) / n) for lbl in labels
    )
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def per_class_pr_f1(y_true: List[str], y_pred: List[str], cls: str) -> Tuple[float, float, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def compute_metrics(df: pd.DataFrame) -> Dict:
    valid = df[df["is_valid_label"]].copy()
    invalid_n = len(df) - len(valid)
    y_true = valid["true_label"].tolist()
    y_pred = valid["predicted"].tolist()

    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(valid) if len(valid) else 0.0
    kappa = cohens_kappa(y_true, y_pred)
    p_e, r_e, f1_e = per_class_pr_f1(y_true, y_pred, "EPISTEMIC")
    p_a, r_a, f1_a = per_class_pr_f1(y_true, y_pred, "ALEATORIC")

    per_sub: Dict[str, Dict] = {}
    for sub, group in valid.groupby("subclass"):
        if len(group) == 0:
            continue
        gt = group["true_label"].tolist()
        gp = group["predicted"].tolist()
        per_sub[sub] = {
            "n":        len(group),
            "accuracy": sum(t == p for t, p in zip(gt, gp)) / len(group),
        }

    confusion = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1
    confusion_dict = {t: dict(d) for t, d in confusion.items()}

    return {
        "n_total":     len(df),
        "n_valid":     len(valid),
        "n_invalid":   invalid_n,
        "accuracy":    accuracy,
        "cohens_kappa": kappa,
        "epistemic":   {"precision": p_e, "recall": r_e, "f1": f1_e},
        "aleatoric":   {"precision": p_a, "recall": r_a, "f1": f1_a},
        "per_subclass": per_sub,
        "confusion":   confusion_dict,
    }


def compute_self_consistency(predictions_by_temp: Dict[float, pd.DataFrame]) -> Dict:
    """Cross-temperature agreement: % of items with identical labels across all runs."""
    temps = sorted(predictions_by_temp.keys())
    if len(temps) < 2:
        return {"temps": temps, "pairwise_agreement": {}, "all_agree_pct": None}

    # Pivot: index=eval_id, columns=temperature, values=predicted
    long = []
    for t, df_t in predictions_by_temp.items():
        for _, r in df_t.iterrows():
            long.append({"eval_id": r["eval_id"], "T": t, "pred": r["predicted"]})
    wide = pd.DataFrame(long).pivot(index="eval_id", columns="T", values="pred")

    # All agree
    all_agree = (wide.nunique(axis=1) == 1).mean()

    # Pairwise agreement
    pairwise: Dict[str, float] = {}
    for i, t1 in enumerate(temps):
        for t2 in temps[i + 1:]:
            agree = (wide[t1] == wide[t2]).mean()
            pairwise[f"{t1:.1f}_vs_{t2:.1f}"] = float(agree)

    return {
        "temps": temps,
        "all_agree_pct": float(all_agree),
        "pairwise_agreement": pairwise,
    }


# ── Figures ─────────────────────────────────────────────────────────────────

def fig_confusion_matrix(metrics: Dict, out_path: Path) -> None:
    classes = ["EPISTEMIC", "ALEATORIC"]
    matrix = np.array([
        [metrics["confusion"].get(t, {}).get(p, 0) for p in classes]
        for t in classes
    ])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — accuracy {metrics['accuracy']:.1%}")

    for i in range(len(classes)):
        for j in range(len(classes)):
            v = matrix[i, j]
            color = "white" if v > matrix.max() / 2 else "black"
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_per_subclass(metrics: Dict, out_path: Path) -> None:
    sub_data = metrics["per_subclass"]
    if not sub_data:
        return
    order = ["NK", "ICL", "polysemy", "co-reference", "whom", "what", "when", "where"]
    items = [(s, sub_data[s]) for s in order if s in sub_data]
    extras = [(s, d) for s, d in sub_data.items() if s not in order]
    items.extend(extras)

    labels = [s for s, _ in items]
    accs   = [d["accuracy"] for _, d in items]
    ns     = [d["n"] for _, d in items]
    epi_subs = {"NK", "ICL"}
    colors = ["#3b6ab8" if s in epi_subs else "#cc6633" for s in labels]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, accs, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, ls="--", color="gray", lw=0.8, label="random baseline")
    ax.axhline(0.85, ls="--", color="green", lw=0.8, label="acceptance threshold")

    for bar, n in zip(bars, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"n={n}", ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-CLAMBER-subclass accuracy (T=0.0)\nblue = epistemic gold, orange = aleatoric gold")
    ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_self_consistency(self_consistency: Dict, predictions_by_temp: Dict[float, pd.DataFrame], out_path: Path) -> None:
    temps = self_consistency["temps"]
    if len(temps) < 2:
        return

    # Per-temperature accuracy bars
    accs = []
    for t in temps:
        df_t = predictions_by_temp[t]
        valid = df_t[df_t["is_valid_label"]]
        if len(valid):
            acc = (valid["true_label"] == valid["predicted"]).mean()
        else:
            acc = 0.0
        accs.append(acc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: accuracy across temperatures
    ax1.bar([f"T={t:.1f}" for t in temps], accs, color="#3b6ab8", edgecolor="black", linewidth=0.5)
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy across temperatures")
    ax1.axhline(0.85, ls="--", color="green", lw=0.8, label="acceptance threshold")
    for i, a in enumerate(accs):
        ax1.text(i, a + 0.02, f"{a:.1%}", ha="center", va="bottom", fontsize=9)
    ax1.legend(fontsize=8)

    # Right: pairwise agreement
    pairs = list(self_consistency["pairwise_agreement"].keys())
    agrees = list(self_consistency["pairwise_agreement"].values())
    ax2.bar(pairs, agrees, color="#cc6633", edgecolor="black", linewidth=0.5)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Agreement rate")
    ax2.set_title(f"Pairwise label agreement\n(all-agree: {self_consistency['all_agree_pct']:.1%})")
    for i, a in enumerate(agrees):
        ax2.text(i, a + 0.02, f"{a:.1%}", ha="center", va="bottom", fontsize=9)
    plt.setp(ax2.get_xticklabels(), rotation=15, ha="right")

    fig.suptitle("Judge consistency across temperatures", y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_metric_summary(metrics_t0: Dict, out_path: Path) -> None:
    items = [
        ("Accuracy",       metrics_t0["accuracy"]),
        ("Cohen's κ",      metrics_t0["cohens_kappa"]),
        ("F1 EPISTEMIC",   metrics_t0["epistemic"]["f1"]),
        ("F1 ALEATORIC",   metrics_t0["aleatoric"]["f1"]),
    ]
    labels = [l for l, _ in items]
    values = [v for _, v in items]

    thresholds = [0.85, 0.70, 0.80, 0.80]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(labels, values, color="#3b6ab8", edgecolor="black", linewidth=0.5)
    for bar, val, thr in zip(bars, values, thresholds):
        c = "green" if val >= thr else "red"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10, color=c, fontweight="bold")
        ax.hlines(thr, bar.get_x(), bar.get_x() + bar.get_width(),
                  colors="green", linestyles="--", linewidth=1)

    ax.set_ylim(0, 1.1)
    ax.set_title("Phase 0.2 acceptance gate (T=0.0)\nGreen dashes = required threshold")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ── Report ──────────────────────────────────────────────────────────────────

def write_report(
    out_dir: Path,
    metrics_by_temp: Dict[float, Dict],
    self_consistency: Dict,
    config: Dict,
) -> None:
    metrics_t0 = metrics_by_temp[min(metrics_by_temp.keys())]
    gate_pass = (
        metrics_t0["accuracy"] >= 0.85
        and metrics_t0["cohens_kappa"] >= 0.70
        and metrics_t0["epistemic"]["f1"] >= 0.80
        and metrics_t0["aleatoric"]["f1"] >= 0.80
    )

    lines: List[str] = []
    lines.append(f"# Phase 0.2 — Judge Validation Report")
    lines.append("")
    lines.append(f"**Run ID:** {config['run_id']}")
    lines.append(f"**Model:** {config['model_id']}")
    lines.append(f"**Temperatures:** {config['temperatures']}")
    lines.append(f"**Eval set:** {config['eval_set']} ({metrics_t0['n_total']} records)")
    lines.append(f"**Few-shot:** {config['few_shot_summary']}")
    lines.append("")
    lines.append("## Acceptance Gate")
    lines.append("")
    lines.append(f"**Overall: {'PASS' if gate_pass else 'FAIL'}**")
    lines.append("")
    lines.append("| Metric | Value | Threshold | Pass |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Accuracy | {metrics_t0['accuracy']:.4f} | ≥ 0.85 | {'✓' if metrics_t0['accuracy'] >= 0.85 else '✗'} |")
    lines.append(f"| Cohen's κ | {metrics_t0['cohens_kappa']:.4f} | ≥ 0.70 | {'✓' if metrics_t0['cohens_kappa'] >= 0.70 else '✗'} |")
    lines.append(f"| F1 EPISTEMIC | {metrics_t0['epistemic']['f1']:.4f} | ≥ 0.80 | {'✓' if metrics_t0['epistemic']['f1'] >= 0.80 else '✗'} |")
    lines.append(f"| F1 ALEATORIC | {metrics_t0['aleatoric']['f1']:.4f} | ≥ 0.80 | {'✓' if metrics_t0['aleatoric']['f1'] >= 0.80 else '✗'} |")
    lines.append("")
    lines.append("## Per-Class Performance (T=0.0)")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| EPISTEMIC | {metrics_t0['epistemic']['precision']:.4f} | {metrics_t0['epistemic']['recall']:.4f} | {metrics_t0['epistemic']['f1']:.4f} |")
    lines.append(f"| ALEATORIC | {metrics_t0['aleatoric']['precision']:.4f} | {metrics_t0['aleatoric']['recall']:.4f} | {metrics_t0['aleatoric']['f1']:.4f} |")
    lines.append("")
    lines.append("## Per-Subclass Accuracy (T=0.0)")
    lines.append("")
    lines.append("| Subclass | n | Accuracy |")
    lines.append("|---|---|---|")
    for sub, d in sorted(metrics_t0["per_subclass"].items(), key=lambda x: -x[1]["accuracy"]):
        lines.append(f"| {sub} | {d['n']} | {d['accuracy']:.4f} |")
    lines.append("")
    lines.append("## Cross-Temperature Self-Consistency")
    lines.append("")
    if self_consistency["all_agree_pct"] is None:
        lines.append("_Skipped — only one temperature was run._")
    else:
        lines.append(f"**Items with identical label across all temps:** {self_consistency['all_agree_pct']:.1%}")
        lines.append("")
        lines.append("| Pair | Agreement |")
        lines.append("|---|---|")
        for k, v in self_consistency["pairwise_agreement"].items():
            lines.append(f"| {k} | {v:.1%} |")
    lines.append("")
    lines.append("## Per-Temperature Accuracy")
    lines.append("")
    lines.append("| Temperature | Accuracy | Cohen's κ | F1 EPI | F1 ALE | n_invalid |")
    lines.append("|---|---|---|---|---|---|")
    for t in sorted(metrics_by_temp.keys()):
        m = metrics_by_temp[t]
        lines.append(
            f"| {t:.1f} | {m['accuracy']:.4f} | {m['cohens_kappa']:.4f} | "
            f"{m['epistemic']['f1']:.4f} | {m['aleatoric']['f1']:.4f} | {m['n_invalid']} |"
        )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `figures/01_acceptance_gate.pdf` — gate metrics vs thresholds")
    lines.append("- `figures/02_confusion_matrix.pdf` — confusion matrix at T=0.0")
    lines.append("- `figures/03_per_subclass_accuracy.pdf` — accuracy per CLAMBER subclass")
    lines.append("- `figures/04_self_consistency.pdf` — accuracy + pairwise agreement across temperatures")

    (out_dir / "judge_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the LLM judge on CLAMBER")
    parser.add_argument("--instruction-path", type=Path, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    parser.add_argument("--delay-s", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit eval to first N rows (for smoke testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan and exit without calling the model")
    parser.add_argument("--resume-run-id", type=str, default=None,
                        help="Reuse an existing run dir; skip temperatures whose predictions CSV already exists")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / "uncertainty_benchmark" / ".env")

    if args.resume_run_id:
        run_id = args.resume_run_id
        logger.info("Resuming existing run: %s", run_id)
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_eval_set(args.eval_csv)
    if args.limit:
        df = df.head(args.limit).copy()
    logger.info("Loaded %d eval examples after exclusion", len(df))
    logger.info("%s", few_shot_summary())

    config = {
        "run_id":            run_id,
        "model_id":          args.model_id,
        "temperatures":      args.temperatures,
        "eval_set":          str(args.eval_csv.relative_to(PROJECT_ROOT)),
        "n_eval":            len(df),
        "few_shot_summary":  few_shot_summary(),
        "instruction_path":  str(args.instruction_path.relative_to(PROJECT_ROOT)),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.dry_run:
        logger.info("DRY RUN — config written to %s. Exiting before any LLM calls.", run_dir / "config.json")
        return 0

    predictions_by_temp: Dict[float, pd.DataFrame] = {}
    metrics_by_temp: Dict[float, Dict] = {}

    judge: Optional[LLMJudge] = None  # build lazily — only if some temp needs running

    for t in args.temperatures:
        pred_path = run_dir / f"predictions_T{t:.1f}.csv"
        if args.resume_run_id and pred_path.exists():
            df_t = pd.read_csv(pred_path)
            df_t["is_valid_label"] = df_t["predicted"].isin(VALID_LABELS)
            logger.info("T=%.1f: loaded %d rows from existing %s", t, len(df_t), pred_path.name)
        else:
            if judge is None:
                provider = GeminiProvider(model_id=args.model_id)
                judge = LLMJudge(
                    provider=provider,
                    instructions_path=args.instruction_path,
                    few_shot_examples=FEW_SHOT_EXAMPLES,
                )
            df_t = run_judge_at_temperature(judge, df, temperature=t, delay_s=args.delay_s)
            df_t.to_csv(pred_path, index=False)

        predictions_by_temp[t] = df_t
        metrics_by_temp[t] = compute_metrics(df_t)
        logger.info(
            "T=%.1f → acc=%.4f κ=%.4f F1_E=%.4f F1_A=%.4f",
            t, metrics_by_temp[t]["accuracy"], metrics_by_temp[t]["cohens_kappa"],
            metrics_by_temp[t]["epistemic"]["f1"], metrics_by_temp[t]["aleatoric"]["f1"],
        )

    self_consistency = compute_self_consistency(predictions_by_temp)

    metrics_t0 = metrics_by_temp[min(metrics_by_temp.keys())]
    fig_metric_summary(metrics_t0, figures_dir / "01_acceptance_gate.pdf")
    fig_confusion_matrix(metrics_t0, figures_dir / "02_confusion_matrix.pdf")
    fig_per_subclass(metrics_t0, figures_dir / "03_per_subclass_accuracy.pdf")
    fig_self_consistency(self_consistency, predictions_by_temp, figures_dir / "04_self_consistency.pdf")

    (run_dir / "metrics.json").write_text(
        json.dumps({
            "by_temperature":   {f"{t:.1f}": m for t, m in metrics_by_temp.items()},
            "self_consistency": self_consistency,
        }, indent=2), encoding="utf-8",
    )

    write_report(run_dir, metrics_by_temp, self_consistency, config)

    logger.info("Done. Outputs in %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
