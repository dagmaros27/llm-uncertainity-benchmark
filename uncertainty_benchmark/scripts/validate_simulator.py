"""Phase 0.3 — Validate the simulator on existing pilot CQs.

Samples 20 cases per dataset (medqa, msdialog, sharc) from pilot run logs,
reconstructs each case's Layer 1 simulator_context, sends (CQ, context) to the
gemini-3.1-pro-preview simulator, and saves results for manual faithfulness
labelling.

Outputs:
    outputs/simulator_validation/<run_id>/
      ├── config.json
      ├── samples_input.csv          # what we will send (review pre-run)
      ├── results.csv                # simulator outputs + auto-detected hedge
      ├── per_dataset_metrics.json   # hedge rate per dataset
      ├── simulator_validation_report.md
      └── figures/01_hedge_rate_by_dataset.pdf
                  02_response_length_by_dataset.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from uncertainty_benchmark.src.pipelines import Simulator, is_hedged, DATASET_CONTEXT_LABELS  # noqa: E402
from uncertainty_benchmark.src.providers import GeminiProvider  # noqa: E402
from uncertainty_benchmark.src.utils import clean_simulator_context, load_dotenv  # noqa: E402

logger = logging.getLogger("validate_simulator")


# ── Config ──────────────────────────────────────────────────────────────────

PILOT_ROOT = PROJECT_ROOT / "pilot_study"
PROMPTS_DIR = PROJECT_ROOT / "uncertainty_benchmark" / "prompts" / "simulator"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "uncertainty_benchmark" / "outputs" / "simulator_validation"
DEFAULT_MODEL_ID = "gemini-3.1-pro-preview"
DEFAULT_N_PER_DATASET = 20
SEED = 42

DATASETS = ["medqa", "msdialog", "sharc"]


# ── Sampling helpers ────────────────────────────────────────────────────────

def _stratified_sample(df: pd.DataFrame, n: int, by: str, seed: int) -> pd.DataFrame:
    """Stratify by `by` column if present; else uniform sample."""
    if by not in df.columns:
        return df.sample(min(n, len(df)), random_state=seed)
    groups = df.groupby(by)
    if len(groups) == 0:
        return df.sample(min(n, len(df)), random_state=seed)
    per = max(1, n // len(groups))
    parts = [g.sample(min(per, len(g)), random_state=seed) for _, g in groups]
    out = pd.concat(parts).head(n)
    if len(out) < n:
        rest = df[~df.index.isin(out.index)].sample(min(n - len(out), len(df) - len(out)), random_state=seed)
        out = pd.concat([out, rest])
    return out.head(n)


def sample_medqa(n: int) -> List[Dict]:
    """Sample n cases from MedQA singleturn results; reconstruct contexts."""
    results_csv = PILOT_ROOT / "outputs" / "medqa" / "gemini-2.5-flash" / "phase1_singleturn_results.csv"
    cases_jsonl = PILOT_ROOT / "datasets" / "medqa" / "cases.jsonl"

    df = pd.read_csv(results_csv)
    df = df[df["clarifying_question"].fillna("").str.strip().ne("")]

    # case_id mapping
    case_by_id: Dict[str, dict] = {}
    with cases_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            case_by_id[str(r["case_id"])] = r

    df = df[df["id"].astype(str).isin(case_by_id.keys())]
    sampled = _stratified_sample(df, n, by="difficulty", seed=SEED)

    out: List[Dict] = []
    for _, row in sampled.iterrows():
        case_id = str(row["id"])
        case = case_by_id[case_id]
        # Combine the three context sections, cleaning out diagnostic conclusions.
        ctx = "\n\n---\n\n".join(
            clean_simulator_context(case.get(k, "") or "")
            for k in ("patient_context", "nurse_context", "specialist_context")
            if (case.get(k, "") or "").strip()
        )
        out.append({
            "dataset":            "medqa",
            "case_id":            case_id,
            "stratum":            row.get("difficulty", ""),
            "clarifying_question": row["clarifying_question"].strip(),
            "simulator_context":  ctx,
        })
    return out


def sample_msdialog(n: int) -> List[Dict]:
    results_csv = PILOT_ROOT / "outputs" / "ms-dialog" / "gemini-2.5-flash" / "phase1_flex_results.csv"
    cases_jsonl = PILOT_ROOT / "datasets" / "ms-dialog" / "msdialog_100.jsonl"

    df = pd.read_csv(results_csv)
    df = df[df["cq_1"].fillna("").str.strip().ne("")]

    case_by_id: Dict[str, dict] = {}
    with cases_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            case_by_id[str(r["case_id"])] = r

    df = df[df["id"].astype(str).isin(case_by_id.keys())]
    sampled = _stratified_sample(df, n, by="category", seed=SEED)

    out: List[Dict] = []
    for _, row in sampled.iterrows():
        case_id = str(row["id"])
        case = case_by_id[case_id]
        ctx = (case.get("simulator_context") or "").strip()
        if not ctx:
            continue
        out.append({
            "dataset":            "msdialog",
            "case_id":            case_id,
            "stratum":            row.get("category", ""),
            "clarifying_question": row["cq_1"].strip(),
            "simulator_context":  ctx,
        })
    return out[:n]


def sample_sharc(n: int) -> List[Dict]:
    results_csv = PILOT_ROOT / "outputs" / "sharc" / "gemini-2.5-flash" / "phase1_flex_results.csv"
    context_jsonl = PILOT_ROOT / "datasets" / "sharc" / "sharc_context_cache.jsonl"

    df = pd.read_csv(results_csv)
    df = df[df["cq_1"].fillna("").str.strip().ne("")]

    ctx_by_id: Dict[str, str] = {}
    with context_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            ctx_by_id[str(r["id"])] = (r.get("context_essay") or "").strip()

    df = df[df["id"].astype(str).isin(ctx_by_id.keys())]
    # Stratify by tree_id if it exists; otherwise random
    sampled = _stratified_sample(df, n, by="tree_id", seed=SEED)

    out: List[Dict] = []
    for _, row in sampled.iterrows():
        case_id = str(row["id"])
        ctx = ctx_by_id.get(case_id, "")
        if not ctx:
            continue
        out.append({
            "dataset":            "sharc",
            "case_id":            case_id,
            "stratum":            str(row.get("tree_id", "")),
            "clarifying_question": row["cq_1"].strip(),
            "simulator_context":  ctx,
        })
    return out[:n]


SAMPLERS = {
    "medqa":    sample_medqa,
    "msdialog": sample_msdialog,
    "sharc":    sample_sharc,
}


# ── Run simulator ──────────────────────────────────────────────────────────

def run(samples: List[Dict], simulators: Dict[str, Simulator], delay_s: float) -> List[Dict]:
    rows: List[Dict] = []
    for i, s in enumerate(samples, 1):
        sim = simulators[s["dataset"]]
        t0 = time.monotonic()
        try:
            answer = sim.answer(s["clarifying_question"], s["simulator_context"])
            err = ""
        except Exception as exc:
            answer = ""
            err = str(exc)
        latency = round(time.monotonic() - t0, 2)
        hedged = is_hedged(answer)
        rows.append({
            **s,
            "answer":     answer,
            "hedged":     hedged,
            "latency_s":  latency,
            "error":      err,
        })
        logger.info(
            "[%d/%d] %s/%s — %s — %.1fs — %s",
            i, len(samples), s["dataset"], s["case_id"][:12],
            "HEDGED" if hedged else "ANSWERED",
            latency, answer[:80].replace("\n", " "),
        )
        if delay_s > 0:
            time.sleep(delay_s)
    return rows


# ── Metrics + figures ──────────────────────────────────────────────────────

def per_dataset_metrics(rows: List[Dict]) -> Dict[str, Dict]:
    df = pd.DataFrame(rows)
    out: Dict[str, Dict] = {}
    for ds, g in df.groupby("dataset"):
        n = len(g)
        n_hedged = int(g["hedged"].sum())
        n_errored = int((g["error"].fillna("") != "").sum())
        out[ds] = {
            "n":              n,
            "n_hedged":       n_hedged,
            "hedge_rate":     n_hedged / n if n else 0.0,
            "n_errored":      n_errored,
            "mean_latency_s": float(g["latency_s"].mean()) if n else 0.0,
            "mean_answer_chars": float(g["answer"].fillna("").str.len().mean()) if n else 0.0,
        }
    return out


def fig_hedge_rate(metrics: Dict[str, Dict], out_path: Path) -> None:
    ds = list(metrics.keys())
    rates = [metrics[d]["hedge_rate"] for d in ds]
    ns    = [metrics[d]["n"] for d in ds]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(ds, rates, color="#3b6ab8", edgecolor="black", linewidth=0.5)
    ax.axhline(0.20, ls="--", color="red", lw=1, label="hedge rate gate (≤ 0.20)")
    for bar, r, n in zip(bars, rates, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{r:.0%} ({metrics[ds[bars.index(bar)]]['n_hedged']}/{n})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(0.4, max(rates) + 0.05))
    ax.set_ylabel("Hedge rate")
    ax.set_title("Simulator hedge rate by dataset (gemini-3.1-pro-preview)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_response_length(rows: List[Dict], out_path: Path) -> None:
    df = pd.DataFrame(rows)
    df["len_chars"] = df["answer"].fillna("").str.len()
    fig, ax = plt.subplots(figsize=(7, 4))
    datasets = sorted(df["dataset"].unique())
    data = [df[df["dataset"] == d]["len_chars"].tolist() for d in datasets]
    ax.boxplot(data, labels=datasets, showmeans=True)
    ax.set_ylabel("Answer length (chars)")
    ax.set_title("Simulator response length by dataset")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ── Report ──────────────────────────────────────────────────────────────────

def write_report(out_dir: Path, metrics: Dict[str, Dict], config: Dict, rows: List[Dict]) -> None:
    lines: List[str] = []
    lines.append("# Phase 0.3 — Simulator Validation Report")
    lines.append("")
    lines.append(f"**Run ID:** {config['run_id']}")
    lines.append(f"**Model:** {config['model_id']}")
    lines.append(f"**Samples per dataset:** {config['n_per_dataset']}")
    lines.append("")

    lines.append("## Per-dataset hedge rates (auto-detected)")
    lines.append("")
    lines.append("| Dataset | n | Hedged | Hedge rate | Mean latency (s) | Mean answer chars |")
    lines.append("|---|---|---|---|---|---|")
    for ds, m in metrics.items():
        lines.append(
            f"| {ds} | {m['n']} | {m['n_hedged']} | {m['hedge_rate']:.1%} | "
            f"{m['mean_latency_s']:.1f} | {m['mean_answer_chars']:.0f} |"
        )
    lines.append("")
    lines.append(f"**Hedge gate (≤ 20% per dataset):** {'PASS' if all(m['hedge_rate'] < 0.20 for m in metrics.values()) else 'CHECK FAILED CASES'}")
    lines.append("")

    lines.append("## Manual labelling instructions")
    lines.append("")
    lines.append("Open `results.csv` and add a column `faithfulness_label` per row using:")
    lines.append("- **FAITHFUL** — every factual claim in the answer is explicitly supported by `simulator_context`")
    lines.append("- **EXTRAPOLATED** — answer goes beyond context (interpretation, synthesis, or implication not literally present)")
    lines.append("- **HALLUCINATED** — claim not in context or directly contradicts it")
    lines.append("- **HEDGED** — refusal, no claim made (already auto-flagged in `hedged` column)")
    lines.append("")
    lines.append("Optional `notes` column for one-line reasoning.")
    lines.append("")

    lines.append("## Figures")
    lines.append("- `figures/01_hedge_rate_by_dataset.pdf`")
    lines.append("- `figures/02_response_length_by_dataset.pdf`")

    (out_dir / "simulator_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate simulator on pilot CQs")
    parser.add_argument("--output-dir",     type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id",       type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--n-per-dataset",  type=int, default=DEFAULT_N_PER_DATASET)
    parser.add_argument("--delay-s",        type=float, default=0.5)
    parser.add_argument("--datasets",       type=str, nargs="+", default=DATASETS)
    parser.add_argument("--dry-run",        action="store_true",
                        help="Sample + write samples_input.csv but don't call the model")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / "uncertainty_benchmark" / ".env")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Sample ────────────────────────────────────────────────────────────
    random.seed(SEED)
    all_samples: List[Dict] = []
    for ds in args.datasets:
        if ds not in SAMPLERS:
            logger.warning("Unknown dataset %r — skipping", ds)
            continue
        ds_samples = SAMPLERS[ds](args.n_per_dataset)
        logger.info("Sampled %d %s cases", len(ds_samples), ds)
        all_samples.extend(ds_samples)

    samples_df = pd.DataFrame(all_samples)
    samples_df.to_csv(run_dir / "samples_input.csv", index=False)
    logger.info("Wrote %d samples → %s", len(all_samples), run_dir / "samples_input.csv")

    config = {
        "run_id":          run_id,
        "model_id":        args.model_id,
        "n_per_dataset":   args.n_per_dataset,
        "datasets":        args.datasets,
        "n_total":         len(all_samples),
        "seed":            SEED,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.dry_run:
        logger.info("DRY RUN — samples + config written. Exiting before LLM calls.")
        return 0

    # ── Build simulators ────────────────────────────────────────────────────
    provider = GeminiProvider(model_id=args.model_id)
    simulators: Dict[str, Simulator] = {}
    for ds in args.datasets:
        prompt_path = PROMPTS_DIR / f"{ds}.txt"
        if not prompt_path.exists():
            logger.error("Missing simulator prompt for %s: %s", ds, prompt_path)
            return 1
        simulators[ds] = Simulator(
            provider=provider,
            instructions_path=prompt_path,
            context_label=DATASET_CONTEXT_LABELS.get(ds, "Situation summary"),
        )

    # ── Run ────────────────────────────────────────────────────────────────
    rows = run(all_samples, simulators, delay_s=args.delay_s)

    # Persist results CSV with a `faithfulness_label` column ready for manual fill
    df_out = pd.DataFrame(rows)
    df_out["faithfulness_label"] = ""
    df_out["faithfulness_notes"] = ""
    df_out.to_csv(run_dir / "results.csv", index=False)
    logger.info("Wrote results → %s", run_dir / "results.csv")

    # ── Metrics + figures ──────────────────────────────────────────────────
    metrics = per_dataset_metrics(rows)
    (run_dir / "per_dataset_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig_hedge_rate(metrics, figures_dir / "01_hedge_rate_by_dataset.pdf")
    fig_response_length(rows, figures_dir / "02_response_length_by_dataset.pdf")

    write_report(run_dir, metrics, config, rows)

    logger.info("Done. Outputs in %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
