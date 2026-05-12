"""ShARC CQ-type analysis — extends flex analysis with EPISTEMIC vs ALEATORIC breakdown.

Joins flex results with judge classifications to answer:
  - What types of CQs does the model ask?
  - Does CQ type relate to recovery success?
  - How does CQ type evolve across turns?
  - Is the model's first CQ type predictive of final accuracy?

Output:
  outputs/sharc/<model>/cqtypes_fig.png
"""

from __future__ import annotations

import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent.resolve()
MODEL_ID = "gemini-2.5-flash"
RESULTS_CSV   = ROOT / "outputs" / "sharc" / MODEL_ID / "phase1_flex_results.csv"
CLASSIFIED_CSV = ROOT / "outputs" / "sharc" / MODEL_ID / "phase1_flex_classified.csv"
FIG_PATH      = ROOT / "outputs" / "sharc" / MODEL_ID / "cqtypes_fig.png"

EPI_COLOR  = "#1f77b4"  # blue
ALE_COLOR  = "#ff7f0e"  # orange


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(RESULTS_CSV)
    df = df[df["n_cqs_asked"] >= 0].copy()
    df["is_correct_preliminary"] = df["is_correct_preliminary"].astype(bool)
    df["is_correct_final"]       = df["is_correct_final"].astype(bool)
    df["n_cqs_asked"]            = df["n_cqs_asked"].astype(int)

    cls = pd.read_csv(CLASSIFIED_CSV)
    # Some judge schemas use 'question'/'label' columns; standardise
    q_col = "question" if "question" in cls.columns else "clarifying_question"
    cls = cls.rename(columns={q_col: "clarifying_question"})
    cls["label"] = cls["label"].str.upper()
    return df, cls


def attach_first_cq_type(df: pd.DataFrame, cls: pd.DataFrame) -> pd.DataFrame:
    first_cq = cls[cls["turn"] == 1][["id", "label"]].rename(columns={"label": "first_cq_type"})
    df = df.merge(first_cq, on="id", how="left")
    return df


def print_summary(df: pd.DataFrame, cls: pd.DataFrame) -> None:
    print("=" * 70)
    print("CQ TYPE SUMMARY")
    print("=" * 70)
    print(f"Total CQs classified : {len(cls)}")
    print(f"Across cases         : {cls['id'].nunique()}")
    print()
    print("Overall label distribution:")
    print(cls["label"].value_counts().to_string())
    print()
    print("By turn:")
    for t in sorted(cls["turn"].unique()):
        sub = cls[cls["turn"] == t]
        epi = (sub["label"] == "EPISTEMIC").sum()
        ale = (sub["label"] == "ALEATORIC").sum()
        print(f"   Turn {t}: total={len(sub):3d} | EPI={epi:3d} ({epi/len(sub):.0%}) | ALE={ale:3d} ({ale/len(sub):.0%})")
    print()
    cases_with_ale = cls[cls["label"] == "ALEATORIC"]["id"].unique()
    print(f"Cases with at least 1 ALEATORIC CQ: {len(cases_with_ale)}/{cls['id'].nunique()}")

    # Final accuracy by first CQ type
    print()
    print("Final accuracy by first CQ type:")
    for t, sub in df.dropna(subset=["first_cq_type"]).groupby("first_cq_type"):
        acc = sub["is_correct_final"].mean()
        print(f"   {t}: {acc:.0%}  (n={len(sub)})")

    # Recovery rate by first CQ type
    wrong = df[~df["is_correct_preliminary"] & df["first_cq_type"].notna()]
    print()
    print("Recovery rate (prelim wrong → final right) by first CQ type:")
    for t, sub in wrong.groupby("first_cq_type"):
        rec = sub["is_correct_final"].mean()
        print(f"   {t}: {rec:.0%}  (n={len(sub)})")


def make_figure(df: pd.DataFrame, cls: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # ── A. Overall label distribution ─────────────────────────────────────
    ax = axes[0, 0]
    counts = cls["label"].value_counts()
    colors = {"EPISTEMIC": EPI_COLOR, "ALEATORIC": ALE_COLOR}
    bars = ax.bar(counts.index, counts.values,
                  color=[colors.get(l, "#888") for l in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, f"{v}\n({v/len(cls):.0%})",
                ha="center", fontsize=10)
    ax.set_title(f"A. CQ Type Distribution (n={len(cls)})")
    ax.set_ylabel("CQs")

    # ── B. Label by turn ──────────────────────────────────────────────────
    ax = axes[0, 1]
    by_turn = pd.crosstab(cls["turn"], cls["label"])
    for label in ["EPISTEMIC", "ALEATORIC"]:
        if label not in by_turn.columns:
            by_turn[label] = 0
    by_turn = by_turn[["EPISTEMIC", "ALEATORIC"]]
    by_turn.plot(kind="bar", stacked=True, ax=ax,
                 color=[EPI_COLOR, ALE_COLOR], width=0.6, legend=False)
    for i, t in enumerate(by_turn.index):
        total = by_turn.loc[t].sum()
        ax.text(i, total + 0.5, f"n={total}", ha="center", fontsize=9)
    ax.set_title("B. CQ Type by Turn")
    ax.set_xlabel("Turn"); ax.set_ylabel("CQs")
    ax.legend(["EPISTEMIC", "ALEATORIC"], loc="upper right")
    ax.tick_params(axis="x", rotation=0)

    # ── C. Final accuracy by first CQ type ────────────────────────────────
    ax = axes[0, 2]
    sub = df.dropna(subset=["first_cq_type"])
    grp = sub.groupby("first_cq_type")["is_correct_final"].agg(["mean", "count"])
    grp = grp.reindex(["EPISTEMIC", "ALEATORIC"]).dropna()
    bars = ax.bar(grp.index, grp["mean"],
                  color=[colors[l] for l in grp.index])
    for b, (k, row) in zip(bars, grp.iterrows()):
        ax.text(b.get_x()+b.get_width()/2, row["mean"]+0.02,
                f"{row['mean']:.0%}\n(n={int(row['count'])})", ha="center", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("C. Final Accuracy by First CQ Type")
    ax.set_ylabel("Final accuracy")

    # ── D. Recovery rate by first CQ type ────────────────────────────────
    ax = axes[1, 0]
    wrong = df[~df["is_correct_preliminary"] & df["first_cq_type"].notna()]
    grp = wrong.groupby("first_cq_type")["is_correct_final"].agg(["mean", "count"])
    grp = grp.reindex(["EPISTEMIC", "ALEATORIC"]).dropna()
    if len(grp):
        bars = ax.bar(grp.index, grp["mean"],
                      color=[colors[l] for l in grp.index])
        for b, (k, row) in zip(bars, grp.iterrows()):
            ax.text(b.get_x()+b.get_width()/2, row["mean"]+0.02,
                    f"{row['mean']:.0%}\n(n={int(row['count'])})",
                    ha="center", fontsize=10)
        ax.set_ylim(0, 1.1)
    ax.set_title("D. Recovery Rate by First CQ Type\n(only prelim-wrong cases)")
    ax.set_ylabel("Recovery rate")

    # ── E. CQs per case, stratified by type ──────────────────────────────
    ax = axes[1, 1]
    per_case = cls.groupby("id")["label"].value_counts().unstack(fill_value=0)
    for col in ["EPISTEMIC", "ALEATORIC"]:
        if col not in per_case.columns:
            per_case[col] = 0
    per_case["total"] = per_case["EPISTEMIC"] + per_case["ALEATORIC"]
    bins = sorted(per_case["total"].unique())
    epi_means = [per_case[per_case["total"] == b]["EPISTEMIC"].mean() for b in bins]
    ale_means = [per_case[per_case["total"] == b]["ALEATORIC"].mean() for b in bins]
    x = np.arange(len(bins))
    width = 0.4
    ax.bar(x - width/2, epi_means, width, label="EPISTEMIC", color=EPI_COLOR)
    ax.bar(x + width/2, ale_means, width, label="ALEATORIC", color=ALE_COLOR)
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in bins])
    ax.set_xlabel("Total CQs in case")
    ax.set_ylabel("Avg CQs of this type")
    ax.set_title("E. Type Mix Within Multi-CQ Cases")
    ax.legend()

    # ── F. Confidence by first CQ type ────────────────────────────────────
    ax = axes[1, 2]
    sub = df.dropna(subset=["first_cq_type"])
    types = ["EPISTEMIC", "ALEATORIC"]
    types = [t for t in types if (sub["first_cq_type"] == t).any()]
    x = np.arange(len(types))
    width = 0.38
    p_means = [sub[sub["first_cq_type"] == t]["preliminary_confidence"].mean() for t in types]
    f_means = [sub[sub["first_cq_type"] == t]["final_confidence"].mean() for t in types]
    ax.bar(x - width/2, p_means, width, label="Preliminary", color="#bbbbbb")
    ax.bar(x + width/2, f_means, width, label="Final", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(types)
    ax.set_ylim(0, 105); ax.set_ylabel("Mean confidence")
    ax.set_title("F. Confidence Dynamics by First CQ Type")
    ax.legend()

    plt.tight_layout()
    fig.suptitle(f"ShARC Flex CQ Types — {MODEL_ID}", y=1.01, fontsize=14)
    plt.savefig(FIG_PATH, dpi=140, bbox_inches="tight")
    print(f"\nSaved figure → {FIG_PATH}")


def main() -> None:
    df, cls = load()
    df = attach_first_cq_type(df, cls)
    print_summary(df, cls)
    make_figure(df, cls)


if __name__ == "__main__":
    main()
