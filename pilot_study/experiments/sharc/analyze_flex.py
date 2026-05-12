"""ShARC Flex experiment analysis — accuracy-based (Yes/No) + confidence dynamics.

Produces an 8-panel overview figure and prints summary stats:
  A. n_cqs_asked distribution
  B. Accuracy: preliminary vs final
  C. Accuracy by n_cqs_asked (final)
  D. Confidence by n_cqs_asked (preliminary vs final)
  E. Confidence calibration (final): correctness rate by confidence bin
  F. Confidence gain (final − preliminary) by accuracy outcome
  G. Reference n_cqs (gold history) vs model n_cqs_asked
  H. Decision matrix: asking helped / hurt / neutral

Output:
  outputs/sharc/<model>/flex_fig_overview.png
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
from scipy import stats

ROOT = Path(__file__).parent.parent.parent.resolve()
MODEL_ID = "gemini-2.5-flash"
CSV_PATH = ROOT / "outputs" / "sharc" / MODEL_ID / "phase1_flex_results.csv"
FIG_PATH = ROOT / "outputs" / "sharc" / MODEL_ID / "flex_fig_overview.png"

COLORS = {0: "#888888", 1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH.name}")
    df = df[df["n_cqs_asked"] >= 0].copy()  # drop parse errors
    print(f"Valid rows after filter: {len(df)}")
    df["is_correct_preliminary"] = df["is_correct_preliminary"].astype(bool)
    df["is_correct_final"]       = df["is_correct_final"].astype(bool)
    df["preliminary_confidence"] = df["preliminary_confidence"].astype(float)
    df["final_confidence"]       = df["final_confidence"].astype(float)
    df["confidence_gain"]        = df["final_confidence"] - df["preliminary_confidence"]
    df["n_cqs_asked"]            = df["n_cqs_asked"].astype(int)
    df["n_history_cqs"]          = pd.to_numeric(df["n_history_cqs"], errors="coerce").fillna(0).astype(int)
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"N records                 : {len(df)}")
    print(f"Gold Yes / No             : {(df['gold_answer']=='Yes').sum()} / {(df['gold_answer']=='No').sum()}")
    print(f"Preliminary accuracy      : {df['is_correct_preliminary'].mean():.1%}")
    print(f"Final accuracy            : {df['is_correct_final'].mean():.1%}")
    print(f"Net improvement           : +{(df['is_correct_final'].mean() - df['is_correct_preliminary'].mean())*100:.1f}pp")
    print()
    print("n_cqs_asked distribution  :")
    for n, c in df["n_cqs_asked"].value_counts().sort_index().items():
        print(f"   {n} CQs : {c:3d} cases  ({c/len(df):.0%})")
    print()
    print("Final accuracy by n_cqs_asked:")
    for n, sub in df.groupby("n_cqs_asked"):
        print(f"   {n} CQs : {sub['is_correct_final'].mean():.1%}  (n={len(sub)})")
    print()
    print("Mean confidence by n_cqs_asked:")
    for n, sub in df.groupby("n_cqs_asked"):
        print(f"   {n} CQs : prelim={sub['preliminary_confidence'].mean():.0f}  final={sub['final_confidence'].mean():.0f}")

    # Stat tests
    n_cq_groups = [df[df["n_cqs_asked"] == k]["is_correct_final"].astype(int).values
                   for k in sorted(df["n_cqs_asked"].unique())]
    if len(n_cq_groups) >= 2:
        h_stat, h_p = stats.kruskal(*n_cq_groups)
        print(f"\nKruskal-Wallis (final accuracy by n_cqs): H={h_stat:.2f} p={h_p:.4f}")

    # Did asking help in cases where prelim was wrong?
    wrong_prelim = df[~df["is_correct_preliminary"]]
    asked = wrong_prelim[wrong_prelim["n_cqs_asked"] > 0]
    didnt = wrong_prelim[wrong_prelim["n_cqs_asked"] == 0]
    print(f"\nWhen prelim was wrong (n={len(wrong_prelim)}):")
    print(f"   Asked CQs: {len(asked)}   recovered correctly: {asked['is_correct_final'].sum()} ({asked['is_correct_final'].mean():.0%})" if len(asked) else "   Asked CQs: 0")
    print(f"   No CQs   : {len(didnt)}   recovered correctly: {didnt['is_correct_final'].sum()} ({didnt['is_correct_final'].mean():.0%})" if len(didnt) else "   No CQs: 0")


def make_figure(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # ── A. n_cqs_asked distribution ─────────────────────────────────────────
    ax = axes[0, 0]
    counts = df["n_cqs_asked"].value_counts().sort_index()
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=[COLORS[k] for k in counts.index])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, str(v), ha="center", fontsize=9)
    ax.set_title("A. CQ Count Distribution")
    ax.set_xlabel("n_cqs_asked"); ax.set_ylabel("cases")

    # ── B. Accuracy: preliminary vs final ──────────────────────────────────
    ax = axes[0, 1]
    p = df["is_correct_preliminary"].mean()
    f = df["is_correct_final"].mean()
    ax.bar(["Preliminary", "Final"], [p, f],
           color=["#bbbbbb", "#2ca02c"])
    ax.set_ylim(0, 1)
    ax.set_title(f"B. Accuracy: {p:.0%} → {f:.0%}")
    ax.set_ylabel("Accuracy")
    for i, v in enumerate([p, f]):
        ax.text(i, v+0.02, f"{v:.0%}", ha="center", fontsize=10, fontweight="bold")

    # ── C. Final accuracy by n_cqs_asked ──────────────────────────────────
    ax = axes[0, 2]
    by_n = df.groupby("n_cqs_asked")["is_correct_final"].agg(["mean", "count"])
    bars = ax.bar(by_n.index.astype(str), by_n["mean"],
                  color=[COLORS[k] for k in by_n.index])
    for b, (k, row) in zip(bars, by_n.iterrows()):
        ax.text(b.get_x()+b.get_width()/2, row["mean"]+0.02,
                f"{row['mean']:.0%}\n(n={row['count']})", ha="center", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("C. Final Accuracy by n_cqs_asked")
    ax.set_xlabel("n_cqs_asked"); ax.set_ylabel("Accuracy")

    # ── D. Confidence by n_cqs_asked ───────────────────────────────────────
    ax = axes[0, 3]
    ns = sorted(df["n_cqs_asked"].unique())
    x = np.arange(len(ns))
    width = 0.38
    prelim_means = [df[df["n_cqs_asked"] == n]["preliminary_confidence"].mean() for n in ns]
    final_means  = [df[df["n_cqs_asked"] == n]["final_confidence"].mean() for n in ns]
    ax.bar(x - width/2, prelim_means, width, label="Preliminary", color="#bbbbbb")
    ax.bar(x + width/2, final_means,  width, label="Final",       color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in ns])
    ax.set_ylim(0, 105)
    ax.set_xlabel("n_cqs_asked"); ax.set_ylabel("Mean confidence")
    ax.set_title("D. Confidence by n_cqs_asked")
    ax.legend()

    # ── E. Confidence calibration (final) ──────────────────────────────────
    ax = axes[1, 0]
    bins = [0, 50, 70, 85, 95, 100.001]
    labels = ["0-50", "51-70", "71-85", "86-95", "96-100"]
    df["_conf_bin"] = pd.cut(df["final_confidence"], bins=bins, labels=labels, include_lowest=True)
    cal = df.groupby("_conf_bin", observed=True)["is_correct_final"].agg(["mean", "count"])
    bars = ax.bar(cal.index.astype(str), cal["mean"], color="#1f77b4")
    for b, (k, row) in zip(bars, cal.iterrows()):
        ax.text(b.get_x()+b.get_width()/2, row["mean"]+0.02,
                f"{row['mean']:.0%}\n(n={row['count']})", ha="center", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title("E. Calibration: Accuracy by Final Confidence")
    ax.set_xlabel("Final confidence bin"); ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=15)

    # ── F. Confidence gain by outcome ──────────────────────────────────────
    ax = axes[1, 1]
    df["outcome"] = np.where(
        df["is_correct_preliminary"] & df["is_correct_final"], "Both correct",
        np.where(~df["is_correct_preliminary"] & df["is_correct_final"], "Recovered",
        np.where(df["is_correct_preliminary"] & ~df["is_correct_final"], "Regressed",
                 "Both wrong")))
    order = ["Both correct", "Recovered", "Regressed", "Both wrong"]
    cols  = ["#2ca02c", "#1f77b4", "#d62728", "#888888"]
    data = [df[df["outcome"] == o]["confidence_gain"].dropna().values for o in order]
    counts = [len(d) for d in data]
    parts = ax.boxplot(data, labels=[f"{o}\n(n={c})" for o, c in zip(order, counts)],
                       patch_artist=True)
    for p, c in zip(parts["boxes"], cols):
        p.set_facecolor(c); p.set_alpha(0.6)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Confidence gain (final − prelim)")
    ax.set_title("F. Confidence Gain by Outcome")
    ax.tick_params(axis="x", labelsize=8)

    # ── G. Reference n_history_cqs vs model n_cqs_asked ────────────────────
    ax = axes[1, 2]
    cross = pd.crosstab(df["n_history_cqs"], df["n_cqs_asked"])
    im = ax.imshow(cross.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(cross.shape[1])); ax.set_xticklabels(cross.columns)
    ax.set_yticks(range(cross.shape[0])); ax.set_yticklabels(cross.index)
    ax.set_xlabel("Model n_cqs_asked"); ax.set_ylabel("Reference (gold) n_history_cqs")
    ax.set_title("G. Reference vs Model CQ Counts")
    for i in range(cross.shape[0]):
        for j in range(cross.shape[1]):
            v = cross.values[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > cross.values.max()/2 else "black",
                        fontsize=9)

    # ── H. Decision matrix ─────────────────────────────────────────────────
    ax = axes[1, 3]
    cells = {
        "Asked,\nCorrect": ((df["n_cqs_asked"] > 0) & df["is_correct_final"]).sum(),
        "Asked,\nWrong":   ((df["n_cqs_asked"] > 0) & ~df["is_correct_final"]).sum(),
        "Didn't ask,\nCorrect": ((df["n_cqs_asked"] == 0) & df["is_correct_final"]).sum(),
        "Didn't ask,\nWrong":   ((df["n_cqs_asked"] == 0) & ~df["is_correct_final"]).sum(),
    }
    bars = ax.bar(list(cells.keys()), list(cells.values()),
                  color=["#2ca02c", "#d62728", "#888888", "#444444"])
    for b, v in zip(bars, cells.values()):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, str(v), ha="center", fontsize=10)
    ax.set_title("H. Decision × Outcome")
    ax.set_ylabel("cases")
    ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    fig.suptitle(f"ShARC Flex — {MODEL_ID} (n={len(df)})", y=1.01, fontsize=14)
    plt.savefig(FIG_PATH, dpi=140, bbox_inches="tight")
    print(f"\nSaved figure → {FIG_PATH}")


def main() -> None:
    df = load_clean()
    print_summary(df)
    make_figure(df)


if __name__ == "__main__":
    main()
