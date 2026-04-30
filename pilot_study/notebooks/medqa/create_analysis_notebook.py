"""Generate medqa_analysis_gemini-2.5-flash.ipynb — model-specific analysis notebook."""
import json, uuid, pathlib

ROOT = pathlib.Path(__file__).parent
NOTEBOOK_NAME = "medqa_analysis_gemini-2.5-flash.ipynb"

def uid(): return str(uuid.uuid4())[:8]
def md(source):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": source}
def code(source):
    return {"cell_type": "code", "id": uid(), "metadata": {},
            "outputs": [], "execution_count": None, "source": source}

cells = [

# ── Title ─────────────────────────────────────────────────────────────────────
md("""\
# MedQA — CLAMBER Uncertainty Analysis
## Model: `gemini-2.5-flash` | Dataset: MedQA

Covers **Experiment 2** (single-turn, 1 CQ) and **Experiment 3** (multi-turn, 3 CQs)
on the same 100 mixed-difficulty cases (50 easy / 30 medium / 20 hard).

Sections:
1. Setup & Load Data
2. **Single-Turn Analysis** — accuracy, per-difficulty, CQ type, confidence delta, calibration, McNemar
3. **Multi-Turn Analysis** — accuracy progression, per-difficulty, CQ type, calibration, McNemar, simulator, CQ diversity
4. **Cross-Experiment Comparison**
5. Export Summary CSV

*To reuse for a different model: change `MODEL_ID` in the Setup cell.*
"""),

# ── 1. Setup ──────────────────────────────────────────────────────────────────
md("## 1. Setup"),
code("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path("../../").resolve()))

DATASET  = "medqa"
MODEL_ID = "gemini-2.5-flash"
ROOT     = Path("../../").resolve()
OUTPUTS  = ROOT / "outputs" / DATASET / MODEL_ID

import warnings, logging
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from IPython.display import display
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
matplotlib.use("Agg")        # headless — safe for nbconvert
sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (9, 5)

print(f"Model:   {MODEL_ID}")
print(f"Outputs: {OUTPUTS}")
print(f"Exists:  {OUTPUTS.exists()}")
"""),

# ── 2. Load Data ──────────────────────────────────────────────────────────────
md("## 2. Load Data"),
code("""\
# ── Experiment 2: single-turn, mixed difficulty ───────────────────────────────
st_results = pd.read_csv(OUTPUTS / "phase1_singleturn_results.csv")
st_labels  = pd.read_csv(OUTPUTS / "phase1_singleturn_classified.csv")

# ── Experiment 3: multi-turn (3 CQs), same cases ─────────────────────────────
mt_results = pd.read_csv(OUTPUTS / "phase1_multiturn_results.csv")
mt_labels  = pd.read_csv(OUTPUTS / "phase1_multiturn_classified.csv")

# Drop blocked rows
st = st_results[~st_results["was_blocked"]].copy()
mt = mt_results[~mt_results["was_blocked"]].copy()

print(f"Single-turn:  {len(st_results)} rows, {int(st_results['was_blocked'].sum())} blocked → {len(st)} usable")
print(f"Multi-turn:   {len(mt_results)} rows, {int(mt_results['was_blocked'].sum())} blocked → {len(mt)} usable")

assert set(st["id"]) == set(mt["id"]), "Dataset mismatch — different case IDs"
print(f"\\nCase overlap check PASSED — same {len(st)} cases in both experiments")

# ── Helpers ───────────────────────────────────────────────────────────────────
def pct(s): return s.mean() * 100

def mcnemar(before, after, name):
    b = int(((~before) & after).sum())
    c = int((before & (~after)).sum())
    total = b + c
    if total == 0:
        print(f"  {name}: no discordant pairs")
        return
    p = binomtest(b, total, 0.5, alternative="greater").pvalue
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {name}: +{b} gained, -{c} lost  p={p:.4f} {sig}")

def compute_ece(confs, correct, n_bins=10):
    bins = np.linspace(0, 100, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs >= lo) & (confs < hi)
        if not m.any():
            rows.append({"mid": (lo+hi)/2, "acc": np.nan, "conf": np.nan, "n": 0})
            continue
        a = correct[m].mean() * 100
        c_avg = confs[m].mean()
        ece += m.sum() * abs(a - c_avg)
        rows.append({"mid": (lo+hi)/2, "acc": a, "conf": c_avg, "n": int(m.sum())})
    return ece / len(confs), pd.DataFrame(rows)
"""),

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-TURN
# ══════════════════════════════════════════════════════════════════════════════
md("---\n## 3. Single-Turn Analysis (Experiment 2)\n*One CQ round — 100 mixed-difficulty cases*"),

# 3.1 Accuracy
md("### 3.1 Overall Accuracy"),
code("""\
prelim_acc  = pct(st["is_correct_preliminary"])
updated_acc = pct(st["is_correct_updated"])
print(f"Preliminary (before CQ): {prelim_acc:.1f}%  ({int(st['is_correct_preliminary'].sum())}/{len(st)})")
print(f"Updated     (after CQ):  {updated_acc:.1f}%  ({int(st['is_correct_updated'].sum())}/{len(st)})")
print(f"Net gain:                +{updated_acc - prelim_acc:.1f} pp")

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Preliminary", "Updated (post-CQ)"], [prelim_acc, updated_acc],
       color=["steelblue", "seagreen"], width=0.45)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title(f"Single-Turn Accuracy — {MODEL_ID} (n={len(st)})")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
for i, v in enumerate([prelim_acc, updated_acc]):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUTS / "st_fig_accuracy.png", dpi=150)
plt.show()
print("Saved: st_fig_accuracy.png")
"""),

# 3.2 Per-difficulty
md("### 3.2 Per-Difficulty Accuracy"),
code("""\
diffs = ["easy", "medium", "hard"]
st_diff = []
for d in diffs:
    sub = st[st["difficulty"] == d]
    st_diff.append({
        "difficulty": d, "n": len(sub),
        "Prelim (%)":   round(pct(sub["is_correct_preliminary"]), 1),
        "Updated (%)":  round(pct(sub["is_correct_updated"]), 1),
        "Gain (pp)":    round(pct(sub["is_correct_updated"]) - pct(sub["is_correct_preliminary"]), 1),
    })

st_diff_df = pd.DataFrame(st_diff).set_index("difficulty")
print("Single-turn accuracy by difficulty:")
display(st_diff_df)
"""),

# 3.3 CQ type distribution
md("### 3.3 CQ Type Distribution"),
code("""\
q_col_st = "question" if "question" in st_labels.columns else "clarifying_question"
st_valid = st_labels[st_labels["label"].isin({"EPISTEMIC", "ALEATORIC"})].copy()

print(f"Single-turn CQs classified: {len(st_valid)}")
print()
vc = st_valid["label"].value_counts()
print(vc.to_string())
print(f"\\nEPISTEMIC: {vc.get('EPISTEMIC', 0)} ({100*vc.get('EPISTEMIC', 0)/len(st_valid):.1f}%)")
print(f"ALEATORIC: {vc.get('ALEATORIC', 0)} ({100*vc.get('ALEATORIC', 0)/len(st_valid):.1f}%)")
"""),

# 3.4 Confidence delta — THE KEY NEW SECTION
md("### 3.4 Confidence Delta Analysis\n`confidence_delta = updated_confidence − preliminary_confidence`"),
code("""\
# ── Overall delta distribution ────────────────────────────────────────────────
delta = st["confidence_delta"]
print("=== Confidence Delta (updated − preliminary) ===")
print(f"  Mean:   {delta.mean():+.1f}")
print(f"  Median: {delta.median():+.1f}")
print(f"  Std:    {delta.std():.1f}")
print(f"  Range:  [{delta.min():.0f}, {delta.max():.0f}]")
print(f"  Positive (gained confidence): {(delta > 0).sum()}")
print(f"  Zero    (no change):          {(delta == 0).sum()}")
print(f"  Negative (lost confidence):   {(delta < 0).sum()}")
print()

# ── Delta by correctness transition ──────────────────────────────────────────
st["transition"] = (
    st["is_correct_preliminary"].map({True: "Right", False: "Wrong"}) + "→" +
    st["is_correct_updated"].map({True: "Right", False: "Wrong"})
)
trans_order = ["Wrong→Right", "Right→Right", "Wrong→Wrong", "Right→Wrong"]
trans_stats = st.groupby("transition")["confidence_delta"].agg(["mean", "median", "count"]).round(2)
trans_stats = trans_stats.reindex([t for t in trans_order if t in trans_stats.index])
print("=== Confidence Delta by Outcome Transition ===")
display(trans_stats.rename(columns={"mean": "Mean Δ", "median": "Median Δ", "count": "n"}))
print()

# ── Delta by difficulty ───────────────────────────────────────────────────────
diff_delta = st.groupby("difficulty")["confidence_delta"].agg(["mean", "median", "count"]).round(2)
print("=== Confidence Delta by Difficulty ===")
display(diff_delta.rename(columns={"mean": "Mean Δ", "median": "Median Δ", "count": "n"}))
print()

# ── Delta by CQ type ─────────────────────────────────────────────────────────
if len(st_valid) > 0:
    st_with_label = st.merge(
        st_valid[[q_col_st, "label"]].rename(columns={q_col_st: "clarifying_question"}),
        on="clarifying_question", how="left"
    )
    label_delta = st_with_label.groupby("label")["confidence_delta"].agg(["mean", "median", "count"]).round(2)
    print("=== Confidence Delta by CQ Type ===")
    display(label_delta.rename(columns={"mean": "Mean Δ", "median": "Median Δ", "count": "n"}))
    print()
"""),
code("""\
# ── Visualisations ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# (a) Histogram of delta
axes[0].hist(delta, bins=20, color="steelblue", edgecolor="white")
axes[0].axvline(0, color="red", lw=1.5, linestyle="--", label="No change")
axes[0].axvline(delta.mean(), color="orange", lw=1.5, linestyle="-", label=f"Mean={delta.mean():+.1f}")
axes[0].set_xlabel("Confidence delta (pp)")
axes[0].set_ylabel("Cases")
axes[0].set_title("Δ Confidence Distribution")
axes[0].legend(fontsize=8)

# (b) Delta by outcome transition
trans_means = (
    st.groupby("transition")["confidence_delta"].mean()
    .reindex([t for t in trans_order if t in st["transition"].values])
)
colors = {"Wrong→Right": "seagreen", "Right→Right": "steelblue",
          "Wrong→Wrong": "salmon",   "Right→Wrong": "firebrick"}
bar_colors = [colors.get(t, "gray") for t in trans_means.index]
axes[1].bar(trans_means.index, trans_means.values, color=bar_colors)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_ylabel("Mean Δ confidence (pp)")
axes[1].set_title("Mean Δ by Outcome Transition")
axes[1].tick_params(axis="x", labelsize=9)

# (c) Scatter: preliminary vs updated confidence, coloured by updated correctness
correct_color = st["is_correct_updated"].map({True: "seagreen", False: "salmon"})
axes[2].scatter(st["preliminary_confidence"], st["updated_confidence"],
                c=correct_color, alpha=0.65, s=40, edgecolors="none")
axes[2].plot([0, 100], [0, 100], "k--", lw=0.8, label="No change")
axes[2].set_xlabel("Preliminary confidence")
axes[2].set_ylabel("Updated confidence")
axes[2].set_title("Conf: Before vs After CQ")
from matplotlib.patches import Patch
axes[2].legend(handles=[
    Patch(color="seagreen", label="Updated correct"),
    Patch(color="salmon",   label="Updated wrong"),
], fontsize=8)

fig.suptitle(f"Single-Turn Confidence Analysis — {MODEL_ID}", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUTS / "st_fig_confidence_delta.png", dpi=150)
plt.show()
print("Saved: st_fig_confidence_delta.png")
"""),

# 3.5 ECE
md("### 3.5 Calibration (ECE)"),
code("""\
print("Expected Calibration Error — lower is better:")
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (label, conf_col, corr_col) in zip(axes, [
    ("Preliminary", "preliminary_confidence", "is_correct_preliminary"),
    ("Updated",     "updated_confidence",     "is_correct_updated"),
]):
    confs  = st[conf_col].values.astype(float)
    corrs  = st[corr_col].values.astype(bool)
    ece_val, bin_df = compute_ece(confs, corrs)
    print(f"  {label}: ECE = {ece_val:.2f} pp")
    vb = bin_df.dropna()
    ax.bar(vb["mid"], vb["acc"], width=8, alpha=0.7)
    ax.plot([0, 100], [0, 100], "k--", lw=1)
    ax.scatter(vb["conf"], vb["acc"], s=40, color="red", zorder=5)
    ax.set_title(f"{label}\\nECE={ece_val:.1f}pp")
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
axes[0].set_ylabel("Accuracy (%)")
fig.suptitle(f"Single-Turn Reliability Diagrams — {MODEL_ID}", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUTS / "st_fig_calibration.png", dpi=150)
plt.show()
print("Saved: st_fig_calibration.png")
"""),

# 3.6 McNemar
md("### 3.6 Statistical Significance (McNemar's Test)"),
code("""\
print("Single-turn McNemar (one-sided: did accuracy improve?):")
mcnemar(st["is_correct_preliminary"], st["is_correct_updated"], "Prelim → Updated")
"""),

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-TURN
# ══════════════════════════════════════════════════════════════════════════════
md("---\n## 4. Multi-Turn Analysis (Experiment 3)\n*Three CQ rounds — same 100 cases*"),

# 4.1 Accuracy progression
md("### 4.1 Overall Accuracy Progression"),
code("""\
mt_acc = {
    "Prelim":    pct(mt["is_correct_preliminary"]),
    "After CQ1": pct(mt["is_correct_1"]),
    "After CQ2": pct(mt["is_correct_2"]),
    "Final":     pct(mt["is_correct_final"]),
}
print("Multi-turn accuracy at each checkpoint:")
for k, v in mt_acc.items():
    n = int(mt[{"Prelim":"is_correct_preliminary","After CQ1":"is_correct_1",
                "After CQ2":"is_correct_2","Final":"is_correct_final"}[k]].sum())
    print(f"  {k:<12}: {v:.1f}%  ({n}/{len(mt)})")
print(f"  Net gain: +{pct(mt['is_correct_final']) - pct(mt['is_correct_preliminary']):.1f} pp")

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(list(mt_acc.keys()), list(mt_acc.values()), "o-", lw=2, ms=8, color="steelblue", label="Multi-turn")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(40, 100)
ax.set_title(f"Multi-Turn Accuracy Progression — {MODEL_ID} (n={len(mt)})")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig(OUTPUTS / "mt_fig_accuracy_progression.png", dpi=150)
plt.show()
print("Saved: mt_fig_accuracy_progression.png")
"""),

# 4.2 Per-difficulty
md("### 4.2 Per-Difficulty Accuracy"),
code("""\
diffs = ["easy", "medium", "hard"]
mt_ckpts = {"Prelim":"is_correct_preliminary","CQ1":"is_correct_1",
            "CQ2":"is_correct_2","Final":"is_correct_final"}
mt_diff_rows = []
for d in diffs:
    sub = mt[mt["difficulty"] == d]
    row = {"difficulty": d, "n": len(sub)}
    for lbl, col in mt_ckpts.items():
        row[lbl] = round(pct(sub[col]), 1)
    mt_diff_rows.append(row)

mt_diff_df = pd.DataFrame(mt_diff_rows).set_index("difficulty")
print("Multi-turn accuracy by difficulty:")
display(mt_diff_df)

fig, ax = plt.subplots(figsize=FIGSIZE)
x, w = np.arange(len(diffs)), 0.2
for i, (lbl, col) in enumerate(mt_ckpts.items()):
    vals = [pct(mt[mt["difficulty"]==d][col]) for d in diffs]
    ax.bar(x + i*w, vals, w, label=lbl)
ax.set_xticks(x + w*1.5)
ax.set_xticklabels([f"{d}\\n(n={len(mt[mt['difficulty']==d])})" for d in diffs])
ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Multi-Turn Accuracy by Difficulty — {MODEL_ID}")
ax.legend(loc="lower right")
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig(OUTPUTS / "mt_fig_accuracy_by_difficulty.png", dpi=150)
plt.show()
print("Saved: mt_fig_accuracy_by_difficulty.png")
"""),

# 4.3 CQ type
md("### 4.3 CQ Type Distribution"),
code("""\
q_col_mt = "question" if "question" in mt_labels.columns else "clarifying_question"

# Attach turn numbers if needed
if "turn" not in mt_labels.columns:
    long_rows = []
    for _, r in mt.iterrows():
        for t in range(1, 4):
            long_rows.append({"id": r["id"], "turn": t, "question": r[f"cq_{t}"]})
    long_df = pd.DataFrame(long_rows)
    mt_labels_m = mt_labels.merge(long_df, on=["id", "question"], how="left")
else:
    mt_labels_m = mt_labels.copy()

mt_valid = mt_labels_m[mt_labels_m["label"].isin({"EPISTEMIC", "ALEATORIC"})].copy()
print(f"Multi-turn CQs classified: {len(mt_valid)}")
vc_mt = mt_valid["label"].value_counts()
print(vc_mt.to_string())
print(f"\\nEPISTEMIC: {vc_mt.get('EPISTEMIC',0)} ({100*vc_mt.get('EPISTEMIC',0)/len(mt_valid):.1f}%)")
print(f"ALEATORIC: {vc_mt.get('ALEATORIC',0)} ({100*vc_mt.get('ALEATORIC',0)/len(mt_valid):.1f}%)")
print()
print("By turn:")
display(mt_valid.groupby(["turn","label"]).size().unstack(fill_value=0))
"""),

# 4.4 ECE
md("### 4.4 Calibration (ECE)"),
code("""\
mt_conf_ckpts = [
    ("Prelim",  "preliminary_confidence", "is_correct_preliminary"),
    ("CQ1",     "confidence_1",           "is_correct_1"),
    ("CQ2",     "confidence_2",           "is_correct_2"),
    ("Final",   "final_confidence",       "is_correct_final"),
]
print("Expected Calibration Error (ECE) — lower is better:")
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, (lbl, cc, corr) in zip(axes, mt_conf_ckpts):
    confs = mt[cc].values.astype(float)
    corrs = mt[corr].values.astype(bool)
    ece_val, bin_df = compute_ece(confs, corrs)
    print(f"  {lbl}: ECE = {ece_val:.2f} pp")
    vb = bin_df.dropna()
    ax.bar(vb["mid"], vb["acc"], width=8, alpha=0.7)
    ax.plot([0,100],[0,100],"k--",lw=1)
    ax.scatter(vb["conf"], vb["acc"], s=40, color="red", zorder=5)
    ax.set_title(f"{lbl}\\nECE={ece_val:.1f}pp")
    ax.set_xlabel("Confidence")
    ax.set_xlim(0,100); ax.set_ylim(0,100)
axes[0].set_ylabel("Accuracy (%)")
fig.suptitle(f"Multi-Turn Reliability Diagrams — {MODEL_ID}", fontsize=13)
plt.tight_layout()
plt.savefig(OUTPUTS / "mt_fig_calibration.png", dpi=150)
plt.show()
print("Saved: mt_fig_calibration.png")
"""),

# 4.5 McNemar
md("### 4.5 Statistical Significance (McNemar's Test)"),
code("""\
print("Multi-turn McNemar (one-sided: did accuracy improve?):")
mcnemar(mt["is_correct_preliminary"], mt["is_correct_1"],    "Prelim  → CQ1")
mcnemar(mt["is_correct_1"],           mt["is_correct_2"],    "CQ1     → CQ2")
mcnemar(mt["is_correct_2"],           mt["is_correct_final"],"CQ2     → Final")
print()
mcnemar(mt["is_correct_preliminary"], mt["is_correct_final"],"Prelim  → Final (overall)")
"""),

# 4.6 Simulator grounding
md("### 4.6 Simulator Grounding Rate"),
code("""\
print("Fraction of simulator responses containing 'not available':")
for t in [1, 2, 3]:
    responses = mt[f"patient_response_{t}"].fillna("").str.lower()
    n_unav = responses.str.contains("not available").sum()
    print(f"  CQ{t}: {n_unav}/{len(mt)} ({100*n_unav/len(mt):.1f}%)")
"""),

# 4.7 CQ diversity
md("### 4.7 CQ Lexical Diversity (Jaccard)"),
code("""\
def jaccard(a, b):
    ta = set(str(a).lower().split()); tb = set(str(b).lower().split())
    if not ta and not tb: return 1.0
    return len(ta & tb) / len(ta | tb)

print("Mean token Jaccard similarity between CQ pairs (lower = more diverse):")
for ca, cb in [("cq_1","cq_2"), ("cq_2","cq_3"), ("cq_1","cq_3")]:
    sims = mt.apply(lambda r: jaccard(r[ca], r[cb]), axis=1)
    print(f"  {ca} vs {cb}: {sims.mean():.3f}  (median {sims.median():.3f})")
"""),

# ══════════════════════════════════════════════════════════════════════════════
# CROSS-EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════
md("---\n## 5. Cross-Experiment Comparison\n*Single-turn vs Multi-turn on the same 100 cases*"),
code("""\
print("Accuracy comparison (same 100 cases):")
print(f"{'Checkpoint':<30} {'Single-turn':>12} {'Multi-turn':>12}")
print("-" * 55)
pairs = [
    ("Preliminary (T0)",   "is_correct_preliminary", "is_correct_preliminary"),
    ("After CQ1",          "is_correct_updated",     "is_correct_1"),
    ("After CQ2 (MT only)",None,                     "is_correct_2"),
    ("Final (MT only)",    None,                     "is_correct_final"),
]
for label, st_col, mt_col in pairs:
    st_val = f"{pct(st[st_col]):.1f}%" if st_col else "—"
    mt_val = f"{pct(mt[mt_col]):.1f}%" if mt_col else "—"
    print(f"  {label:<28} {st_val:>12} {mt_val:>12}")

print()
print("Confidence delta (single-turn):")
print(f"  Mean: {st['confidence_delta'].mean():+.1f} pp  |  "
      f"Median: {st['confidence_delta'].median():+.1f} pp")
print()
print("Confidence gain proxy (multi-turn, final − prelim):")
mt_conf_gain = mt["final_confidence"] - mt["preliminary_confidence"]
print(f"  Mean: {mt_conf_gain.mean():+.1f} pp  |  Median: {mt_conf_gain.median():+.1f} pp")

# Side-by-side bar chart
fig, ax = plt.subplots(figsize=FIGSIZE)
bars_st = [pct(st["is_correct_preliminary"]), pct(st["is_correct_updated"])]
bars_mt = [pct(mt["is_correct_preliminary"]), pct(mt["is_correct_1"]), pct(mt["is_correct_final"])]
ax.bar([0-0.18, 1-0.18], bars_st, 0.35, label="Single-turn", color="steelblue")
ax.bar([0+0.18, 1+0.18, 2+0.18], bars_mt, 0.35, label="Multi-turn", color="seagreen")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["Preliminary", "After CQ1", "MT Final"])
ax.set_ylabel("Accuracy (%)")
ax.set_title(f"Single-turn vs Multi-turn — {MODEL_ID}")
ax.set_ylim(0, 100)
ax.legend()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig(OUTPUTS / "fig_comparison.png", dpi=150)
plt.show()
print("Saved: fig_comparison.png")
"""),

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════
md("---\n## 6. Export Summary CSV"),
code("""\
# Single-turn summary
st_export = st[["id","difficulty","correct_option",
                "is_correct_preliminary","is_correct_updated",
                "preliminary_confidence","updated_confidence","confidence_delta"]].copy()

# Bridge id → cq_type via clarifying_question column in st (results df)
if len(st_valid) > 0 and "clarifying_question" in st.columns:
    cq_col_in_labels = q_col_st  # "question" or "clarifying_question" in st_labels
    st_id_cqtype = st[["id","clarifying_question"]].merge(
        st_valid[[cq_col_in_labels,"label"]].rename(
            columns={cq_col_in_labels:"clarifying_question","label":"cq_type"}),
        on="clarifying_question", how="left"
    )[["id","cq_type"]]
    st_export = st_export.merge(st_id_cqtype, on="id", how="left")

st_out = OUTPUTS / "st_analysis_summary.csv"
st_export.to_csv(st_out, index=False)
print(f"Single-turn summary: {st_out.name}  ({len(st_export)} rows)")

# Multi-turn summary
mt_type_wide = {}
for t in [1, 2, 3]:
    tdf = mt_valid[mt_valid["turn"]==t][["id","label"]].rename(columns={"label":f"cq{t}_type"})
    mt_type_wide[t] = tdf

mt_export = mt[["id","difficulty","correct_option",
                "is_correct_preliminary","is_correct_1","is_correct_2","is_correct_final",
                "preliminary_confidence","confidence_1","confidence_2","final_confidence"]].copy()
for t, df in mt_type_wide.items():
    mt_export = mt_export.merge(df, on="id", how="left")
mt_out = OUTPUTS / "mt_analysis_summary.csv"
mt_export.to_csv(mt_out, index=False)
print(f"Multi-turn summary:  {mt_out.name}  ({len(mt_export)} rows)")
display(mt_export.head(3))
"""),

]  # end cells

NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "aims_project", "language": "python", "name": "aims_project"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = ROOT / NOTEBOOK_NAME
with open(out, "w", encoding="utf-8") as f:
    json.dump(NB, f, indent=1, ensure_ascii=False)

print(f"Written: {out}")
print(f"Cells:   {len(cells)}")
