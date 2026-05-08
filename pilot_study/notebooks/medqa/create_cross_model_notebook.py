"""Generate medqa_cross_model_analysis.ipynb — compare gemini-2.5-flash vs gemma-3-12b-it."""
import json, uuid, pathlib

ROOT = pathlib.Path(__file__).parent
NOTEBOOK_NAME = "medqa_cross_model_analysis.ipynb"

def uid(): return str(uuid.uuid4())[:8]
def md(source):
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": source}
def code(source):
    return {"cell_type": "code", "id": uid(), "metadata": {},
            "outputs": [], "execution_count": None, "source": source}

cells = [

md("""\
# MedQA — Cross-Model Analysis
## gemini-2.5-flash vs gemma-3-12b-it

Compares the two models on the **same 100 mixed-difficulty cases** (50 easy / 30 medium / 20 hard).

- **Clinician model** differs: Gemini-2.5-flash vs Gemma-3-12b-it
- **Simulator + Judge** are identical: Gemini-2.5-flash (controlled)

Sections:
1. Setup & Load Data
2. Single-Turn Accuracy Comparison
3. Multi-Turn Accuracy Progression Comparison
4. Per-Difficulty Breakdown
5. Confidence Delta & Overconfidence Analysis
6. CQ Type Distribution
7. Calibration (ECE)
8. Statistical Tests (McNemar)
9. Summary Table & Export
"""),

# ── 1. Setup ──────────────────────────────────────────────────────────────────
md("## 1. Setup & Load Data"),
code("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path("../../").resolve()))

DATASET   = "medqa"
ROOT      = Path("../../").resolve()
OUTPUTS   = ROOT / "outputs" / DATASET
CROSS_OUT = OUTPUTS / "cross_model"
CROSS_OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "gemini-2.5-flash": OUTPUTS / "gemini-2.5-flash",
    "gemma-3-12b-it":   OUTPUTS / "gemma-3-12b-it",
}
MODEL_LABELS = {
    "gemini-2.5-flash": "Gemini-2.5-flash",
    "gemma-3-12b-it":   "Gemma-3-12b-it",
}
COLORS = {
    "gemini-2.5-flash": "steelblue",
    "gemma-3-12b-it":   "darkorange",
}

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
matplotlib.use("Agg")
sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (10, 5)

print(f"Output dir: {CROSS_OUT}")
"""),

code("""\
# Load single-turn and multi-turn results for each model
st, mt, st_clf, mt_clf = {}, {}, {}, {}

for mid, mdir in MODELS.items():
    st_raw  = pd.read_csv(mdir / "phase1_singleturn_results.csv")
    mt_raw  = pd.read_csv(mdir / "phase1_multiturn_results.csv")
    st_c    = pd.read_csv(mdir / "phase1_singleturn_classified.csv")
    mt_c    = pd.read_csv(mdir / "phase1_multiturn_classified.csv")

    st[mid]     = st_raw[~st_raw["was_blocked"]].copy()
    mt[mid]     = mt_raw[~mt_raw["was_blocked"]].copy()
    st_clf[mid] = st_c[st_c["label"].isin({"EPISTEMIC","ALEATORIC"})].copy()
    mt_clf[mid] = mt_c[mt_c["label"].isin({"EPISTEMIC","ALEATORIC"})].copy()

    print(f"{MODEL_LABELS[mid]}")
    print(f"  ST: {len(st[mid])} rows | MT: {len(mt[mid])} rows")
    print(f"  ST CQs classified: {len(st_clf[mid])} | MT CQs classified: {len(mt_clf[mid])}")

# Verify same cases
ids = [set(st[m]["id"]) for m in MODELS]
assert ids[0] == ids[1], f"Case mismatch between models!"
print(f"\\nCase overlap check PASSED — same {len(ids[0])} cases across models")

def pct(s): return s.mean() * 100

def mcnemar_test(before, after, name):
    b = int(((~before) & after).sum())
    c = int((before & (~after)).sum())
    total = b + c
    if total == 0:
        return f"  {name}: no discordant pairs"
    p = binomtest(b, total, 0.5, alternative="greater").pvalue
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    return f"  {name}: +{b} gained, -{c} lost  p={p:.4f} {sig}"

def compute_ece(confs, correct, n_bins=10):
    bins = np.linspace(0, 100, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs >= lo) & (confs < hi)
        if not m.any(): continue
        a = correct[m].mean() * 100
        c_avg = confs[m].mean()
        ece += m.sum() * abs(a - c_avg)
    return ece / len(confs)
"""),

# ── 2. Single-Turn Accuracy ───────────────────────────────────────────────────
md("---\n## 2. Single-Turn Accuracy Comparison"),
code("""\
print(f"{'Model':<22} {'Prelim':>10} {'Post-CQ':>10} {'Gain':>8}")
print("-" * 55)
for mid in MODELS:
    s = st[mid]
    prelim = pct(s["is_correct_preliminary"])
    post   = pct(s["is_correct_updated"])
    print(f"  {MODEL_LABELS[mid]:<20} {prelim:>9.1f}% {post:>9.1f}% {post-prelim:>+7.1f}pp")

fig, ax = plt.subplots(figsize=(6, 4))
x = np.array([0, 1])
w = 0.35
for i, mid in enumerate(MODELS):
    s = st[mid]
    vals = [pct(s["is_correct_preliminary"]), pct(s["is_correct_updated"])]
    bars = ax.bar(x + (i-0.5)*w, vals, w, label=MODEL_LABELS[mid], color=COLORS[mid], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(["Preliminary\\n(before CQ)", "Post-CQ\\n(after 1 CQ)"])
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("Single-Turn Accuracy — Model Comparison (n=100)")
ax.legend()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig(CROSS_OUT / "st_accuracy_comparison.png", dpi=150)
plt.show()
print("Saved: st_accuracy_comparison.png")
"""),

# ── 3. Multi-Turn Accuracy Progression ───────────────────────────────────────
md("---\n## 3. Multi-Turn Accuracy Progression Comparison"),
code("""\
ckpts = ["Prelim", "After CQ1", "After CQ2", "Final"]
ckpt_cols = ["is_correct_preliminary", "is_correct_1", "is_correct_2", "is_correct_final"]

print(f"{'Checkpoint':<14}", end="")
for mid in MODELS:
    print(f"  {MODEL_LABELS[mid]:>20}", end="")
print()
print("-" * 60)
for lbl, col in zip(ckpts, ckpt_cols):
    print(f"  {lbl:<12}", end="")
    for mid in MODELS:
        v = pct(mt[mid][col])
        n = int(mt[mid][col].sum())
        print(f"  {v:>18.1f}%", end="")
    print()

fig, ax = plt.subplots(figsize=FIGSIZE)
for mid in MODELS:
    vals = [pct(mt[mid][c]) for c in ckpt_cols]
    ax.plot(ckpts, vals, "o-", lw=2.5, ms=8, label=MODEL_LABELS[mid], color=COLORS[mid])
    for x, v in zip(ckpts, vals):
        ax.annotate(f"{v:.1f}%", (x, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=COLORS[mid])

ax.set_ylabel("Accuracy (%)")
ax.set_ylim(40, 90)
ax.set_title("Multi-Turn Accuracy Progression — Model Comparison (n=100)")
ax.legend()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig(CROSS_OUT / "mt_accuracy_progression.png", dpi=150)
plt.show()
print("Saved: mt_accuracy_progression.png")
"""),

# ── 4. Per-Difficulty ─────────────────────────────────────────────────────────
md("---\n## 4. Per-Difficulty Breakdown"),
code("""\
diffs = ["easy", "medium", "hard"]

print("=== Single-Turn — Final Accuracy by Difficulty ===")
st_diff_rows = []
for d in diffs:
    row = {"difficulty": d}
    for mid in MODELS:
        sub = st[mid][st[mid]["difficulty"] == d]
        row[f"{MODEL_LABELS[mid]} Prelim"] = round(pct(sub["is_correct_preliminary"]), 1)
        row[f"{MODEL_LABELS[mid]} Post-CQ"] = round(pct(sub["is_correct_updated"]), 1)
        row["n"] = len(sub)
    st_diff_rows.append(row)
st_diff_df = pd.DataFrame(st_diff_rows).set_index("difficulty")
display(st_diff_df)

print("\\n=== Multi-Turn — Final Accuracy by Difficulty ===")
mt_diff_rows = []
for d in diffs:
    row = {"difficulty": d}
    for mid in MODELS:
        sub = mt[mid][mt[mid]["difficulty"] == d]
        row[f"{MODEL_LABELS[mid]} Prelim"] = round(pct(sub["is_correct_preliminary"]), 1)
        row[f"{MODEL_LABELS[mid]} Final"]  = round(pct(sub["is_correct_final"]), 1)
        row[f"{MODEL_LABELS[mid]} Gain"]   = round(pct(sub["is_correct_final"]) - pct(sub["is_correct_preliminary"]), 1)
        row["n"] = len(sub)
    mt_diff_rows.append(row)
mt_diff_df = pd.DataFrame(mt_diff_rows).set_index("difficulty")
display(mt_diff_df)
"""),

code("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
n_diffs = len(diffs)
x = np.arange(n_diffs)
w = 0.2

for ax, (exp_label, acc_col, prelim_col) in zip(axes, [
    ("Single-Turn", "is_correct_updated",  "is_correct_preliminary"),
    ("Multi-Turn",  "is_correct_final",    "is_correct_preliminary"),
]):
    data = {mid: {} for mid in MODELS}
    for mid in MODELS:
        for d in diffs:
            sub = (st if exp_label == "Single-Turn" else mt)[mid]
            sub = sub[sub["difficulty"] == d]
            data[mid]["prelim_" + d] = pct(sub["is_correct_preliminary"])
            data[mid]["final_" + d]  = pct(sub[acc_col])

    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
    for i, (mid, linetype, alpha) in enumerate([
        (list(MODELS.keys())[0], "prelim", 0.45),
        (list(MODELS.keys())[0], "final",  0.9),
        (list(MODELS.keys())[1], "prelim", 0.45),
        (list(MODELS.keys())[1], "final",  0.9),
    ]):
        vals = [data[mid][f"{linetype}_{d}"] for d in diffs]
        label = f"{MODEL_LABELS[mid]} {'Prelim' if linetype=='prelim' else ('Post-CQ' if exp_label=='Single-Turn' else 'Final')}"
        ax.bar(x + offsets[i], vals, w, color=COLORS[mid], alpha=alpha, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\\n(n={len(st[list(MODELS.keys())[0]][st[list(MODELS.keys())[0]]['difficulty']==d])})" for d in diffs])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{exp_label} — Accuracy by Difficulty")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=8)

plt.suptitle("Per-Difficulty Accuracy — Model Comparison", fontsize=13)
plt.tight_layout()
plt.savefig(CROSS_OUT / "per_difficulty_comparison.png", dpi=150)
plt.show()
print("Saved: per_difficulty_comparison.png")
"""),

# ── 5. Confidence Delta ───────────────────────────────────────────────────────
md("---\n## 5. Confidence Delta & Overconfidence Analysis"),
code("""\
print("=== Single-Turn Confidence Delta ===")
for mid in MODELS:
    d = st[mid]["confidence_delta"]
    print(f"\\n{MODEL_LABELS[mid]}:")
    print(f"  Mean delta:   {d.mean():+.1f} pp")
    print(f"  Median delta: {d.median():+.1f} pp")
    print(f"  Std:          {d.std():.1f}")
    print(f"  Increased: {(d > 0).sum()} | Unchanged: {(d == 0).sum()} | Decreased: {(d < 0).sum()}")

print("\\n=== Confidence vs Accuracy Gap (Overconfidence) ===")
for mid in MODELS:
    s = st[mid]
    for label, conf_col, corr_col in [
        ("Prelim",  "preliminary_confidence", "is_correct_preliminary"),
        ("Post-CQ", "updated_confidence",     "is_correct_updated"),
    ]:
        mean_conf = s[conf_col].mean()
        acc       = pct(s[corr_col])
        gap       = mean_conf - acc
        print(f"  {MODEL_LABELS[mid]:<22} {label:<8}: conf={mean_conf:.1f}%  acc={acc:.1f}%  gap={gap:+.1f}pp")

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# (a) Confidence delta distribution side by side
for mid in MODELS:
    axes[0].hist(st[mid]["confidence_delta"], bins=20, alpha=0.6,
                 label=MODEL_LABELS[mid], color=COLORS[mid], edgecolor="none")
axes[0].axvline(0, color="black", lw=1.5, linestyle="--")
axes[0].set_xlabel("Confidence delta (pp)")
axes[0].set_ylabel("Cases")
axes[0].set_title("ST Confidence Delta Distribution")
axes[0].legend()

# (b) Mean confidence vs accuracy at each checkpoint
checkpoints = ["Prelim", "Post-CQ"]
conf_cols_st = ["preliminary_confidence", "updated_confidence"]
corr_cols_st = ["is_correct_preliminary", "is_correct_updated"]

x2 = np.arange(len(checkpoints))
w2 = 0.25
for i, mid in enumerate(MODELS):
    confs = [st[mid][c].mean() for c in conf_cols_st]
    accs  = [pct(st[mid][c]) for c in corr_cols_st]
    axes[1].bar(x2 + (i*2)*w2,   confs, w2, color=COLORS[mid], alpha=0.9, label=f"{MODEL_LABELS[mid]} Conf")
    axes[1].bar(x2 + (i*2+1)*w2, accs,  w2, color=COLORS[mid], alpha=0.4, label=f"{MODEL_LABELS[mid]} Acc")

axes[1].set_xticks(x2 + 1.5*w2)
axes[1].set_xticklabels(checkpoints)
axes[1].set_ylabel("Value (%)")
axes[1].set_ylim(0, 100)
axes[1].set_title("Mean Confidence vs Accuracy (ST)")
axes[1].legend(fontsize=7, ncol=2)
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())

plt.suptitle("Confidence Analysis — Model Comparison", fontsize=13)
plt.tight_layout()
plt.savefig(CROSS_OUT / "confidence_comparison.png", dpi=150)
plt.show()
print("Saved: confidence_comparison.png")
"""),

# ── 6. CQ Type Distribution ───────────────────────────────────────────────────
md("---\n## 6. CQ Type Distribution (EPISTEMIC vs ALEATORIC)"),
code("""\
print("=== Single-Turn CQ Type Distribution ===")
for mid in MODELS:
    vc = st_clf[mid]["label"].value_counts()
    total = len(st_clf[mid])
    print(f"\\n{MODEL_LABELS[mid]}  (n={total}):")
    for lbl in ["EPISTEMIC", "ALEATORIC"]:
        n = vc.get(lbl, 0)
        print(f"  {lbl}: {n} ({100*n/total:.1f}%)")

print("\\n=== Multi-Turn CQ Type Distribution (300 CQs per model) ===")
for mid in MODELS:
    vc = mt_clf[mid]["label"].value_counts()
    total = len(mt_clf[mid])
    print(f"\\n{MODEL_LABELS[mid]}  (n={total}):")
    for lbl in ["EPISTEMIC", "ALEATORIC"]:
        n = vc.get(lbl, 0)
        print(f"  {lbl}: {n} ({100*n/total:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
label_order = ["EPISTEMIC", "ALEATORIC"]
for ax, (exp_label, clf_dict) in zip(axes, [
    ("Single-Turn (100 CQs)", st_clf),
    ("Multi-Turn (300 CQs)",  mt_clf),
]):
    x3 = np.arange(len(label_order))
    w3 = 0.35
    for i, mid in enumerate(MODELS):
        vc = clf_dict[mid]["label"].value_counts()
        total = len(clf_dict[mid])
        vals = [100 * vc.get(l, 0) / total for l in label_order]
        ax.bar(x3 + (i-0.5)*w3, vals, w3, label=MODEL_LABELS[mid], color=COLORS[mid], alpha=0.85)
    ax.set_xticks(x3)
    ax.set_xticklabels(label_order)
    ax.set_ylabel("Proportion (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"CQ Types — {exp_label}")
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

plt.suptitle("CQ Type Distribution — Model Comparison", fontsize=13)
plt.tight_layout()
plt.savefig(CROSS_OUT / "cq_type_comparison.png", dpi=150)
plt.show()
print("Saved: cq_type_comparison.png")
"""),

# ── 7. Calibration ────────────────────────────────────────────────────────────
md("---\n## 7. Calibration (ECE) Comparison"),
code("""\
print("Expected Calibration Error (ECE, lower = better):")
print()
print("Single-Turn:")
for mid in MODELS:
    s = st[mid]
    for lbl, cc, corr in [("Prelim", "preliminary_confidence", "is_correct_preliminary"),
                           ("Post-CQ","updated_confidence",    "is_correct_updated")]:
        ece = compute_ece(s[cc].values.astype(float), s[corr].values.astype(bool))
        print(f"  {MODEL_LABELS[mid]:<22} {lbl:<8}: ECE = {ece:.2f} pp")

print()
print("Multi-Turn:")
mt_ckpts = [("Prelim","preliminary_confidence","is_correct_preliminary"),
            ("CQ1","confidence_1","is_correct_1"),
            ("CQ2","confidence_2","is_correct_2"),
            ("Final","final_confidence","is_correct_final")]
for mid in MODELS:
    m = mt[mid]
    for lbl, cc, corr in mt_ckpts:
        ece = compute_ece(m[cc].values.astype(float), m[corr].values.astype(bool))
        print(f"  {MODEL_LABELS[mid]:<22} {lbl:<8}: ECE = {ece:.2f} pp")
"""),

# ── 8. McNemar ────────────────────────────────────────────────────────────────
md("---\n## 8. Statistical Tests (McNemar)"),
code("""\
print("=== Single-Turn McNemar (Prelim → Post-CQ, one-sided p: did it improve?) ===")
for mid in MODELS:
    s = st[mid]
    print(f"\\n{MODEL_LABELS[mid]}:")
    print(mcnemar_test(s["is_correct_preliminary"], s["is_correct_updated"], "Prelim → Post-CQ"))

print()
print("=== Multi-Turn McNemar ===")
for mid in MODELS:
    m = mt[mid]
    print(f"\\n{MODEL_LABELS[mid]}:")
    print(mcnemar_test(m["is_correct_preliminary"], m["is_correct_1"],    "Prelim  → CQ1"))
    print(mcnemar_test(m["is_correct_1"],           m["is_correct_2"],    "CQ1     → CQ2"))
    print(mcnemar_test(m["is_correct_2"],           m["is_correct_final"],"CQ2     → Final"))
    print(mcnemar_test(m["is_correct_preliminary"], m["is_correct_final"],"Prelim  → Final (overall)"))

print()
print("=== Cross-Model McNemar (does Gemini outperform Gemma?) ===")
# Compare final outcomes on the same cases (matched pairs)
# Align by case id
gem_st  = st["gemini-2.5-flash"].set_index("id").sort_index()
gmm_st  = st["gemma-3-12b-it"].set_index("id").sort_index()
common  = gem_st.index.intersection(gmm_st.index)
gem_st  = gem_st.loc[common]
gmm_st  = gmm_st.loc[common]

print("Single-turn post-CQ (Gemini better than Gemma?):")
b = int(( gem_st["is_correct_updated"] & ~gmm_st["is_correct_updated"]).sum())
c = int((~gem_st["is_correct_updated"] &  gmm_st["is_correct_updated"]).sum())
total = b + c
if total > 0:
    p = binomtest(b, total, 0.5, alternative="greater").pvalue
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  Gemini wins: {b}, Gemma wins: {c}  p={p:.4f} {sig}")

gem_mt  = mt["gemini-2.5-flash"].set_index("id").sort_index()
gmm_mt  = mt["gemma-3-12b-it"].set_index("id").sort_index()
common2 = gem_mt.index.intersection(gmm_mt.index)
gem_mt  = gem_mt.loc[common2]
gmm_mt  = gmm_mt.loc[common2]

print("Multi-turn final (Gemini better than Gemma?):")
b2 = int(( gem_mt["is_correct_final"] & ~gmm_mt["is_correct_final"]).sum())
c2 = int((~gem_mt["is_correct_final"] &  gmm_mt["is_correct_final"]).sum())
total2 = b2 + c2
if total2 > 0:
    p2 = binomtest(b2, total2, 0.5, alternative="greater").pvalue
    sig2 = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else "n.s."
    print(f"  Gemini wins: {b2}, Gemma wins: {c2}  p={p2:.4f} {sig2}")
"""),

# ── 9. Summary Table & Export ─────────────────────────────────────────────────
md("---\n## 9. Summary Table & Export"),
code("""\
rows = []
for mid in MODELS:
    s, m = st[mid], mt[mid]
    rows.append({
        "model":                 MODEL_LABELS[mid],
        "st_prelim_acc":         round(pct(s["is_correct_preliminary"]), 1),
        "st_post_cq_acc":        round(pct(s["is_correct_updated"]), 1),
        "st_acc_gain_pp":        round(pct(s["is_correct_updated"]) - pct(s["is_correct_preliminary"]), 1),
        "st_mean_conf_delta":    round(s["confidence_delta"].mean(), 1),
        "st_prelim_ece":         round(compute_ece(s["preliminary_confidence"].values.astype(float), s["is_correct_preliminary"].values.astype(bool)), 2),
        "st_postcq_ece":         round(compute_ece(s["updated_confidence"].values.astype(float), s["is_correct_updated"].values.astype(bool)), 2),
        "st_pct_epistemic":      round(100 * st_clf[mid]["label"].value_counts().get("EPISTEMIC", 0) / len(st_clf[mid]), 1),
        "mt_prelim_acc":         round(pct(m["is_correct_preliminary"]), 1),
        "mt_cq1_acc":            round(pct(m["is_correct_1"]), 1),
        "mt_cq2_acc":            round(pct(m["is_correct_2"]), 1),
        "mt_final_acc":          round(pct(m["is_correct_final"]), 1),
        "mt_acc_gain_pp":        round(pct(m["is_correct_final"]) - pct(m["is_correct_preliminary"]), 1),
        "mt_final_ece":          round(compute_ece(m["final_confidence"].values.astype(float), m["is_correct_final"].values.astype(bool)), 2),
        "mt_pct_epistemic":      round(100 * mt_clf[mid]["label"].value_counts().get("EPISTEMIC", 0) / len(mt_clf[mid]), 1),
    })

summary_df = pd.DataFrame(rows).set_index("model").T
print("=== Full Summary Table ===")
display(summary_df)

out_path = CROSS_OUT / "cross_model_summary.csv"
summary_df.to_csv(out_path)
print(f"\\nSaved: {out_path.name}")
"""),

code("""\
# Final combined figure: all key metrics side by side
metrics = [
    ("ST Prelim Acc (%)",  "st_prelim_acc"),
    ("ST Post-CQ Acc (%)", "st_post_cq_acc"),
    ("ST Acc Gain (pp)",   "st_acc_gain_pp"),
    ("ST Conf Delta (pp)", "st_mean_conf_delta"),
    ("MT Prelim Acc (%)",  "mt_prelim_acc"),
    ("MT Final Acc (%)",   "mt_final_acc"),
    ("MT Acc Gain (pp)",   "mt_acc_gain_pp"),
]

summary_dict = {row["model"]: row for row in rows}
model_names  = [MODEL_LABELS[mid] for mid in MODELS]

fig, ax = plt.subplots(figsize=(14, 5))
x4 = np.arange(len(metrics))
w4 = 0.35
for i, mid in enumerate(MODELS):
    row  = summary_dict[MODEL_LABELS[mid]]
    vals = [row[key] for _, key in metrics]
    bars = ax.bar(x4 + (i-0.5)*w4, vals, w4, label=MODEL_LABELS[mid],
                  color=COLORS[mid], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0) + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7)

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x4)
ax.set_xticklabels([lbl for lbl, _ in metrics], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Value")
ax.set_title("Cross-Model Summary — Key Metrics", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(CROSS_OUT / "cross_model_summary_chart.png", dpi=150)
plt.show()
print("Saved: cross_model_summary_chart.png")
print("\\nCross-model analysis complete.")
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
