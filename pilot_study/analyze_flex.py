"""Standalone analysis for the MS-Dialog Gemini Flex experiment.
Run after phase1_flex_results.csv is produced.
"""
from __future__ import annotations
import sys, io, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from rouge_score import rouge_scorer as rs
from scipy.stats import spearmanr, mannwhitneyu, kruskal

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT       = Path(__file__).parent
OUTPUTS    = ROOT / "outputs" / "ms-dialog" / "gemini-2.5-flash"
CASES_PATH = ROOT / "datasets" / "ms-dialog" / "msdialog_100.jsonl"
FLEX_CSV   = OUTPUTS / "phase1_flex_results.csv"
FIG_DIR    = OUTPUTS

# ── Load ──────────────────────────────────────────────────────────────────────
flex = pd.read_csv(FLEX_CSV)
print(f"Loaded {len(flex)} rows from {FLEX_CSV.name}")
flex["was_blocked"] = flex["was_blocked"].astype(str).str.upper() == "TRUE"
flex["n_cqs_asked"] = pd.to_numeric(flex["n_cqs_asked"], errors="coerce")
# Drop blocked rows and parse-error rows (n_cqs=-1)
fv = flex[~flex["was_blocked"] & (flex["n_cqs_asked"] >= 0)].copy()
print(f"Valid (non-blocked, non-error): {len(fv)}")

# ── ROUGE-L ───────────────────────────────────────────────────────────────────
scorer = rs.RougeScorer(["rougeL"], use_stemmer=False)

def rouge_l(hyp, ref):
    if pd.isna(hyp) or pd.isna(ref) or not str(hyp).strip() or not str(ref).strip():
        return np.nan
    return scorer.score(str(ref), str(hyp))["rougeL"].fmeasure

fv["rouge_prelim"] = fv.apply(lambda r: rouge_l(r["preliminary_solution"], r["accepted_answer"]), axis=1)
fv["rouge_final"]  = fv.apply(lambda r: rouge_l(r["final_solution"],        r["accepted_answer"]), axis=1)
fv["rouge_gain"]   = fv["rouge_final"] - fv["rouge_prelim"]
fv["conf_gain"]    = fv["final_confidence"] - fv["preliminary_confidence"]
fv["n_cqs_asked"]  = fv["n_cqs_asked"].astype(int)
fv = fv[fv["n_cqs_asked"] >= 0].copy()
fv["asked_any"]    = fv["n_cqs_asked"] > 0

# ── 1. Distribution of n_cqs_asked ───────────────────────────────────────────
print("\n=== 1. CQ Selection Distribution ===")
dist = fv["n_cqs_asked"].value_counts().sort_index()
for k, v in dist.items():
    print(f"  n_cqs={k}: {v} cases ({v/len(fv)*100:.1f}%)")

# ── 2. Preliminary confidence by n_cqs_asked ─────────────────────────────────
print("\n=== 2. Preliminary Confidence by n_cqs_asked ===")
g = fv.groupby("n_cqs_asked")["preliminary_confidence"]
print(g.agg(["mean","median","std","count"]).round(1).to_string())
rho, p = spearmanr(fv["n_cqs_asked"], fv["preliminary_confidence"])
print(f"\nSpearman ρ(n_cqs, prelim_conf) = {rho:.3f}  p={p:.4f}")

# ── 3. Final ROUGE-L by n_cqs_asked ──────────────────────────────────────────
print("\n=== 3. Final ROUGE-L by n_cqs_asked ===")
g2 = fv.groupby("n_cqs_asked")["rouge_final"]
print(g2.agg(["mean","median","std","count"]).round(3).to_string())
groups = [fv[fv["n_cqs_asked"]==k]["rouge_final"].dropna().values for k in sorted(fv["n_cqs_asked"].unique())]
if len(groups) >= 2:
    stat, pval = kruskal(*groups)
    print(f"\nKruskal-Wallis H={stat:.3f}  p={pval:.4f}")

# ── 4. Confidence gain by n_cqs_asked ────────────────────────────────────────
print("\n=== 4. Confidence Gain (final − preliminary) by n_cqs_asked ===")
g3 = fv.groupby("n_cqs_asked")["conf_gain"]
print(g3.agg(["mean","median","std"]).round(2).to_string())

# Asked any vs not
asked    = fv[fv["asked_any"]]["rouge_final"].dropna()
not_asked = fv[~fv["asked_any"]]["rouge_final"].dropna()
if len(asked) > 0 and len(not_asked) > 0:
    u, up = mannwhitneyu(asked, not_asked, alternative="two-sided")
    print(f"\nMann-Whitney rouge_final: asked({len(asked)}) vs not({len(not_asked)})")
    print(f"  asked mean={asked.mean():.3f}  not_asked mean={not_asked.mean():.3f}  p={up:.4f}")

# ── 5. ROUGE gain per turn (for cases that asked) ────────────────────────────
print("\n=== 5. ROUGE-L arc (mean) ===")
sol_cols = ["rouge_prelim"]
for k in [1, 2]:
    col = f"solution_{k}"
    if col in fv.columns:
        fv[f"rouge_sol{k}"] = fv.apply(lambda r: rouge_l(r.get(col, ""), r["accepted_answer"]), axis=1)
        sol_cols.append(f"rouge_sol{k}")
sol_cols.append("rouge_final")
print(fv[sol_cols].mean().round(3).to_string())

# ── 6. needed_clarification_0 vs actual confidence ───────────────────────────
print("\n=== 6. Did nc_0 prediction match confidence? ===")
fv["nc0"] = fv["needed_clarification_0"].astype(str).str.upper().isin({"TRUE","1"})
nc_conf = fv.groupby("nc0")["preliminary_confidence"].agg(["mean","median","count"])
print(nc_conf.round(1).to_string())
if fv["nc0"].nunique() == 2:
    u2, up2 = mannwhitneyu(
        fv[fv["nc0"]]["preliminary_confidence"],
        fv[~fv["nc0"]]["preliminary_confidence"],
        alternative="two-sided",
    )
    print(f"Mann-Whitney p={up2:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {0: "#4C72B0", 1: "#55A868", 2: "#C44E52", 3: "#8172B2"}
NC = sorted(fv["n_cqs_asked"].unique())
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Panel A: CQ count bar chart ───────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
vals  = [dist.get(k, 0) for k in range(4)]
bars  = ax_a.bar(range(4), vals, color=[COLORS[k] for k in range(4)], edgecolor="white", linewidth=0.8)
ax_a.set_xticks(range(4))
ax_a.set_xticklabels(["0 CQs\n(direct)", "1 CQ", "2 CQs", "3 CQs"])
ax_a.set_ylabel("Cases")
ax_a.set_title("A  CQ Selection Distribution", fontweight="bold", loc="left")
for bar, v in zip(bars, vals):
    if v > 0:
        ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                  f"{v}\n({v/len(fv)*100:.0f}%)", ha="center", va="bottom", fontsize=8)

# ── Panel B: preliminary confidence by n_cqs_asked ───────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
data_b = [fv[fv["n_cqs_asked"]==k]["preliminary_confidence"].values for k in NC]
bp = ax_b.boxplot(data_b, patch_artist=True, widths=0.5,
                  medianprops=dict(color="black", linewidth=1.5))
for patch, k in zip(bp["boxes"], NC):
    patch.set_facecolor(COLORS[k])
    patch.set_alpha(0.75)
ax_b.set_xticklabels([f"{k} CQ{'s' if k!=1 else ''}" for k in NC])
ax_b.set_ylabel("Preliminary Confidence")
ax_b.set_title(f"B  Prelim Confidence by n_cqs\n(ρ={rho:.2f}, p={p:.3f})", fontweight="bold", loc="left")
ax_b.set_ylim(0, 105)

# ── Panel C: Final ROUGE-L by n_cqs_asked ────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
data_c = [fv[fv["n_cqs_asked"]==k]["rouge_final"].dropna().values for k in NC]
bp2 = ax_c.boxplot(data_c, patch_artist=True, widths=0.5,
                   medianprops=dict(color="black", linewidth=1.5))
for patch, k in zip(bp2["boxes"], NC):
    patch.set_facecolor(COLORS[k])
    patch.set_alpha(0.75)
ax_c.set_xticklabels([f"{k} CQ{'s' if k!=1 else ''}" for k in NC])
ax_c.set_ylabel("Final ROUGE-L F1")
plab = f"KW p={pval:.3f}" if len(groups) >= 2 else ""
ax_c.set_title(f"C  Solution Quality by n_cqs\n({plab})", fontweight="bold", loc="left")
ax_c.set_ylim(0, 0.55)

# ── Panel D: ROUGE-L arc by n_cqs group ──────────────────────────────────────
ax_d = fig.add_subplot(gs[1, :2])
arc_map = {
    0: (["rouge_prelim", "rouge_final"], ["Prelim", "Final"]),
    1: (["rouge_prelim", "rouge_sol1", "rouge_final"],   ["Prelim", "After CQ1", "Final"]),
    2: (["rouge_prelim", "rouge_sol1", "rouge_sol2", "rouge_final"], ["Prelim","After CQ1","After CQ2","Final"]),
    3: (["rouge_prelim", "rouge_sol1", "rouge_sol2", "rouge_final"], ["Prelim","After CQ1","After CQ2","Final(forced)"]),
}
x_max = 0
for k in NC:
    sub = fv[fv["n_cqs_asked"]==k]
    if len(sub) == 0:
        continue
    cols, labels = arc_map.get(k, arc_map[min(k, 3)])
    cols   = [c for c in cols if c in sub.columns]
    labels = labels[:len(cols)]
    means  = [sub[c].dropna().mean() for c in cols]
    xs     = list(range(len(cols)))
    x_max  = max(x_max, len(cols) - 1)
    ax_d.plot(xs, means, "o-", color=COLORS[k], linewidth=1.8, markersize=5,
              label=f"{k} CQ{'s' if k!=1 else ''} (n={len(sub)})")
ax_d.set_xlim(-0.2, x_max + 0.2)
ax_d.set_xticks(range(x_max + 1))
ax_d.set_xticklabels(["Prelim", "After CQ1", "After CQ2", "Final"][:x_max+1])
ax_d.set_ylabel("Mean ROUGE-L F1")
ax_d.set_title("D  Solution Quality Arc by CQ Group", fontweight="bold", loc="left")
ax_d.legend(fontsize=8)
ax_d.set_ylim(0, 0.55)
ax_d.grid(axis="y", alpha=0.3)

# ── Panel E: Confidence arc ───────────────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 2])
conf_arc_map = {
    0: (["preliminary_confidence", "final_confidence"],              ["Prelim", "Final"]),
    1: (["preliminary_confidence", "confidence_1", "final_confidence"], ["Prelim","T1","Final"]),
    2: (["preliminary_confidence", "confidence_1", "confidence_2", "final_confidence"], ["P","T1","T2","F"]),
    3: (["preliminary_confidence", "confidence_1", "confidence_2", "final_confidence"], ["P","T1","T2","F"]),
}
for k in NC:
    sub = fv[fv["n_cqs_asked"]==k]
    if len(sub) == 0:
        continue
    cols, labels = conf_arc_map.get(k, conf_arc_map[min(k, 3)])
    cols  = [c for c in cols if c in sub.columns]
    means = [sub[c].dropna().mean() for c in cols]
    xs    = [i/(len(cols)-1) for i in range(len(cols))] if len(cols) > 1 else [0]
    ax_e.plot(xs, means, "o-", color=COLORS[k], linewidth=1.8, markersize=5,
              label=f"{k} CQ{'s' if k!=1 else ''}")
ax_e.set_xticks([0, 0.5, 1])
ax_e.set_xticklabels(["Start", "Mid", "End"])
ax_e.set_ylabel("Mean Confidence")
ax_e.set_title("E  Confidence Arc by CQ Group", fontweight="bold", loc="left")
ax_e.legend(fontsize=8)
ax_e.set_ylim(0, 105)
ax_e.grid(axis="y", alpha=0.3)

# ── Panel F: nc_0 decision vs prelim confidence ───────────────────────────────
ax_f = fig.add_subplot(gs[2, 0])
nc_true  = fv[fv["nc0"]]["preliminary_confidence"].values
nc_false = fv[~fv["nc0"]]["preliminary_confidence"].values
ax_f.violinplot([nc_false, nc_true], positions=[0, 1], showmedians=True)
ax_f.set_xticks([0, 1])
ax_f.set_xticklabels(["Said NO\n(went direct)", "Said YES\n(asked CQ)"])
ax_f.set_ylabel("Preliminary Confidence")
_up2_label = f"MW p={up2:.3f}" if "up2" in locals() and not np.isnan(up2) else ""
ax_f.set_title(f"F  nc_0 Decision vs Confidence\n({_up2_label})", fontweight="bold", loc="left")
ax_f.set_ylim(0, 105)

# ── Panel G: Confidence gain by n_cqs ────────────────────────────────────────
ax_g = fig.add_subplot(gs[2, 1])
gain_data = [fv[fv["n_cqs_asked"]==k]["conf_gain"].values for k in NC]
bp3 = ax_g.boxplot(gain_data, patch_artist=True, widths=0.5,
                   medianprops=dict(color="black", linewidth=1.5))
for patch, k in zip(bp3["boxes"], NC):
    patch.set_facecolor(COLORS[k])
    patch.set_alpha(0.75)
ax_g.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax_g.set_xticklabels([f"{k} CQ{'s' if k!=1 else ''}" for k in NC])
ax_g.set_ylabel("Confidence gain (final − prelim)")
ax_g.set_title("G  Confidence Gain by n_cqs", fontweight="bold", loc="left")

# ── Panel H: ROUGE gain (final − prelim) by n_cqs ────────────────────────────
ax_h = fig.add_subplot(gs[2, 2])
rgain_data = [fv[fv["n_cqs_asked"]==k]["rouge_gain"].dropna().values for k in NC]
bp4 = ax_h.boxplot(rgain_data, patch_artist=True, widths=0.5,
                   medianprops=dict(color="black", linewidth=1.5))
for patch, k in zip(bp4["boxes"], NC):
    patch.set_facecolor(COLORS[k])
    patch.set_alpha(0.75)
ax_h.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax_h.set_xticklabels([f"{k} CQ{'s' if k!=1 else ''}" for k in NC])
ax_h.set_ylabel("ROUGE-L gain (final − prelim)")
ax_h.set_title("H  ROUGE Gain by n_cqs", fontweight="bold", loc="left")

fig.suptitle("MS-Dialog Flex Experiment — Gemini-2.5-Flash\n(Model chooses whether to ask CQs, 0–3 max)",
             fontsize=13, fontweight="bold", y=1.01)

out_path = FIG_DIR / "flex_fig_overview.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved → {out_path}")
print("\nDone.")
