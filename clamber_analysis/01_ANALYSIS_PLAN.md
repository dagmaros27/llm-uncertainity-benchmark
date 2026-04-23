# 01 — Analysis Plan

## Overview

This document outlines all analysis steps to run on the CLAMBER dataset. Each phase builds on the previous. Every step that produces a numeric result should also produce a saved figure. All figures should be saved to a `figures/` subfolder.

The dataset is a JSONL file (one JSON object per line). Some entries may be double-encoded strings — handle that in the loader. The `predict_is_ambiguous_response` field is a JSON string nested inside the main JSON and needs to be parsed separately to extract `Output` (True/False) and `Confidence` (integer 1–5).

---

## Phase 1 — EDA: Dataset Structure and Distributions

Get a complete picture of what the dataset contains before doing any evaluation.

### 1.1 Basic sanity checks
- Total number of entries
- Count of missing or null values per field
- Count of entries where `predict_clarifying_question` is empty or missing even though `require_clarification = 1`
- Count of entries where `predict_is_ambiguous_response` failed to parse

### 1.2 Label distributions
- Distribution of `require_clarification` (how balanced is ambiguous vs not ambiguous overall)
- Distribution of `category` (MC, FD, LA)
- Distribution of `subclass` (all eight types)
- Cross-tabulation of `category` × `require_clarification` to see balance within each category

**Figures:** Bar chart for category counts. Bar chart for subclass counts. Stacked bar showing ambiguous vs not ambiguous split per category. Pie chart for overall ambiguous vs not ambiguous.

### 1.3 Text length analysis
- Word count distribution for: `question`, `clarifying_question` (ground truth), `predict_clarifying_question` (LLM output)
- Report mean, median, min, max for each
- Compare whether the LLM's clarifying questions are systematically longer or shorter than the ground truth

**Figures:** Three side-by-side histograms — one per text field. Boxplot comparing ground truth vs predicted CQ length, broken down by category.

### 1.4 Confidence score distribution
- Parse `Confidence` from `predict_is_ambiguous_response`
- Distribution of confidence scores across the full dataset
- Distribution broken down by `require_clarification` label — does the model express higher confidence when it is correct?

**Figure:** Histogram of confidence scores overall. Side-by-side histogram comparing confidence when prediction is correct vs incorrect.

---

## Phase 2 — Ambiguity Detection Performance

Evaluate how well the LLM identifies which queries are ambiguous.

### 2.1 Overall classification metrics
- Compute accuracy, precision, recall, and weighted F1 comparing `predict_ambiguous` vs `require_clarification`
- Print these clearly — they will be used in Phase 5 to identify the model

### 2.2 Confusion matrix
- Full confusion matrix (TP, FP, TN, FN)
- Note whether the model skews toward over-predicting or under-predicting ambiguity

**Figure:** Confusion matrix heatmap with counts and percentages.

### 2.3 Per-category and per-subclass performance
- Compute accuracy and weighted F1 separately for each `category` and each `subclass`
- Identify which categories the model handles best and worst

**Figures:** Grouped bar chart showing accuracy and F1 per category. Grouped bar chart showing accuracy and F1 per subclass. Add a dashed line at 0.5 to mark the random baseline.

### 2.4 False negative rate analysis
This is especially important for the research: the false negative cases are where the model fails to surface its uncertainty even though the query is genuinely ambiguous.

- Compute the false negative rate per subclass: proportion of truly ambiguous queries where the model predicted not ambiguous
- Rank subclasses by false negative rate

**Figure:** Horizontal bar chart of false negative rate per subclass, sorted descending.

### 2.5 Acc@1 vs Acc@0
Reproducing Figure 1 from the paper — accuracy on ambiguous queries only vs accuracy on unambiguous queries only.

- Acc@1 = accuracy when `require_clarification = 1`
- Acc@0 = accuracy when `require_clarification = 0`
- Compare — does the model handle one type better than the other?

**Figure:** Simple two-bar chart showing Acc@1 and Acc@0 side by side. Use these values in Phase 5 to identify the model against Figure 1 in the paper.

---

## Phase 3 — Clarifying Question Quality

Evaluate the quality of the LLM's generated clarifying questions against the ground truth.

### 3.1 BERTScore
- Compute BERTScore (Precision, Recall, F1) between `predict_clarifying_question` and `clarifying_question` for all entries where `require_clarification = 1`
- Use `bert-score` Python library with `lang="en"`
- Report mean BERTScore F1 overall

**Figures:** Histogram of BERTScore F1 distribution. Boxplot of BERTScore F1 broken down by `category`. Boxplot of BERTScore F1 broken down by `subclass`.

### 3.2 BLEU and ROUGE (optional complement)
- Compute BLEU-1, BLEU-4, ROUGE-1, ROUGE-L as lightweight lexical overlap metrics
- Report per category and per subclass as a table

### 3.3 Does detecting ambiguity improve CQ quality?
- Split entries where `require_clarification = 1` into two groups: those where `predict_ambiguous = 1` (model detected it) and those where `predict_ambiguous = 0` (model missed it)
- Compare BERTScore F1 between these two groups
- Hypothesis: when the model correctly detects ambiguity, it generates a better clarifying question

**Figure:** Side-by-side boxplot of BERTScore F1 for "model detected ambiguity" vs "model missed ambiguity."

### 3.4 CQ quality vs confidence
- Scatter plot of `Confidence` score vs BERTScore F1
- Is the model better calibrated when it is more confident?

**Figure:** Scatter plot with a trend line.

---

## Phase 4 — Uncertainty Lens (Core Research Analysis)

This is the most important phase. Re-read the uncertainty type definitions in `00_CONTEXT.md` before proceeding.

The goal is to look at the `predict_clarifying_question` field and try to infer what type of uncertainty the model was experiencing when it generated that question — epistemic, aleatoric, or preferential.

### 4.1 Heuristic uncertainty type classification
Build a simple keyword and pattern-based classifier that labels each predicted CQ with one of: `epistemic`, `aleatoric`, `preferential`, or `unclear`.

Guidelines for labelling:
- **Epistemic**: The question asks what something means, asks for clarification about an unfamiliar entity, or signals that the model doesn't know something. Key phrases: "what do you mean by", "could you clarify what X is", "I'm not familiar with", "what is X", "please explain".
- **Aleatoric**: The model cannot produce a single answer because multiple valid outputs exist. This covers both discrete disambiguation (the model lists known options and asks the user to pick) and open-ended cases (the answer depends entirely on user context or preference). Key phrases: "are you referring to X or Y", "which version", "which one", "which year", "what are your interests", "what do you prefer", "what style", "could you specify".

Apply this classifier to every entry that has a non-empty `predict_clarifying_question`. Report the label distribution.

**Figure:** Bar chart of uncertainty type distribution across the full dataset.

### 4.2 Uncertainty type vs CLAMBER taxonomy
- Cross-tabulate inferred `uncertainty_type` against `subclass`
- This tests the mapping hypothesis: does the CLAMBER prompt-level category predict what uncertainty type the model expresses?
- Key question: does `NK` (unfamiliar entity) reliably produce epistemic questions? Does `MC` (aleatoric output) reliably produce aleatoric questions? Or does the model's expressed uncertainty diverge from the prompt-level label?

**Figure:** Heatmap of `subclass` vs `uncertainty_type` with row-normalised proportions. This is the central figure for the research motivation.

### 4.3 Mismatch analysis — silent uncertainty
These are the most interesting cases: entries where `require_clarification = 1` (truly ambiguous) but `predict_ambiguous = 0` (model did not flag it), yet the model still generated a clarifying question.

- Count how many such entries exist
- Apply the uncertainty type classifier to their `predict_clarifying_question`
- What type of uncertainty appears in these "silent" cases?
- Which `subclass` do they mostly belong to?

**Figures:** Bar chart of uncertainty type distribution in silent uncertain cases. Comparison bar chart of subclass distribution between "model missed it" vs "model detected it."

### 4.4 Question framing analysis
Look at how the model opens its clarifying questions — this reveals the frame through which it is expressing uncertainty.

- Extract the first 3–5 words of each `predict_clarifying_question`
- Count the top 20–30 most common opening phrases
- Observe: does the model predominantly ask "Could you please specify..." (preferential framing) vs "What do you mean by..." (epistemic framing) vs "Are you referring to X or Y..." (aleatoric framing)?

**Figure:** Horizontal bar chart of top opening phrases ranked by frequency.

### 4.5 Manual inspection sample
Select a stratified sample of ~30 entries — a few from each subclass — and print the full `question`, `clarifying_question` (ground truth), and `predict_clarifying_question` side by side. This is for qualitative reading, not automated analysis. Look for:
- Cases where the model asks something very different from the ground truth
- Cases where the model's question reveals a different type of uncertainty than CLAMBER's label suggests
- Surprising or interesting failure patterns

No figure needed — just readable printed output or a saved CSV of the sample.

---

## Phase 5 — Model Identification

Use the metrics computed in earlier phases to infer which LLM generated the `predict_*` fields.

### 5.1 Match overall metrics to Table 3 in the paper
Table 3 reports Accuracy and weighted F1 per model, per prompting scheme. Compare the overall accuracy and F1 computed in Phase 2.1 to all cells in Table 3. The closest match indicates the model and prompting scheme.

### 5.2 Match per-subclass metrics to Table 6 in the paper
Table 6 reports Accuracy and F1 per model per subclass under the Few-shot w/o CoT setting. Compare the per-subclass values computed in Phase 2.3. A consistent match across subclasses narrows down the model.

### 5.3 Match Acc@1 vs Acc@0 to Figure 1 in the paper
Figure 1 in the paper shows Acc@1 vs Acc@0 for each model under Zero-shot w/o CoT. Match the values from Phase 2.5 to this figure.

### 5.4 Confidence score format as a clue
The `predict_is_ambiguous_response` field contains a structured JSON with `Output` and `Confidence`. The format suggests it was generated with a structured output prompt. Note whether the confidence range (1–5) and output format align with any model-specific details mentioned in the paper's Appendix.

### 5.5 Conclusion
State the most likely model and prompting scheme based on the evidence, and note any ambiguity in the match.

---

## Phase 6 — Summary

### 6.1 Key findings table
Produce a clean summary table covering:
- Dataset balance per category
- Model detection accuracy per subclass
- False negative rates per subclass
- Mean BERTScore per subclass
- Most common inferred uncertainty type per subclass

### 6.2 Key takeaways for my research
Write a brief paragraph (in the notebook or a separate notes file) answering:
- Does the CLAMBER taxonomy map onto my uncertainty types in a predictable way?
- Which subclasses are the most epistemically interesting (model is uncertain but doesn't know it)?
- What patterns in question framing should I look for when building my own annotation scheme?
- What should I do differently when generating my own dataset?

---

## Output Files Expected

By the end of this analysis the `figures/` folder should contain at minimum:

| Filename | What it shows |
|---|---|
| `01_category_subclass_distribution.png` | Dataset composition |
| `02_ambiguous_split_per_category.png` | Label balance |
| `03_text_length_distributions.png` | Query and CQ lengths |
| `04_cq_length_by_category.png` | GT vs predicted CQ length |
| `05_confidence_distribution.png` | Confidence scores |
| `06_confusion_matrix.png` | Detection performance |
| `07_accuracy_f1_per_category.png` | Per-category performance |
| `08_accuracy_f1_per_subclass.png` | Per-subclass performance |
| `09_false_negative_rate.png` | Suppressed uncertainty |
| `10_acc1_vs_acc0.png` | Figure 1 reproduction |
| `11_bertscore_distribution.png` | CQ quality overall |
| `12_bertscore_by_category.png` | CQ quality per category |
| `13_bertscore_by_subclass.png` | CQ quality per subclass |
| `14_bertscore_detected_vs_missed.png` | Detection impact on quality |
| `15_uncertainty_type_distribution.png` | Core research figure |
| `16_uncertainty_type_vs_subclass_heatmap.png` | Mapping hypothesis test |
| `17_silent_uncertainty_analysis.png` | Mismatch cases |
| `18_cq_opening_phrases.png` | Framing patterns |
