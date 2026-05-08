# Uncertainty Benchmark — Progress Checklist

Update this file as each task is completed. Mark done with `[x]`.

---

## Phase 0 — Pre-flight + Validate trusted components

### 0.0 Colab small-model pre-flight
- [ ] Load Qwen2.5-4B-Instruct on Colab T4
- [ ] Test flex-turn JSON schema compliance (native, no constraints)
- [ ] Test single-turn JSON schema compliance (native, no constraints)
- [ ] Test grammar-constrained decoding (`outlines` or `JSONLogitsProcessor`)
- [ ] Confirm logprob extraction via `output_scores=True` + `return_dict_in_generate=True`
- [ ] Test 3-sample generation at T=0.7 (semantic entropy feasibility)
- [ ] Document token budget / truncation point
- [ ] Write findings cell summarising pass/fail per probe
- [ ] **Decision recorded:** Qwen-4B IN or OUT of main matrix

**Deliverable:** `colab/small_model_preflight.ipynb`

---

### 0.1 Fix judge prompt
- [ ] Rewrite `prompts/judge.txt` — add WH-specification aleatoric patterns
- [ ] Add "Are you referring to…" / "Do you specifically…" patterns
- [ ] Add adversarial few-shot pairs (same surface form, different label)
- [ ] One pair per CLAMBER subclass (8 pairs total)

---

### 0.2 Validate judge on CLAMBER
- [ ] Run judge at T=0 on full CLAMBER labelled set
- [ ] Compute accuracy, per-class P/R/F1, confusion matrix
- [ ] Run 5 batches at T=0.0 / 0.3 / 0.5 — compute self-consistency rate
- [ ] Compute Cohen's κ vs gold
- [ ] Save `analysis/judge_validation_report.md`
- [ ] **Gate passed:** accuracy ≥ 85%, κ ≥ 0.70, per-class F1 ≥ 0.80

---

### 0.3 Validate simulator
- [ ] Spot-check 20 cases per dataset (60 total)
- [ ] Compute hedge rate per dataset
- [ ] Flag any simulator answer contradicting layer 1 content
- [ ] Revise simulator prompt if hedge rate ≥ 20%
- [ ] Save `outputs/simulator_validation/` (hedge rates CSV, spot-check CSV)
- [ ] **Gate passed:** hedge rate < 20% per dataset, no systematic contradictions

---

### 0.4 Re-run validated judge on pilot CSVs
- [ ] Re-classify MS-Dialog pilot CQs with new judge
- [ ] Re-classify ShARC pilot CQs with new judge
- [ ] Save updated classified CSVs

---

## Phase 1 — Infrastructure

### Providers
- [ ] `src/providers/gemini.py` — gemini-2.5-flash + gemini-3.1-pro-preview (API)
- [ ] `src/providers/gemma.py` — gemma-3-12b-it (local, transformers, logprobs)
- [ ] `src/providers/llama.py` — Llama-3.3-70B (local, transformers, logprobs)
- [ ] `src/providers/qwen.py` — Qwen-4B (local, conditional on pre-flight)
- [ ] Smoke test: each provider completes 1 structured-output call

### Robust JSON handling
- [ ] `src/parsing.py` — `parse_with_schema()` with retry ladder
- [ ] Test: deliberately broken response triggers retry then parse_error row

### UQ utilities
- [ ] `src/uq/verbalized.py` — extract `confidence` from JSON
- [ ] `src/uq/logprob_entropy.py` — per-token entropy from top-k logprobs
- [ ] `src/uq/semantic_entropy.py` — N-sample clustering (stretch)

### WandB
- [ ] `src/tracking.py` — wrapper around wandb
- [ ] Test: WandB run appears in dashboard with config + metrics

### Dataset curation
- [ ] `sharc_200.jsonl` — already done, verify
- [ ] `preprocess_medqa.py` → `medqa_200.jsonl`
- [ ] `preprocess_msdialog.py` → `msdialog_200.jsonl`

### Pipelines
- [ ] `src/pipelines/single_turn.py`
- [ ] `src/pipelines/flex_turn.py`
- [ ] `src/pipelines/simulator.py`

---

## Phase 2 — Experiment runs

### Dry runs (5 records each — all configs before full runs)
- [ ] MedQA × gemini-2.5-flash × single-turn
- [ ] MedQA × gemini-2.5-flash × flex
- [ ] MedQA × gemini-3.1-pro-preview × single-turn
- [ ] MedQA × gemini-3.1-pro-preview × flex
- [ ] MedQA × gemma-3-12b-it × single-turn
- [ ] MedQA × gemma-3-12b-it × flex
- [ ] MedQA × llama-3.3-70b × single-turn
- [ ] MedQA × llama-3.3-70b × flex
- [ ] MedQA × qwen-4B × single-turn (if in matrix)
- [ ] MedQA × qwen-4B × flex (if in matrix)
- [ ] MS-Dialog × gemini-2.5-flash × single-turn
- [ ] MS-Dialog × gemini-2.5-flash × flex
- [ ] MS-Dialog × gemini-3.1-pro-preview × single-turn
- [ ] MS-Dialog × gemini-3.1-pro-preview × flex
- [ ] MS-Dialog × gemma-3-12b-it × single-turn
- [ ] MS-Dialog × gemma-3-12b-it × flex
- [ ] MS-Dialog × llama-3.3-70b × single-turn
- [ ] MS-Dialog × llama-3.3-70b × flex
- [ ] MS-Dialog × qwen-4B × single-turn (if in matrix)
- [ ] MS-Dialog × qwen-4B × flex (if in matrix)
- [ ] ShARC × gemini-2.5-flash × single-turn
- [ ] ShARC × gemini-2.5-flash × flex
- [ ] ShARC × gemini-3.1-pro-preview × single-turn
- [ ] ShARC × gemini-3.1-pro-preview × flex
- [ ] ShARC × gemma-3-12b-it × single-turn
- [ ] ShARC × gemma-3-12b-it × flex
- [ ] ShARC × llama-3.3-70b × single-turn
- [ ] ShARC × llama-3.3-70b × flex
- [ ] ShARC × qwen-4B × single-turn (if in matrix)
- [ ] ShARC × qwen-4B × flex (if in matrix)

### Full runs (200 records each)
- [ ] MedQA × gemini-2.5-flash × single-turn
- [ ] MedQA × gemini-2.5-flash × flex
- [ ] MedQA × gemini-3.1-pro-preview × single-turn
- [ ] MedQA × gemini-3.1-pro-preview × flex
- [ ] MedQA × gemma-3-12b-it × single-turn
- [ ] MedQA × gemma-3-12b-it × flex
- [ ] MedQA × llama-3.3-70b × single-turn
- [ ] MedQA × llama-3.3-70b × flex
- [ ] MedQA × qwen-4B × single-turn (if in matrix)
- [ ] MedQA × qwen-4B × flex (if in matrix)
- [ ] MS-Dialog × gemini-2.5-flash × single-turn
- [ ] MS-Dialog × gemini-2.5-flash × flex
- [ ] MS-Dialog × gemini-3.1-pro-preview × single-turn
- [ ] MS-Dialog × gemini-3.1-pro-preview × flex
- [ ] MS-Dialog × gemma-3-12b-it × single-turn
- [ ] MS-Dialog × gemma-3-12b-it × flex
- [ ] MS-Dialog × llama-3.3-70b × single-turn
- [ ] MS-Dialog × llama-3.3-70b × flex
- [ ] MS-Dialog × qwen-4B × single-turn (if in matrix)
- [ ] MS-Dialog × qwen-4B × flex (if in matrix)
- [ ] ShARC × gemini-2.5-flash × single-turn
- [ ] ShARC × gemini-2.5-flash × flex
- [ ] ShARC × gemini-3.1-pro-preview × single-turn
- [ ] ShARC × gemini-3.1-pro-preview × flex
- [ ] ShARC × gemma-3-12b-it × single-turn
- [ ] ShARC × gemma-3-12b-it × flex
- [ ] ShARC × llama-3.3-70b × single-turn
- [ ] ShARC × llama-3.3-70b × flex
- [ ] ShARC × qwen-4B × single-turn (if in matrix)
- [ ] ShARC × qwen-4B × flex (if in matrix)

---

## Phase 3 — Judge classification

- [ ] `experiments/run_judge.py` written and tested
- [ ] MedQA all runs classified → `medqa_*_classified.csv`
- [ ] MS-Dialog all runs classified → `msdialog_*_classified.csv`
- [ ] ShARC all runs classified → `sharc_*_classified.csv`
- [ ] `outputs/cq_types_summary.csv` generated (model × dataset × type → count)

---

## Phase 4 — Analysis

### Required figures (all PDF)
- [ ] `figures/accuracy_matrix.pdf` — cross-model accuracy per dataset
- [ ] `figures/method_comparison.pdf` — single-turn vs flex per model × dataset
- [ ] `figures/cq_asking_rate.pdf` — CQ-asking rate by model × dataset (flex only)
- [ ] `figures/recovery.pdf` — recovery rate by model × dataset
- [ ] `figures/calibration_grid.pdf` — confidence calibration per model × dataset
- [ ] `figures/cq_types.pdf` — CQ type distribution by model × dataset
- [ ] `figures/conf_correctness.pdf` — verbalized confidence ↔ correctness

### Open-model figures (Gemma + Llama)
- [ ] `figures/token_entropy_by_type.pdf`
- [ ] `figures/entropy_decomposition.pdf`
- [ ] `figures/uq_method_comparison.pdf` + AUC table

### Final deliverables
- [ ] `analysis/notebooks/final_report.ipynb` complete
- [ ] `BENCHMARK_REPORT.md` at project root with all figures embedded
- [ ] All 24+ `results.csv` files in `outputs/`
- [ ] All runs visible in WandB
