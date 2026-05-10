# Uncertainty Benchmark — Progress Checklist

Update this file as each task is completed. Mark done with `[x]`.

---

## Phase 0 — Pre-flight + Validate trusted components

### 0.0 Colab small-model pre-flight ✅ COMPLETE
- [x] Load Qwen3-4B on Colab T4 (8.0 / 15.6 GB VRAM used)
- [x] Test flex-turn JSON schema compliance — **100% native pass**
- [x] Test single-turn JSON schema compliance — **100% native pass**
- [x] Test grammar-constrained decoding — outlines FAILED (API changed); not needed since native is 100%
- [x] Confirm logprob extraction — works, vocab_size 151,936, 46 steps captured
- [x] Test 3-sample generation at T=0.7 — 3/3 unique CQs in 14.3s (semantic entropy feasible)
- [x] Document token budget — 30 tokens used; max_new_tokens=64 sufficient (thinking off)
- [x] Write findings cell summarising pass/fail per probe
- [x] **Decision recorded: Qwen3-4B IN the main model matrix**

**Findings:**
- Thinking mode produces 927-char `<think>` blocks; suppressed cleanly with `enable_thinking=False`
- Mean per-token entropy at T=0 is 0.133 — model is highly deterministic at T=0
- ⚠️ **Behavioural flag:** in flex-turn schema, model returned `ask_cq=False` in all 5 trials — refuses to ask CQs on the toy case. Needs monitoring during real experiments.
- Outlines library API is broken in current version; use native JSON + regex fallback (works perfectly)

**Deliverable:** `uncertainty_benchmark/colab/small_model_preflight.ipynb` ✅

---

### 0.1 Fix judge prompt ✅ COMPLETE
- [x] Rewrote `prompts/judge.txt` — core decision rule + WH-specification aleatoric patterns
- [x] Added "Are you referring to…" / "Do you specifically…" patterns
- [x] Added 3 adversarial pairs (version, where, generic-spec) — same surface form, different label
- [x] 8 canonical examples — one per CLAMBER subclass — in `src/judge/few_shot_examples.py`
- [x] Built validation script `scripts/validate_judge_clamber.py` with 4 PDF figures + resume support
- [x] User reviewed prompt + few-shots — approved

---

### 0.2 Validate judge on CLAMBER ✅ COMPLETE — GATE PASSED
- [x] Run judge (**gemini-3.1-pro-preview**) at T=0.0, 0.3, 0.5 on full CLAMBER labelled set (200 records each)
- [x] Compute accuracy, per-class P/R/F1, confusion matrix
- [x] Compute self-consistency: 97.0% all-temp agreement, pairwise 97.0–99.5%
- [x] Compute Cohen's κ vs gold = 0.920 at T=0.0
- [x] Saved `outputs/judge_validation/20260508T165406Z/judge_validation_report.md` + 4 PDFs
- [x] **Gate passed at T=0.0:** acc 0.960 ≥ 0.85 ✓ | κ 0.920 ≥ 0.70 ✓ | F1 EPI 0.962 ≥ 0.80 ✓ | F1 ALE 0.958 ≥ 0.80 ✓

**Notes for downstream interpretation:**
- Judge has slight epistemic-leaning bias: EPISTEMIC recall = 1.0 / precision = 0.926; ALEATORIC precision = 1.0 / recall = 0.920
- All errors are ALEATORIC→EPISTEMIC misclassifications (8 at T=0.0)
- Errors concentrated in MC subclasses (when 0.87, where 0.87, what 0.89, whom 0.94); LA + FD subclasses are perfect (1.0)
- Pattern: when an MC question reads like a fact-request rather than a scope-choice, judge slips epistemic. Document as a known judge bias in the paper

**Earlier (invalid) flash run archived as `INVALID_flash_run_20260508T152617Z` — used the wrong model.**

---

### 0.3 Validate simulator ✅ COMPLETE
- [x] 60 simulator calls on gemini-3.1-pro-preview (20 per dataset)
- [x] Hedge rates: medqa 25%, msdialog 80%, sharc 10%
- [x] Claude labelled all 60 rows: 0 HALLUCINATED, 0 EXTRAPOLATED, 37 FAITHFUL, 23 HEDGED
- [x] Faithfulness rate on answered cases: **100% across all three datasets**
- [x] Saved `outputs/simulator_validation/20260508T183433Z/` with results.csv, per_dataset_metrics.json, 2 PDFs, report
- [x] Wrote new SHARC simulator prompt (domain-specific, eligibility/regulatory framing)
- [x] User trusts Claude's labelling (no manual re-review)
- [x] Fixed sharc_006 corrupted context — regenerated at T=0 with anti-meta prompt; cleanup done
- [x] Re-selected & re-synthesized 200 ms-dialog cases (`uncertainty_benchmark/datasets/msdialog/msdialog_200.jsonl`)
  - New selection: ≥2 substantive user turns; mean 3.1 substantive user turns/case (vs pilot ~1.4)
  - New synthesis: pulls ALL substantive user utterances (FD/NF/FQ/CQ/IR/RQ + content-bearing PF), not just FD-tagged
  - 200 calls on gemini-3.1-pro-preview, mean output 1430 chars/case
- [x] A/B comparison test on the original 20 ms-dialog cases:
  - **Hedge rate: 80% → 70% (-10pp), 0 quality regressions**
  - 2 flips HEDGED→ANSWER (msd_051 controller connection, msd_003 user attempt notes)
  - Remaining 14 hedges all structural (specialist asks "Have you tried X?" / requests info user genuinely never volunteered)

**Final verdict:** Strict 20% hedge gate fails on MS-Dialog (70%) but residual is structural — accepted as a real signal per user direction. Simulator faithfulness is 100% across all three datasets.

---

### 0.4 Re-run validated judge on pilot CSVs — ⏭️ SKIPPED
> **Decision:** We are past the pilot study. The pilot CSVs are not used in the main experiment.
> The validated judge (gemini-3.1-pro-preview, T=0.0) will be applied fresh to all experiment outputs in Phase 3.
- [~] Re-classify MS-Dialog pilot CQs with new judge — *skipped*
- [~] Re-classify ShARC pilot CQs with new judge — *skipped*
- [~] Save updated classified CSVs — *skipped*

---

## Phase 1 — Infrastructure ✅ COMPLETE

### Providers
- [x] `src/providers/gemini.py` — gemini-2.5-flash + gemini-3.1-pro-preview; updated with `expect_json`, `call_multiturn`
- [x] `src/providers/gemma.py` — gemma-3-12b-it (local, transformers, `call_with_logprobs`, 4-bit BnB optional)
- [x] `src/providers/llama.py` — DeepSeek-R1-Distill-Llama-70B (local, device_map=auto, `call_with_logprobs`, 4-bit default, `<think>` stripping; swapped from Llama-3.3-70B which is gated)
- [x] `src/providers/qwen.py` — Qwen3-4B (local, `enable_thinking=False`, `call_with_logprobs`, validated in pre-flight)
- [x] `src/providers/base.py` — updated with `call_with_logprobs()`, `call_multiturn()`, `supports_logprobs` property
- [ ] Smoke test: each provider completes 1 structured-output call *(run on GPU node)*

### Robust JSON handling
- [x] `src/parsing.py` — `parse_with_schema()` with 4-step retry ladder (direct → fence-strip → brace-extract → LLM self-repair)
- [x] Smoke tested: all 4 paths verified, missing-keys → None confirmed

### UQ utilities
- [x] `src/uq/verbalized.py` — `extract_confidence()`, normalises 0–100 to 0.0–1.0
- [x] `src/uq/logprob_entropy.py` — `token_entropy()`, `mean_entropy()`, `response_entropy_stats()`
- [x] `src/uq/semantic_entropy.py` — `SemanticEntropyEstimator` (NLI clustering, stretch goal)

### WandB
- [x] `src/tracking.py` — `WandBTracker` + `make_tracker()` convenience factory
- [x] Smoke tested in disabled mode; live test deferred to GPU node

### Dataset curation
- [x] `datasets/sharc/sharc_200.jsonl` — 200 records with `context_essay` merged from cache
- [x] `scripts/preprocess_medqa.py` → `datasets/medqa/medqa_200.jsonl` — 200 records, all 3 context layers, difficulty-stratified (71 easy / 122 medium / 7 hard)
- [x] `datasets/msdialog/msdialog_200.jsonl` — done in Phase 0.3 (1430 chars/case avg)

### Pipelines
- [x] `src/pipelines/single_turn.py` — `SingleTurnPipeline` (all 3 datasets, forced CQ, WandB + CSV)
- [x] `src/pipelines/flex_turn.py` — `FlexTurnPipeline` (all 3 datasets, optional CQ, WandB + CSV)
- [x] `src/pipelines/simulator.py` — already done in Phase 0.3

### Prompts
- [x] `prompts/medqa/single_turn.txt`, `flex_turn.txt`
- [x] `prompts/msdialog/single_turn.txt`, `flex_turn.txt`
- [x] `prompts/sharc/single_turn.txt`, `flex_turn.txt`

**Notes:**
- Local providers (Gemma, Llama, Qwen) need GPU node to smoke test — load + 1 call each
- WandB live test deferred to first dry run (Phase 2)
- Semantic entropy (stretch) requires `transformers` + NLI model on inference node

---

## Phase 2 — Experiment runs

### Dry runs (5 records each — all configs before full runs)
- [ ] MedQA × gemini-2.5-flash × single-turn
- [ ] MedQA × gemini-2.5-flash × flex
- [ ] MedQA × gemini-3.1-pro-preview × single-turn
- [ ] MedQA × gemini-3.1-pro-preview × flex
- [ ] MedQA × gemma-3-12b-it × single-turn
- [ ] MedQA × gemma-3-12b-it × flex
- [ ] MedQA × deepseek-r1-distill-70b × single-turn
- [ ] MedQA × deepseek-r1-distill-70b × flex
- [ ] MedQA × qwen3-4B × single-turn- [ ] MedQA × qwen3-4B × flex- [ ] MS-Dialog × gemini-2.5-flash × single-turn
- [ ] MS-Dialog × gemini-2.5-flash × flex
- [ ] MS-Dialog × gemini-3.1-pro-preview × single-turn
- [ ] MS-Dialog × gemini-3.1-pro-preview × flex
- [ ] MS-Dialog × gemma-3-12b-it × single-turn
- [ ] MS-Dialog × gemma-3-12b-it × flex
- [ ] MS-Dialog × deepseek-r1-distill-70b × single-turn
- [ ] MS-Dialog × deepseek-r1-distill-70b × flex
- [ ] MS-Dialog × qwen3-4B × single-turn- [ ] MS-Dialog × qwen3-4B × flex- [ ] ShARC × gemini-2.5-flash × single-turn
- [ ] ShARC × gemini-2.5-flash × flex
- [ ] ShARC × gemini-3.1-pro-preview × single-turn
- [ ] ShARC × gemini-3.1-pro-preview × flex
- [ ] ShARC × gemma-3-12b-it × single-turn
- [ ] ShARC × gemma-3-12b-it × flex
- [ ] ShARC × deepseek-r1-distill-70b × single-turn
- [ ] ShARC × deepseek-r1-distill-70b × flex
- [ ] ShARC × qwen3-4B × single-turn- [ ] ShARC × qwen3-4B × flex
### Full runs (200 records each)
- [ ] MedQA × gemini-2.5-flash × single-turn
- [ ] MedQA × gemini-2.5-flash × flex
- [ ] MedQA × gemini-3.1-pro-preview × single-turn
- [ ] MedQA × gemini-3.1-pro-preview × flex
- [ ] MedQA × gemma-3-12b-it × single-turn
- [ ] MedQA × gemma-3-12b-it × flex
- [ ] MedQA × deepseek-r1-distill-70b × single-turn
- [ ] MedQA × deepseek-r1-distill-70b × flex
- [ ] MedQA × qwen3-4B × single-turn- [ ] MedQA × qwen3-4B × flex- [ ] MS-Dialog × gemini-2.5-flash × single-turn
- [ ] MS-Dialog × gemini-2.5-flash × flex
- [ ] MS-Dialog × gemini-3.1-pro-preview × single-turn
- [ ] MS-Dialog × gemini-3.1-pro-preview × flex
- [ ] MS-Dialog × gemma-3-12b-it × single-turn
- [ ] MS-Dialog × gemma-3-12b-it × flex
- [ ] MS-Dialog × deepseek-r1-distill-70b × single-turn
- [ ] MS-Dialog × deepseek-r1-distill-70b × flex
- [ ] MS-Dialog × qwen3-4B × single-turn- [ ] MS-Dialog × qwen3-4B × flex- [ ] ShARC × gemini-2.5-flash × single-turn
- [ ] ShARC × gemini-2.5-flash × flex
- [ ] ShARC × gemini-3.1-pro-preview × single-turn
- [ ] ShARC × gemini-3.1-pro-preview × flex
- [ ] ShARC × gemma-3-12b-it × single-turn
- [ ] ShARC × gemma-3-12b-it × flex
- [ ] ShARC × deepseek-r1-distill-70b × single-turn
- [ ] ShARC × deepseek-r1-distill-70b × flex
- [ ] ShARC × qwen3-4B × single-turn- [ ] ShARC × qwen3-4B × flex
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
