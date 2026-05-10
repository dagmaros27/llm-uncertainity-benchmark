# Uncertainty Benchmark — Project Plan

This file lives in the project root (`D:/final_project/`) and governs the new
**uncertainty benchmark** project. The existing `pilot_study/` stays as a frozen
reference; the benchmark project takes useful code from it but is structured
cleanly from day one for GCP execution.

---

## 1. Project objective

Measure whether LLMs **recognise their own uncertainty** and ask **appropriate
clarifying questions** before answering. Decompose uncertainty into:

- **Epistemic** — fact gaps, reducible by one factual answer
- **Aleatoric** — interpretive ambiguity in the user's own framing, often irreducible

Compare across:
- **Datasets** (3): MedQA, MS-Dialog, ShARC
- **Models** (4 core + 1 conditional): gemini-2.5-flash, gemini-3.1-pro-preview (API), gemma-3-12b-it (local GCP), llama-3.3-70b (local GCP), **qwen3-4B (local GCP — enters matrix only if it passes the Phase 0 Colab pre-flight)**
- **Generation methods** (core): single-turn (1 CQ then answer) + flex-turn (0–3 CQs, model decides when to stop)
- **UQ methods**: verbalized confidence (all models) + token/semantic entropy (open models only)

**Judge:** gemini-3.1-pro-preview is used as the LLM judge. Using the same model
family for both experiments and judging is acknowledged as a limitation and
documented as a design caution (see section 5). The few-shot constraint on the
judge prompt mitigates free-form reasoning bias. This is the pragmatic budget
decision and will be stated explicitly in the paper.

Output: one cross-model × cross-domain benchmark with validated components,
WandB-tracked runs, robustness to API failures, and analysis-ready CSVs.

---

## 2. Current state — what exists in `pilot_study/`

### Reusable as-is or with light edits
| Component | Path | Status |
|---|---|---|
| `GeminiProvider`, `GemmaProvider` | `src/providers.py` | Reuse, extend with `LlamaProvider` |
| `LLMJudge`, `CSVBatchClassifier` | `src/judge.py` | Reuse, rewrite prompt + few-shot |
| `UserSimulator` | `src/pipeline.py` | Reuse, validate in Phase 0 |
| Flex pipeline classes (MS-Dialog, ShARC) | `src/pipeline.py` | Reuse pattern, generalise |
| Schemas (Gemini structured output) | `src/pipeline.py` | Reuse |
| Specialist prompts per dataset | `prompts/<dataset>/` | Reuse, audit |
| Curated datasets | `datasets/sharc/sharc_200.jsonl`, etc. | Reuse, scale to 200 each |
| Context essay synthesis | `scripts/build_sharc_context.py` | Reuse pattern for any dataset |
| ShARC preprocessor | `scripts/preprocess_sharc.py` | Reuse |
| Analysis utilities (ROUGE-L, calibration plots) | `experiments/<dataset>/analyze_*.py` | Reuse functions, restructure |

### Needs rewriting
| Component | Reason |
|---|---|
| Judge prompt + few-shot | Misses aleatoric WH-questions and "do you specifically..." patterns |
| JSON parsing | No schema validation, no retry-on-malformed, no fallback ladder |
| Logprob extraction | Not implemented |
| Semantic entropy computation | Not implemented |
| Per-experiment runners | One file per (model × method) is wasteful — replace with one runner per dataset that loops |
| WandB | Not integrated |
| Resume/checkpoint | CSV-based only, fragile |
| Config | Hard-coded paths, no env-var overrides for GCP |

### Drop entirely
- Single-turn and multi-turn pipelines from pilot — superseded by unified flex runner
- `pilot_study/notebooks/` — keep as historical reference but don't migrate

---

## 3. Target benchmark project layout

```
uncertainty_benchmark/                      # new folder, GCP-deployable
├── README.md
├── requirements.txt                        # frozen for reproducibility
├── pyproject.toml
├── .env.example                            # GEMINI_API_KEY, WANDB_API_KEY, HF_TOKEN
├── config/
│   ├── default.yaml                        # all knobs in one place
│   └── models.yaml                         # provider configs, max_tokens, temperatures
├── src/
│   ├── __init__.py
│   ├── providers/
│   │   ├── base.py                         # LLMProvider ABC
│   │   ├── gemini.py                       # gemini-2.5-flash, gemini-3.1-pro-preview (API)
│   │   ├── gemma.py                        # gemma-3-12b-it (local, via transformers)
│   │   ├── llama.py                        # Llama-3.3-70B (local, via transformers)
│   │   └── qwen.py                         # Qwen-4B (local, conditional on pre-flight)
│   ├── pipelines/
│   │   ├── base.py                         # shared SpecialistPipeline ABC
│   │   ├── flex_turn.py                    # core pipeline (0–3 CQs)
│   │   ├── single_turn.py                  # core pipeline (1 CQ)
│   │   └── simulator.py                    # UserSimulator
│   ├── judge/
│   │   ├── classifier.py                   # LLMJudge
│   │   └── validation.py                   # CLAMBER scoring, kappa, F1
│   ├── uq/
│   │   ├── verbalized.py                   # extract verbalized confidence
│   │   ├── logprob_entropy.py              # token entropy (open models)
│   │   └── semantic_entropy.py             # N-sample clustering (open models)
│   ├── schemas.py                          # all JSON schemas in one place
│   ├── parsing.py                          # robust JSON extraction with retry/fallback
│   ├── tracking.py                         # WandB wrapper
│   └── utils.py
├── prompts/
│   ├── medqa/
│   ├── msdialog/
│   ├── sharc/
│   └── judge.txt                           # single judge prompt across datasets
├── datasets/
│   ├── medqa/medqa_200.jsonl
│   ├── msdialog/msdialog_200.jsonl
│   └── sharc/sharc_200.jsonl
├── scripts/
│   ├── preprocess_medqa.py
│   ├── preprocess_msdialog.py
│   ├── preprocess_sharc.py
│   ├── build_simulator_contexts.py
│   └── validate_judge_clamber.py
├── experiments/
│   ├── run_medqa.py                        # loops all model configs × 200 records
│   ├── run_msdialog.py
│   ├── run_sharc.py
│   └── run_judge.py                        # classifies CQs from all experiment outputs
├── analysis/
│   ├── load.py                             # load + merge CSVs across runs
│   ├── metrics.py                          # accuracy, recovery, calibration, etc.
│   ├── cross_model.py                      # cross-model figures
│   ├── cross_domain.py                     # cross-domain figures
│   ├── entropy_decomposition.py            # token vs semantic entropy plots
│   └── notebooks/
│       └── final_report.ipynb
├── outputs/
│   ├── medqa/<model>/results.csv
│   ├── msdialog/<model>/results.csv
│   └── sharc/<model>/results.csv
├── figures/                                # all figures saved as PDF
├── logs/
└── colab/
    └── small_model_preflight.ipynb         # Phase 0.0 deliverable
```

---

## 4. Phases

The chain `specialist → simulator → judge` has three sensitive components.
Every result depends on all three being trustworthy. **Phase 0 validates all
three before any experiments run.** It is a hard blocker.

---

### Phase 0 — Pre-flight + Validate trusted components (BLOCKER for everything else)

**Goal:** verify open-model viability (Colab), then establish that the judge
and simulator are reliable enough to trust downstream results.

#### 0.0 Colab small-model pre-flight notebook
Run this in Google Colab (free T4 tier) before building any infrastructure.

**Purpose:** understand whether open models can comply with our JSON schemas and
whether logprob extraction works — so the infrastructure in Phase 1 is designed
around known facts, not guesses.

**Work:**
- Load **Qwen2.5-4B-Instruct** via `transformers` on a Colab T4
- Test the flex-turn JSON schema (5 keys + boolean `ask_cq`) and single-turn
  schema (3 keys) on toy inputs
- Probes:
  - Native JSON compliance without constraints
  - Grammar/regex-constrained decoding (`outlines` or `transformers` `JSONLogitsProcessor`)
  - Token budget: at what `max_tokens` does the model truncate mid-JSON?
  - Logprob extraction via `model.generate(..., output_scores=True, return_dict_in_generate=True)` — confirm we can extract top-k per step
  - 3-sample generation at T=0.7 for semantic entropy feasibility
- Document all failure modes and recommended per-model settings

**Gate — Qwen-4B enters the main model matrix if and only if:**
- It can produce valid schema-compliant JSON (natively or with constraints)
- Logprob extraction works and returns meaningful per-token distributions
- It runs within Colab T4 memory limits

**Deliverable:** `colab/small_model_preflight.ipynb` — runnable end-to-end on
Colab T4 free tier, with a brief findings cell at the end.

---

#### 0.1 Fix the judge prompt
- Rewrite `prompts/judge.txt` to capture missing aleatoric patterns:
  - WH-specification questions ("Which X are you referring to?")
  - "Do you specifically…", "Are you referring to…"
  - Option-listing ("Is it A, B, or C?") when options reflect *interpretation* not facts
- Add adversarial few-shot pairs: same surface form, different label, one short
  explanation per pair (one pair per CLAMBER subclass)

#### 0.2 Validate judge accuracy on CLAMBER
CLAMBER has gold epistemic/aleatoric labels — use them.
- Run judge at T=0 on full CLAMBER labelled set
- Report: accuracy, per-class precision/recall/F1, confusion matrix
- Run 5 batches at T=0.0 / 0.3 / 0.5 → measure self-consistency rate
- Compute Cohen's κ vs gold across runs
- Save validation report to `analysis/judge_validation_report.md`

**Acceptance gate (must pass before Phase 2):**
- Accuracy ≥ 85% on CLAMBER
- Cohen's κ ≥ 0.70 vs gold
- Per-class F1 ≥ 0.80 for both EPISTEMIC and ALEATORIC

If gate not met: revise the prompt and re-run. There is no plan B — the judge
must pass or the project pauses until it does.

#### 0.3 Validate the simulator
- Spot-check 20 cases per dataset (60 total): does the simulator answer
  correctly from the context essay? Does it hedge when it should?
- Compute hedge rate per dataset (proportion of "that information is not
  available" responses)
- Sanity bounds: hedge rate should be < 20% overall; a systematic hedge on
  a specific question type is a red flag requiring prompt revision
- Flag any case where the simulator's answer contradicts layer 1 content

**Acceptance gate:**
- Hedge rate < 20% per dataset
- No systematic contradiction with layer 1 content in the spot-check sample

#### 0.4 Re-run validated judge on existing pilot CSVs
- Rerun the Phase 0.2-validated judge on MS-Dialog and ShARC pilot outputs
- Replaces the current potentially biased CQ-type results in those files

**Deliverables:**
- New `prompts/judge.txt` + few-shot list in `src/judge/classifier.py`
- `scripts/validate_judge_clamber.py`
- `outputs/judge_validation/` — confusion matrices, kappa table, self-consistency
- `analysis/judge_validation_report.md`
- `outputs/simulator_validation/` — per-dataset hedge rates, spot-check CSV

---

### Phase 1 — Infrastructure

**Goal:** all building blocks the experiment runners need.

1. **Provider layer**
   - `GeminiProvider`: gemini-2.5-flash and gemini-3.1-pro-preview via API key
   - `GemmaProvider`: gemma-3-12b-it loaded **locally on GCP GPU VM** via `transformers`;
     exposes logprobs via `output.scores`
   - `LlamaProvider`: Llama-3.3-70B loaded **locally on GCP GPU VM** via `transformers`;
     exposes logprobs via `output.scores`; same interface as GemmaProvider
   - `QwenProvider` (conditional): Qwen-4B loaded **locally on GCP GPU VM** — only
     built if Qwen passed the Phase 0.0 Colab pre-flight
   - Standard interface: `call(system, user, schema, max_tokens, temperature, return_logprobs)`
   - All providers expose the same `Response` object: `text`, `finish_reason`, optional `logprobs`

2. **Robust JSON handling** (`src/parsing.py`)
   - `parse_with_schema(raw, schema)` → validated dict or raises `ParseError`
   - Retry ladder: schema-validate → 1 reprompt with error message → regex extraction → log as `parse_error` row

3. **UQ utilities** (`src/uq/`)
   - `verbalized.py`: pull `confidence` from parsed JSON (all models)
   - `logprob_entropy.py`: per-token entropy using top-k logprobs (open models: Gemma, Llama)
   - `semantic_entropy.py`: N-sample (default 5) at T=0.7, cluster by string match or NLI (stretch)

4. **WandB tracking** (`src/tracking.py`)
   - One run = (dataset × model)
   - Log config, per-record metrics streamed, summary at end

5. **Curate datasets to 200 each**
   - `sharc_200.jsonl` already done
   - Write `preprocess_medqa.py` → `medqa_200.jsonl`
   - Write `preprocess_msdialog.py` → `msdialog_200.jsonl`

**Acceptance:**
- Each provider passes a smoke test (1 structured-output call)
- Schema-validation + retry ladder catches a deliberately broken response
- WandB run appears in the dashboard with config, metrics, summary
- Gemma loads locally and returns logprobs on a test input

---

### Phase 2 — Experiment runs (core)

**Structure:** one runner per dataset, loops through 4 model configurations.

**Core matrix (must finish):**

| Models | Access | Methods |
|---|---|---|
| gemini-2.5-flash | API | single-turn + flex |
| gemini-3.1-pro-preview | API | single-turn + flex |
| gemma-3-12b-it | local GCP | single-turn + flex |
| llama-3.3-70b | local GCP | single-turn + flex |
| **qwen3-4B** | **local GCP** | **single-turn + flex (if pre-flight passed)** |

**3 datasets × 4 models × 2 methods = 24 runs** (minimum).
**+ Qwen-4B if pre-flight passes → 3 datasets × 5 models × 2 methods = 30 runs.**

**Run protocol:**
1. **5-record dry run** for all 12 configurations sequentially
2. Inspect: JSON parse success rate, token usage, simulator hedge rate on dry run, no crashes
3. **Stop and fix** if any config fails any check
4. **Full 200-record run** — sequential, each resumable from CSV checkpoint
5. WandB logs per run; one parent group per dataset

**For Gemma and Llama specifically:** capture top-k logprobs per generation
step in a parallel `<run>_logprobs.jsonl` cache for Phase 4 entropy analysis.

**Deliverables:**
- `outputs/<dataset>/<model>/<method>/results.csv` — 24 CSVs total
- 24 WandB runs grouped by dataset
- For Gemma + Llama: 12 logprobs caches (3 datasets × 2 open models × 2 methods)

**Acceptance:**
- ≥ 95% non-error rows per run (≤ 10 parse errors / 200)
- All 24 runs land cleanly in WandB
- Resume from any checkpoint produces identical row count

**Stretch (only if Phase 2 core is complete and time allows):**
- Multi-seed variance bars on 2–3 critical figures
- Cross-domain transfer analysis

---

### Phase 3 — Judge classification

**Work:**
- `experiments/run_judge.py` reads every Phase 2 CSV, extracts non-empty CQs
  (0–3 per case for flex), classifies each with the Phase 0-validated judge,
  writes `<run>_classified.csv`

**Deliverables:**
- 12 classified CSVs
- One summary CSV: `outputs/cq_types_summary.csv` (model × dataset × type → count)

**Acceptance:**
- All non-empty CQs classified (no skips)

---

### Phase 4 — Analysis

All figures saved as PDF to `figures/`.

**Required (must produce — these are the deliverable):**

| Analysis | Output |
|---|---|
| Cross-model accuracy table per dataset (single-turn + flex) | `figures/accuracy_matrix.pdf` + CSV |
| Single-turn vs flex comparison per model × dataset | `figures/method_comparison.pdf` |
| CQ-asking rate by model × dataset (flex only) | `figures/cq_asking_rate.pdf` |
| Recovery rate (prelim wrong → final right) by model × dataset | `figures/recovery.pdf` |
| Confidence calibration (final conf bin → accuracy) per model × dataset | `figures/calibration_grid.pdf` |
| CQ type distribution (epistemic / aleatoric) per model × dataset | `figures/cq_types.pdf` |
| Verbalized-confidence ↔ correctness correlation (Pearson, Spearman) | `figures/conf_correctness.pdf` + table |

**Open-model-only (Gemma + Llama):**

| Analysis | Output |
|---|---|
| Token entropy distribution per CQ type | `figures/token_entropy_by_type.pdf` |
| Entropy decomposition map: X=token entropy, Y=semantic entropy, color=judge label | `figures/entropy_decomposition.pdf` |
| UQ-method comparison: verbalized vs token entropy vs semantic entropy → AUC | `figures/uq_method_comparison.pdf` + AUC table |

**Stretch (only if everything else green):**
- Single-turn vs flex comparison per model × dataset (requires stretch Phase 2 runs)
- Cross-domain transfer trends
- Per-rule analysis on ShARC

**Deliverables:**
- All required figures (PDF) + CSVs
- `analysis/notebooks/final_report.ipynb` synthesising findings
- `BENCHMARK_REPORT.md` at project root with all required figures embedded

---

## 5. Cross-cutting concerns

### Judge + same-model-family bias (design caution)
The judge (`gemini-3.1-pro-preview`) shares a model family with one of the
evaluated models (`gemini-2.5-flash`, `gemini-3.1-pro-preview`). This creates
a potential systematic bias: the judge may parse Gemini-family phrasings more
reliably than open-model phrasings. Mitigations:
- The judge uses a constrained few-shot prompt with fixed classification
  criteria — this limits free-form reasoning and reduces stylistic bias
- This limitation is documented explicitly in the benchmark paper

### WandB
- One **project**: `uncertainty-benchmark`
- One **group** per dataset (`group=medqa`)
- One **run** per (dataset × model): `name=medqa_llama70b_flex`
- Tags: `phase=experiment`, `model_size=70b`
- Logs: every parsed JSON output, per-row latency, finish_reason, parse errors

### Robustness to failures
- Every API call wrapped in `tenacity` retry (max 6, exponential backoff)
- Every CSV writer flushes after each row
- Resume by reading the existing CSV's `id` column on startup
- On unrecoverable error: log a `parse_error` row and continue (never crash)

### GCP execution
- `Dockerfile` based on `python:3.11-slim`; mount config + outputs
- Single GPU VM: both Gemma-3-12b and Llama-3.3-70B run locally — need sufficient VRAM
  - Gemma-3-12b: ~24 GB VRAM (L4 sufficient)
  - Llama-3.3-70B: ~140 GB VRAM in fp16 (A100 80GB × 2 or single H100 recommended); use 4-bit quantisation (`bitsandbytes`) if needed to fit on a single A100
  - Qwen-4B: ~8 GB VRAM (fits alongside other models or on any GPU)
- API models (Gemini) run from the same VM with no GPU requirement
- Environment variables: `GEMINI_API_KEY`, `WANDB_API_KEY`, `HF_TOKEN`
- Outputs to mounted GCS bucket via `gcsfs` or `gsutil rsync`

### Determinism
- All experiments at T=0 unless otherwise specified
- Semantic entropy sampling at T=0.7 with fixed `seed`
- All file IO UTF-8

---

## 6. Scope policy

### Minimum complete (defines "done")
- Phase 0: judge passes acceptance gate, simulator passes sanity check
- Phase 1: infrastructure working
- Phase 2: 3 datasets × 4 models × 2 methods = 24 runs
- Phase 3: all CQs classified
- Phase 4: all required figures, no stretch

### Cuts in priority order if time runs out
1. Drop semantic entropy (keep token entropy) → simplifies open-model UQ
2. Drop one Gemini variant → 18 runs instead of 24
3. Drop single-turn (keep flex) → 12 runs instead of 24
4. Drop Llama/DeepSeek → all-Google + Gemma matrix, no cross-provider comparison
5. Drop Gemma → API-only matrix, no logprob analysis at all

### Stretch (only if everything else green)
- Multi-seed variance bars on critical comparisons
- Cross-domain transfer analysis
- Per-rule analysis on ShARC

---

## 7. Migration tasks (pilot_study → uncertainty_benchmark)

In order:

1. **Copy + clean** `pilot_study/src/providers.py` → `src/providers/{gemini,gemma,llama,qwen}.py` (split per provider; qwen conditional)
2. **Copy + clean** `pilot_study/src/judge.py` → `src/judge/classifier.py`
3. **Extract + generalise** flex pipeline → `src/pipelines/flex_turn.py` (parameterise over dataset schemas)
4. **Move** `UserSimulator` → `src/pipelines/simulator.py`
5. **Audit + migrate prompts**: keep specialist prompts, rewrite judge prompt (Phase 0)
6. **Migrate datasets**: copy `sharc_200.jsonl`; re-run preprocessing for medqa and msdialog at 200
7. **Reuse analysis utilities**: lift ROUGE-L scorer, calibration plotter, conf-correctness functions from pilot into `analysis/metrics.py`

**Do not migrate:**
- Old per-experiment runners — replaced by new unified runners
- Pilot notebooks
- Pilot logs / outputs (kept in `pilot_study/` as frozen reference)

---

## 8. Open decisions / flagged risks

| Question | Resolution |
|---|---|
| Llama-3.3-70B quantisation on GCP? | Try 4-bit (`bitsandbytes`) on a single A100 80GB first; upgrade to multi-GPU if quality degrades |
| Semantic entropy clustering method? | Start with string-equality clustering; upgrade to NLI-based only if needed |
| MedQA / MS-Dialog at 200 — same selection criteria as pilot? | Match pilot criteria (length, balance) but scale to 200; deduplicate by topic |
| Qwen-4B pre-flight outcome? | Determined in Phase 0.0 — if it fails JSON compliance or logprob extraction, it is dropped |
| Run on free GCP credits? | TBD — depends on budget; Llama-70B local is the main cost driver |

---

## 9. Phase ordering & dependencies

```
Phase 0 (Validate judge + simulator)  ──┐
                                        ├──→ Phase 1 (Infrastructure)
                                        │            │
                                        │            ▼
                                        │   Phase 2 (Experiment runs)
                                        │            │
                                        │            ▼
                                        └──→ Phase 3 (Judge classification)
                                                     │
                                                     ▼
                                            Phase 4 (Analysis)
```

Phase 0 must complete before Phase 1 starts (judge prompt is an input to the
infrastructure). Phase 4 can start partial once the first dataset's Phase 3
results are ready — don't wait for all 12 runs.

---

## 10. Definition of done

The project is complete when:
- `BENCHMARK_REPORT.md` exists at the project root with all required Phase 4 figures embedded
- All 24 (or fewer if scoped down) `results.csv` files are in `outputs/`
- Judge validation passes the acceptance gate, with the report committed
- Simulator validation spot-check passes, with hedge rates documented
- All experiment runs are logged and viewable in WandB
- The benchmark project runs end-to-end on GCP from a fresh checkout (documented in `README.md`)
