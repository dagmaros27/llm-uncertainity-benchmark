# Weekly Research Report — 30 April 2026

> **Project:** LLM Uncertainty Benchmark — Applying CLAMBER to Clinical Question Answering  
> **Branch:** `new_pilot_study`  
> **Report covers:** Initial setup through end of Week 1

---

## 1. Project Overview

The goal of this pilot study is to investigate whether LLMs exhibit **calibrated uncertainty** in clinical decision-making, and to classify the *type* of uncertainty driving each clarifying question (CQ) a clinician-LLM asks before committing to a diagnosis or treatment recommendation.

We adopt the **CLAMBER taxonomy** as the theoretical backbone:

| Category | Subclass | Description |
|---|---|---|
| **EPISTEMIC** | NK (Not Known) | Model lacks knowledge of an entity |
| **EPISTEMIC** | ICL (Inconsistent/Contradictory Labels) | Presentation contains contradictions |
| **ALEATORIC** | Polysemy | Term has multiple valid clinical meanings |
| **ALEATORIC** | Co-reference | Ambiguous pronoun or referent |
| **ALEATORIC** | Whom | Answer depends on patient-specific preference |
| **ALEATORIC** | What | Underspecified request |
| **ALEATORIC** | When | Ambiguous temporal scope |
| **ALEATORIC** | Where | Ambiguous spatial scope |

**Epistemic** uncertainty is reducible — gathering more information fully resolves it.  
**Aleatoric** uncertainty is irreducible — it stems from inherent ambiguity that cannot be eliminated by any external fact.

---

## 2. Infrastructure & Setup

### 2.1 Repository Reorganisation

Early sessions established the project structure:

```
pilot_study/
├── config.py              # Global settings (model, seeds, field schemas)
├── datasets/
│   ├── clamber/           # CLAMBER benchmark cases
│   └── medqa/
│       ├── cases.jsonl          # Full dataset (1,273 cases)
│       ├── unseen_100.jsonl     # Held-out: 100 easy cases (Experiment 1)
│       └── multiturn_100.jsonl  # Held-out: 50 easy / 30 medium / 20 hard (Experiments 2–3)
├── notebooks/
│   ├── clamber/           # Judge calibration notebook
│   └── medqa/             # Phase 1 generation + judge classification notebooks
├── outputs/
│   └── medqa/
│       └── gemini-2.5-flash/    # All outputs namespaced by model
├── prompts/medqa/         # System instructions for each pipeline role
└── src/
    ├── pipeline.py        # Phase1Pipeline + MultiTurnPhase1Pipeline
    ├── providers.py       # GeminiProvider (gemini-2.5-flash, v1beta)
    ├── judge.py           # LLMJudge + CSVBatchClassifier
    └── utils.py           # JSON parsing, formatting helpers
```

Outputs are namespaced under `outputs/medqa/<model-id>/` so that future experiments with different models (e.g., GPT-4o, Claude) write to separate directories without overwriting each other.

### 2.2 Model

All experiments use **gemini-2.5-flash** (a thinking model) via the Google GenAI API (`v1beta`). Structured JSON output is enforced through the Gemini `responseSchema` / enum constraints, ensuring clean letter choices (A/B/C/D) rather than free text.

Key parameter decisions:
- `max_tokens = 4096` for Turn 0 and Turn 1 (thinking model consumes reasoning budget before output; lower values produce truncated JSON)
- `temperature = 0.0` for all calls (fully deterministic)
- `request_interval = 1.0s` between calls

---

## 3. CLAMBER Judge Calibration

Before running MedQA experiments we validated the LLM judge on the CLAMBER benchmark itself.

### 3.1 Setup

- **Eval set:** 200 cases (100 per class: EPISTEMIC / ALEATORIC), drawn from the CLAMBER benchmark pool
- **Exclusion:** The 8 hand-crafted few-shot examples were explicitly excluded from the eval pool before sampling (leakage check via `assert len(overlap) == 0`)
- **Two conditions:** Zero-shot baseline vs. 8-shot few-shot (one example per CLAMBER subclass)

### 3.2 Results

| Condition | Accuracy |
|---|---|
| Zero-shot baseline | 68.0% |
| 8-shot few-shot | **94.5%** |
| Improvement | +26.5pp |

The few-shot examples covering all 8 CLAMBER subclasses gave a large and consistent accuracy boost. This validates the judge as reliable enough for downstream classification of MedQA clarifying questions.

**Note on data integrity:** An early run had a data leakage bug (5 of 6 few-shot examples were present in the eval pool). This was caught, fixed with a runtime assert, and the experiment was rerun cleanly on the corrected 200-case eval set.

---

## 4. MedQA Pilot — Methodology

### 4.1 Dataset

**Source:** A custom clinical MCQ dataset. Each case contains:

| Field | Description |
|---|---|
| `ehr_summary` | Sparse presenting complaint (1–2 sentences, no vitals or labs) |
| `question` | Clinical MCQ task (e.g., "What is the most appropriate next step?") |
| `options` | Four answer choices (A–D) |
| `correct_option` | Ground-truth letter |
| `correct_answer` | Ground-truth text |
| `patient_context` | Detailed patient history (for simulator) |
| `nurse_context` | Nursing observations (for simulator) |
| `specialist_context` | Specialist notes (for simulator) |
| `full_patient_context` | Oracle context including teaching point — **deliberately excluded** from simulator to prevent answer leakage |

**Simulator context** = `patient_context + nurse_context + specialist_context` (concatenated). The `full_patient_context` contains an explicit teaching point that reveals the correct diagnosis, so it is never shown to the simulator.

### 4.2 Two Held-Out Sets (No Overlap)

| Dataset | n | Difficulty | Used for |
|---|---|---|---|
| `unseen_100.jsonl` | 100 | 100% easy | Experiment 1 (easy-only baseline) |
| `multiturn_100.jsonl` | 100 | 50% easy / 30% medium / 20% hard | Experiments 2 & 3 (controlled comparison) |

Zero overlap between the two sets is verified at runtime:  
```python
assert len(new_ids & old_ids) == 0
```

### 4.3 Prompts & Roles

#### Clinician LLM — Turn 0 (Phase 1 Instruction)

The model acts as an experienced clinician. It is shown the sparse EHR summary, the clinical question, and the answer options. It must return:

1. **One clarifying question** — targeted at discriminating between the answer options  
2. **Preliminary answer** — exactly one letter: A, B, C, or D  
3. **Confidence** — 0–100

> *"ask exactly one clarifying question that would most help you choose between the answer options"*

This framing was deliberately chosen to avoid epistemic bias. An earlier version of the instruction said *"identify the most important missing clinical information"*, which systematically biased the model toward asking epistemic (fact-seeking) questions. The current framing is open to both epistemic and aleatoric CQs.

#### Patient Simulator

A separate LLM call acting as a clinical information source. It receives the combined `simulator_context` and the clinician's CQ. It answers using **only** information present in the context and returns a 1–2 sentence response. It will say *"That information is not available"* if the CQ targets something not in the context.

#### Clinician LLM — Turn 1 (Final Assessment)

Sees the full context: EHR + question + options + CQ exchange. Returns a final letter choice (A/B/C/D) and updated confidence.

#### Continuation Instruction (Multi-turn only)

For turns 2 and 3, the model sees the **full accumulated history** of all previous CQs and simulator responses. It must:
1. Update its answer (A/B/C/D)
2. Ask a **new** clarifying question — explicitly forbidden from repeating any previous question

> *"Do not repeat any question you have already asked. The new question should target a different aspect of the case."*

#### LLM Judge

A separate judge call classifies each CQ as EPISTEMIC or ALEATORIC using 8 few-shot examples (one per CLAMBER subclass, adapted to clinical language). The judge never sees the simulator response or the model's answer — only the clarifying question text.

### 4.4 Correctness Metric

Both preliminary and updated answers are scored as **exact letter match**:

```python
is_correct = answer.upper() == correct_option.upper()
```

This is clean and unambiguous. Earlier experiments used substring matching on free-text assessments (when options were not shown in Turn 0), which was an unreliable lower bound.

---

## 5. Results

### 5.1 Experiment 1 — Easy-Only Dataset (`unseen_100.jsonl`, Single-Turn)

**Pipeline design iteration:** Two sub-experiments on the same 100 easy cases.

#### v1 — Options NOT shown in Turn 0

The model generates a free-text preliminary assessment without seeing the answer choices. Correctness was checked via substring match (unreliable).

| Metric | Value |
|---|---|
| Preliminary correct (substring, unreliable) | 19% |
| Updated correct (after CQ + options shown) | 76% |
| Mean confidence delta | +31.7 |

**Problem identified:** Without seeing the options, the model asked clinically reasonable but *option-irrelevant* CQs. For example, when the question was about disclosing a medical error (ethics), the model asked about prior carpal tunnel treatments. It had no way to target its question toward what would discriminate between A/B/C/D.

#### v2 — Options shown in Turn 0 (Current Design)

| Metric | Value |
|---|---|
| Preliminary correct (exact letter match) | 76% |
| Updated correct (after CQ) | **82%** |
| Judge: EPISTEMIC | 96 / 100 |
| Judge: ALEATORIC | 4 / 100 |
| Errors | 0 |

Showing options in Turn 0 improved post-CQ accuracy (+6pp) and made the preliminary metric meaningful (exact letter match instead of unreliable substring).

---

### 5.2 Experiments 2 & 3 — Mixed Difficulty, Controlled Comparison (`multiturn_100.jsonl`)

Same 100 cases run through both single-turn and multi-turn pipelines for a direct comparison.

#### Single-Turn (1 CQ round)

| Metric | Value |
|---|---|
| Preliminary correct | 63% |
| Updated correct (after CQ1) | 73% |
| Improvement | +10pp |
| Judge: EPISTEMIC | 98 / 100 |
| Judge: ALEATORIC | 2 / 100 |
| Errors | 0 |

#### Multi-Turn (3 CQ rounds)

| Checkpoint | Accuracy | Δ from previous |
|---|---|---|
| Preliminary (Turn 0) | 61% | — |
| After CQ1 | 75% | +14pp |
| After CQ2 | 80% | +5pp |
| After CQ3 (Final) | 79% | −1pp |

| Judge metric | Value |
|---|---|
| Total CQs classified | 300 (100 cases × 3) |
| EPISTEMIC | 297 / 300 (99%) |
| ALEATORIC | 3 / 300 (1%) |
| Errors | 0 |

#### Direct Comparison (Same Dataset)

| | Single-turn | Multi-turn |
|---|---|---|
| Preliminary | 63% | 61% |
| After CQ1 | **73%** | **75%** |
| After CQ2 | — | **80%** |
| After CQ3 | — | 79% |
| EPISTEMIC % | 98% | 99% |

---

## 6. Discussion & Findings

### 6.1 Value of Showing Options in Turn 0

The single most impactful design change was showing the answer options to the model before it asks its clarifying question. Without options, the model targets clinically plausible but option-irrelevant information. With options, it asks questions that discriminate between the specific choices — this is what actually improves post-CQ accuracy.

### 6.2 Diminishing Returns Across CQ Rounds

The multi-turn experiment reveals a clear diminishing-returns pattern:

- **CQ1 → CQ2:** +5pp gain — there is still useful information to extract on the second question
- **CQ2 → CQ3:** −1pp — effectively noise at this sample size; the third question adds nothing on average

This suggests that **2 CQ rounds is the practical optimum** for this task. A third round does not justify the additional latency and API cost.

### 6.3 Epistemic Dominance in the Medical MCQ Domain

Across all three experiments, EPISTEMIC CQs account for 96–99% of all clarifying questions. This is a robust finding and is scientifically coherent:

- Medical MCQs are constructed so that the correct answer is determinable from complete clinical information
- The sparse EHR summary deliberately omits key facts (vitals, labs, history)
- The natural response to a sparse clinical presentation is to ask for those missing facts — an epistemic act

Aleatoric questions (targeting genuine ambiguity in the patient's language or preferences) are rare in this setting because the MCQ format has a single correct answer. This finding is dataset-specific and may differ in open-ended clinical reasoning tasks.

### 6.4 Effect of Difficulty Mix

Comparing Experiment 1 (100% easy) to Experiments 2 & 3 (50% easy / 30% medium / 20% hard):

| | Easy-only | Mixed difficulty |
|---|---|---|
| Single-turn preliminary | 76% | 63% |
| Single-turn post-CQ | 82% | 73% |
| Drop | — | −9pp |

The 9pp accuracy drop from adding medium and hard cases is expected. It also confirms the easy-only results were not representative of real-world performance.

---

## 7. Next Steps

- [ ] Analyse per-difficulty accuracy breakdown (easy vs medium vs hard separately) in multi-turn
- [ ] Investigate whether ALEATORIC CQs are more or less effective at improving accuracy than EPISTEMIC ones
- [ ] Run experiments with a second model (e.g., GPT-4o or Claude) using the same datasets for cross-model comparison
- [ ] Consider 2-turn pipeline as the production design given diminishing returns at turn 3
- [ ] Begin writing up the methodology section for the formal paper

---

*Report generated: 30 April 2026 | Model: gemini-2.5-flash | Branch: new_pilot_study*
