# Project Description: Benchmarking LLM Uncertainty Through Clarifying Questions

> **Note:** This file is a project overview. The authoritative implementation
> plan is in `PLAN.md` at the project root. If there is any conflict between
> this file and `PLAN.md`, trust `PLAN.md`.

---

## Research Goal

This project builds a benchmark for evaluating how LLMs express and manage
internal uncertainty by analysing the clarifying questions they generate when
faced with incomplete information. The core hypothesis: a model's clarifying
question is an observable diagnostic signal of its internal uncertainty state.

Two uncertainty types:

**Epistemic uncertainty** — the model lacks a specific fact. Answering the
clarifying question fully resolves the uncertainty. Example: "What is the
patient's current temperature?"

**Aleatoric uncertainty** — the problem is inherently ambiguous. Multiple valid
answers exist; no single fact resolves it — the user must make a choice.
Example: "Should the API return JSON or XML?"

The benchmark evaluates three things simultaneously: whether the model correctly
detects that clarification is needed, whether the type of CQ reflects the type
of uncertainty, and whether receiving the answer actually reduces uncertainty
(measured by confidence change and accuracy improvement).

---

## What Makes This Different from Existing Work

Most UQ methods measure uncertainty from outside the model (token probabilities,
output entropy, consistency across samples). CLAMBER (2024) taxonomises
ambiguity at the prompt level — classifying what is wrong with the user's input.
This project flips that paradigm: it classifies the model's uncertainty state by
reading the question the model generates. The same prompt can trigger epistemic
uncertainty in one model and aleatoric uncertainty in another. This captures
model-level variation that CLAMBER cannot.

---

## Domains and Datasets

Three datasets, each contributing 200 curated records:

- **MedQA** — clinical USMLE vignettes. Layer 0 = sparse presenting complaint.
  Layer 1 = hidden clinical details (vitals, labs, exam findings). Expected to
  produce predominantly epistemic CQs (missing objective clinical data).

- **MS-Dialog** — Microsoft product support dialogues. Layer 0 = initial
  underspecified user request. Layer 1 = hidden technical context and
  preferences. Expected to produce a mix of epistemic and aleatoric CQs.

- **ShARC** — regulatory eligibility QA (benefits, compliance). Layer 0 =
  underspecified eligibility question. Layer 1 = hidden rule conditions and
  user-specific facts. Expected to produce predominantly epistemic CQs.

---

## Model Matrix

| Model | Type | Access | Logprobs |
|---|---|---|---|
| gemini-2.5-flash | Closed | API key | No |
| gemini-3.1-pro-preview | Closed (judge + experiment) | API key | No |
| gemma-3-12b-it | Open (Google) | Local via `transformers` | Yes |
| llama-3.3-70b (or DeepSeek) | Open | Together AI / DeepInfra | Yes |

**Note on judge:** `gemini-3.1-pro-preview` is used as the LLM judge and also
appears in the evaluation matrix. This is a budget-driven design decision
acknowledged as a limitation. The few-shot-constrained judge prompt reduces
stylistic bias.

---

## Pipeline

1. **Layer split** — each dataset case is split into Layer 0 (shown to model)
   and Layer 1 (hidden, used by simulator).
2. **Flex turn** — model receives Layer 0, decides whether to ask 0–3
   clarifying questions before committing to an answer.
3. **Simulation** — a separate LLM call acting as the user/patient answers each
   CQ using only Layer 1 content.
4. **Judge classification** — a validated LLM judge classifies each CQ as
   EPISTEMIC or ALEATORIC.
5. **Evaluation** — preliminary accuracy, updated accuracy, confidence delta,
   CQ type, hedge rate.

---

## UQ Methods

- **Verbalized confidence** (all models): `confidence` field in structured JSON output
- **Token entropy** (open models only: Gemma, Llama): computed from top-k
  logprobs captured per generation step
- **Semantic entropy** (stretch): N-sample clustering at T=0.7

---

## Validation (Phase 0 — blocker)

Before any experiments run, both the judge and the simulator are validated:

- **Judge**: ≥ 85% accuracy on CLAMBER gold labels, Cohen's κ ≥ 0.70,
  per-class F1 ≥ 0.80 for both EPISTEMIC and ALEATORIC.
- **Simulator**: hedge rate < 20% per dataset on a 20-case spot-check;
  no systematic contradiction with Layer 1 content.

---

## Infrastructure

- Python, GCP (single VM), WandB for experiment tracking
- Gemini/Llama: API calls with `tenacity` retry + exponential backoff
- Gemma: local `transformers` load on GPU VM (L4 or A100)
- Incremental CSV writing after each row; full resume support
- All figures saved as PDF

---

## Output

A per-model uncertainty profile across three domains capturing: CQ-asking rate,
CQ type distribution, accuracy improvement from clarification, confidence
calibration, and (for open models) token/semantic entropy decomposition.
