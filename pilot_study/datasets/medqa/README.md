# MedQA Processed Dataset

Clinical MCQ cases derived from the MedQA (USMLE-style) dataset, processed for use with the Human-API (H-API) experimental framework.

## Files

| File | Records | Description |
|---|---|---|
| `cases.jsonl` | 1,273 | Full processed dataset |
| `unseen_100.jsonl` | 100 | Held-out evaluation subset |
| `manifest.json` | — | Dataset metadata (counts, seed) |

## Difficulty Distribution

| Difficulty | Count |
|---|---|
| Easy | 449 |
| Medium | 781 |
| Hard | 43 |

Seed used for processing/splitting: `42`.

## Schema (JSONL)

Each line is a JSON object with the following fields:

| Field | Type | Description |
|---|---|---|
| `case_id` | string | Unique identifier, e.g. `"medqa_0481"` |
| `ehr_summary` | string | Brief EHR presenting complaint |
| `question` | string | MCQ question text |
| `options` | object | Answer choices keyed `"A"`–`"D"` |
| `correct_option` | string | Correct option letter |
| `correct_answer` | string | Full text of the correct answer |
| `difficulty` | string | `"easy"`, `"medium"`, or `"hard"` |
| `difficulty_score` | int | Numeric difficulty score |
| `full_patient_context` | string | Complete simulated patient profile (demographics, HPI, PMH, labs, imaging, teaching point) |
| `age` | string | Patient age (may be empty) |
| `gender` | string | Patient gender (may be empty) |
| `chief_complaint` | string | Chief complaint (may be empty) |
| `full_vignette` | string | Original MedQA vignette text |
| `patient_context` | string | Information accessible to the patient source (history, ROS, medications, key findings) |
| `nurse_context` | string | Information accessible to the nurse source (vitals, physical exam, lab results) |
| `specialist_context` | string | Information accessible to the specialist source (imaging, advanced tests) |
| `context_costs` | object | Sensing cost per source in registry points, e.g. `{"patient": 10, "nurse": 26}` |
| `context_actions` | array | Sensing actions relevant to the case, e.g. `["hx_chief_complaint", "lab_cbc"]` |
| `source` | string | Dataset origin, always `"medqa"` |
| `meta` | object | Metadata, e.g. `{"original_idx": 481}` |

## Context Partitioning

Each case partitions clinical information across three human sensor sources used by the H-API framework:

- **Patient** — history, review of systems, medications, allergies, key positive/negative findings
- **Nurse** — vitals, physical examination, laboratory results
- **Specialist** — imaging and advanced diagnostic tests

## Usage for Clarifying Questions in Active Feature Acquisition

Beyond standard MCQ evaluation, this dataset supports research on **clarifying questions** within an active feature acquisition (AFA) loop. An agent that does not yet have enough information to answer the clinical question can query individual human sources (patient, nurse, specialist) to acquire missing features before committing to a diagnosis. The `context_costs` and `context_actions` fields make it possible to study cost-sensitive acquisition strategies — agents must decide *which* source to query and *what* to ask, trading off informational value against sensing cost.

The key fields for clarifying-question research are:

- **`patient_context`** / **`nurse_context`** / **`specialist_context`** — the ground-truth information each source can provide; used to simulate source responses and to judge whether a clarifying question is answerable.
- **`full_patient_context`** — the complete clinical picture; serves as the oracle for evaluating whether a question was necessary and whether the answer is correct.
- **`context_actions`** — the set of sensing actions relevant to the case; defines the action space an agent can choose from when formulating queries.
- **`context_costs`** — the cost per source; enables cost-sensitive acquisition strategies where agents must justify the value of each clarifying question.
- **`question`** / **`options`** / **`correct_option`** — the downstream task; determines when the agent has acquired enough information to stop asking and commit to an answer.

This setup naturally gives rise to two categories of clarifying questions. **Epistemic** clarifying questions target information that the human source possesses but has not yet provided — the uncertainty is reducible and the answer exists within one of the context partitions (e.g., asking the nurse for a lab value that was ordered but not yet communicated). **Aleatoric** clarifying questions target inherent ambiguity that no single source can fully resolve — the uncertainty is irreducible given the available evidence (e.g., asking the patient to characterise intermittent symptoms whose presentation varies). The partitioned context design of this dataset allows researchers to distinguish between the two: if the answer to a question is contained in an unqueried source's context, the question is epistemic; if it requires reasoning beyond what any source context provides, it is aleatoric.
