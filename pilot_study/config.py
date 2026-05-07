# ═══════════════════════════════════════════════════════════════════════════
# Global configuration — edit only this file to tune experiment parameters.
# ═══════════════════════════════════════════════════════════════════════════

# ── Model ──────────────────────────────────────────────────────────────────
GEMINI_MODEL_ID         = "gemini-2.5-flash"          # MedQA clinician model
MSDIALOG_GEMINI_MODEL_ID = "gemini-2.5-flash"         # MS-Dialog clinician model (update to preview ID here)
GEMINI_API_VERSION      = "v1beta"

# ── Shared infrastructure models (constant across all experiments) ──────────
# Change these two to swap judge/simulator for every script and notebook at once.
JUDGE_MODEL_ID     = "gemini-3.1-pro-preview"  # 98.5% CLAMBER accuracy vs 94.5% flash
SIMULATOR_MODEL_ID = "gemini-3.1-pro-preview"  # patient responder

# ── Request settings ───────────────────────────────────────────────────────
TEMPERATURE        = 0.0
REQUEST_INTERVAL   = 1.0    # seconds between API calls (rate-limit courtesy)
MAX_RETRIES        = 6
MAX_OUTPUT_TOKENS  = 4096

# ── Experiment defaults ────────────────────────────────────────────────────
N_RECORDS          = 20
RANDOM_SEED        = 42
N_CQ_TURNS         = 3   # number of clarifying question rounds in multi-turn pipeline

# ── MS-Dialog Phase 1 CSV output schemas ──────────────────────────────────

MSDIALOG_PHASE1_FIELDS = [
    "id",
    "title",
    "category",
    "original_question",
    "clarifying_question",
    "cq_type",               # filled by LLM judge
    "user_response",         # simulated from synthesised context
    "preliminary_solution",
    "preliminary_confidence",
    "updated_solution",
    "updated_confidence",
    "accepted_answer",       # ground truth for semantic evaluator
    "provider",
    "model_id",
    "finish_reason",
    "was_blocked",
]

MSDIALOG_PHASE1_MULTITURN_FIELDS = [
    "id",
    "title",
    "category",
    "original_question",
    # Turn 0 — model sees problem, no history
    "preliminary_solution",
    "preliminary_confidence",
    # Turn 1
    "cq_1",
    "user_response_1",
    "solution_1",
    "confidence_1",
    # Turn 2
    "cq_2",
    "user_response_2",
    "solution_2",
    "confidence_2",
    # Turn 3 — final
    "cq_3",
    "user_response_3",
    "final_solution",
    "final_confidence",
    # Ground truth & metadata
    "accepted_answer",
    "provider",
    "model_id",
    "finish_reason",
    "was_blocked",
]

# ── MS-Dialog Flex (optional-CQ) CSV output schema ────────────────────────
# One row per case; CQ columns are empty when that turn was not reached.
# needed_clarification_N = did the model choose to ask CQ(N+1)?
# n_cqs_asked = 0..3 (number of CQs actually asked before final solution)

MSDIALOG_FLEX_FIELDS = [
    "id",
    "title",
    "category",
    "original_question",
    # Turn 0 — model sees problem only
    "preliminary_solution",
    "preliminary_confidence",
    "needed_clarification_0",       # True → model asked CQ1
    # After CQ1
    "cq_1",
    "user_response_1",
    "solution_1",
    "confidence_1",
    "needed_clarification_1",       # True → model asked CQ2
    # After CQ2
    "cq_2",
    "user_response_2",
    "solution_2",
    "confidence_2",
    "needed_clarification_2",       # True → model wanted CQ3 (may have been forced)
    # After CQ3 (forced final if reached)
    "cq_3",
    "user_response_3",
    "final_solution",               # always the last solution produced
    "final_confidence",
    # Summary
    "n_cqs_asked",                  # 0 / 1 / 2 / 3
    "accepted_answer",
    "provider",
    "model_id",
    "finish_reason",
    "was_blocked",
]

# ── Phase 1 single-turn CSV output schema ─────────────────────────────────
PHASE1_FIELDS = [
    "id",
    "ehr_summary",
    "question",
    "clarifying_question",
    "cq_type",                  # filled by LLM judge in phase 2
    "patient_response",         # simulated from partitioned contexts
    "preliminary_assessment",
    "preliminary_confidence",
    "updated_assessment",
    "updated_confidence",
    "correct_option",           # letter: A/B/C/D
    "correct_answer",           # full text of correct option
    "is_correct_preliminary",
    "is_correct_updated",
    "confidence_delta",
    "provider",
    "model_id",
    "difficulty",
    "finish_reason",
    "was_blocked",
]

# ── Phase 1 multi-turn CSV output schema (one row per case, N_CQ_TURNS=3) ─
PHASE1_MULTITURN_FIELDS = [
    "id",
    "ehr_summary",
    "question",
    "difficulty",
    "correct_option",
    "correct_answer",
    # Turn 0 — model sees options but no history
    "preliminary_assessment",       # letter A/B/C/D
    "preliminary_confidence",
    "is_correct_preliminary",
    # Turn 1 — after CQ1 answer
    "cq_1",
    "patient_response_1",
    "assessment_1",                 # letter A/B/C/D
    "confidence_1",
    "is_correct_1",
    # Turn 2 — after CQ2 answer
    "cq_2",
    "patient_response_2",
    "assessment_2",
    "confidence_2",
    "is_correct_2",
    # Turn 3 — after CQ3 answer (final)
    "cq_3",
    "patient_response_3",
    "final_assessment",
    "final_confidence",
    "is_correct_final",
    # Metadata
    "provider",
    "model_id",
    "finish_reason",
    "was_blocked",
]
