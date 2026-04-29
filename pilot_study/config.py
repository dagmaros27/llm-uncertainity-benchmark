# ═══════════════════════════════════════════════════════════════════════════
# Global configuration — edit only this file to tune experiment parameters.
# ═══════════════════════════════════════════════════════════════════════════

# ── Model ──────────────────────────────────────────────────────────────────
GEMINI_MODEL_ID    = "gemini-2.5-flash"
GEMINI_API_VERSION = "v1beta"

# ── Request settings ───────────────────────────────────────────────────────
TEMPERATURE        = 0.0
REQUEST_INTERVAL   = 3.0    # seconds between API calls (rate-limit courtesy)
MAX_RETRIES        = 6
MAX_OUTPUT_TOKENS  = 1024

# ── Experiment defaults ────────────────────────────────────────────────────
N_RECORDS          = 20
RANDOM_SEED        = 42
N_CQ_TURNS         = 3   # number of clarifying question rounds in multi-turn pipeline

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
