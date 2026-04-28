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

# ── Phase 1 CSV output schema ──────────────────────────────────────────────
PHASE1_FIELDS = [
    "id",
    "layer_0",
    "layer_1",
    "clarifying_question",
    "cq_type",                  # filled by LLM judge
    "patient_response",         # simulated from layer_1
    "preliminary_assessment",
    "preliminary_confidence",
    "updated_assessment",
    "updated_confidence",
    "correct_answer_text",
    "correct_answer_idx",
    "is_correct_preliminary",
    "is_correct_updated",
    "confidence_delta",
    "provider",
    "model_id",
    "meta_info",
    "finish_reason",
    "was_blocked",
]
