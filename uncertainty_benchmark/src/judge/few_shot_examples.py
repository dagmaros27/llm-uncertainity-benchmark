"""Few-shot examples for the LLM judge.

Strategy:
  • 8 canonical examples — one per CLAMBER subclass — covering the full taxonomy
  • 3 adversarial pairs — same surface form, different label — to teach the
    model the decision boundary on cases the pilot judge got wrong

All example strings here are EXCLUDED from the validation pool to avoid leakage.
Keep `FEW_SHOT_EXCLUSION_SET` in sync with the inputs of `FEW_SHOT_EXAMPLES`.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FewShotExample:
    input: str
    expected_output: str  # "EPISTEMIC" or "ALEATORIC"
    explanation: str
    subclass: str  # CLAMBER subclass tag for traceability


# ────────────────────────────────────────────────────────────────────────────
# 8 canonical examples — one per CLAMBER subclass
# ────────────────────────────────────────────────────────────────────────────

CANONICAL_EXAMPLES: List[FewShotExample] = [
    # ── EPISTEMIC ────────────────────────────────────────────────────────────
    FewShotExample(
        subclass="NK",
        input="What is the context or field in which 'Comallcium' is typically used or referenced?",
        expected_output="EPISTEMIC",
        explanation=(
            "The model has never encountered this entity. It is asking for a specific "
            "piece of information to fill a knowledge gap. A definitive answer exists "
            "in principle and would fully resolve the uncertainty."
        ),
    ),
    FewShotExample(
        subclass="ICL",
        input="Is the category either animal or outdoor location?",
        expected_output="EPISTEMIC",
        explanation=(
            "The model is checking which factual category applies to a given example. "
            "The two listed labels point to the same external truth — the model is "
            "asking 'which factual class is this?' not 'which interpretation do you want?'"
        ),
    ),

    # ── ALEATORIC ────────────────────────────────────────────────────────────
    FewShotExample(
        subclass="polysemy",
        input="Are you referring to 'bank' as in a financial institution, or as in the side of a river?",
        expected_output="ALEATORIC",
        explanation=(
            "Both senses of the word are equally valid linguistically. No external "
            "fact can determine which the user meant; only the user can pick. "
            "Disambiguation among valid interpretations = aleatoric."
        ),
    ),
    FewShotExample(
        subclass="co-reference",
        input="What does 'she' refer to — the sister-in-law or Amanda?",
        expected_output="ALEATORIC",
        explanation=(
            "The pronoun supports two valid referents in the surface form. From the "
            "model's perspective, both are legitimate readings; the user must choose. "
            "Co-reference disambiguation is aleatoric, not epistemic."
        ),
    ),
    FewShotExample(
        subclass="whom",
        input="What factors are most important to you in a job — salary, location, career growth, or company culture?",
        expected_output="ALEATORIC",
        explanation=(
            "The answer depends entirely on the user's personal preferences. Even with "
            "complete world knowledge, no observer can produce the 'right' answer for "
            "a given user. This is preference-driven uncertainty."
        ),
    ),
    FewShotExample(
        subclass="what",
        input="What specific aspect of climate change would you like the article to focus on?",
        expected_output="ALEATORIC",
        explanation=(
            "The original prompt is underspecified about the scope of the task. "
            "Multiple valid focuses exist (causes, effects, policy, mitigation) and "
            "only the user can decide which one to write about."
        ),
    ),
    FewShotExample(
        subclass="when",
        input="What time period are you interested in — recent trends, the past decade, or historical context?",
        expected_output="ALEATORIC",
        explanation=(
            "Several time frames are equally valid for the original question. No "
            "external fact narrows it down — the user must pick the intended scope."
        ),
    ),
    FewShotExample(
        subclass="where",
        input="In which city or region would you like recommendations?",
        expected_output="ALEATORIC",
        explanation=(
            "The location is a user-specific scope choice. Many valid locations exist; "
            "the user must specify which one is intended for their request."
        ),
    ),
]


# ────────────────────────────────────────────────────────────────────────────
# 3 adversarial pairs — same surface form, different label
# These teach the decision boundary that pilot judges most often missed.
# ────────────────────────────────────────────────────────────────────────────

ADVERSARIAL_EXAMPLES: List[FewShotExample] = [
    # Pair 1 — "Which version" ─────────────────────────────────────────────
    FewShotExample(
        subclass="adv-version-fact",
        input="Which exact version of TensorFlow are you running (e.g. 2.14.0, 2.15.0)?",
        expected_output="EPISTEMIC",
        explanation=(
            "There is one true answer — the version number is a fact about the user's "
            "environment. The model needs that fact to proceed; no choice is involved."
        ),
    ),
    FewShotExample(
        subclass="adv-version-pref",
        input="Which version of the library would you prefer for your project — the stable one or the latest beta?",
        expected_output="ALEATORIC",
        explanation=(
            "Same 'which version' surface form, but here the answer depends on what "
            "the user prefers (trade-off between stability and features). No fact "
            "determines the answer — it is a preference."
        ),
    ),

    # Pair 2 — "Where" ─────────────────────────────────────────────────────
    FewShotExample(
        subclass="adv-where-fact",
        input="Where is the file saved — on the local disk or a network drive?",
        expected_output="EPISTEMIC",
        explanation=(
            "The file has one actual location. The model is asking the user to report "
            "an objective fact about the system state."
        ),
    ),
    FewShotExample(
        subclass="adv-where-scope",
        input="Where would you like the analysis to focus — North America, Europe, or globally?",
        expected_output="ALEATORIC",
        explanation=(
            "Same 'where' surface form, but here the answer is a scope choice. Multiple "
            "valid scopes exist and only the user can pick the intended one."
        ),
    ),

    # Pair 3 — generic specification request ───────────────────────────────
    FewShotExample(
        subclass="adv-spec-fact",
        input="Could you specify what 'Glembrosium' refers to in your question?",
        expected_output="EPISTEMIC",
        explanation=(
            "The CQ targets a specific unrecognised term. The model is asking the user "
            "to fill a clear knowledge gap — once defined, the uncertainty is resolved."
        ),
    ),
    FewShotExample(
        subclass="adv-spec-pref",
        input="Could you please provide more details so I can tailor the response?",
        expected_output="ALEATORIC",
        explanation=(
            "Generic specification request with no specific knowledge target. The model "
            "is asking the user to supply preferences, scope, or context — a choice, "
            "not a fact. Default for vague specification requests is aleatoric."
        ),
    ),
]


FEW_SHOT_EXAMPLES: List[FewShotExample] = CANONICAL_EXAMPLES + ADVERSARIAL_EXAMPLES


# ────────────────────────────────────────────────────────────────────────────
# Exclusion set — these CQ strings MUST be filtered out of the eval pool.
# ────────────────────────────────────────────────────────────────────────────
FEW_SHOT_EXCLUSION_SET = frozenset(ex.input for ex in FEW_SHOT_EXAMPLES)


def summary() -> str:
    """Return a short text summary of the few-shot composition."""
    n_epi = sum(1 for ex in FEW_SHOT_EXAMPLES if ex.expected_output == "EPISTEMIC")
    n_ale = sum(1 for ex in FEW_SHOT_EXAMPLES if ex.expected_output == "ALEATORIC")
    return (
        f"Few-shot examples: {len(FEW_SHOT_EXAMPLES)} total "
        f"({n_epi} EPISTEMIC, {n_ale} ALEATORIC) — "
        f"{len(CANONICAL_EXAMPLES)} canonical (one per CLAMBER subclass) + "
        f"{len(ADVERSARIAL_EXAMPLES)} adversarial."
    )


if __name__ == "__main__":
    print(summary())
    for ex in FEW_SHOT_EXAMPLES:
        print(f"\n[{ex.expected_output}] ({ex.subclass})")
        print(f"  input: {ex.input}")
        print(f"  why:   {ex.explanation}")
