"""Select and format 200 MedQA cases for the benchmark.

Strategy (mirrors the pilot):
  • Source: pilot_study/datasets/medqa/cases.jsonl (1273 LLM-generated cases)
  • Context composition: patient_context + nurse_context + specialist_context
    concatenated with --- separators (same three-layer structure as pilot)
  • clean_simulator_context() strips any leaked diagnoses / summary sections
  • Stratified by difficulty: proportional to availability in the all-3-context pool
  • Deterministic: sorted by (difficulty_score desc within stratum, then case_id)

Output
------
uncertainty_benchmark/datasets/medqa/medqa_200.jsonl
  Fields: case_id, ehr_summary, question, options, correct_option, correct_answer,
          difficulty, difficulty_score, simulator_context, source_case_id
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from uncertainty_benchmark.src.utils import clean_simulator_context

logger = logging.getLogger(__name__)

SOURCE_JSONL = PROJECT_ROOT / "pilot_study" / "datasets" / "medqa" / "cases.jsonl"
OUT_DIR      = PROJECT_ROOT / "uncertainty_benchmark" / "datasets" / "medqa"
OUT_JSONL    = OUT_DIR / "medqa_200.jsonl"

TARGET_N     = 200
DIFFICULTY_ORDER = ["easy", "medium", "hard"]


# ── Context composition ──────────────────────────────────────────────────────

def compose_simulator_context(record: dict) -> str:
    """Concatenate all available context layers with --- separators.

    Layer order: patient → nurse → specialist.
    clean_simulator_context() removes any leaked diagnoses or summary sections.
    """
    layers = []
    for field in ("patient_context", "nurse_context", "specialist_context"):
        text = (record.get(field) or "").strip()
        if text:
            layers.append(text)
    if not layers:
        return ""
    raw = "\n\n---\n\n".join(layers)
    return clean_simulator_context(raw)


def context_richness(record: dict) -> int:
    """Score = number of non-empty context layers (0–3)."""
    score = 0
    for field in ("patient_context", "nurse_context", "specialist_context"):
        if (record.get(field) or "").strip():
            score += 1
    return score


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    # 1. Load source
    if not SOURCE_JSONL.exists():
        logger.error("Source not found: %s", SOURCE_JSONL)
        return 1

    all_records: list[dict] = []
    with SOURCE_JSONL.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                all_records.append(json.loads(line))
    logger.info("Loaded %d source records from %s", len(all_records), SOURCE_JSONL.name)

    # 2. Filter: must have ehr_summary, question, options, correct_option
    valid = [
        r for r in all_records
        if r.get("ehr_summary", "").strip()
        and r.get("question", "").strip()
        and r.get("options")
        and r.get("correct_option", "").strip()
    ]
    logger.info("Valid records (have EHR+question+options): %d", len(valid))

    # 3. Annotate richness; exclude records with no context at all
    for r in valid:
        r["_richness"] = context_richness(r)

    valid = [r for r in valid if r["_richness"] > 0]
    logger.info("Records with at least 1 context layer: %d", len(valid))

    # 4. Stratify by difficulty
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in valid:
        diff = (r.get("difficulty") or "medium").lower()
        by_diff[diff].append(r)

    for d, recs in by_diff.items():
        logger.info("  difficulty=%s: %d records", d, len(recs))

    # 5. Compute target counts proportionally, honouring available pool
    total_available = sum(len(v) for v in by_diff.values())
    targets: dict[str, int] = {}
    for d in DIFFICULTY_ORDER:
        pool = by_diff.get(d, [])
        if not pool:
            targets[d] = 0
            continue
        proportion = len(pool) / total_available
        targets[d] = min(round(TARGET_N * proportion), len(pool))

    # Adjust so sum == TARGET_N
    total_assigned = sum(targets.values())
    shortfall = TARGET_N - total_assigned
    # Distribute shortfall to the biggest pools in order
    for d in sorted(DIFFICULTY_ORDER, key=lambda x: len(by_diff.get(x, [])), reverse=True):
        if shortfall == 0:
            break
        capacity = len(by_diff.get(d, [])) - targets.get(d, 0)
        add = min(shortfall, capacity)
        targets[d] = targets.get(d, 0) + add
        shortfall -= add

    for d, n in targets.items():
        logger.info("  Target: difficulty=%s → %d", d, n)

    # 6. Select within each stratum: prefer highest richness, then highest
    #    difficulty_score (hard cases selected over easy within stratum),
    #    then alphabetical case_id for determinism.
    selected: list[dict] = []
    for d in DIFFICULTY_ORDER:
        pool = by_diff.get(d, [])
        n = targets.get(d, 0)
        if n == 0:
            continue
        pool_sorted = sorted(
            pool,
            key=lambda r: (
                -r["_richness"],
                -(r.get("difficulty_score") or 0),
                r.get("case_id", ""),
            ),
        )
        selected.extend(pool_sorted[:n])

    logger.info("Selected %d records total", len(selected))

    # 7. Compose simulator contexts
    empty_ctx = 0
    for r in selected:
        r["_simulator_context"] = compose_simulator_context(r)
        if not r["_simulator_context"]:
            empty_ctx += 1
    if empty_ctx:
        logger.warning("%d records have empty simulator_context after cleaning", empty_ctx)

    # 8. Write output
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out_records = []
    for seq, r in enumerate(selected, start=1):
        out_records.append({
            "case_id":          f"medqa_{seq:03d}",
            "source_case_id":   r.get("case_id", ""),
            "ehr_summary":      r["ehr_summary"].strip(),
            "question":         r["question"].strip(),
            "options":          r["options"],
            "correct_option":   r["correct_option"].strip().upper(),
            "correct_answer":   (r.get("correct_answer") or "").strip(),
            "difficulty":       (r.get("difficulty") or "medium").lower(),
            "difficulty_score": r.get("difficulty_score", 0),
            "simulator_context": r["_simulator_context"],
            "n_context_layers": r["_richness"],
        })

    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for rec in out_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Wrote %d records to %s", len(out_records), OUT_JSONL)

    # Stats
    from collections import Counter
    diff_dist = Counter(r["difficulty"] for r in out_records)
    rich_dist = Counter(r["n_context_layers"] for r in out_records)
    logger.info("Difficulty distribution: %s", dict(diff_dist))
    logger.info("Context-layer distribution: %s", dict(sorted(rich_dist.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
