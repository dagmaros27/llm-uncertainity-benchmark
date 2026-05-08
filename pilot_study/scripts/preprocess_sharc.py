"""Preprocess ShARC dataset → sharc_200.jsonl for LLM context-essay synthesis.

Selection criteria
------------------
- Terminal answer: Yes or No (Irrelevant excluded — too domain-specific to synthesise)
- Non-empty scenario (simulator needs user context)
- History length >= 2 (multiple follow-up questions — richer context for the essay)
- Snippet length >= 20 words (enough rule substance)
- Max 3 Yes + 3 No picks per tree_id (rule diversity, but allow volume)
- Final 200: top-100 Yes + top-100 No by richness score
- Richness score = history_len*3 + evidence_len*4 + min(scenario_words, 30)

Output
------
datasets/sharc/sharc_200.jsonl
  Fields: id, snippet, question, scenario, history, evidence,
          answer, source_url, n_history_cqs, n_evidence_cqs, n_total_cqs

The context_essay field is NOT generated here — run build_sharc_context.py next.

Usage
-----
    python scripts/preprocess_sharc.py
"""

from __future__ import annotations

import json
import random
import sys
import io
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT        = Path(__file__).parent.parent.resolve()  # repo root
SHARC_DIR   = ROOT / "datasets" / "sharc1-official" / "json"
OUT_DIR     = ROOT / "datasets" / "sharc"
OUT_PATH    = OUT_DIR / "sharc_200.jsonl"

RANDOM_SEED = 42
TARGET_PER_CLASS = 100   # 100 Yes + 100 No = 200 total
MIN_HISTORY  = 2
MIN_SNIPPET_WORDS = 20
MAX_PER_TREE_PER_SIDE = 3   # max Yes picks and max No picks per tree_id


def score(ex: dict) -> float:
    """Richness score: prefer more CQs in history + evidence + longer scenarios."""
    return (
        len(ex["history"]) * 3
        + len(ex["evidence"]) * 4
        + min(len(ex["scenario"].split()), 30)
    )


def load_split(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    random.seed(RANDOM_SEED)

    # ── Load dev + test ────────────────────────────────────────────────────────
    dev  = load_split(SHARC_DIR / "sharc_dev.json")
    test = load_split(SHARC_DIR / "sharc_test.json")
    all_data = dev + test
    print(f"Loaded {len(all_data)} utterances (dev + test)")

    # ── Filter ────────────────────────────────────────────────────────────────
    candidates = [
        ex for ex in all_data
        if ex["answer"] in ("Yes", "No")
        and ex["scenario"].strip()
        and len(ex["history"]) >= MIN_HISTORY
        and len(ex["snippet"].split()) >= MIN_SNIPPET_WORDS
    ]
    print(f"Candidates after filter: {len(candidates)}")

    # ── Group by tree_id, pick richest per side ────────────────────────────────
    trees: dict[str, list[dict]] = defaultdict(list)
    for ex in candidates:
        trees[ex["tree_id"]].append(ex)

    pool: list[dict] = []
    for tid, exs in trees.items():
        yes_exs = sorted([e for e in exs if e["answer"] == "Yes"], key=score, reverse=True)
        no_exs  = sorted([e for e in exs if e["answer"] == "No"],  key=score, reverse=True)
        pool.extend(yes_exs[:MAX_PER_TREE_PER_SIDE])
        pool.extend(no_exs[:MAX_PER_TREE_PER_SIDE])

    print(f"Pool after per-tree dedup (cap={MAX_PER_TREE_PER_SIDE}+{MAX_PER_TREE_PER_SIDE}): {len(pool)}")

    # ── Balance Yes/No, keep richest ──────────────────────────────────────────
    yes_pool = sorted([e for e in pool if e["answer"] == "Yes"], key=score, reverse=True)
    no_pool  = sorted([e for e in pool if e["answer"] == "No"],  key=score, reverse=True)

    assert len(yes_pool) >= TARGET_PER_CLASS, f"Not enough Yes cases: {len(yes_pool)}"
    assert len(no_pool)  >= TARGET_PER_CLASS, f"Not enough No cases: {len(no_pool)}"

    final = yes_pool[:TARGET_PER_CLASS] + no_pool[:TARGET_PER_CLASS]
    random.shuffle(final)

    # ── Diagnostics ───────────────────────────────────────────────────────────
    hist_dist   = Counter(len(e["history"]) for e in final)
    ev_dist     = Counter(len(e["evidence"]) for e in final)
    domain_dist = Counter(e["source_url"].split("/")[2] for e in final)
    ans_dist    = Counter(e["answer"] for e in final)
    total_cqs   = [len(e["history"]) + len(e["evidence"]) for e in final]
    unique_trees = len(set(e["tree_id"] for e in final))

    print(f"\nFinal selection: {len(final)} cases")
    print(f"  Answer dist        : {dict(ans_dist)}")
    print(f"  History dist       : {dict(sorted(hist_dist.items()))}")
    print(f"  Evidence dist      : {dict(sorted(ev_dist.items()))}")
    print(f"  Total CQs median   : {sorted(total_cqs)[len(total_cqs)//2]}")
    print(f"  Unique trees       : {unique_trees}")
    print(f"  Domains            : {dict(domain_dist.most_common(6))}")

    # ── Write JSONL ───────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i, ex in enumerate(final):
            record = {
                "id":             f"sharc_{i:03d}",
                "utterance_id":   ex["utterance_id"],
                "tree_id":        ex["tree_id"],
                "source_url":     ex["source_url"],
                "snippet":        ex["snippet"].strip(),
                "question":       ex["question"].strip(),
                "scenario":       ex["scenario"].strip(),
                "history":        ex["history"],      # [{follow_up_question, follow_up_answer}]
                "evidence":       ex["evidence"],     # [{follow_up_question, follow_up_answer}]
                "answer":         ex["answer"],
                "n_history_cqs":  len(ex["history"]),
                "n_evidence_cqs": len(ex["evidence"]),
                "n_total_cqs":    len(ex["history"]) + len(ex["evidence"]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(final)} records → {OUT_PATH}")
    print("Next step: run build_sharc_context.py to generate context_essay for each record.")


if __name__ == "__main__":
    main()
