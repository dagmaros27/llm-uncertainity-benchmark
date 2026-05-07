"""Generate LLM context essays for ShARC cases.

For each record in sharc_200.jsonl, calls the LLM to produce a single coherent
paragraph ("context_essay") that synthesises:
  - the rule snippet
  - the user's scenario
  - all Q&A pairs from history and evidence

This essay becomes the simulator's sole knowledge source — it can answer
the specialist's clarifying questions by drawing from it.

Output
------
datasets/sharc/sharc_context_cache.jsonl
  One line per record: {id, utterance_id, context_essay}
  Idempotent: already-synthesised IDs are skipped on resume.

Usage
-----
    python scripts/build_sharc_context.py
"""

from __future__ import annotations

import json
import logging
import sys
import io
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT      = Path(__file__).parent.parent.resolve()  # repo root
sys.path.insert(0, str(ROOT))

from config import SIMULATOR_MODEL_ID, REQUEST_INTERVAL

DATASET_PATH = ROOT / "datasets" / "sharc" / "sharc_200.jsonl"
CACHE_PATH   = ROOT / "datasets" / "sharc" / "sharc_context_cache.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "build_sharc_context.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTION = """\
You are a precise document synthesiser. Your only job is to write a single fluent \
paragraph that captures all the facts from the inputs you are given. Do not add \
information that is not in the inputs. Do not omit any fact. Do not use bullet points \
or headings — write flowing prose only.\
"""

def build_user_message(record: dict) -> str:
    """Construct the synthesis prompt for one ShARC record."""
    lines = [
        "Synthesise the following information into one concise paragraph.",
        "",
        "ELIGIBILITY RULE:",
        record["snippet"],
        "",
        f"USER QUESTION: {record['question']}",
        "",
        f"USER CONTEXT: {record['scenario']}",
    ]

    all_qa = record["history"] + record["evidence"]
    if all_qa:
        lines.append("")
        lines.append("ADDITIONAL FACTS (established through clarifying dialogue):")
        for qa in all_qa:
            q = qa["follow_up_question"].strip().rstrip("?")
            a = qa["follow_up_answer"].strip()
            lines.append(f"- {q}: {a}")

    lines += [
        "",
        "Write one paragraph that integrates the rule, the user's situation, "
        "and every additional fact above. A support specialist reading this paragraph "
        "should be able to answer any follow-up question about the user's eligibility "
        "without needing any other source.",
    ]
    return "\n".join(lines)


def main() -> None:
    from src.utils import load_dotenv
    from src.providers import GeminiProvider

    load_dotenv(ROOT / ".env")
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)

    # Load records
    with open(DATASET_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    logger.info("Loaded %d records from %s", len(records), DATASET_PATH.name)

    # Load already-synthesised IDs
    done: set[str] = set()
    if CACHE_PATH.exists():
        with open(CACHE_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])
    logger.info("Already synthesised: %d / %d", len(done), len(records))

    remaining = [r for r in records if r["id"] not in done]
    if not remaining:
        logger.info("All records already synthesised. Nothing to do.")
        return

    provider = GeminiProvider(model_id=SIMULATOR_MODEL_ID)

    # Smoke test
    resp = provider.call(
        system_instruction="You are a helpful assistant.",
        user_message="Reply with exactly: SMOKE TEST PASSED",
        temperature=0.0, max_tokens=4000,
    )
    assert "SMOKE" in resp.upper(), f"Smoke test failed: {resp}"
    logger.info("Smoke test PASSED")

    # Generate essays
    errors = 0
    with open(CACHE_PATH, "a", encoding="utf-8") as out:
        for i, record in enumerate(remaining):
            rid = record["id"]
            try:
                user_msg = build_user_message(record)
                essay = provider.call(
                    system_instruction=SYSTEM_INSTRUCTION,
                    user_message=user_msg,
                    temperature=0.0,
                    max_tokens=4000,
                )
                essay = essay.strip()
                entry = {
                    "id":           rid,
                    "utterance_id": record["utterance_id"],
                    "context_essay": essay,
                }
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out.flush()
                logger.info("[%d/%d] %s — essay (%d chars)",
                            i + 1, len(remaining), rid, len(essay))
            except Exception as exc:
                errors += 1
                logger.error("[%d/%d] %s — FAILED: %s", i + 1, len(remaining), rid, exc)

            if i < len(remaining) - 1:
                time.sleep(REQUEST_INTERVAL)

    total_done = len(done) + len(remaining) - errors
    logger.info("Done. %d synthesised, %d errors. Cache: %s", total_done, errors, CACHE_PATH)


if __name__ == "__main__":
    main()
