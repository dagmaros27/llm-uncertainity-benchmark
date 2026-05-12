"""Re-select + re-synthesize 200 MS-Dialog cases for the benchmark.

Improvements over the pilot:
  • Selection: picks dialogs with HIGH USER-CONTENT richness (more user turns +
    longer substantive replies), not just at least one FD tag.
  • Synthesis input: feeds ALL substantive user utterances (FD, NF, FQ, CQ, PF
    with content) into the LLM, not just FD-tagged ones. This recovers the
    user's responses to agent CQs that the pilot was dropping.
  • Synthesis prompt: same SYSTEM + USER_TEMPLATE as the pilot (deliberate —
    we want a like-for-like replacement).

Output:
  uncertainty_benchmark/datasets/msdialog/msdialog_200.jsonl

Resumable cache:
  uncertainty_benchmark/datasets/msdialog/_synthesis_cache.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from uncertainty_benchmark.src.providers import GeminiProvider
from uncertainty_benchmark.src.utils import load_dotenv

logger = logging.getLogger("preprocess_msdialog")

INTENT_PATH    = PROJECT_ROOT / "pilot_study" / "datasets" / "ms-dialog" / "MSDialog-Intent.json"
OUT_DIR        = PROJECT_ROOT / "uncertainty_benchmark" / "datasets" / "msdialog"
OUT_JSONL      = OUT_DIR / "msdialog_200.jsonl"
CACHE_JSONL    = OUT_DIR / "_synthesis_cache.jsonl"

DEFAULT_N      = 200
DEFAULT_MODEL  = "gemini-3.1-pro-preview"
DEFAULT_DELAY  = 1.0

# Tag classification
SUBSTANTIVE_USER_TAGS = {"FD", "NF", "FQ", "CQ", "IR", "RQ"}  # user volunteers info or asks back
GREETING_USER_TAGS    = {"GG", "JK"}
FEEDBACK_USER_TAGS    = {"PF"}      # "thanks, it worked" — keep only if has length

SYNTHESIS_SYSTEM = (
    "You are a technical documentation assistant. Your job is to read a messy "
    "tech-support conversation and rewrite the user's situation as a clean, "
    "structured background document.\n\n"
    "The document will be used by a virtual user (simulator) to answer any "
    "follow-up questions about their problem. It must:\n"
    "- Be written in first person (\"I\", \"my\") — the simulator speaks AS the user\n"
    "- Capture every concrete technical fact from the conversation\n"
    "- Be organised under clear section headers so specific facts are easy to find\n"
    "- Strip all greetings, thanks, and conversational filler\n"
    "- Never include the solution or anything the agent suggested\n\n"
    "Output ONLY the structured summary, no preamble or explanation."
)

SYNTHESIS_USER_TEMPLATE = """Here is a tech-support thread. Synthesise a structured User Situation Summary.

PRODUCT / CATEGORY: {category}
TITLE: {title}

USER'S ORIGINAL QUESTION:
{original_question}

ADDITIONAL INFORMATION FROM THE CONVERSATION:
{raw_context}

---
Write the User Situation Summary using these sections (omit a section if there is no information for it):

CORE PROBLEM
[One or two sentences describing exactly what is broken or not working]

SYSTEM & SOFTWARE INFO
[OS version, device type, software name and version, browser, app version — any specifics mentioned]

SYMPTOMS & ERROR DETAILS
[Exact error messages, what the user sees on screen, how often it happens, when it started]

WHAT I HAVE ALREADY TRIED
[Any steps the user already attempted before asking for help]

ADDITIONAL CONTEXT
[Any other relevant facts: network setup, other devices affected, recent changes, workarounds discovered]"""


# ── Selection ───────────────────────────────────────────────────────────────

def extract_dialog_facts(dialog_id: str, dialog: dict) -> Optional[dict]:
    """Pull the OQ, accepted answer, and all substantive user utterances + agent CQs.

    Returns None if the dialog doesn't satisfy minimal criteria
    (no OQ, no accepted answer, or zero substantive user content beyond OQ).
    """
    utts = dialog.get("utterances", [])
    if not utts:
        return None

    oq_text: str = ""
    accepted_answer: str = ""
    answer_source: str = ""
    user_substantive: List[str] = []     # user utts after OQ that volunteer info or ask back
    agent_cqs: List[str] = []            # agent CQs / IRs / FQs (asking the user)
    last_agent_pa: str = ""

    for u in utts:
        actor = (u.get("actor_type") or "").strip()
        text  = (u.get("utterance") or "").strip()
        if not text:
            continue
        tags  = set((u.get("tags") or "").split())
        is_answer = bool(u.get("is_answer", 0))

        if actor == "User":
            if "OQ" in tags and not oq_text:
                oq_text = text
                continue
            # Substantive user utterance: tagged with FD/NF/FQ/CQ/IR/RQ, OR
            # PA tagged but with content, OR PF with content beyond pure thanks.
            is_subst = bool(tags & SUBSTANTIVE_USER_TAGS)
            is_pf_with_content = ("PF" in tags) and (len(text.split()) > 12)
            is_user_pa = ("PA" in tags) and (len(text.split()) > 5)
            if is_subst or is_pf_with_content or is_user_pa:
                user_substantive.append(text)

        elif actor == "Agent":
            # Agent CQs / IRs / FQs — these are useful framing for the simulator
            # because they show what the agent asked (the user's substantive
            # replies follow).
            if tags & {"CQ", "IR", "FQ", "RQ"}:
                agent_cqs.append(text)
            if "PA" in tags:
                last_agent_pa = text
                if is_answer and not accepted_answer:
                    accepted_answer = text
                    answer_source   = "explicit_is_answer"

    # Fallbacks
    if not accepted_answer and last_agent_pa:
        accepted_answer = last_agent_pa
        answer_source   = "fallback_last_pa"

    if not oq_text or not accepted_answer or not user_substantive:
        return None

    # Richness score = total words across substantive user utterances.
    user_word_count = sum(len(u.split()) for u in user_substantive)
    richness = user_word_count

    return {
        "source_dialog_id":     dialog_id,
        "title":                dialog.get("title", "").strip(),
        "category":             dialog.get("category", "").strip(),
        "original_question":    oq_text,
        "user_substantive":     user_substantive,
        "agent_cqs":            agent_cqs,
        "accepted_answer":      accepted_answer,
        "answer_source":        answer_source,
        "n_user_substantive":   len(user_substantive),
        "n_agent_cqs":          len(agent_cqs),
        "user_word_count":      user_word_count,
        "richness":             richness,
        "n_utterances":         len(utts),
    }


def build_raw_context(record: dict) -> str:
    """Concatenate user substantive utterances + agent CQs into the synthesis input."""
    parts: List[str] = []
    if record["user_substantive"]:
        parts.append("[Further details and replies provided by the user]")
        for u in record["user_substantive"]:
            parts.append(u)
    if record["agent_cqs"]:
        parts.append("\n---\n[Clarifying questions the support agent asked]")
        for q in record["agent_cqs"]:
            parts.append(q)
    return "\n\n".join(parts)


# ── Synthesis ───────────────────────────────────────────────────────────────

def synthesise(provider: GeminiProvider, record: dict) -> str:
    user_msg = SYNTHESIS_USER_TEMPLATE.format(
        category=record["category"] or "Microsoft product",
        title=record["title"] or "(no title)",
        original_question=record["original_question"],
        raw_context=build_raw_context(record),
    )
    return provider.call(
        system_instruction=SYNTHESIS_SYSTEM,
        user_message=user_msg,
        temperature=0.0,
        max_tokens=5000,
    ).strip()


# ── Cache helpers ───────────────────────────────────────────────────────────

def load_cache(path: Path) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                e = json.loads(line)
                cache[e["source_dialog_id"]] = e["synthesised_context"]
    return cache


def append_cache(path: Path, dialog_id: str, essay: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"source_dialog_id": dialog_id,
                             "synthesised_context": essay},
                             ensure_ascii=False) + "\n")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--delay-s", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run selection only; print stats and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    load_dotenv(PROJECT_ROOT / "uncertainty_benchmark" / ".env")

    # 1. Load Intent
    logger.info("Loading %s ...", INTENT_PATH.name)
    with INTENT_PATH.open(encoding="utf-8") as fh:
        intent = json.load(fh)
    logger.info("Loaded %d dialogs", len(intent))

    # 2. Extract facts + filter
    candidates: List[dict] = []
    for did, d in intent.items():
        rec = extract_dialog_facts(did, d)
        if rec is not None:
            candidates.append(rec)
    logger.info("Candidates after filtering (have OQ + answer + ≥1 substantive user turn): %d", len(candidates))

    # 3. Sort by richness desc; require at least 2 substantive user turns when possible
    candidates.sort(key=lambda r: (r["n_user_substantive"], r["richness"]), reverse=True)

    selected = candidates[: args.n]
    logger.info("Selected top %d by (n_user_substantive desc, richness desc)", len(selected))

    # Stats
    n_subst_dist = {}
    word_total = 0
    for r in selected:
        n_subst_dist[r["n_user_substantive"]] = n_subst_dist.get(r["n_user_substantive"], 0) + 1
        word_total += r["user_word_count"]
    logger.info("Selected user_substantive distribution: %s",
                sorted(n_subst_dist.items()))
    logger.info("Mean user words per record: %.1f", word_total / max(1, len(selected)))

    if args.dry_run:
        logger.info("DRY RUN — exiting before any LLM calls.")
        # Save selected (without synthesis) for inspection
        with (OUT_DIR / "_selected_preview.jsonl").open("w", encoding="utf-8") as fh:
            for r in selected:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Wrote %s for inspection.", OUT_DIR / "_selected_preview.jsonl")
        return 0

    # 4. Synthesize (with cache)
    cache = load_cache(CACHE_JSONL)
    logger.info("Cache hits: %d / %d", sum(1 for r in selected if r["source_dialog_id"] in cache), len(selected))

    provider = GeminiProvider(model_id=args.model_id)
    n_pending = sum(1 for r in selected if r["source_dialog_id"] not in cache)
    logger.info("Synthesising %d new contexts on %s ...", n_pending, args.model_id)

    for i, r in enumerate(selected, 1):
        sid = r["source_dialog_id"]
        if sid in cache:
            r["synthesised_context"] = cache[sid]
            continue
        try:
            essay = synthesise(provider, r)
            cache[sid] = essay
            r["synthesised_context"] = essay
            append_cache(CACHE_JSONL, sid, essay)
            logger.info("[%d/%d] %s — %d words", i, len(selected), sid, len(essay.split()))
        except Exception as exc:
            logger.error("[%d/%d] %s — FAILED: %s", i, len(selected), sid, exc)
            r["synthesised_context"] = ""
        if args.delay_s > 0:
            time.sleep(args.delay_s)

    # 5. Write final dataset (msd_001 .. msd_200)
    n_empty = sum(1 for r in selected if not r.get("synthesised_context"))
    if n_empty:
        logger.warning("%d records have empty synthesised_context — they will be dropped.", n_empty)

    out_records = []
    seq = 0
    for r in selected:
        if not r.get("synthesised_context"):
            continue
        seq += 1
        out_records.append({
            "case_id":              f"msd_{seq:03d}",
            "title":                r["title"],
            "category":             r["category"],
            "original_question":    r["original_question"],
            "simulator_context":    r["synthesised_context"],
            "accepted_answer":      r["accepted_answer"],
            "source_dialog_id":     r["source_dialog_id"],
            "answer_source":        r["answer_source"],
            "n_user_substantive":   r["n_user_substantive"],
            "n_agent_cqs":          r["n_agent_cqs"],
            "n_utterances":         r["n_utterances"],
            "user_word_count":      r["user_word_count"],
        })

    with OUT_JSONL.open("w", encoding="utf-8") as fh:
        for rec in out_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records → %s", len(out_records), OUT_JSONL)

    return 0


if __name__ == "__main__":
    sys.exit(main())
