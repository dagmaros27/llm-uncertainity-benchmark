"""Robust JSON parsing with a 4-step retry ladder.

parse_with_schema(raw, required_keys, provider, repair_prompt)
  1. Direct json.loads
  2. Strip markdown fences + retry
  3. Extract first {…} block + retry
  4. Ask the provider to self-repair (if provider supplied)
  5. Return None → caller logs parse_error row

All steps validate required_keys. The first step to produce a valid dict wins.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Compiled fence stripper
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)

_REPAIR_SYSTEM = (
    "You are a JSON repair assistant. The user will give you a malformed JSON string "
    "and the list of required keys. Return ONLY a valid JSON object containing those keys "
    "with sensible values extracted from the broken input. No explanation, no markdown fences."
)


def _try_parse(text: str, required_keys: Optional[list[str]] = None) -> Optional[dict]:
    """Try to parse text as JSON; validate required_keys if provided."""
    try:
        obj = json.loads(text.strip())
        if not isinstance(obj, dict):
            return None
        if required_keys and not all(k in obj for k in required_keys):
            missing = [k for k in required_keys if k not in obj]
            logger.debug("JSON parsed but missing keys: %s", missing)
            return None
        return obj
    except (json.JSONDecodeError, ValueError):
        return None


def parse_with_schema(
    raw: str,
    required_keys: Optional[list[str]] = None,
    provider=None,
    repair_context: Optional[str] = None,
) -> Optional[dict]:
    """Parse raw LLM output as a JSON dict, trying progressively harder approaches.

    Args:
        raw: The raw string returned by the LLM.
        required_keys: If given, the parsed dict must contain all these keys.
        provider: An LLMProvider instance used for self-repair (step 4).
            If None, step 4 is skipped.
        repair_context: Additional context to include in the repair prompt
            (e.g. the original user message). Helps the model produce better repairs.

    Returns:
        Parsed dict on success, None on failure.
    """
    if not raw or not raw.strip():
        logger.warning("parse_with_schema received empty input")
        return None

    # Step 1: Direct parse
    result = _try_parse(raw, required_keys)
    if result is not None:
        return result

    # Step 2: Strip markdown fences
    stripped = _FENCE_RE.sub("", raw).strip()
    result = _try_parse(stripped, required_keys)
    if result is not None:
        logger.debug("parse_with_schema: recovered via fence-strip (step 2)")
        return result

    # Step 3: Extract first {…} block
    match = _BRACE_RE.search(stripped)
    if match:
        result = _try_parse(match.group(0), required_keys)
        if result is not None:
            logger.debug("parse_with_schema: recovered via brace-extraction (step 3)")
            return result

    # Step 4: LLM self-repair
    if provider is not None:
        try:
            keys_str = ", ".join(f'"{k}"' for k in required_keys) if required_keys else "(any valid JSON)"
            user_msg_parts = [
                f"Required keys: {keys_str}",
                f"\nBroken input:\n{raw[:2000]}",
            ]
            if repair_context:
                user_msg_parts.insert(0, f"Context:\n{repair_context[:500]}\n")
            repair_raw = provider.call(
                system_instruction=_REPAIR_SYSTEM,
                user_message="\n".join(user_msg_parts),
                temperature=0.0,
                max_tokens=512,
            )
            result = _try_parse(repair_raw, required_keys)
            if result is not None:
                logger.info("parse_with_schema: recovered via LLM self-repair (step 4)")
                return result
            # Try fence-strip on repair output
            repaired_stripped = _FENCE_RE.sub("", repair_raw).strip()
            result = _try_parse(repaired_stripped, required_keys)
            if result is not None:
                logger.info("parse_with_schema: recovered via LLM self-repair + fence-strip (step 4b)")
                return result
        except Exception as exc:
            logger.warning("parse_with_schema: self-repair call failed: %s", exc)

    logger.warning(
        "parse_with_schema: all steps failed. required=%s raw=%.200s",
        required_keys, raw,
    )
    return None
