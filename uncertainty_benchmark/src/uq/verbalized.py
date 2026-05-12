"""Verbalized confidence extraction.

Models report confidence as an integer 0–100 in their JSON output.
This module normalises it to a float 0.0–1.0 and handles edge cases.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_confidence(
    parsed: Optional[dict],
    key: str = "confidence",
    fallback_keys: tuple[str, ...] = ("updated_confidence", "final_confidence", "preliminary_confidence"),
) -> Optional[float]:
    """Extract and normalise confidence from a parsed JSON dict.

    Args:
        parsed: The parsed model output dict (may be None).
        key: Primary key to look for (default "confidence").
        fallback_keys: Tried in order if primary key is absent.

    Returns:
        Float in [0.0, 1.0], or None if not found / unparseable.
        Values reported as 0–100 are divided by 100.
        Values already in [0.0, 1.0] are returned as-is.
    """
    if not parsed or not isinstance(parsed, dict):
        return None

    # Try primary key then fallbacks
    raw = parsed.get(key)
    if raw is None:
        for fk in fallback_keys:
            raw = parsed.get(fk)
            if raw is not None:
                break

    if raw is None:
        return None

    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.debug("extract_confidence: cannot cast %r to float", raw)
        return None

    if value < 0:
        logger.debug("extract_confidence: negative value %s clamped to 0", value)
        value = 0.0

    # Normalise 0–100 scale to 0.0–1.0
    if value > 1.0:
        value = value / 100.0

    # Clamp to [0, 1]
    return min(max(value, 0.0), 1.0)
