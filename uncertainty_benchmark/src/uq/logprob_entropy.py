"""Token-level entropy from top-k logprob distributions.

Local providers (Gemma, Llama, Qwen) return a list of per-token logprob dicts:
    [{token_str: log_prob}, ...]   — one dict per generated token, top-k only

These functions compute entropy over the truncated top-k distribution.
The entropy is slightly underestimated (mass not in top-k is excluded), but
this is consistent across all tokens and models — sufficient for relative
comparison.

All entropy values are in nats (base-e).
"""

from __future__ import annotations

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def token_entropy(top_k_logprobs: dict[str, float]) -> float:
    """Shannon entropy of one token's top-k distribution (nats).

    The top-k distribution is renormalised to sum to 1 before computing
    entropy, so we measure entropy within the observed support.

    Args:
        top_k_logprobs: {token_str: log_prob} for the top-k tokens at one step.

    Returns:
        Entropy in nats. Returns 0.0 for degenerate inputs.
    """
    if not top_k_logprobs:
        return 0.0

    probs = [math.exp(lp) for lp in top_k_logprobs.values()]
    total = sum(probs)
    if total <= 0.0:
        return 0.0

    # Renormalise to top-k
    probs = [p / total for p in probs]
    entropy = -sum(p * math.log(p + 1e-15) for p in probs if p > 0)
    return max(0.0, entropy)


def mean_entropy(logprobs: list[dict[str, float]]) -> float:
    """Mean per-token entropy across all generated tokens (nats).

    Args:
        logprobs: One dict per generated token (from call_with_logprobs).

    Returns:
        Mean entropy in nats, or 0.0 if logprobs is empty.
    """
    if not logprobs:
        return 0.0
    entropies = [token_entropy(step) for step in logprobs]
    return sum(entropies) / len(entropies)


def response_entropy_stats(
    logprobs: Optional[list[dict[str, float]]],
) -> dict[str, Optional[float]]:
    """Compute entropy summary statistics for a full response.

    Args:
        logprobs: Per-token logprob dicts, or None if unavailable.

    Returns:
        Dict with keys:
          "logprob_mean_entropy"   — mean per-token entropy (nats)
          "logprob_max_entropy"    — max per-token entropy (nats)
          "logprob_min_entropy"    — min per-token entropy (nats)
          "logprob_n_tokens"       — number of generated tokens
        All values are None when logprobs is None (API models).
    """
    if logprobs is None:
        return {
            "logprob_mean_entropy": None,
            "logprob_max_entropy":  None,
            "logprob_min_entropy":  None,
            "logprob_n_tokens":     None,
        }

    if not logprobs:
        return {
            "logprob_mean_entropy": 0.0,
            "logprob_max_entropy":  0.0,
            "logprob_min_entropy":  0.0,
            "logprob_n_tokens":     0,
        }

    entropies = [token_entropy(step) for step in logprobs]
    return {
        "logprob_mean_entropy": sum(entropies) / len(entropies),
        "logprob_max_entropy":  max(entropies),
        "logprob_min_entropy":  min(entropies),
        "logprob_n_tokens":     len(entropies),
    }
