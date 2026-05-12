"""Semantic entropy via N-sample clustering (stretch goal).

Reference: Kuhn et al. 2023 "Semantic Uncertainty: Linguistic Invariances for
Uncertainty Estimation in Natural Language Generation"

Algorithm:
  1. Generate N samples at temperature T (e.g. T=0.7, N=5)
  2. Cluster samples by semantic equivalence using an NLI model
     (pairs that are bidirectionally entailed → same cluster)
  3. Semantic entropy = -Σ p(cluster) * log p(cluster)
     where p(cluster) = cluster_size / N

Requirements:
  - An LLMProvider that supports temperature sampling (local models)
  - A cross-encoder NLI model (e.g. cross-encoder/nli-deberta-v3-base)

Usage (from pipeline code):
    from uncertainty_benchmark.src.uq.semantic_entropy import SemanticEntropyEstimator
    estimator = SemanticEntropyEstimator()
    se = estimator.estimate(provider, system_instruction, user_message, n_samples=5)
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

_NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"


class SemanticEntropyEstimator:
    """Estimates semantic entropy for free-text generations.

    Lazy-loads the NLI model on first call.
    """

    def __init__(self, nli_model_id: str = _NLI_MODEL_ID, device: str = "cpu") -> None:
        self._nli_model_id = nli_model_id
        self._device = device
        self._nli_pipeline = None

    def _load_nli(self):
        if self._nli_pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            self._nli_pipeline = hf_pipeline(
                "text-classification",
                model=self._nli_model_id,
                device=self._device,
            )
            logger.info("NLI model loaded: %s", self._nli_model_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load NLI model {self._nli_model_id}: {exc}"
            ) from exc

    def _entails(self, premise: str, hypothesis: str) -> bool:
        """Return True if premise entails hypothesis (NLI label = ENTAILMENT)."""
        self._load_nli()
        result = self._nli_pipeline(
            f"{premise} [SEP] {hypothesis}",
            truncation=True,
            max_length=512,
        )
        label = result[0]["label"].upper() if result else ""
        return "ENTAIL" in label

    def _bidirectional_entailment(self, a: str, b: str) -> bool:
        """True iff a entails b AND b entails a."""
        return self._entails(a, b) and self._entails(b, a)

    def _cluster(self, samples: list[str]) -> list[list[str]]:
        """Greedy clustering: assign each sample to the first compatible cluster."""
        clusters: list[list[str]] = []
        for s in samples:
            placed = False
            for cluster in clusters:
                if self._bidirectional_entailment(cluster[0], s):
                    cluster.append(s)
                    placed = True
                    break
            if not placed:
                clusters.append([s])
        return clusters

    def estimate(
        self,
        provider,
        system_instruction: str,
        user_message: str,
        n_samples: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Optional[float]:
        """Estimate semantic entropy for a given prompt.

        Args:
            provider: An LLMProvider (must support temperature > 0 sampling).
            system_instruction: System prompt.
            user_message: User turn.
            n_samples: Number of stochastic samples.
            temperature: Sampling temperature (must be > 0).
            max_tokens: Max tokens per sample.

        Returns:
            Semantic entropy in nats, or None on failure.
        """
        if temperature <= 0:
            logger.warning("semantic_entropy requires temperature > 0; got %s", temperature)
            return None

        # Generate N samples
        samples: list[str] = []
        for i in range(n_samples):
            try:
                text = provider.call(
                    system_instruction=system_instruction,
                    user_message=user_message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                samples.append(text.strip())
            except Exception as exc:
                logger.warning("semantic_entropy: sample %d failed: %s", i, exc)

        if len(samples) < 2:
            logger.warning("semantic_entropy: insufficient samples (%d)", len(samples))
            return None

        # Cluster by semantic equivalence
        try:
            clusters = self._cluster(samples)
        except Exception as exc:
            logger.warning("semantic_entropy: clustering failed: %s", exc)
            return None

        # Compute cluster distribution entropy
        n = len(samples)
        entropy = 0.0
        for cluster in clusters:
            p = len(cluster) / n
            if p > 0:
                entropy -= p * math.log(p)

        logger.debug(
            "semantic_entropy: n=%d clusters=%d entropy=%.3f",
            n, len(clusters), entropy,
        )
        return entropy
