"""WandB tracking wrapper for the uncertainty benchmark.

Usage:
    tracker = WandBTracker(
        project="uncertainty-benchmark",
        run_name="medqa-gemini-flash-single",
        config={"dataset": "medqa", "model": "gemini-2.5-flash", "method": "single"},
        tags=["medqa", "gemini", "single"],
    )
    tracker.log({"case_id": "medqa_001", "is_correct_final": True, ...})
    tracker.finish()

Each log() call increments an internal step counter.
NaN/None values are automatically filtered before logging (WandB rejects None).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class WandBTracker:
    """Thin wrapper around wandb.init / wandb.log / wandb.finish."""

    def __init__(
        self,
        project: str,
        run_name: str,
        config: dict[str, Any],
        tags: Optional[list[str]] = None,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._step = 0
        self._run = None

        if not enabled:
            logger.info("WandBTracker disabled — all log() calls are no-ops")
            return

        try:
            import wandb
            self._run = wandb.init(
                project=project,
                name=run_name,
                config=config,
                tags=tags or [],
                reinit=True,
            )
            logger.info(
                "WandB run started — project=%s name=%s url=%s",
                project, run_name, self._run.url if self._run else "N/A",
            )
        except Exception as exc:
            logger.warning("WandB init failed — tracking disabled: %s", exc)
            self._enabled = False

    def log(self, row: dict[str, Any]) -> None:
        """Log a single row of metrics. Filters out None values."""
        if not self._enabled or self._run is None:
            return
        try:
            import wandb
            # WandB can't handle None — replace with wandb.define_metric exclusion
            clean = {k: v for k, v in row.items() if v is not None}
            wandb.log(clean, step=self._step)
            self._step += 1
        except Exception as exc:
            logger.warning("WandB log failed at step %d: %s", self._step, exc)

    def summary(self, metrics: dict[str, Any]) -> None:
        """Set run-level summary metrics (e.g. final accuracy, CQ rate)."""
        if not self._enabled or self._run is None:
            return
        try:
            import wandb
            for k, v in metrics.items():
                if v is not None:
                    wandb.run.summary[k] = v
        except Exception as exc:
            logger.warning("WandB summary failed: %s", exc)

    def finish(self) -> None:
        """Finalise the WandB run."""
        if not self._enabled or self._run is None:
            return
        try:
            import wandb
            wandb.finish()
            logger.info("WandB run finished")
        except Exception as exc:
            logger.warning("WandB finish failed: %s", exc)

    @property
    def step(self) -> int:
        return self._step


def make_tracker(
    dataset: str,
    model_id: str,
    method: str,
    extra_config: Optional[dict] = None,
    project: str = "uncertainty-benchmark",
    enabled: bool = True,
) -> "WandBTracker":
    """Convenience factory used by pipeline run scripts.

    Builds a standardised run name + config dict and returns a WandBTracker.
    """
    config = {
        "dataset":  dataset,
        "model_id": model_id,
        "method":   method,
    }
    if extra_config:
        config.update(extra_config)

    # Shorten model_id for the run name (last component after /)
    short_model = model_id.split("/")[-1]
    run_name = f"{dataset}-{short_model}-{method}"
    tags = [dataset, short_model, method]

    return WandBTracker(
        project=project,
        run_name=run_name,
        config=config,
        tags=tags,
        enabled=enabled,
    )
