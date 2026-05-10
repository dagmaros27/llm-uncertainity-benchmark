"""Single-config experiment runner.

Usage:
    python uncertainty_benchmark/scripts/run_experiment.py \
        --dataset  medqa \
        --model    gemini-2.5-flash \
        --method   single \
        [--n_records 5] \
        [--dry_run] \
        [--no_wandb]

Model aliases (pass the alias — full HF path is resolved internally):
    gemini-2.5-flash
    gemini-3.1-pro-preview
    gemma-3-12b-it          →  google/gemma-3-12b-it
    deepseek-r1-distill-70b →  deepseek-ai/DeepSeek-R1-Distill-Llama-70B
    qwen3-4b                →  Qwen/Qwen3-4B

Dry-run mode (--dry_run) writes to outputs/dry_runs/ and caps at --n_records.
Full-run mode writes to outputs/runs/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from uncertainty_benchmark.src.utils import load_dotenv

load_dotenv(PROJECT_ROOT / "uncertainty_benchmark" / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_experiment")

# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict] = {
    "gemini-2.5-flash": {
        "provider": "gemini",
        "model_id": "gemini-2.5-flash",
    },
    "gemini-3.1-pro-preview": {
        "provider": "gemini",
        "model_id": "gemini-3.1-pro-preview",
    },
    "gemma-3-12b-it": {
        "provider": "gemma",
        "model_id": "google/gemma-3-12b-it",
        "load_in_4bit": True,
    },
    "deepseek-r1-distill-70b": {
        "provider": "llama",
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "load_in_4bit": True,
    },
    "qwen3-4b": {
        "provider": "qwen",
        "model_id": "Qwen/Qwen3-4B",
    },
}

DATASETS    = ("medqa", "msdialog", "sharc")
METHODS     = ("single", "flex")
PROMPT_ROOT = PROJECT_ROOT / "uncertainty_benchmark" / "prompts"
DATA_ROOT   = PROJECT_ROOT / "uncertainty_benchmark" / "datasets"
OUTPUT_ROOT = PROJECT_ROOT / "uncertainty_benchmark" / "outputs"


# ── Provider factory ──────────────────────────────────────────────────────────

def build_provider(alias: str):
    cfg = MODEL_REGISTRY[alias]
    ptype = cfg["provider"]

    if ptype == "gemini":
        from uncertainty_benchmark.src.providers.gemini import GeminiProvider
        return GeminiProvider(model_id=cfg["model_id"])

    elif ptype == "gemma":
        from uncertainty_benchmark.src.providers.gemma import GemmaProvider
        return GemmaProvider(
            model_id=cfg["model_id"],
            load_in_4bit=cfg.get("load_in_4bit", False),
        )

    elif ptype == "llama":
        from uncertainty_benchmark.src.providers.llama import LlamaProvider
        return LlamaProvider(
            model_id=cfg["model_id"],
            load_in_4bit=cfg.get("load_in_4bit", True),
        )

    elif ptype == "qwen":
        from uncertainty_benchmark.src.providers.qwen import QwenProvider
        return QwenProvider(model_id=cfg["model_id"])

    else:
        raise ValueError(f"Unknown provider type: {ptype}")


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_dataset(dataset: str, n_records: int) -> list[dict]:
    paths = {
        "medqa":    DATA_ROOT / "medqa"    / "medqa_200.jsonl",
        "msdialog": DATA_ROOT / "msdialog" / "msdialog_200.jsonl",
        "sharc":    DATA_ROOT / "sharc"    / "sharc_200.jsonl",
    }
    path = paths[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if n_records > 0:
        records = records[:n_records]

    logger.info("Loaded %d records from %s", len(records), path.name)
    return records


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Run one experiment config")
    parser.add_argument("--dataset",   required=True, choices=DATASETS)
    parser.add_argument("--model",     required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--method",    required=True, choices=METHODS)
    parser.add_argument("--n_records", type=int, default=0,
                        help="Cap number of records. 0 = all (200). Use 5 for dry runs.")
    parser.add_argument("--dry_run",   action="store_true",
                        help="Write to outputs/dry_runs/ instead of outputs/runs/")
    parser.add_argument("--no_wandb",  action="store_true",
                        help="Disable WandB tracking (useful for dry runs)")
    args = parser.parse_args()

    dataset  = args.dataset
    model    = args.model
    method   = args.method
    n_recs   = args.n_records if args.n_records > 0 else 200
    is_dry   = args.dry_run
    use_wandb = not args.no_wandb

    model_id  = MODEL_REGISTRY[model]["model_id"]
    subdir    = "dry_runs" if is_dry else "runs"
    short_model = model_id.split("/")[-1]
    csv_name  = f"{dataset}_{short_model}_{method}.csv"
    output_csv = OUTPUT_ROOT / subdir / csv_name

    logger.info("=" * 60)
    logger.info("Config: dataset=%s model=%s method=%s n=%d dry=%s",
                dataset, model, method, n_recs, is_dry)
    logger.info("Output: %s", output_csv)
    logger.info("=" * 60)

    # ── Build provider ────────────────────────────────────────────────────────
    logger.info("Building provider: %s ...", model)
    provider = build_provider(model)

    # ── Build simulator ───────────────────────────────────────────────────────
    from uncertainty_benchmark.src.providers.gemini import GeminiProvider
    from uncertainty_benchmark.src.pipelines.simulator import Simulator, DATASET_CONTEXT_LABELS

    sim_provider = GeminiProvider(model_id="gemini-2.5-flash")
    sim_prompt   = PROMPT_ROOT / "simulator" / f"{dataset}.txt"
    simulator    = Simulator(
        provider=sim_provider,
        instructions_path=sim_prompt,
        context_label=DATASET_CONTEXT_LABELS[dataset],
    )

    # ── Build tracker ─────────────────────────────────────────────────────────
    from uncertainty_benchmark.src.tracking import make_tracker

    tracker = make_tracker(
        dataset=dataset,
        model_id=model_id,
        method=method,
        extra_config={"n_records": n_recs, "dry_run": is_dry},
        enabled=use_wandb and not is_dry,  # disable WandB for dry runs by default
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    records = load_dataset(dataset, n_recs)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    prompt_path = PROMPT_ROOT / dataset / f"{method}_turn.txt"

    if method == "single":
        from uncertainty_benchmark.src.pipelines.single_turn import SingleTurnPipeline
        pipeline = SingleTurnPipeline(
            provider=provider,
            dataset=dataset,
            instruction_path=prompt_path,
            simulator=simulator,
            output_csv=output_csv,
            tracker=tracker,
        )
    else:
        from uncertainty_benchmark.src.pipelines.flex_turn import FlexTurnPipeline
        pipeline = FlexTurnPipeline(
            provider=provider,
            dataset=dataset,
            instruction_path=prompt_path,
            simulator=simulator,
            output_csv=output_csv,
            tracker=tracker,
        )

    pipeline.run(records)

    logger.info("Done — output: %s", output_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
