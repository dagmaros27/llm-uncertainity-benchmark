"""Smoke test: one structured-output call per provider.

Run on the GPU VM after setup_vm.sh:
    python uncertainty_benchmark/scripts/smoke_test_providers.py [--providers gemini,gemma,llama,qwen]

Costs:
  - gemini: ~1 API call (negligible)
  - gemma/llama/qwen: local inference (free, but loads model into VRAM)

Each test asks: "Is 2+2=4? Answer in JSON: {\"answer\": <true|false>, \"confidence\": <0-100>}"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from uncertainty_benchmark.src.utils import load_dotenv
from uncertainty_benchmark.src.parsing import parse_with_schema

load_dotenv(PROJECT_ROOT / "uncertainty_benchmark" / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("smoke_test")

SYSTEM = "You are a helpful assistant. Always respond with valid JSON only."
USER   = 'Is 2+2=4? Answer ONLY with this JSON: {"answer": true, "confidence": 100}'

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def test_gemini() -> bool:
    logger.info("Testing GeminiProvider (gemini-2.5-flash)...")
    try:
        from uncertainty_benchmark.src.providers.gemini import GeminiProvider
        p = GeminiProvider(model_id="gemini-2.5-flash")
        text = p.call(SYSTEM, USER, temperature=0.0, max_tokens=64)
        logger.info("  Response: %s", text[:200])
        obj = parse_with_schema(text, required_keys=["answer", "confidence"])
        assert obj is not None, f"parse_with_schema returned None — raw: {text[:200]}"
        assert obj.get("answer") is True, f"Expected answer=true, got {obj}"
        print(f"  gemini-2.5-flash: {PASS}")
        return True
    except Exception as exc:
        print(f"  gemini-2.5-flash: {FAIL} — {exc}")
        return False


def test_gemma() -> bool:
    logger.info("Testing GemmaProvider (google/gemma-3-12b-it)...")
    try:
        from uncertainty_benchmark.src.providers.gemma import GemmaProvider
        p = GemmaProvider(model_id="google/gemma-3-12b-it", load_in_4bit=True)
        text, logprobs = p.call_with_logprobs(SYSTEM, USER, temperature=0.0, max_tokens=64)
        logger.info("  Response: %s", text[:200])
        logger.info("  Logprob steps: %d", len(logprobs))
        assert logprobs and len(logprobs) > 0
        print(f"  gemma-3-12b-it: {PASS}  (logprob steps={len(logprobs)})")
        return True
    except Exception as exc:
        print(f"  gemma-3-12b-it: {FAIL} — {exc}")
        return False


def test_llama() -> bool:
    logger.info("Testing LlamaProvider (meta-llama/Llama-3.3-70B-Instruct)...")
    try:
        from uncertainty_benchmark.src.providers.llama import LlamaProvider
        p = LlamaProvider(model_id="meta-llama/Llama-3.3-70B-Instruct", load_in_4bit=True)
        text, logprobs = p.call_with_logprobs(SYSTEM, USER, temperature=0.0, max_tokens=64)
        logger.info("  Response: %s", text[:200])
        logger.info("  Logprob steps: %d", len(logprobs))
        assert logprobs and len(logprobs) > 0
        print(f"  Llama-3.3-70B: {PASS}  (logprob steps={len(logprobs)})")
        return True
    except Exception as exc:
        print(f"  Llama-3.3-70B: {FAIL} — {exc}")
        return False


def test_qwen() -> bool:
    logger.info("Testing QwenProvider (Qwen/Qwen3-4B)...")
    try:
        from uncertainty_benchmark.src.providers.qwen import QwenProvider
        p = QwenProvider(model_id="Qwen/Qwen3-4B")
        text, logprobs = p.call_with_logprobs(SYSTEM, USER, temperature=0.0, max_tokens=64)
        logger.info("  Response: %s", text[:200])
        logger.info("  Logprob steps: %d", len(logprobs))
        assert logprobs and len(logprobs) > 0
        print(f"  Qwen3-4B: {PASS}  (logprob steps={len(logprobs)})")
        return True
    except Exception as exc:
        print(f"  Qwen3-4B: {FAIL} — {exc}")
        return False


PROVIDER_MAP = {
    "gemini": test_gemini,
    "gemma":  test_gemma,
    "llama":  test_llama,
    "qwen":   test_qwen,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--providers",
        default="gemini",
        help="Comma-separated list of providers to test. Default: gemini. "
             "Options: gemini,gemma,llama,qwen. "
             "WARNING: gemma/llama/qwen load large models into GPU memory.",
    )
    args = parser.parse_args()

    requested = [p.strip() for p in args.providers.split(",")]
    unknown = [p for p in requested if p not in PROVIDER_MAP]
    if unknown:
        print(f"Unknown providers: {unknown}. Options: {list(PROVIDER_MAP.keys())}")
        return 1

    print(f"\n=== Provider Smoke Test — {', '.join(requested)} ===\n")
    results = {}
    for name in requested:
        t0 = time.monotonic()
        ok = PROVIDER_MAP[name]()
        results[name] = ok
        logger.info("  %s: %.1fs", name, time.monotonic() - t0)
        print()

    print("=== Summary ===")
    all_ok = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {name:15s} {status}")
        if not ok:
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
