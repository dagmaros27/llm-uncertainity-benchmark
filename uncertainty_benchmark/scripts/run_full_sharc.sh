#!/bin/bash
# Full-run launcher: 200 records × 10 configs for ShARC.
#
# Run on the GPU VM inside a tmux session:
#   tmux new -s full_sharc
#   source ~/final_project/.venv/bin/activate
#   cd ~/final_project
#   bash uncertainty_benchmark/scripts/run_full_sharc.sh 2>&1 | tee ~/full_sharc.log
#
# Order: Gemini (API, no GPU) → Qwen → Gemma → DeepSeek
# Estimated wall-clock: ~16 hours

set -euo pipefail

PYTHON="python"
RUNNER="uncertainty_benchmark/scripts/run_experiment.py"
N=200
PASS=0
FAIL=0
FAILURES=()

run_config() {
    local dataset=$1
    local model=$2
    local method=$3
    local label="${dataset} × ${model} × ${method}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  FULL RUN: ${label}"
    echo "  $(date)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if $PYTHON $RUNNER \
            --dataset  "$dataset" \
            --model    "$model" \
            --method   "$method" \
            --n_records "$N"; then

        model_id=$(python -c "
import sys; sys.path.insert(0, '.')
from uncertainty_benchmark.scripts.run_experiment import MODEL_REGISTRY
print(MODEL_REGISTRY['$model']['model_id'].split('/')[-1])
" 2>/dev/null)
        csv="uncertainty_benchmark/outputs/runs/${dataset}_${model_id}_${method}.csv"

        if [ -f "$csv" ]; then
            rows=$(tail -n +2 "$csv" | wc -l | tr -d '[:space:]')
            ok_rows=$(grep -cE ',STOP(_NO_CQ)?,' "$csv" 2>/dev/null | tr -d '[:space:]' || echo 0)
            err_rows=$(grep -cE ',PARSE_ERROR,' "$csv" 2>/dev/null | tr -d '[:space:]' || echo 0)
            echo "  → CSV rows: ${rows} total | ${ok_rows} STOP | ${err_rows} PARSE_ERROR"
            if [ "${ok_rows:-0}" -ge 1 ] 2>/dev/null; then
                echo "  ✅ PASS"
                PASS=$((PASS + 1))
            else
                echo "  ❌ FAIL — 0 successful rows"
                FAIL=$((FAIL + 1))
                FAILURES+=("$label")
            fi
        else
            echo "  ❌ FAIL — output CSV not created"
            FAIL=$((FAIL + 1))
            FAILURES+=("$label")
        fi
    else
        echo "  ❌ FAIL — runner exited with error"
        FAIL=$((FAIL + 1))
        FAILURES+=("$label")
    fi
}

echo "======================================================"
echo "  Uncertainty Benchmark — Full Runs: ShARC (N=${N})"
echo "  $(date)"
echo "======================================================"

# ── Gemini (API-only, no GPU) ─────────────────────────────────────────────────
echo ""
echo "▶▶▶  GEMINI-2.5-FLASH  ◀◀◀"
for method in single flex; do
    run_config "sharc" "gemini-2.5-flash" "$method"
done

echo ""
echo "▶▶▶  GEMINI-3.1-PRO-PREVIEW  ◀◀◀"
for method in single flex; do
    run_config "sharc" "gemini-3.1-pro-preview" "$method"
done

# ── Qwen3-4B ──────────────────────────────────────────────────────────────────
echo ""
echo "▶▶▶  QWEN3-4B  ◀◀◀"
for method in single flex; do
    run_config "sharc" "qwen3-4b" "$method"
done

# ── Gemma-3-12B ───────────────────────────────────────────────────────────────
echo ""
echo "▶▶▶  GEMMA-3-12B-IT  ◀◀◀"
for method in single flex; do
    run_config "sharc" "gemma-3-12b-it" "$method"
done

# ── DeepSeek-R1-Distill-70B ───────────────────────────────────────────────────
echo ""
echo "▶▶▶  DEEPSEEK-R1-DISTILL-70B  ◀◀◀"
for method in single flex; do
    run_config "sharc" "deepseek-r1-distill-70b" "$method"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  SHARC FULL-RUN SUMMARY"
echo "  $(date)"
echo "======================================================"
echo "  PASS: ${PASS} / $((PASS + FAIL))"
echo "  FAIL: ${FAIL} / $((PASS + FAIL))"

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo ""
    echo "  Failed configs:"
    for f in "${FAILURES[@]}"; do
        echo "    ✗ ${f}"
    done
    exit 1
else
    echo ""
    echo "  All ShARC runs PASSED."
    exit 0
fi
