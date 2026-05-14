#!/bin/bash
# MedQA DeepSeek-only re-run (single + flex).
#
# Run on the GPU VM inside a tmux session:
#   tmux new -s medqa_deepseek
#   source ~/final_project/.venv/bin/activate
#   cd ~/final_project
#   bash uncertainty_benchmark/scripts/run_medqa_deepseek.sh 2>&1 | tee ~/medqa_deepseek.log
#
# Estimated wall-clock: ~6-8 hours (T1 verbosity fix applied)

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
        echo "  ✅ PASS"
        PASS=$((PASS + 1))
    else
        echo "  ❌ FAIL — runner exited with error"
        FAIL=$((FAIL + 1))
        FAILURES+=("$label")
    fi
}

echo "======================================================"
echo "  MedQA DeepSeek Re-run (T1 verbosity fix)"
echo "  $(date)"
echo "======================================================"

run_config "medqa" "deepseek-r1-distill-70b" "single"
run_config "medqa" "deepseek-r1-distill-70b" "flex"

echo ""
echo "======================================================"
echo "  SUMMARY"
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
    echo "  All DeepSeek MedQA runs PASSED."
    exit 0
fi
