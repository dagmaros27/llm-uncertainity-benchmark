#!/bin/bash
# Dry-run launcher: 5 records × 30 configs (3 datasets × 5 models × 2 methods).
#
# Run on the GPU VM inside a tmux session:
#   tmux new -s dry_runs
#   source ~/final_project/.venv/bin/activate
#   cd ~/final_project
#   bash uncertainty_benchmark/scripts/run_dry_runs.sh 2>&1 | tee ~/dry_runs.log
#
# Order: Gemini (API, no GPU) → Qwen → Gemma → DeepSeek
# Each model's runs are batched together so the model is loaded once.
# If any config produces 0 successful rows, the script prints a warning and
# continues (so you see all failures, not just the first).

set -euo pipefail

PYTHON="python"
RUNNER="uncertainty_benchmark/scripts/run_experiment.py"
N=5          # records per dry-run config
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
    echo "  DRY RUN: ${label}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if $PYTHON $RUNNER \
            --dataset  "$dataset" \
            --model    "$model" \
            --method   "$method" \
            --n_records "$N" \
            --dry_run \
            --no_wandb; then

        # Check that the output CSV has at least 1 data row
        model_id=$(python -c "
import sys; sys.path.insert(0, '.')
from uncertainty_benchmark.scripts.run_experiment import MODEL_REGISTRY
print(MODEL_REGISTRY['$model']['model_id'].split('/')[-1])
" 2>/dev/null)
        csv="uncertainty_benchmark/outputs/dry_runs/${dataset}_${model_id}_${method}.csv"

        if [ -f "$csv" ]; then
            rows=$(tail -n +2 "$csv" | wc -l)
            # Count rows that actually succeeded (STOP or STOP_NO_CQ)
            ok_rows=$(grep -cE ',STOP(_NO_CQ)?,' "$csv" 2>/dev/null || echo 0)
            err_rows=$(grep -cE ',PARSE_ERROR,' "$csv" 2>/dev/null || echo 0)
            echo "  → CSV rows: ${rows} total, ${ok_rows} STOP, ${err_rows} PARSE_ERROR"
            if [ "$ok_rows" -ge 1 ]; then
                echo "  ✅ PASS"
                PASS=$((PASS + 1))
            else
                echo "  ❌ FAIL — 0 successful rows (all PARSE_ERROR or empty)"
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
echo "  Uncertainty Benchmark — Dry Runs (N=${N} each)"
echo "  $(date)"
echo "======================================================"

# ── Gemini (API-only, no GPU load) ────────────────────────────────────────────
echo ""
echo "▶▶▶  GEMINI-2.5-FLASH  ◀◀◀"
for dataset in medqa msdialog sharc; do
    for method in single flex; do
        run_config "$dataset" "gemini-2.5-flash" "$method"
    done
done

echo ""
echo "▶▶▶  GEMINI-3.1-PRO-PREVIEW  ◀◀◀"
for dataset in medqa msdialog sharc; do
    for method in single flex; do
        run_config "$dataset" "gemini-3.1-pro-preview" "$method"
    done
done

# ── Qwen3-4B (smallest local model) ──────────────────────────────────────────
echo ""
echo "▶▶▶  QWEN3-4B  ◀◀◀"
for dataset in medqa msdialog sharc; do
    for method in single flex; do
        run_config "$dataset" "qwen3-4b" "$method"
    done
done

# ── Gemma-3-12B ───────────────────────────────────────────────────────────────
echo ""
echo "▶▶▶  GEMMA-3-12B-IT  ◀◀◀"
for dataset in medqa msdialog sharc; do
    for method in single flex; do
        run_config "$dataset" "gemma-3-12b-it" "$method"
    done
done

# ── DeepSeek-R1-Distill-Llama-70B (largest, last) ────────────────────────────
echo ""
echo "▶▶▶  DEEPSEEK-R1-DISTILL-70B  ◀◀◀"
for dataset in medqa msdialog sharc; do
    for method in single flex; do
        run_config "$dataset" "deepseek-r1-distill-70b" "$method"
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo "  DRY-RUN SUMMARY"
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
    echo ""
    echo "  Fix the above before starting full runs."
    exit 1
else
    echo ""
    echo "  All dry runs PASSED. Ready for full experiment runs."
    exit 0
fi
