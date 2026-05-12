#!/bin/bash
# Check progress of full experiment runs.
# Run anytime from the project root:
#   bash uncertainty_benchmark/scripts/check_progress.sh [dataset]
#
# With no argument: shows all CSVs in outputs/runs/
# With argument:    filters to that dataset (msdialog / sharc / medqa)

RUNS_DIR="uncertainty_benchmark/outputs/runs"
FILTER="${1:-}"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Experiment Progress  —  $(date)"
echo "══════════════════════════════════════════════════════════════"
printf "  %-55s %6s %6s %6s %5s\n" "Config" "Total" "STOP" "ERR" "OK%"
echo "  ──────────────────────────────────────────────────────────────"

total_rows=0
total_stop=0
total_err=0

for csv in "$RUNS_DIR"/*.csv; do
    [ -f "$csv" ] || continue
    name=$(basename "$csv" .csv)

    # Apply dataset filter if given
    if [ -n "$FILTER" ] && [[ "$name" != ${FILTER}* ]]; then
        continue
    fi

    rows=$(tail -n +2 "$csv" | wc -l | tr -d '[:space:]')
    stop=$(grep -cE ',STOP(_NO_CQ)?,' "$csv" 2>/dev/null | tr -d '[:space:]' || echo 0)
    err=$(grep -cE ',PARSE_ERROR,' "$csv" 2>/dev/null | tr -d '[:space:]' || echo 0)

    if [ "${rows:-0}" -gt 0 ] 2>/dev/null; then
        pct=$(( stop * 100 / rows ))
    else
        pct=0
    fi

    # Progress bar (out of 200)
    done_of_200=$(( rows * 100 / 200 ))
    bar=$(printf '%0.s█' $(seq 1 $((done_of_200 / 10))) 2>/dev/null)
    pad=$(printf '%0.s░' $(seq 1 $((10 - done_of_200 / 10))) 2>/dev/null)

    printf "  %-55s %6s %6s %6s %4s%%\n" "$name" "$rows/200" "$stop" "$err" "$pct"

    total_rows=$((total_rows + rows))
    total_stop=$((total_stop + stop))
    total_err=$((total_err + err))
done

echo "  ──────────────────────────────────────────────────────────────"
if [ "$total_rows" -gt 0 ]; then
    total_pct=$(( total_stop * 100 / total_rows ))
    printf "  %-55s %6s %6s %6s %4s%%\n" "TOTAL" "$total_rows" "$total_stop" "$total_err" "$total_pct"
fi
echo ""
