#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs
exec 9>logs/subpanel_full_refresh.lock
if ! flock -n 9; then
    echo "已有子图补充任务正在运行。"
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mpl-subpanel-full-refresh
PYTHON_BIN="${PYTHON_BIN:-/home/heol/anaconda3/bin/python}"

STATUS=results/E22/subpanel_full_refresh_status.json
COMPLETED=0
trap 'if [[ "$COMPLETED" -eq 0 ]]; then printf '\''{"status":"failed","stage":"see logs"}\n'\'' > "$STATUS"; fi' EXIT
printf '{"status":"running","stage":"IEEE-33 direct training and evaluation"}\n' > "$STATUS"

nice -n 10 "$PYTHON_BIN" -u -m \
    src.extra.ieee33_device_day_simulation.figures.run_ieee33_missing_direct_training \
    --seed-count 30 \
    > logs/subpanel_ieee33_direct.log 2>&1

printf '{"status":"running","stage":"parallel supplementary experiments"}\n' > "$STATUS"

nice -n 10 "$PYTHON_BIN" -u tools/run_subpanel_direct_supplement_03_04.py \
    --workers 2 --seed-count 30 \
    > logs/subpanel_pairwise_eps_direct.log 2>&1 &
PAIRWISE_EPS_PID=$!

nice -n 10 "$PYTHON_BIN" -u tools/run_subpanel_baseline_completion.py \
    --workers 2 --seed-count 30 \
    > logs/subpanel_pairwise_baselines.log 2>&1 &
PAIRWISE_BASELINE_PID=$!

nice -n 10 "$PYTHON_BIN" -u -m \
    src.extra.ieee33_device_day_simulation.figures.run_e24_ieee123_audit \
    --model-source results/E24/trained_eps_ieee123_direct/models \
    --workers 2 --seed-count 30 --bootstrap-draws 4000 \
    > logs/subpanel_ieee123_audit.log 2>&1 &
IEEE123_PID=$!

nice -n 10 "$PYTHON_BIN" -u -m \
    src.extra.ieee33_device_day_simulation.figures.run_single_constraint_effect \
    --workers 6 --seed-count 30 \
    > logs/subpanel_single_constraint.log 2>&1 &
SINGLE_CONSTRAINT_PID=$!

FAILED=0
for PID in "$PAIRWISE_EPS_PID" "$PAIRWISE_BASELINE_PID" "$IEEE123_PID" "$SINGLE_CONSTRAINT_PID"; do
    if ! wait "$PID"; then
        FAILED=1
    fi
done

if [[ "$FAILED" -ne 0 ]]; then
    printf '{"status":"failed","stage":"parallel supplementary experiments"}\n' > "$STATUS"
    exit 1
fi

printf '{"status":"running","stage":"figure generation"}\n' > "$STATUS"
nice -n 10 "$PYTHON_BIN" -u tools/generate_e22_subpanel_figures.py \
    > logs/subpanel_figure_generation.log 2>&1
nice -n 10 "$PYTHON_BIN" -u tools/build_e22_subpanel_presentation.py \
    > logs/subpanel_presentation_generation.log 2>&1

printf '{"status":"completed","stage":"all outputs generated"}\n' > "$STATUS"
COMPLETED=1
