#!/usr/bin/env bash
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

nohup "$PYTHON_BIN" -m src.extra.nc_excel_experiments.run \
  --protocol "$CONFIGS/protocol_e9_phase1.yaml" \
  --experiments E9 > "$LOGS/e9_phase1.log" 2>&1 &
echo "E9 phase 1 pid=$!"

nohup "$PYTHON_BIN" -m src.extra.nc_excel_experiments.run \
  --protocol "$CONFIGS/protocol_supp_full.yaml" \
  --experiments E13 E14 > "$LOGS/supp_full.log" 2>&1 &
echo "E13+E14 pid=$!"
