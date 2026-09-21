#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "[$(stamp)] unit tests"
"$PYTHON_BIN" -m pytest tests -q -o addopts=""

echo "[$(stamp)] standalone mixture generator self-test"
"$PYTHON_BIN" -m src.extra.dataset_combinations.self_test --data-root data

echo "[$(stamp)] E1 smoke"
"$PYTHON_BIN" -m src.extra.nc_excel_experiments.run --protocol "$CONFIGS/e1_smoke.yaml" --experiments E1

echo "[$(stamp)] E2 and E3 smoke"
"$PYTHON_BIN" -m src.extra.nc_excel_experiments.run --protocol "$CONFIGS/e2e3_smoke.yaml" --experiments E2 E3

echo "[$(stamp)] E6, E9, E15 and E13/E14 smoke"
"$PYTHON_BIN" -m src.extra.nc_excel_experiments.run --protocol "$CONFIGS/protocol_e6_smoke.yaml" --experiments E6
"$PYTHON_BIN" -m src.extra.nc_excel_experiments.run --protocol "$CONFIGS/protocol_e9_smoke.yaml" --experiments E9
"$PYTHON_BIN" -m src.extra.nc_excel_experiments.run --protocol "$CONFIGS/protocol_e15_smoke.yaml" --experiments E15
"$PYTHON_BIN" -m src.extra.nc_excel_experiments.run --protocol "$CONFIGS/protocol_supp_smoke.yaml" --experiments E13 E14

echo "[$(stamp)] smoke suite finished"
