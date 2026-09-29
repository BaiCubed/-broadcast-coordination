#!/usr/bin/env bash
set -u
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

nohup "$PYTHON_BIN" -m src.extra.population_experiments.run \
  --protocol "$CONFIGS/long_horizon_state.yaml" \
  --experiments long_horizon_state > "$LOGS/long_horizon_state.log" 2>&1 &
echo "long-horizon state pid=$!"

nohup "$PYTHON_BIN" -m src.extra.population_experiments.run \
  --protocol "$CONFIGS/response_mechanisms_and_capacity_concentration.yaml" \
  --experiments response_mechanisms capacity_concentration > "$LOGS/response_mechanisms_and_capacity_concentration.log" 2>&1 &
echo "response mechanisms and capacity concentration pid=$!"
