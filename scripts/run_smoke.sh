#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "[$(stamp)] unit tests"
"$PYTHON_BIN" -m pytest tests -q -o addopts=""

echo "[$(stamp)] standalone mixture generator self-test"
"$PYTHON_BIN" -m src.extra.dataset_combinations.self_test --data-root data

echo "[$(stamp)] E1 smoke"
"$PYTHON_BIN" -m src.extra.population_experiments.run --protocol "$CONFIGS/smoke/E1_population_scale.yaml" --experiments E1

echo "[$(stamp)] E2 and phase coherence smoke"
"$PYTHON_BIN" -m src.extra.population_experiments.run --protocol "$CONFIGS/smoke/E2_controller_synchronization_and_phase_coherence.yaml" --experiments E2 phase_coherence

echo "[$(stamp)] availability, long-horizon, response-mechanism and capacity-concentration smoke"
"$PYTHON_BIN" -m src.extra.population_experiments.run --protocol "$CONFIGS/smoke/structured_availability.yaml" --experiments structured_availability
"$PYTHON_BIN" -m src.extra.population_experiments.run --protocol "$CONFIGS/smoke/long_horizon_state.yaml" --experiments long_horizon_state
"$PYTHON_BIN" -m src.extra.population_experiments.run --protocol "$CONFIGS/smoke/availability_second_family.yaml" --experiments availability_second_family
"$PYTHON_BIN" -m src.extra.population_experiments.run --protocol "$CONFIGS/smoke/response_mechanisms_and_capacity_concentration.yaml" --experiments response_mechanisms capacity_concentration

echo "[$(stamp)] smoke suite finished"
