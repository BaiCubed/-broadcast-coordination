#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

E22_SEEDS="${E22_SEEDS:-30}"
E22_BOOTSTRAP="${E22_BOOTSTRAP:-4000}"
E23_SEEDS="${E23_SEEDS:-30}"
E23_BOOTSTRAP="${E23_BOOTSTRAP:-2000}"
E24_SEEDS="${E24_SEEDS:-10}"
E24_BOOTSTRAP="${E24_BOOTSTRAP:-1000}"
FIGURES="src.extra.ieee33_device_day_simulation.figures"

case "${1:-all}" in
  --ieee69-complexity)
    "$PYTHON_BIN" -m "$FIGURES.run_e22_ieee69_complexity" \
      --seed-count "$E22_SEEDS" --bootstrap-draws "$E22_BOOTSTRAP"
    ;;
  --ieee69-boundary)
    "$PYTHON_BIN" -m "$FIGURES.run_e23_ieee69_relative_boundary" \
      --seed-count "$E23_SEEDS" --bootstrap-draws "$E23_BOOTSTRAP"
    ;;
  --ieee123-audit)
    "$PYTHON_BIN" -m "$FIGURES.run_e24_ieee123_audit" \
      --seed-count "$E24_SEEDS" --workers "${E24_WORKERS:-2}" --bootstrap-draws "$E24_BOOTSTRAP"
    ;;
  all|--all)
    "$PYTHON_BIN" -m "$FIGURES.run_e23_ieee69_relative_boundary" \
      --seed-count "$E23_SEEDS" --bootstrap-draws "$E23_BOOTSTRAP"
    "$PYTHON_BIN" -m "$FIGURES.run_e24_ieee123_audit" \
      --seed-count "$E24_SEEDS" --workers "${E24_WORKERS:-2}" --bootstrap-draws "$E24_BOOTSTRAP"
    ;;
  *)
    printf '%s\n' "usage: bash run_e23_e24.sh [--ieee69-complexity|--ieee69-boundary|--ieee123-audit|--all]" >&2
    exit 2
    ;;
esac
