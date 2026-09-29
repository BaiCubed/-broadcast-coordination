#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

IEEE69_SEEDS="${IEEE69_SEEDS:-30}"
IEEE69_BOOTSTRAP="${IEEE69_BOOTSTRAP:-4000}"
STRESS_SEEDS="${STRESS_SEEDS:-30}"
STRESS_BOOTSTRAP="${STRESS_BOOTSTRAP:-2000}"
IEEE123_SEEDS="${IEEE123_SEEDS:-30}"
IEEE123_BOOTSTRAP="${IEEE123_BOOTSTRAP:-4000}"
NETWORK="src.extra.ieee33_device_day_simulation.network_experiments"

ieee69() {
  "$PYTHON_BIN" -m "$NETWORK.ieee69_network_implementation" \
    --seed-count "$IEEE69_SEEDS" --bootstrap-draws "$IEEE69_BOOTSTRAP"
}
stress() {
  "$PYTHON_BIN" -m "$NETWORK.network_stress_boundary" \
    --seed-count "$STRESS_SEEDS" --bootstrap-draws "$STRESS_BOOTSTRAP"
}
ieee123() {
  "$PYTHON_BIN" -m "$NETWORK.ieee123_safety_audit" \
    --model-source results/ieee123_safety_audit/trained_eps_ieee123_direct/models \
    --seed-count "$IEEE123_SEEDS" --workers "${IEEE123_WORKERS:-2}" --bootstrap-draws "$IEEE123_BOOTSTRAP"
}

case "${1:-all}" in
  --ieee69-implementation) ieee69 ;;
  --stress-boundary) stress ;;
  --ieee123-audit) ieee123 ;;
  all|--all) ieee69; stress; ieee123 ;;
  *)
    printf '%s\n' "usage: bash scripts/run_network_experiments.sh [--ieee69-implementation|--stress-boundary|--ieee123-audit|--all]" >&2
    exit 2
    ;;
esac
