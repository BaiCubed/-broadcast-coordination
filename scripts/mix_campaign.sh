#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

WORKERS="${1:-110}"
SEED="${2:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for E in availability_second_family E4; do
  echo "[$(stamp)] --- $E ---"
  bash "$HERE/run_mix_any.sh" "$E" "$WORKERS" "$SEED" "$SEED"
done
echo "[$(stamp)] === all mixed-population runs finished ==="
