#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

W_SMALL="${1:-25}"
W_FULL="${2:-110}"
SEED="${3:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[$(stamp)] === stage 1 ($W_SMALL workers): E9 -> E3 -> E6 ==="
for E in E9 E3 E6; do
  echo "[$(stamp)] --- $E ---"
  bash "$HERE/run_mix_any.sh" "$E" "$W_SMALL" "$SEED" "$SEED"
done
echo "[$(stamp)] === stage 1 finished ==="

echo "[$(stamp)] === stage 2 ($W_FULL workers): E13 -> E14 -> E15 -> E4 ==="
for E in E13 E14 E15 E4; do
  echo "[$(stamp)] --- $E ---"
  bash "$HERE/run_mix_any.sh" "$E" "$W_FULL" "$SEED" "$SEED"
done
echo "[$(stamp)] === all stages finished ==="
