#!/usr/bin/env bash
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

cp -f "$CONFIGS/e1_full.yaml"  "$CONFIGS/e1_full_v2.yaml"
cp -f "$CONFIGS/protocol.yaml" "$CONFIGS/e4_v2.yaml"
cp -f "$CONFIGS/e2_full.yaml"  "$CONFIGS/e2_full_v2.yaml"
cp -f "$CONFIGS/e3_full.yaml"  "$CONFIGS/e3_full_v2.yaml"

sed -i 's|^results_root: .*|results_root: results/e1_full_v2|' "$CONFIGS/e1_full_v2.yaml"
sed -i 's|^results_root: .*|results_root: results/e4_v2|'      "$CONFIGS/e4_v2.yaml"
sed -i 's|^results_root: .*|results_root: results/e2_full_v2|' "$CONFIGS/e2_full_v2.yaml"
sed -i 's|^results_root: .*|results_root: results/e3_full_v2|' "$CONFIGS/e3_full_v2.yaml"

grep -H '^results_root' \
  "$CONFIGS/e1_full_v2.yaml" "$CONFIGS/e4_v2.yaml" \
  "$CONFIGS/e2_full_v2.yaml" "$CONFIGS/e3_full_v2.yaml"
