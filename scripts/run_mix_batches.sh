#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

EXPERIMENT="${1:?experiment, E1 or E2}"
WORKERS="${2:?number of worker processes}"
SEED_FROM="${3:-0}"
SEED_TO="${4:-29}"

case "$EXPERIMENT" in
  E1|E2) ;;
  *) echo "only E1 and E2 are supported here; use run_mix_any.sh for the others"; exit 1 ;;
esac

LOWER=$(echo "$EXPERIMENT" | tr 'A-Z' 'a-z')
POOLS=data/dataset_combination_pools/pools_manifest.json

for seed in $(seq "$SEED_FROM" "$SEED_TO"); do
  tag=$(printf "s%02d" "$seed")
  root="results/${LOWER}_mix_${tag}"
  status="$root/run_status.json"
  if [ -f "$status" ] && grep -q '"status": "completed"' "$status"; then
    echo "[$(stamp)] $EXPERIMENT $tag already complete, skipping"
    continue
  fi

  names=$("$PYTHON_BIN" - "$POOLS" "$seed" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
seed = int(sys.argv[2])
print(",".join(e["name"] for e in manifest["entries"] if e["seed_index"] == seed))
PY
)
  protocol="$CONFIGS/${LOWER}_mix_${tag}.yaml"
  "$PYTHON_BIN" tools/protocols/make_mix_protocol.py --experiment "$EXPERIMENT" --names "$names" \
    --results-root "$root" --workers "$WORKERS" --out "$protocol" > /dev/null

  echo "[$(stamp)] $EXPERIMENT $tag starting (119 mixtures, $WORKERS workers)"
  "$PYTHON_BIN" -m src.extra.nc_excel_experiments.run --protocol "$protocol" \
    --experiments "$EXPERIMENT" >> "$LOGS/${LOWER}_mix_${tag}.log" 2>&1 || {
      echo "[$(stamp)] $EXPERIMENT $tag failed, see $LOGS/${LOWER}_mix_${tag}.log"
      continue
    }
  echo "[$(stamp)] $EXPERIMENT $tag done -> $ROOT/$root"
done

echo "[$(stamp)] $EXPERIMENT all batches finished (seeds $SEED_FROM..$SEED_TO)"
