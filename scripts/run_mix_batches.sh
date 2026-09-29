#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

WORKERS="${1:?number of worker processes}"
SEED_FROM="${2:-0}"
SEED_TO="${3:-29}"

POOLS=data/dataset_combination_pools/pools_manifest.json

for seed in $(seq "$SEED_FROM" "$SEED_TO"); do
  tag=$(printf "s%02d" "$seed")
  name="E1_population_scale_mixed_${tag}"
  root="results/${name}"
  status="$root/run_status.json"
  if [ -f "$status" ] && grep -q '"status": "completed"' "$status"; then
    echo "[$(stamp)] E1 $tag already complete, skipping"
    continue
  fi

  names=$("$PYTHON_BIN" - "$POOLS" "$seed" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
seed = int(sys.argv[2])
print(",".join(e["name"] for e in manifest["entries"] if e["seed_index"] == seed))
PY
)
  protocol="$CONFIGS/${name}.yaml"
  "$PYTHON_BIN" tools/protocols/make_mixed_protocol.py --experiment E1 --names "$names" \
    --results-root "$root" --workers "$WORKERS" --out "$protocol" > /dev/null

  echo "[$(stamp)] E1 $tag starting (119 mixed populations, $WORKERS workers)"
  "$PYTHON_BIN" -m src.extra.population_experiments.run --protocol "$protocol" \
    --experiments E1 >> "$LOGS/${name}.log" 2>&1 || {
      echo "[$(stamp)] E1 $tag failed, see $LOGS/${name}.log"
      continue
    }
  echo "[$(stamp)] E1 $tag done -> $ROOT/$root"
done

echo "[$(stamp)] E1 all batches finished (seeds $SEED_FROM..$SEED_TO)"
