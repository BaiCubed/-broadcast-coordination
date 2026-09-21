#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

EXPERIMENT="${1:?experiment id, e.g. E13}"
WORKERS="${2:?number of worker processes}"
SEED_FROM="${3:-0}"
SEED_TO="${4:-0}"

case "$EXPERIMENT" in
  E3|E4|E6|E9|E13|E14|E15) ;;
  *) echo "unknown experiment: $EXPERIMENT"; exit 1 ;;
esac

LOWER=$(echo "$EXPERIMENT" | tr 'A-Z' 'a-z')

for seed in $(seq "$SEED_FROM" "$SEED_TO"); do
  tag=$(printf "s%02d" "$seed")
  root="results/${LOWER}_mix_${tag}"
  status="$root/run_status.json"
  if [ -f "$status" ] && grep -q '"status": "completed"' "$status"; then
    echo "[$(stamp)] $EXPERIMENT $tag already complete, skipping"
    continue
  fi

  protocol="$CONFIGS/${LOWER}_mix_${tag}.yaml"
  "$PYTHON_BIN" tools/protocols/make_mix_protocol_v2.py \
    --experiment "$EXPERIMENT" --seed-index "$seed" \
    --results-root "$root" --workers "$WORKERS" --out "$protocol" > /dev/null

  echo "[$(stamp)] $EXPERIMENT $tag starting (119 mixtures, $WORKERS workers)"
  "$PYTHON_BIN" -m src.extra.nc_excel_experiments.run --protocol "$protocol" \
    --experiments "$EXPERIMENT" >> "$LOGS/${LOWER}_mix_${tag}.log" 2>&1 || {
      echo "[$(stamp)] $EXPERIMENT $tag failed, see $LOGS/${LOWER}_mix_${tag}.log"
      continue
    }
  done_n=$("$PYTHON_BIN" - "$root" <<'PY'
import csv, sys, pathlib
p = list(pathlib.Path(sys.argv[1]).glob("*/dataset_summary.csv"))
if not p:
    print(0); raise SystemExit
rows = list(csv.DictReader(open(p[0])))
print(len({r["dataset"] for r in rows if r.get("status") == "completed"}))
PY
)
  echo "[$(stamp)] $EXPERIMENT $tag done -> $ROOT/$root ($done_n/119 mixtures completed)"
  if [ "$done_n" -lt 119 ]; then
    echo "[$(stamp)] WARNING: $EXPERIMENT $tag only $done_n/119 completed, check the status column of dataset_summary.csv"
  fi
done

echo "[$(stamp)] $EXPERIMENT all batches finished (seeds $SEED_FROM..$SEED_TO)"
