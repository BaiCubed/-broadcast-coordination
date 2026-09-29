#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

EXPERIMENT="${1:?experiment id, E4 or availability_second_family}"
WORKERS="${2:?number of worker processes}"
SEED_FROM="${3:-0}"
SEED_TO="${4:-0}"

case "$EXPERIMENT" in
  E4) BASE=E4_controller_drift ;;
  availability_second_family) BASE=availability_second_family ;;
  *) echo "unknown experiment: $EXPERIMENT"; exit 1 ;;
esac

for seed in $(seq "$SEED_FROM" "$SEED_TO"); do
  tag=$(printf "s%02d" "$seed")
  if [ "$seed" -eq 0 ]; then
    name="${BASE}_mixed"
  else
    name="${BASE}_mixed_${tag}"
  fi
  root="results/${name}"
  status="$root/run_status.json"
  if [ -f "$status" ] && grep -q '"status": "completed"' "$status"; then
    echo "[$(stamp)] $EXPERIMENT $tag already complete, skipping"
    continue
  fi

  protocol="$CONFIGS/${name}.yaml"
  "$PYTHON_BIN" tools/protocols/make_mixed_protocol.py \
    --experiment "$EXPERIMENT" --seed-index "$seed" \
    --results-root "$root" --workers "$WORKERS" --out "$protocol" > /dev/null

  echo "[$(stamp)] $EXPERIMENT $tag starting (119 mixed populations, $WORKERS workers)"
  "$PYTHON_BIN" -m src.extra.population_experiments.run --protocol "$protocol" \
    --experiments "$EXPERIMENT" >> "$LOGS/${name}.log" 2>&1 || {
      echo "[$(stamp)] $EXPERIMENT $tag failed, see $LOGS/${name}.log"
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
  echo "[$(stamp)] $EXPERIMENT $tag done -> $ROOT/$root ($done_n/119 mixed populations completed)"
  if [ "$done_n" -lt 119 ]; then
    echo "[$(stamp)] WARNING: $EXPERIMENT $tag only $done_n/119 completed, check the status column of dataset_summary.csv"
  fi
done

echo "[$(stamp)] $EXPERIMENT all batches finished (seeds $SEED_FROM..$SEED_TO)"
