#!/usr/bin/env bash
set -o pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo "=== [$(stamp)] stage 1: verify checksums and extract archives ==="
bash download_data.sh > "$LOGS/verify_extract.log" 2>&1
echo "download_data.sh exit=$? at $(stamp)"
tail -25 "$LOGS/verify_extract.log"

echo "=== [$(stamp)] stage 2: per-dataset configurations ==="
"$PYTHON_BIN" tools/data/prepare_dataset_configs.py --all \
  > "$LOGS/prepare_configs.jsonl" 2> "$LOGS/prepare_configs.err"
echo "prepare exit=$? at $(stamp)"
"$PYTHON_BIN" - "$LOGS/prepare_configs.jsonl" <<'PY'
import json, sys
from pathlib import Path
ready, blocked = [], []
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    (ready if row["status"] == "ready" else blocked).append(row)
print(f"ready={len(ready)} not_ready={len(blocked)}")
for row in blocked:
    print("  NOT READY", row["dataset"], row["status"], str(row.get("error", ""))[:200])
PY

echo "=== [$(stamp)] stage 3: the population experiments on the 15 published populations ==="
run_one() {
  "$PYTHON_BIN" -m src.extra.population_experiments.run --protocol "$CONFIGS/$1" --experiments "${@:2}" \
    >> "$LOGS/experiments.json" 2>> "$LOGS/experiments.err"
}
run_one E1_population_scale.yaml E1
run_one E2_controller_synchronization.yaml E2
run_one phase_coherence.yaml phase_coherence
run_one E4_controller_drift.yaml E4
run_one response_mechanisms_and_capacity_concentration.yaml response_mechanisms capacity_concentration
run_one structured_availability.yaml structured_availability
run_one availability_second_family.yaml availability_second_family
run_one long_horizon_state.yaml long_horizon_state
echo "experiments exit=$? at $(stamp)"
tail -40 "$LOGS/experiments.err"
cat "$LOGS/experiments.json"
echo "=== [$(stamp)] done ==="
