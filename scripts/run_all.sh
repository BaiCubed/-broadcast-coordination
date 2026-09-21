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

echo "=== [$(stamp)] stage 3: E1-E4 on the 15 published fleets ==="
"$PYTHON_BIN" -m src.extra.nc_excel_experiments.run \
  --protocol "$CONFIGS/protocol.yaml" \
  --experiments E1 E2 E3 E4 > "$LOGS/experiments.json" 2> "$LOGS/experiments.err"
echo "experiments exit=$? at $(stamp)"
tail -40 "$LOGS/experiments.err"
cat "$LOGS/experiments.json"
echo "=== [$(stamp)] done ==="
