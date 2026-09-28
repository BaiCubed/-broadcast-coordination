#!/usr/bin/env bash
# Chain the remaining E1-E4 pipeline: finish downloads, verify and extract,
# build the 15 dataset configs, then run the four experiments.
set -o pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate reviewer3-real
ROOT=/data/zhaojiaxing/nc_run
REPO=$ROOT/broadcast-coordination
LOGS=$ROOT/logs
cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHON_BIN="$(command -v python)"

stamp() { date -Is; }

echo "=== [$(stamp)] stage 1: wait for the parallel prefetch to drain ==="
while pgrep -f par_download.py >/dev/null; do sleep 60; done
echo "prefetch drained at $(stamp)"

echo "=== [$(stamp)] stage 2: checksum verification and archive extraction ==="
# download_data.sh skips every file already on disk, so this pass only verifies
# checksums, retries anything still missing, and unpacks zip/rar5/7z archives.
bash download_data.sh > "$LOGS/verify_extract.log" 2>&1
echo "download_data.sh exit=$? at $(stamp)"
tail -25 "$LOGS/verify_extract.log"

echo "=== [$(stamp)] stage 3: per-dataset configs ==="
python -m tools.prepare_dataset_configs --all > "$LOGS/prepare_configs.jsonl" 2> "$LOGS/prepare_configs.err"
echo "prepare exit=$? at $(stamp)"
python - <<'PY'
import json
from pathlib import Path
ready, blocked = [], []
for line in Path("/data/zhaojiaxing/nc_run/logs/prepare_configs.jsonl").read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    (ready if row["status"] == "ready" else blocked).append(row)
print(f"ready={len(ready)} not_ready={len(blocked)}")
for row in blocked:
    print("  NOT READY", row["dataset"], row["status"], str(row.get("error", ""))[:200])
PY

echo "=== [$(stamp)] stage 4: E1-E4 ==="
python -m src.extra.nc_excel_experiments.run \
  --protocol src/extra/nc_excel_experiments/configs/protocol.yaml \
  --experiments E1 E2 E3 E4 > "$LOGS/experiments.json" 2> "$LOGS/experiments.err"
echo "experiments exit=$? at $(stamp)"
tail -40 "$LOGS/experiments.err"
cat "$LOGS/experiments.json"
echo "=== [$(stamp)] done ==="
