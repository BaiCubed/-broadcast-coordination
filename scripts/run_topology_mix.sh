#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

"$PYTHON_BIN" tools/protocols/make_mix_protocols.py
mkdir -p "$LOGS/e1mix"
for topo in ieee33 ieee69 ieee123; do
  ( export NC_TOPOLOGY=$topo NC_NETWORK_FEEDBACK=1
    nohup "$PYTHON_BIN" -u -m src.extra.nc_excel_experiments.run \
      --protocol "$CONFIGS/e1_mix_s00_${topo}_fb.yaml" \
      --experiments E1 > "$LOGS/e1mix/${topo}.log" 2>&1 ) &
done
wait
echo "topology mixture arms finished at $(stamp)"
