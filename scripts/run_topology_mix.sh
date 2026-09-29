#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

"$PYTHON_BIN" tools/protocols/make_feeder_topology_protocols.py
mkdir -p "$LOGS/E1_feeder_topology"
for topo in ieee33 ieee69 ieee123; do
  ( export NC_TOPOLOGY=$topo NC_NETWORK_FEEDBACK=1
    nohup "$PYTHON_BIN" -u -m src.extra.population_experiments.run \
      --protocol "$CONFIGS/E1_feeder_topology_${topo}.yaml" \
      --experiments E1 > "$LOGS/E1_feeder_topology/${topo}.log" 2>&1 ) &
done
wait
echo "feeder topology runs finished at $(stamp)"
