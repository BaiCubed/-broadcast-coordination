#!/usr/bin/env bash
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

launch() {
  setsid nohup "$PYTHON_BIN" -m src.extra.population_experiments.run \
      --protocol "$CONFIGS/$1" --experiments "$2" \
      > "$LOGS/$3.log" 2> "$LOGS/$3.err" < /dev/null &
  echo "started $3 (pid $!)  config=$1"
}

launch E1_population_scale.yaml           E1              E1_population_scale
launch E4_controller_drift.yaml           E4              E4_controller_drift
launch E2_controller_synchronization.yaml E2              E2_controller_synchronization
launch phase_coherence.yaml               phase_coherence phase_coherence

sleep 20
echo "--- processes still running after 20 s ---"
pgrep -af "population_experiments.run" | sed 's/--protocol.*configs\//cfg=/' | head -20
echo "--- tail of each log ---"
for t in E1_population_scale E2_controller_synchronization phase_coherence E4_controller_drift; do
  echo "== $t"; tail -3 "$LOGS/$t.log" 2>/dev/null; tail -3 "$LOGS/$t.err" 2>/dev/null
done
