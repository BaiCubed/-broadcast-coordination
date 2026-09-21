#!/usr/bin/env bash
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

launch() {
  setsid nohup "$PYTHON_BIN" -m src.extra.nc_excel_experiments.run \
      --protocol "$CONFIGS/$1" --experiments "$2" \
      > "$LOGS/$3.log" 2> "$LOGS/$3.err" < /dev/null &
  echo "started $3 (pid $!)  config=$1"
}

launch e1_full_v2.yaml E1 e1_v2
launch e4_v2.yaml      E4 e4_v2
launch e2_full_v2.yaml E2 e2_v2
launch e3_full_v2.yaml E3 e3_v2

sleep 20
echo "--- processes still running after 20 s ---"
pgrep -af "nc_excel_experiments.run" | sed 's/--protocol.*configs\//cfg=/' | head -20
echo "--- tail of each log ---"
for t in e1_v2 e2_v2 e3_v2 e4_v2; do
  echo "== $t"; tail -3 "$LOGS/$t.log" 2>/dev/null; tail -3 "$LOGS/$t.err" 2>/dev/null
done
