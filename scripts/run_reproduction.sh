#!/usr/bin/env bash
set -euo pipefail
stage=all
if [[ $# -gt 0 ]]; then
  case "$1" in
    --check) stage=check;; --data) stage=data;; --population) stage=population;; --network-inputs) stage=network-inputs;; --network) stage=network;; --figures) stage=figures;; --all) stage=all;; *) echo "Usage: $0 [--check|--data|--population|--network-inputs|--network|--figures|--all]" >&2; exit 2;;
  esac
fi
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
exec "$PYTHON_BIN" scripts/run_reproduction.py --stage "$stage"
