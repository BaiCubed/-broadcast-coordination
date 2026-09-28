#!/usr/bin/env bash
set -euo pipefail
stage=all
if [[ $# -gt 0 ]]; then
  case "$1" in
    --check) stage=check;; --data) stage=data;; --e1-e4) stage=e1-e4;; --experiments) stage=e20-e24;; --artifacts|--figures|--tables) stage=artifacts;; --all) stage=all;; *) echo "Usage: $0 [--check|--data|--e1-e4|--experiments|--artifacts|--all]" >&2; exit 2;;
  esac
fi
cd "$(dirname "$0")/.."
exec python scripts/run_reproduction.py --stage "$stage"
