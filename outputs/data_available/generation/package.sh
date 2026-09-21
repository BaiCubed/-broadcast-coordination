#!/usr/bin/env bash

set -e

# 只打包独立生成器代码，不包含任何数据或实验文件。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/dist"
ARCHIVE="${OUTPUT_DIR}/mixed_dataset_generator.tar.gz"

mkdir -p "${OUTPUT_DIR}"
rm -f "${ARCHIVE}" "${ARCHIVE}.sha256"

# 显式列出文件，避免把 __pycache__、CSV、NPZ 或临时文件装入压缩包。
tar \
  --sort=name \
  --mtime='UTC 2026-07-14' \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  -czf "${ARCHIVE}" \
  -C "${PROJECT_ROOT}" \
  src/__init__.py \
  src/extra/__init__.py \
  src/extra/dataset_combinations/__init__.py \
  src/extra/dataset_combinations/__main__.py \
  src/extra/dataset_combinations/catalog.py \
  src/extra/dataset_combinations/cli.py \
  src/extra/dataset_combinations/constants.py \
  src/extra/dataset_combinations/generator.py \
  src/extra/dataset_combinations/package.sh \
  src/extra/dataset_combinations/README.md \
  src/extra/dataset_combinations/requirements.txt \
  src/extra/dataset_combinations/self_test.py

if command -v sha256sum >/dev/null 2>&1; then
  (cd "${OUTPUT_DIR}" && sha256sum "$(basename "${ARCHIVE}")" > "$(basename "${ARCHIVE}").sha256")
else
  (cd "${OUTPUT_DIR}" && shasum -a 256 "$(basename "${ARCHIVE}")" > "$(basename "${ARCHIVE}").sha256")
fi
printf '%s\n' "${ARCHIVE}"
