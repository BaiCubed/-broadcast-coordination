ROOT="${NC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOGS="${NC_LOGS:-$ROOT/logs}"
CONFIGS="src/extra/nc_excel_experiments/configs"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "$LOGS"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
stamp() { date -u +%FT%TZ; }
