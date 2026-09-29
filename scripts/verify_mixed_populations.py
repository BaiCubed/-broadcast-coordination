#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/mixed_populations"
CHECKSUMS = ROOT / "datasets/mixed_populations/summary_checksums.json"


def _run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT, check=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    _run("check_mixed_populations.py")
    _run("check_english_docs.py", "--root", str(ROOT))
    _run("summarize_mixed_populations.py")
    _run("plot_mixed_populations.py")

    scenarios = sorted((DATA / "mixed_scenarios").glob("*.csv"))
    pairwise = sorted((DATA / "pairwise").glob("*.csv"))
    if len(scenarios) != 14 or len(pairwise) != 105:
        raise SystemExit("Derived-data count check failed")

    expected = json.loads(CHECKSUMS.read_text(encoding="utf-8"))
    actual = {}
    for relative in expected["files"]:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty reproduced artifact: {relative}")
        actual[relative] = _sha256(path)
    if actual != expected["sha256"]:
        raise SystemExit("Summary table checksum mismatch; inspect datasets/mixed_populations/summary_checksums.json")

    for relative in expected["figure_files"]:
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty reproduced figure: {relative}")

    print(json.dumps({
        "status": "pass",
        "scope": "summaries and figures of the 119 mixed populations",
        "scenario_csv_count": len(scenarios),
        "pairwise_csv_count": len(pairwise),
        "long_experiments_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
