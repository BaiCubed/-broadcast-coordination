#!/usr/bin/env python3
"""Wait for the complete E22 constraint sweep, then refresh Figure 01."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/E22/single_constraint_effect"
RAW = OUTPUT / "data/raw"
PROTOCOL = "E22_single_physical_constraint_effect_v2_extended"
EXPECTED_FILES = 17 * 3 * 30
EXPECTED_ROWS = 17 * 3 * 30 * 65
SERVICE = "e22-single-constraint-v2.service"


def complete() -> bool:
    manifest_path = OUTPUT / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not (
        manifest.get("protocol") == PROTOCOL
        and manifest.get("status") == "completed"
        and int(manifest.get("seed_count", 0)) == 30
        and int(manifest.get("row_count", 0)) == EXPECTED_ROWS
    ):
        return False
    paths = sorted(RAW.glob("*/*/seed_*.csv"))
    if len(paths) != EXPECTED_FILES:
        return False
    for path in paths:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, csv.Error):
            return False
        if len(rows) != 65 or any(row.get("protocol") != PROTOCOL for row in rows):
            return False
    return True


def service_active() -> bool:
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", SERVICE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def refresh() -> None:
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp/matplotlib-e22-figure01"
    subprocess.run(
        [sys.executable, "-c", "from tools import generate_e22_subpanel_figures as f; f.OUT.mkdir(parents=True, exist_ok=True); f.make_constraint_audit()"],
        cwd=ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    while True:
        if complete():
            refresh()
            print(json.dumps({"status": "complete", "figure": str(ROOT / "outputs/figs/subpanel/01_ieee69_constraint_audit.png")}), flush=True)
            return
        if not service_active():
            raise SystemExit("实验服务已停止，但完整 v2 汇总尚未就绪")
        print(json.dumps({"status": "waiting", "service": SERVICE}), flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
