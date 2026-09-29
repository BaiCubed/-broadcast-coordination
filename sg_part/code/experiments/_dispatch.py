"""Small dispatch layer for experiment-owned public entrypoints."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dispatch(experiment: str) -> None:
    """Run exactly one experiment entrypoint from the release root."""
    commands = {
        "E1": ["-m", "src.extra.nc_excel_experiments.run", "--protocol", "src/extra/nc_excel_experiments/configs/protocol.yaml", "--experiments", "E1"],
        "E2": ["-m", "src.extra.nc_excel_experiments.run", "--protocol", "src/extra/nc_excel_experiments/configs/protocol.yaml", "--experiments", "E2"],
        "E3": ["-m", "src.extra.nc_excel_experiments.run", "--protocol", "src/extra/nc_excel_experiments/configs/protocol.yaml", "--experiments", "E3"],
        "E4": ["-m", "src.extra.nc_excel_experiments.run", "--protocol", "src/extra/nc_excel_experiments/configs/protocol.yaml", "--experiments", "E4"],
        "E20": ["-m", "src.extra.ieee33_device_day_simulation.figures.run_e20_transfer"],
        "E21": ["-m", "src.extra.ieee33_device_day_simulation.figures.run_e21_mixed_scenarios"],
        "E22": ["-m", "src.extra.ieee33_device_day_simulation.figures.run_e22_ieee69_complexity"],
        "E23": ["-m", "src.extra.ieee33_device_day_simulation.figures.run_e23_ieee69_relative_boundary"],
        "E24": ["-m", "src.extra.ieee33_device_day_simulation.figures.run_e24_ieee123_direct_training"],
    }
    if experiment not in commands:
        raise SystemExit(f"Unknown experiment: {experiment}")
    subprocess.run([sys.executable, *commands[experiment]], cwd=ROOT, check=True)
