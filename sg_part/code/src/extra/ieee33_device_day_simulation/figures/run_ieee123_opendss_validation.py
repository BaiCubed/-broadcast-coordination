"""使用官方 IEEE-123 OpenDSS 算例执行三相潮流安全校核。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _require_opendss() -> Any:
    try:
        import opendssdirect as dss
    except ImportError as exc:
        raise RuntimeError(
            "缺少完整多相验证依赖，请先安装 requirements-e24-multiphase.txt 中的 opendssdirect.py。"
        ) from exc
    return dss


def _solve(dss: Any, master: Path) -> dict[str, Any]:
    dss.Basic.ClearAll()
    dss.Text.Command(f'compile [{master.resolve()}]')
    dss.Solution.Solve()
    if not bool(dss.Solution.Converged()):
        raise RuntimeError(f"OpenDSS 未收敛：{master}")
    bus_names = list(dss.Circuit.AllBusNames())
    voltages = np.asarray(dss.Circuit.AllBusVmagPu(), dtype=float)
    line_names = list(dss.Lines.AllNames())
    line_loading = []
    for name in line_names:
        dss.Lines.Name(name)
        currents = np.asarray(dss.CktElement.CurrentsMagAng(), dtype=float)[::2]
        norm_amps = float(np.max(currents)) if currents.size else float("nan")
        line_loading.append({"element": name, "max_current_a": norm_amps})
    transformer_names = list(dss.Transformers.AllNames())
    transformers = []
    for name in transformer_names:
        dss.Transformers.Name(name)
        powers = np.asarray(dss.CktElement.Powers(), dtype=float)
        transformers.append({"element": name, "apparent_power_kva": float(np.sum(np.abs(powers[0::2] + 1j * powers[1::2])) / 1000.0)})
    return {
        "converged": True,
        "bus_count": len(bus_names),
        "bus_names": bus_names,
        "minimum_voltage_pu": float(np.min(voltages)) if voltages.size else float("nan"),
        "maximum_voltage_pu": float(np.max(voltages)) if voltages.size else float("nan"),
        "line_count": len(line_names),
        "line_measurements": line_loading,
        "transformer_measurements": transformers,
        "validation": {
            "voltage_limits_pu": [0.95, 1.05],
            "voltage_violations": int(np.sum((voltages < 0.95) | (voltages > 1.05))),
            "phase_resolved": True,
            "solver": "OpenDSS",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True, help="官方 IEEE123Master.dss 文件")
    parser.add_argument("--output", type=Path, default=Path("results/E24/data/ieee123_multiphase_validation.json"))
    args = parser.parse_args()
    if not args.master.is_file():
        raise FileNotFoundError(f"找不到 OpenDSS 主文件：{args.master}")
    result = _solve(_require_opendss(), args.master)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output), "bus_count": result["bus_count"], "minimum_voltage_pu": result["minimum_voltage_pu"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
