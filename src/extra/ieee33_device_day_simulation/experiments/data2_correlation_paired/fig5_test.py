from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ...config_loader import load_config
from ...experiments.run_experiment import run_resource_response_diagnostics
from ...figures.data2_figure_labels import rewrite_data2_figure4
from ...figures.plot_figures import plot_all
from ..ieee33_real_snapshot.compare_pure_simulation import generate_comparisons


ROOT = Path(__file__).resolve().parents[5]
RESULTS_ROOT = ROOT / "results" / "data2_ieee33_correlation_paired"
CONFIG_ROOT = ROOT / "src" / "extra" / "ieee33_device_day_simulation" / "experiments" / "data2_correlation_paired" / "configs"
GROUPS = (
    ("network_weak_data2", CONFIG_ROOT / "default_network_weak_data2.yaml"),
    ("network_stress_data2", CONFIG_ROOT / "default_network_stress_data2.yaml"),
)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")


def run_group(name: str, config_path: Path) -> None:
    config = load_config(config_path)
    output = RESULTS_ROOT / name
    scan = list(config["experiment"]["experiments_rho_scan"])
    diagnostics: dict[str, dict[str, Any]] = {}
    for spec in scan:
        path = run_resource_response_diagnostics(
            config_path,
            mode="network_stress",
            result_root=output,
            snapshot_sampling="aligned",
            replications=10,
            weak_rho_threshold=0.25 / (3000 - 1),
            zone_correlation=float(spec["zone_correlation"]),
            condition_label=str(spec["label"]),
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["experiments_parameters"] = spec
        diagnostics[str(spec["label"])] = payload
        _write(output / "data" / f"fig5_diagnostic_{spec['label']}.json", payload)

    nscale = json.loads((output / "data" / "n_scaling.json").read_text(encoding="utf-8"))
    scan_rows: dict[str, Any] = {}
    for spec in scan:
        diagnostic = diagnostics[str(spec["label"])]
        signed = float(diagnostic["conditional_residual_mean_pairwise_rho"])
        nominal = float(spec["nominal_rho"])
        n = 3000
        zone_count = len(config["zones"]["zones"])
        n_eff = n / (1.0 + (n / zone_count - 1.0) * nominal) if nominal else float(n)
        aggregate_cv = float(diagnostic.get("aggregate_response_cv", 0.0))
        scan_rows[str(spec["label"])] = {
            "rho_definition": "experiments nominal rho; measured residual retained separately",
            "rho_within": nominal,
            "nominal_rho": nominal,
            "measured_residual_rho_signed": signed,
            "measured_residual_rho_abs": abs(signed),
            "zone_correlation_parameter": float(spec["zone_correlation"]),
            "N": n,
            "K": zone_count,
            "N_eff": float(n_eff),
            "N_eff_over_N": float(n_eff / n),
            "cv_at_max_N": aggregate_cv,
            "cv_sqrt_neff": aggregate_cv * np.sqrt(n_eff),
        }

    aligned = {
        "iid": scan_rows["iid"],
        "weak": scan_rows["rho~0.006"],
        "moderate": scan_rows["rho~0.023"],
    }
    old = nscale
    for key in aligned:
        aligned[key]["scaling_data"] = old.get(key, {}).get("scaling_data", [])
        aligned[key]["loglog_slope"] = old.get(key, {}).get("loglog_slope", -0.5)
        aligned[key]["sigma_hat"] = old.get(key, {}).get("sigma_hat", 0.0)
    rho_output = {**aligned, "rho_scan": scan_rows}
    _write(output / "data" / "rho_sensitivity.json", rho_output)
    _write(output / "data" / "fig5_experiments_aligned_diagnostics.json", {
        "protocol": "IEEE33 test-only Figure 5 using experiments rho settings",
        "groups": {name: {"parameters": scan, "diagnostics": diagnostics}},
    })
    figures = plot_all(output)
    for path in rewrite_data2_figure4(output):
        (output / "paper_figures" / path.name).write_bytes(path.read_bytes())
    generate_comparisons(output / "Figs", output / "comparison_pure_simulation", name)
    print(name, figures)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the IEEE33 Figure 5 test sweep with experiments-aligned rho settings.")
    parser.add_argument("--group", choices=[name for name, _ in GROUPS], default=None)
    args = parser.parse_args()
    groups = tuple(
        (name, config_path)
        for name, config_path in GROUPS
        if args.group is None or name == args.group
    )
    for name, config_path in groups:
        run_group(name, config_path)


if __name__ == "__main__":
    main()
