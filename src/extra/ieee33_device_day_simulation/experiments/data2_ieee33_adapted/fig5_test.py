from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...config_loader import load_config
from ...experiments.run_experiment import run_resource_response_diagnostics
from ...figures.data2_figure_labels import rewrite_data2_figure4
from ...figures.plot_figures import plot_all
from ..data2_correlation_paired.run_experiment import _write_json
from ..ieee33_real_snapshot.compare_pure_simulation import generate_comparisons


ROOT = Path(__file__).resolve().parents[5]
BASE_RESULTS = ROOT / "results" / "data2_ieee33_adapted"
CONFIG_ROOT = ROOT / "src" / "extra" / "ieee33_device_day_simulation" / "experiments" / "data2_ieee33_adapted" / "configs"

def run_group(profile: str, name: str, config_path: Path, *, reuse_diagnostics: bool = False) -> None:
    output = BASE_RESULTS / profile / name
    config = load_config(config_path)
    scan_specs = config["experiment"].get("experiments_rho_scan")
    if not scan_specs:
        raise ValueError(f"{config_path} must define experiments_rho_scan")
    rho_scan = tuple(
        (
            str(spec["label"]),
            float(spec["zone_correlation"]),
            float(spec["nominal_rho"]),
        )
        for spec in scan_specs
    )
    diagnostics: dict[str, dict[str, Any]] = {}
    for label, zone_correlation, nominal_rho in rho_scan:
        diagnostic_path = output / "data" / f"fig5_diagnostic_{label}.json"
        if reuse_diagnostics and diagnostic_path.exists():
            payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        else:
            path = run_resource_response_diagnostics(
                config_path,
                mode="network_stress",
                result_root=output,
                snapshot_sampling="aligned",
                replications=int(config["experiment"].get("figure5_rho_replications", 10)),
                weak_rho_threshold=0.25 / (int(config["experiment"].get("figure5_rho_resources", 5000)) - 1),
                zone_correlation=zone_correlation,
                condition_label=label,
                diagnostic_resources=int(config["experiment"].get("figure5_rho_resources", 5000)),
                diagnostic_steps=int(config["experiment"].get("figure5_rho_steps", 100)),
                diagnostic_scenarios=["peak_shaving"],
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        payload["nominal_rho"] = nominal_rho
        diagnostics[label] = payload
        _write_json(output / "data" / f"fig5_diagnostic_{label}.json", payload)

    scan_rows: dict[str, Any] = {}
    for label, zone_correlation, nominal_rho in rho_scan:
        diagnostic = diagnostics[label]
        n = int(config["experiment"].get("figure5_rho_resources", 5000))
        zone_count = int(config["experiment"].get("figure5_rho_theoretical_regions", 5))
        n_eff = n / (1.0 + (n / zone_count - 1.0) * nominal_rho) if nominal_rho else float(n)
        aggregate_cv = float(diagnostic.get("aggregate_response_cv", 0.0))
        scan_rows[label] = {
            "rho_within": nominal_rho,
            "nominal_rho": nominal_rho,
            "measured_residual_rho_signed": float(diagnostic["conditional_residual_mean_pairwise_rho"]),
            "zone_correlation_parameter": zone_correlation,
            "N": n,
            "K": zone_count,
            "N_eff": float(n_eff),
            "N_eff_over_N": float(n_eff / n),
            "cv_at_max_N": aggregate_cv,
            "cv_sqrt_neff": aggregate_cv * n_eff ** 0.5,
        }

    nscale = json.loads((output / "data" / "n_scaling.json").read_text(encoding="utf-8"))
    aligned = {
        "iid": scan_rows["iid"],
        "weak": scan_rows["rho~0.006"],
        "moderate": scan_rows["rho~0.023"],
        "rho_scan": scan_rows,
    }
    for key in ("iid", "weak", "moderate"):
        aligned[key]["scaling_data"] = nscale.get(key, {}).get("scaling_data", [])
        aligned[key]["loglog_slope"] = nscale.get(key, {}).get("loglog_slope", -0.5)
        aligned[key]["sigma_hat"] = nscale.get(key, {}).get("sigma_hat", 0.0)
    _write_json(output / "data" / "rho_sensitivity.json", aligned)
    _write_json(output / "data" / "fig5_experiments_aligned_diagnostics.json", {
        "profile": profile,
        "rho_scan": [row[0] for row in rho_scan],
        "test_only": True,
    })

    plot_all(output)
    for path in rewrite_data2_figure4(output):
        (output / "paper_figures" / path.name).write_bytes(path.read_bytes())
    generate_comparisons(output / "Figs", output / "comparison_pure_simulation", name)
    print(f"{profile}/{name}: refreshed Figure 5 and Figure 4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the adapted data2 Figure 5 rho scan without retraining.")
    parser.add_argument("--profile", choices=("observed_demand", "input_counterfactual"), default="observed_demand")
    parser.add_argument("--reuse-diagnostics", action="store_true", help="reuse completed six-point diagnostic JSON files")
    args = parser.parse_args()
    suffix = "input_counterfactual" if args.profile == "input_counterfactual" else "adapted"
    for name, filename in (
        ("network_weak_data2_adapted", f"default_network_weak_data2_{suffix}.yaml"),
        ("network_stress_data2_adapted", f"default_network_stress_data2_{suffix}.yaml"),
    ):
        run_group(args.profile, name, CONFIG_ROOT / filename, reuse_diagnostics=args.reuse_diagnostics)


if __name__ == "__main__":
    main()
