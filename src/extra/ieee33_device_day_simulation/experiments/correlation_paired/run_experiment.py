from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ...config_loader import load_config
from ...experiments.run_experiment import (
    _correlation_conditions,
    run_default,
    run_resource_response_diagnostics,
)
from ...figures.plot_figures import plot_all
from ..ieee33_real_snapshot.compare_pure_simulation import generate_comparisons


ROOT = Path(__file__).resolve().parents[5]
RESULTS_ROOT = ROOT / "results" / "ieee33_real_snapshot_correlation_paired"
CONFIG_ROOT = ROOT / "src" / "extra" / "ieee33_device_day_simulation" / "experiments" / "correlation_paired" / "configs"
PURE_FIGURES = ROOT / "results" / "experiments" / "Figs"

GROUPS: tuple[dict[str, Any], ...] = (
    {
        "name": "network_weak_strict",
        "config": CONFIG_ROOT / "default_network_weak_strict.yaml",
        "description": "IEEE33 active, eight-fold capacity headroom, paired low-pressure candidate.",
    },
    {
        "name": "network_stress_strict",
        "config": CONFIG_ROOT / "default_network_stress_strict.yaml",
        "description": "IEEE33 active at standard capacity, paired network-stress condition.",
    },
)

STRICT_RHO = 0.25 / (3000 - 1)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")


def _network_summary(result_dir: Path) -> dict[str, float]:
    trace = json.loads((result_dir / "data" / "network_timeseries.json").read_text(encoding="utf-8"))
    scale = np.asarray(trace["network_scale"], dtype=float)
    return {
        "mean_scale": float(np.mean(scale)),
        "min_scale": float(np.min(scale)),
        "fraction_scale_at_least_0_95": float(np.mean(scale >= 0.95)),
        "mean_transformer_loading": float(np.mean(trace["transformer_loading"])),
        "mean_overloaded_branches": float(np.mean(trace["overloaded_branches"])),
        "mean_voltage_violations": float(np.mean(trace["voltage_violations"])),
    }


def _update_empirical_rho(result_dir: Path, diagnostics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    nscale = json.loads((result_dir / "data" / "n_scaling.json").read_text(encoding="utf-8"))
    rows: dict[str, Any] = {}
    n = 3000
    for label, diagnostic in diagnostics.items():
        signed = float(diagnostic["conditional_residual_mean_pairwise_rho"])
        rho_abs = abs(signed)
        n_eff = min(float(n), n / (1.0 + (n - 1) * rho_abs))
        cv = float(nscale[label]["scaling_data"][-1]["cv"])
        rows[label] = {
            "rho_definition": "absolute measured conditional residual pairwise correlation",
            "nominal_rho": diagnostic.get("nominal_rho"),
            "zone_correlation_parameter": diagnostic.get("zone_correlation_parameter"),
            "measured_residual_rho_signed": signed,
            "rho_within": rho_abs,
            "N": n,
            "N_eff": n_eff,
            "N_eff_over_N": n_eff / n,
            "cv_at_max_N": cv,
            "cv_sqrt_neff": cv * np.sqrt(n_eff),
        }
    _write_json(result_dir / "data" / "rho_sensitivity.json", rows)
    return rows


def _strict_status(rho: dict[str, Any], network: dict[str, float]) -> dict[str, Any]:
    residual = float(rho["rho_within"])
    n_eff_ratio = float(rho["N_eff_over_N"])
    return {
        "strict_residual_weak": residual <= STRICT_RHO and n_eff_ratio >= 0.80,
        "strict_network_low_stress": (
            network["fraction_scale_at_least_0_95"] >= 0.95
            and network["mean_scale"] >= 0.98
        ),
        "strict_weak_group": (
            residual <= STRICT_RHO
            and n_eff_ratio >= 0.80
            and network["fraction_scale_at_least_0_95"] >= 0.95
            and network["mean_scale"] >= 0.98
        ),
        "thresholds": {
            "rho_abs_max": STRICT_RHO,
            "N_eff_over_N_min": 0.80,
            "fraction_scale_at_least_0_95_min": 0.95,
            "mean_scale_min": 0.98,
        },
    }


def run_suite(
    results_root: Path = RESULTS_ROOT,
    selected_group: str | None = None,
) -> dict[str, Any]:
    results_root.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "protocol": "strict_paired_real_snapshot_correlation_and_ieee33",
        "results_root_is_direct_child_of_results": True,
        "paired_control": "same device-days, timestamps, EPS, allocation and seeds; network capacity only differs between groups",
        "strict_rho_threshold_at_N_3000": STRICT_RHO,
        "groups": {},
    }
    existing_index = results_root / "run_index.json"
    if selected_group and existing_index.exists():
        previous = json.loads(existing_index.read_text(encoding="utf-8"))
        index["groups"].update(previous.get("groups", {}))
    groups = tuple(group for group in GROUPS if selected_group in {None, group["name"]})
    if not groups:
        raise ValueError(f"unknown group: {selected_group}")
    for group in groups:
        config_path = group["config"]
        config = load_config(config_path)
        conditions = _correlation_conditions(config, real_snapshot_only=False)
        result_dir = results_root / group["name"]
        output = run_default(
            config_path,
            mode="network_stress",
            results_root=result_dir,
            snapshot_protocol=True,
            snapshot_sampling="aligned",
            real_snapshot_only=False,
        )
        diagnostics: dict[str, dict[str, Any]] = {}
        for condition in conditions:
            path = run_resource_response_diagnostics(
                config_path,
                mode="network_stress",
                result_root=output,
                snapshot_sampling="aligned",
                replications=8,
                weak_rho_threshold=STRICT_RHO,
                zone_correlation=float(condition["zone_correlation"]),
                condition_label=str(condition["label"]),
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["nominal_rho"] = condition["nominal_rho"]
            diagnostics[str(condition["label"])] = payload
            _write_json(output / "data" / f"resource_response_diagnostics_{condition['label']}.json", payload)
        rho = _update_empirical_rho(output, diagnostics)
        network = _network_summary(output)
        classification = {
            label: _strict_status(rho[label], network)
            for label in rho
        }
        _write_json(output / "data" / "correlation_diagnostics.json", {
            "conditions": diagnostics,
            "empirical_rho_for_fig5": rho,
            "network_summary": network,
            "strict_classification": classification,
        })
        figures = plot_all(output)
        comparisons = generate_comparisons(
            output / "Figs",
            output / "comparison_pure_simulation",
            group["name"],
        )
        index["groups"][group["name"]] = {
            **group,
            "config": str(config_path),
            "result_dir": str(output),
            "figures": [str(path) for path in figures],
            "comparisons": [str(path) for path in comparisons],
            "network_summary": network,
            "strict_classification": classification,
        }
    index["plots_generated"] = True
    _write_json(results_root / "run_index.json", index)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the strict paired real-snapshot correlation and IEEE33 experiment.")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--group", choices=[group["name"] for group in GROUPS], default=None)
    args = parser.parse_args()
    run_suite(Path(args.results_root), selected_group=args.group)


if __name__ == "__main__":
    main()
