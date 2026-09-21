from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...config_loader import load_config
from ...experiments.run_experiment import _correlation_conditions, run_default, run_resource_response_diagnostics
from ...figures.plot_figures import plot_all
from ...figures.data2_figure_labels import rewrite_data2_figure4
from ..correlation_paired.run_experiment import (
    RESULTS_ROOT as _OLD_ROOT,
    STRICT_RHO,
    _network_summary,
    _strict_status,
    _update_empirical_rho,
    _write_json,
)
from ..ieee33_real_snapshot.compare_pure_simulation import generate_comparisons


ROOT = Path(__file__).resolve().parents[5]
RESULTS_ROOT = ROOT / "results" / "data2_ieee33_correlation_paired"
CONFIG_ROOT = ROOT / "src" / "extra" / "ieee33_device_day_simulation" / "experiments" / "data2_correlation_paired" / "configs"

GROUPS: tuple[dict[str, Any], ...] = (
    {"name": "network_weak_data2", "config": CONFIG_ROOT / "default_network_weak_data2.yaml", "description": "data2 charging sessions, IEEE33 active, eight-fold capacity headroom."},
    {"name": "network_stress_data2", "config": CONFIG_ROOT / "default_network_stress_data2.yaml", "description": "data2 charging sessions, IEEE33 active, standard capacity."},
)


def run_suite(
    results_root: Path = RESULTS_ROOT,
    selected_group: str | None = None,
    *,
    groups: tuple[dict[str, Any], ...] | None = None,
    protocol_name: str | None = None,
    data2_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results_root.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "protocol": protocol_name or "data2_transaction_session_strict_paired_ieee33",
        "results_root_is_direct_child_of_results": True,
        "data2_input": data2_input or {
            "workbook": "data/data2/processed_data_longer_than_30.xlsx",
            "resource_unit": "transaction-session",
            "deduplicate_key": "transaction_id",
            "time_alignment": "absolute_time_of_day_to_5_minute_bins",
            "pv_available": False,
            "household_load_available": False,
            "background_load": "zero",
            "external_energy_input": "observed_out_power",
            "energy_input_semantics": "observed charging power replaces PV in the generic input channel",
            "power_scale": 1.0,
            "capacity": "derived_from_delta_energy_delta_soc_with_fallback",
            "network_location": "fixed_random_ieee33_assignment",
        },
        "paired_control": "same data2 sessions, clock bins, EPS, allocation and seeds; network capacity only differs",
        "strict_rho_threshold_at_N_3000": STRICT_RHO,
        "groups": {},
    }
    existing = results_root / "run_index.json"
    if selected_group and existing.exists():
        index["groups"].update(json.loads(existing.read_text(encoding="utf-8")).get("groups", {}))
    active_groups = groups or GROUPS
    active_groups = tuple(group for group in active_groups if selected_group in {None, group["name"]})
    if not active_groups:
        raise ValueError(f"unknown group: {selected_group}")
    for group in active_groups:
        config_path = group["config"]
        config = load_config(config_path)
        output = run_default(
            config_path,
            mode="network_stress",
            results_root=results_root / group["name"],
            snapshot_protocol=True,
            snapshot_sampling="aligned",
            real_snapshot_only=False,
        )
        conditions = _correlation_conditions(config, real_snapshot_only=False)
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
            diagnostics[condition["label"]] = payload
            _write_json(output / "data" / f"resource_response_diagnostics_{condition['label']}.json", payload)
        rho = _update_empirical_rho(output, diagnostics)
        network = _network_summary(output)
        classification = {label: _strict_status(rho[label], network) for label in rho}
        _write_json(output / "data" / "correlation_diagnostics.json", {
            "conditions": diagnostics,
            "empirical_rho_for_fig5": rho,
            "network_summary": network,
            "strict_classification": classification,
        })
        _write_json(output / "data" / "data2_input_metadata.json", index["data2_input"])
        figures = plot_all(output)
        figures_by_name = {path.name: path for path in figures}
        for path in rewrite_data2_figure4(output):
            figures_by_name[path.name] = path
            (output / "paper_figures" / path.name).write_bytes(path.read_bytes())
        figures = list(figures_by_name.values())
        comparisons = generate_comparisons(output / "Figs", output / "comparison_pure_simulation", group["name"])
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
    parser = argparse.ArgumentParser(description="Run the complete data2 charging-session strict paired experiment.")
    parser.add_argument("--results-root", default=str(RESULTS_ROOT))
    parser.add_argument("--group", choices=[group["name"] for group in GROUPS], default=None)
    args = parser.parse_args()
    run_suite(Path(args.results_root), selected_group=args.group)


if __name__ == "__main__":
    main()
