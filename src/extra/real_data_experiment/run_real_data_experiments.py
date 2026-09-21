#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from . import original_run_experiment as core
from . import original_paper_figures as figures


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "nextgen"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "real_data_simulation"


def _load_nextgen_dataset(data_dir: Path) -> dict[str, Any]:

    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No NextGen CSV files found in {data_dir}")

    hour_load = np.zeros(24, dtype=float)
    hour_solar = np.zeros(24, dtype=float)
    hour_battery = np.zeros(24, dtype=float)
    hour_self_consumption = np.zeros(24, dtype=float)
    hour_count = np.zeros(24, dtype=np.int64)
    local_tz = ZoneInfo("Australia/Canberra")
    first_soc: list[float] = []
    capacities: list[float] = []
    peak_powers: list[float] = []

    for path in files:
        first_row = True
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    timestamp = int(float(row["original index"]))
                    load = float(row["load power (kW)"])
                    solar = -float(row["solar power (kW)"])
                    battery = float(row["battery power (kW)"])
                    soc_kwh = float(row["battery SoC (kWh)"])
                    capacity = float(row["battery capacity (kWh)"])
                    peak = float(row["battery peak power (kW)"])
                except (KeyError, TypeError, ValueError):
                    continue

                hour = datetime.fromtimestamp(timestamp, tz=local_tz).hour
                hour_load[hour] += load
                hour_solar[hour] += solar
                hour_battery[hour] += battery
                hour_self_consumption[hour] += min(max(solar, 0.0), max(load, 0.0))
                hour_count[hour] += 1

                if first_row:
                    first_soc.append(np.clip(soc_kwh / max(capacity, 1e-9), 0.1, 0.95))
                    capacities.append(capacity)
                    peak_powers.append(peak)
                    first_row = False

    if not np.all(hour_count > 0):
        raise ValueError(f"Incomplete hourly profile; counts={hour_count.tolist()}")

    mean_load = hour_load / hour_count
    mean_solar = hour_solar / hour_count
    mean_battery = hour_battery / hour_count
    mean_self_consumption = hour_self_consumption / hour_count

    return {
        "n_devices": len(files),
        "capacities": np.asarray(capacities, dtype=float),
        "peak_powers": np.asarray(peak_powers, dtype=float),
        "c_rates": np.asarray(peak_powers, dtype=float) / np.maximum(capacities, 1e-9),
        "initial_soc": np.asarray(first_soc, dtype=float),
        "profiles": {
            "solar": mean_solar / max(float(mean_solar.max()), 1e-9),
            "load": mean_load / max(float(mean_load.max()), 1e-9),
        },
        "observed_profiles": {
            "load_kw": mean_load,
            "solar_kw": mean_solar,
            "battery_kw": mean_battery,
            "self_consumption_kw": mean_self_consumption,
        },
        "profile_counts": hour_count,
    }


def _real_parameter_sample(dataset: dict[str, Any], n_devices: int, seed: int) -> dict[str, Any]:

    rng = np.random.default_rng(seed)
    n_real = int(dataset["n_devices"])
    if n_devices <= n_real:
        indices = rng.choice(n_real, size=n_devices, replace=False)
    else:
        indices = rng.choice(n_real, size=n_devices, replace=True)
    return {
        "capacities": dataset["capacities"][indices],
        "peak_powers": dataset["peak_powers"][indices],
        "c_rates": dataset["c_rates"][indices],
        "initial_soc": dataset["initial_soc"][indices],
        "n_devices": n_devices,
    }


def _install_real_data_hooks(
    dataset: dict[str, Any],
    output_root: Path,
    sample_scale: float = 1.0,
    target_devices: int | None = None,
) -> None:

    import src.simulation as simulation_api
    import src.simulation.simulator as simulator_module

    original_simulator = simulation_api.EPSSimulator
    original_simulator_module = simulator_module.EPSSimulator

    def real_solar_factors(real_profiles=None):
        if real_profiles and "solar" in real_profiles:
            return np.asarray(real_profiles["solar"], dtype=float), np.zeros(24, dtype=float)
        return dataset["profiles"]["solar"], np.zeros(24, dtype=float)

    def real_load_factors(real_profiles=None):
        if real_profiles and "load" in real_profiles:
            return np.asarray(real_profiles["load"], dtype=float)
        return dataset["profiles"]["load"]

    core._get_renewable_factors = real_solar_factors
    core._get_load_factors = real_load_factors
    core.REAL_DATA_SAMPLE_SCALE = sample_scale

    def load_real_params():
        n_target = int(target_devices or dataset["n_devices"])
        sampled = _real_parameter_sample(dataset, n_target, seed=1042)
        sampled["source_n_devices"] = int(dataset["n_devices"])
        sampled["sampling_method"] = (
            "without_replacement"
            if n_target <= int(dataset["n_devices"])
            else "empirical_bootstrap_with_replacement"
        )
        return sampled

    core._load_nextgen_params = load_real_params

    if sample_scale <= 0:
        raise ValueError("sample_scale must be positive")
    original_generate = core._generate_training_data
    original_generate_real = core._generate_training_data_real_params
    original_generate_region = core._generate_region_training_data
    original_generate_eval = core._generate_evaluation_set
    original_generate_replicated = core._generate_replicated_eval_data
    minimum_samples = max(100, int(target_devices or dataset["n_devices"]))

    def scaled_generate(n_samples, *args, **kwargs):
        scaled = max(minimum_samples, int(round(int(n_samples) * sample_scale)))
        return original_generate(scaled, *args, **kwargs)

    def scaled_generate_real(n_samples, *args, **kwargs):
        scaled = max(minimum_samples, int(round(int(n_samples) * sample_scale)))
        return original_generate_real(scaled, *args, **kwargs)

    def scaled_generate_region(n_samples, *args, **kwargs):
        scaled = max(minimum_samples, int(round(int(n_samples) * sample_scale)))
        return original_generate_region(scaled, *args, **kwargs)

    def scaled_generate_eval(n_samples, *args, **kwargs):
        scaled = max(minimum_samples, int(round(int(n_samples) * sample_scale)))
        return original_generate_eval(scaled, *args, **kwargs)

    def scaled_generate_replicated(n_unique_signals, replications, *args, **kwargs):
        target = max(minimum_samples, int(round(n_unique_signals * replications * sample_scale)))
        scaled_replications = max(1, min(5, int(replications)))
        scaled_unique_signals = int(np.ceil(target / scaled_replications))
        return original_generate_replicated(
            scaled_unique_signals,
            scaled_replications,
            *args,
            **kwargs,
        )

    core._generate_training_data = scaled_generate
    core._generate_training_data_real_params = scaled_generate_real
    core._generate_region_training_data = scaled_generate_region
    core._generate_evaluation_set = scaled_generate_eval
    core._generate_replicated_eval_data = scaled_generate_replicated
    core.REAL_DATA_MIN_SAMPLES = minimum_samples
    logging.getLogger("src.simulation.simulator").setLevel(logging.WARNING)

    class RealDataSimulator:

        def __init__(self, config):
            self._inner = original_simulator(config)
            self.config = config

        def initialize(self):
            result = self._inner.initialize()
            real_params = _real_parameter_sample(
                dataset, int(self.config.num_devices), int(self.config.random_seed or 0)
            )
            core._inject_real_params(self._inner, real_params)
            for battery, state, soc in zip(
                self._inner._population.batteries,
                self._inner._population.battery_states,
                real_params["initial_soc"],
            ):
                battery.soc = float(soc)
                state.soc = float(soc)
                state.capacity_kwh = battery.params.nominal_capacity_kwh
                state.max_charge_kw = battery.params.max_charge_power_kw
                state.max_discharge_kw = battery.params.max_discharge_power_kw
            return result

        @property
        def rng(self):
            return self._inner.rng

        @rng.setter
        def rng(self, value):
            self._inner.rng = value
            if getattr(self._inner, "_signal_generator", None) is not None:
                self._inner._signal_generator.rng = value

        def __getattr__(self, name):
            return getattr(self._inner, name)

    simulation_api.EPSSimulator = RealDataSimulator
    simulator_module.EPSSimulator = RealDataSimulator

    sequence = {"value": 0}

    def isolated_setup(base_dir: str, experiment_name: str = "experiment", config=None):
        sequence["value"] += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{stamp}_{sequence['value']:02d}_{experiment_name}_real"
        experiment_dir = output_root / experiment_id
        for name in ("data", "figures", "tables", "logs", "estimation"):
            (experiment_dir / name).mkdir(parents=True, exist_ok=True)
        metadata = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "data_source": "NextGen ACT Australia CSV",
            "n_real_devices": int(dataset["n_devices"]),
            "wind_source": "not available; set to zero",
            "simulator": "existing EPSSimulator with empirical parameter injection",
            "status": "running",
        }
        (experiment_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if config:
            (experiment_dir / "config.json").write_text(
                json.dumps(config, indent=2, default=str), encoding="utf-8"
            )
        return experiment_dir, experiment_id

    core._setup_experiment_directory = isolated_setup

    core._real_data_original_simulator = original_simulator
    core._real_data_original_simulator_module = original_simulator_module


def _write_manifest(dataset: dict[str, Any], output_root: Path, args: argparse.Namespace) -> None:
    payload = {
        "data_source": "NextGen ACT Australia (Zenodo 14885589)",
        "csv_files": int(dataset["n_devices"]),
        "records_are_used_for": [
            "empirical battery parameters",
            "initial SOC",
            "24-hour measured PV profile",
            "24-hour measured load profile",
            "observed baseline battery power and PV self-consumption statistics",
        ],
        "not_available": ["broadcast signal history", "wind power", "grid curtailment"],
        "simulator_model": "original EPSSimulator; no new physical model",
        "n_devices": args.n_devices,
        "n_runs": args.n_runs,
        "sample_scale": args.sample_scale,
        "population_expansion": {
            "method": "empirical bootstrap with replacement" if args.n_devices > int(dataset["n_devices"]) else "measured-device selection",
            "source_devices": int(dataset["n_devices"]),
            "target_devices": int(args.n_devices),
            "joint_parameter_sampling": True,
            "preserved_fields": ["capacity", "peak_power", "c_rate", "initial_soc"],
            "interpretation": "target fleet is a bootstrap pseudo-population, not independent measured sites" if args.n_devices > int(dataset["n_devices"]) else "target fleet selected from measured devices",
        },
        "created_at": datetime.now().isoformat(),
        "profile": {
            "solar_normalized": dataset["profiles"]["solar"].tolist(),
            "load_normalized": dataset["profiles"]["load"].tolist(),
            "observed_load_kw": dataset["observed_profiles"]["load_kw"].tolist(),
            "observed_solar_kw": dataset["observed_profiles"]["solar_kw"].tolist(),
            "observed_battery_kw": dataset["observed_profiles"]["battery_kw"].tolist(),
            "observed_self_consumption_kw": dataset["observed_profiles"]["self_consumption_kw"].tolist(),
        },
    }
    (output_root / "real_data_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _generate_figures(output_root: Path, figure_dir: Path) -> list[Path]:
    result1 = max(
        (p for p in output_root.iterdir() if "result1_scaling_law" in p.name),
        key=lambda p: p.stat().st_mtime,
    )
    result3 = max(
        (p for p in output_root.iterdir() if "result3_robustness" in p.name),
        key=lambda p: p.stat().st_mtime,
    )
    generated = [
        figures.make_figure2(result1, figure_dir),
        figures.make_figure3(result1, figure_dir),
        figures.make_figure4(result1, figure_dir),
        figures.make_figure4_self_consumption(result1, figure_dir),
        figures.make_figure5(result3, figure_dir),
    ]
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated real-data-driven paper experiments")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--n-devices", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=8)
    parser.add_argument(
        "--sample-scale",
        type=float,
        default=1.0,
        help="Scale repeated training sample counts; 1.0 preserves the original protocol",
    )
    parser.add_argument("--only", choices=("all", "result1", "result2", "result3", "result4"), default="all")
    args = parser.parse_args()

    dataset = _load_nextgen_dataset(args.data_dir)
    if args.n_devices < 1:
        raise ValueError("--n-devices must be positive")
    output_root = args.output_root or (
        DEFAULT_OUTPUT_ROOT / f"n{args.n_devices}_runs{args.n_runs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_manifest(dataset, output_root, args)
    _install_real_data_hooks(
        dataset,
        output_root,
        sample_scale=args.sample_scale,
        target_devices=args.n_devices,
    )

    config = core.ExperimentConfig(
        num_devices=args.n_devices,
        num_runs=args.n_runs,
        random_seed=42,
        enable_online_learning=True,
        enable_closed_loop=True,
    )

    if args.only in ("all", "result1"):
        core._run_result1(config)
    if args.only in ("all", "result2"):
        core._run_result2(config)
    if args.only in ("all", "result3"):
        core._run_result3(config)
    if args.only in ("all", "result4"):
        core._run_result4(config)

    result_dirs = [p for p in output_root.iterdir() if p.is_dir()]
    for result_dir in result_dirs:
        metadata_file = result_dir / "metadata.json"
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            metadata["status"] = "completed"
            metadata["completed_at"] = datetime.now().isoformat()
            metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.only == "all":
        figure_dir = output_root / "paper_figures"
        generated = _generate_figures(output_root, figure_dir)
        for path in generated:
            print(path)


if __name__ == "__main__":
    main()
