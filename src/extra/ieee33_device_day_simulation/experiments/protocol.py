from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ..network.ieee33_distflow import IEEE33DistFlow
from ..original_adapter.eps_adapter import OriginalEPSAdapter
from ..population.device_day_loader import DeviceDay


@dataclass
class SimulationTrace:
    time_index: list[int]
    load_kw: list[float]
    pv_kw: list[float]
    energy_input_kw: list[float]
    baseline_battery_kw: list[float]
    desired_control_kw: list[float]
    accepted_control_kw: list[float]
    net_grid_kw: list[float]
    intensity: list[float]
    direction: list[int]
    network_scale: list[float]
    baseline_transformer_loading: list[float]
    baseline_maximum_branch_loading: list[float]
    baseline_minimum_voltage_pu: list[float]
    baseline_maximum_voltage_pu: list[float]
    transformer_loading: list[float]
    minimum_voltage_pu: list[float]
    maximum_voltage_pu: list[float]
    overloaded_branches: list[int]
    voltage_violations: list[int]
    self_consumption_baseline_kwh: list[float]
    self_consumption_eps_kwh: list[float]
    zone_control_kw: dict[str, list[float]]
    bus_control_kw: dict[str, list[float]]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def concatenate_traces(traces: list[SimulationTrace], sample_count: int) -> SimulationTrace:
    if not traces:
        raise ValueError("at least one trace is required")
    scalar_fields = [
        name
        for name in SimulationTrace.__annotations__
        if name not in {"time_index", "zone_control_kw", "bus_control_kw"}
    ]
    values: dict[str, Any] = {name: [] for name in scalar_fields}
    for trace in traces:
        for name in scalar_fields:
            values[name].extend(getattr(trace, name))
    values["time_index"] = list(range(min(sample_count, len(values[scalar_fields[0]]))))

    for grouped_field in ("zone_control_kw", "bus_control_kw"):
        keys = sorted({key for trace in traces for key in getattr(trace, grouped_field)})
        grouped: dict[str, list[float]] = {key: [] for key in keys}
        for trace in traces:
            trace_length = len(trace.time_index)
            source = getattr(trace, grouped_field)
            for key in keys:
                grouped[key].extend(source.get(key, [0.0] * trace_length))
        values[grouped_field] = grouped

    for name in scalar_fields:
        values[name] = values[name][:sample_count]
    values["time_index"] = values["time_index"][:sample_count]
    for grouped_field in ("zone_control_kw", "bus_control_kw"):
        values[grouped_field] = {
            key: series[:sample_count] for key, series in values[grouped_field].items()
        }
    if len(values["time_index"]) != sample_count:
        raise ValueError(f"requested {sample_count} validation samples, got {len(values['time_index'])}")
    return SimulationTrace(**values)


def _signal_values(net_supply_kw: float, total_load_kw: float, scenario: str, rng: np.random.Generator) -> tuple[int, int]:
    if scenario == "peak_shaving":
        direction = 10
        magnitude = 0.82
    elif scenario == "valley_filling":
        direction = 5
        magnitude = 0.82
    elif scenario in {"emergency", "emergency_grid_stability", "emergency_supply_shortage"}:
        direction = 14
        magnitude = 0.95
    else:
        direction = 6 if net_supply_kw >= 0 else 10
        magnitude = min(0.95, max(0.12, abs(net_supply_kw) / max(total_load_kw, 1.0)))
    intensity = int(np.clip(round(magnitude * 4095 + rng.normal(0, 15)), 0, 4095))
    return direction, intensity


def _energy_input(record: DeviceDay, profile_index: int, config: dict[str, Any]) -> float:
    field = str(config.get("simulation", {}).get("energy_input_field", "energy_input_kw"))
    if field == "pv_kw":
        value = record.pv_kw[profile_index]
    elif field == "energy_input_kw":
        value = record.energy_input_kw[profile_index]
    else:
        raise ValueError(
            f"unsupported simulation.energy_input_field={field!r}; "
            "the adapter must expose the source as energy_input_kw or pv_kw"
        )
    return max(0.0, float(value))


def _input_multiplier(zone_config: dict[str, Any]) -> float:
    return float(zone_config.get("energy_input_multiplier", zone_config.get("pv_multiplier", 1.0)))


def build_real_signal_schedule(
    records: list[DeviceDay],
    config: dict[str, Any],
    profile_steps: list[int],
    *,
    seed: int,
    scenario: str = "normal",
) -> list[list[tuple[int, int]]]:
    zones_cfg = config["zones"]["zones"]
    zone_ids = sorted(zones_cfg)
    rng = np.random.default_rng(seed)
    schedule: list[list[tuple[int, int]]] = []
    for profile_step in profile_steps:
        specs: list[tuple[int, int]] = []
        for zone in zone_ids:
            zcfg = zones_cfg[zone]
            indices = [i for i, record in enumerate(records) if record.zone_id == zone]
            load = sum(
                float(records[i].load_kw[profile_step]) * float(zcfg["load_multiplier"])
                for i in indices
            )
            energy_input = sum(
                _energy_input(records[i], profile_step, config) * _input_multiplier(zcfg)
                for i in indices
            )
            specs.append(_signal_values(energy_input - load, load, scenario, rng))
        schedule.append(specs)
    return schedule


def build_condition_signal_schedule(
    records: list[DeviceDay],
    config: dict[str, Any],
    profile_steps: list[int],
    scenarios: list[str],
    *,
    seed: int,
) -> list[list[tuple[int, int]]]:
    if not scenarios:
        raise ValueError("at least one threshold signal scenario is required")
    schedule = []
    for index, profile_step in enumerate(profile_steps):
        schedule.extend(build_real_signal_schedule(
            records,
            config,
            [profile_step],
            seed=seed + index,
            scenario=scenarios[index % len(scenarios)],
        ))
    return schedule


def build_scenario_profile_signal_schedule(
    config: dict[str, Any],
    scenario: str,
    sample_count: int,
    *,
    seed: int | None = None,
    ratio_noise_std: float = 0.0,
) -> list[list[tuple[int, int]]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if scenario not in {
        "peak_shaving",
        "valley_filling",
        "emergency_grid_stability",
        "emergency_supply_shortage",
    }:
        raise ValueError(f"unsupported scenario profile: {scenario}")

    zone_count = len(config["zones"]["zones"])
    step_hours = float(config["original_model"]["time_step_seconds"]) / 3600.0
    cycle_hours = {
        "peak_shaving": 3.0,
        "valley_filling": 4.0,
        "emergency_grid_stability": 1.0,
        "emergency_supply_shortage": 1.0,
    }[scenario]
    rng = np.random.default_rng(seed)
    schedule: list[list[tuple[int, int]]] = []
    for index in range(sample_count):
        phase = ((index * step_hours) % cycle_hours) / cycle_hours
        if scenario == "peak_shaving":
            r = 0.70 - 0.20 * np.sin(np.pi * phase)
            dispatch = 0.50 + 0.50 * np.sin(np.pi * phase)
        elif scenario == "valley_filling":
            r = 1.20 + 0.30 * np.sin(np.pi * phase)
            dispatch = 0.50 + 0.40 * np.sin(np.pi * phase)
        else:
            transition_end = 1.0 / 12.0
            sustained_end = 7.0 / 12.0
            if phase < transition_end:
                fraction = phase / transition_end
            elif phase < sustained_end:
                fraction = 1.0
            else:
                fraction = max(0.0, 1.0 - (phase - sustained_end) / (5.0 / 12.0))
            if scenario == "emergency_grid_stability":
                r = 1.0 + 0.50 * fraction
            else:
                r = 1.0 - 0.40 * fraction
            dispatch = 1.0 if phase < sustained_end else 0.50 + 0.50 * fraction

        if ratio_noise_std > 0:
            r *= 1.0 + float(rng.normal(0.0, ratio_noise_std))
        score = float(np.clip((r - 1.0) * dispatch, -1.0, 1.0))
        magnitude = abs(score)
        intensity = int(np.clip(round(magnitude * 4095), 0, 4095))
        if score >= 0:
            direction = int(np.clip(round(7 * (1.0 - magnitude)), 0, 7))
        else:
            direction = int(np.clip(round(8 + 7 * magnitude), 8, 15))
        schedule.append([(direction, intensity)] * zone_count)
    return schedule


def build_experiments_training_signal_schedule(
    config: dict[str, Any],
    sample_count: int,
    *,
    seed: int,
) -> list[list[tuple[int, int]]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rng = np.random.default_rng(seed)
    zone_count = len(config["zones"]["zones"])
    n_scenario = int(sample_count * 0.80)
    n_time_aware = int(sample_count * 0.10)
    profiles = (
        "valley_filling",
        "peak_shaving",
        "emergency_grid_stability",
        "emergency_supply_shortage",
    )
    per_profile = max(n_scenario // len(profiles), 1)
    schedule: list[list[tuple[int, int]]] = []
    for index in range(sample_count):
        if index < n_scenario:
            scenario = profiles[min(index // per_profile, len(profiles) - 1)]
            phase = float(rng.uniform(0.0, 1.0))
            if scenario == "peak_shaving":
                r = 0.70 - 0.20 * np.sin(np.pi * phase)
                dispatch = 0.50 + 0.50 * np.sin(np.pi * phase)
            elif scenario == "valley_filling":
                r = 1.20 + 0.30 * np.sin(np.pi * phase)
                dispatch = 0.50 + 0.40 * np.sin(np.pi * phase)
            else:
                transition_end = 1.0 / 12.0
                sustained_end = 7.0 / 12.0
                if phase < transition_end:
                    fraction = phase / transition_end
                elif phase < sustained_end:
                    fraction = 1.0
                else:
                    fraction = max(
                        0.0,
                        1.0 - (phase - sustained_end) / (5.0 / 12.0),
                    )
                r = 1.0 + 0.50 * fraction if scenario == "emergency_grid_stability" else 1.0 - 0.40 * fraction
                dispatch = 1.0 if phase < sustained_end else 0.50 + 0.50 * fraction
            score = float(np.clip((r - 1.0) * dispatch + rng.normal(0.0, 0.02), -1.0, 1.0))
            intensity = int(np.clip(round(abs(score) * 4095), 0, 4095))
            direction = (
                int(np.clip(round(7 * (1.0 - abs(score))), 0, 7))
                if score >= 0
                else int(np.clip(round(8 + 7 * abs(score)), 8, 15))
            )
        elif index < n_scenario + n_time_aware:
            hour = int(rng.integers(0, 24))
            if 7 <= hour <= 11 or 17 <= hour <= 21:
                intensity = int(rng.normal(2800, 500) if rng.random() < 0.5 else rng.normal(1000, 400))
                direction = int(rng.normal(11, 2))
            elif hour <= 6 or hour >= 22:
                intensity = int(rng.normal(1000, 300))
                direction = int(rng.normal(3, 2))
            else:
                intensity = int(rng.normal(2048, 600))
                direction = int(rng.normal(8, 2))
            intensity = int(np.clip(intensity, 0, 4095))
            direction = int(np.clip(direction, 0, 15))
        else:
            intensity = int(rng.uniform(0, 4095))
            direction = int(rng.uniform(0, 16))
        schedule.append([(direction, intensity)] * zone_count)
    return schedule


def build_experiments_transfer_signal_schedule(
    config: dict[str, Any],
    sample_count: int,
    *,
    seed: int,
) -> tuple[list[list[tuple[int, int]]], list[int]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rng = np.random.default_rng(seed)
    zone_count = len(config["zones"]["zones"])
    time_aware_count = int(sample_count * 0.70)
    schedule: list[list[tuple[int, int]]] = []
    hours: list[int] = []
    for index in range(sample_count):
        hour = int(rng.integers(0, 24))
        if index < time_aware_count:
            if 7 <= hour <= 11 or 17 <= hour <= 21:
                intensity = int(rng.normal(3000, 500))
                direction = int(rng.normal(12, 2))
            elif hour <= 6 or hour >= 22:
                intensity = int(rng.normal(1000, 300))
                direction = int(rng.normal(3, 2))
            else:
                intensity = int(rng.normal(2048, 600))
                direction = int(rng.normal(8, 2))
        else:
            intensity = int(rng.uniform(0, 4095))
            direction = int(rng.uniform(0, 16))
        intensity = int(np.clip(intensity, 0, 4095))
        direction = int(np.clip(direction, 0, 15))
        schedule.append([(direction, intensity)] * zone_count)
        hours.append(hour)
    return schedule, hours


def build_stratified_signal_schedule(
    config: dict[str, Any],
    sample_count: int,
    *,
    seed: int,
    min_abs_score: float = 0.05,
    max_abs_score: float = 0.95,
) -> list[list[tuple[int, int]]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not 0.0 <= min_abs_score < max_abs_score <= 1.0:
        raise ValueError("signal score bounds must satisfy 0 <= min < max <= 1")

    zone_count = len(config["zones"]["zones"])
    if zone_count <= 0:
        raise ValueError("at least one zone is required")

    rng = np.random.default_rng(seed)
    direction_counts = (sample_count // 2, sample_count - sample_count // 2)
    signed_scores: list[float] = []
    for sign, count in zip((1.0, -1.0), direction_counts):
        fractions = (np.arange(count, dtype=float) + 0.5) / count
        magnitudes = min_abs_score + fractions * (max_abs_score - min_abs_score)
        signed_scores.extend((sign * magnitudes).tolist())
    rng.shuffle(signed_scores)

    schedule: list[list[tuple[int, int]]] = []
    for score in signed_scores:
        magnitude = min(abs(float(score)), 1.0)
        intensity = int(np.clip(round(magnitude * 4095), 0, 4095))
        if score >= 0:
            direction = int(np.clip(round(7 * (1.0 - magnitude)), 0, 7))
        else:
            direction = int(np.clip(round(8 + 7 * magnitude), 8, 15))
        schedule.append([(direction, intensity)] * zone_count)
    return schedule


def _sample_original_region_perturbations(
    rng: np.random.Generator,
    region_count: int,
    global_std: float,
    regional_std: float,
) -> dict[int, float] | None:
    if global_std < 0.0 or regional_std < 0.0:
        raise ValueError("original correlation standard deviations must be nonnegative")
    if global_std == 0.0 and regional_std == 0.0:
        return None
    persistent_global_shock = rng.normal(0.0, global_std)
    return {
        region: persistent_global_shock + rng.normal(0.0, regional_std)
        for region in range(region_count)
    }


def simulate_day(
    records: list[DeviceDay],
    config: dict[str, Any],
    *,
    steps: int,
    seed: int,
    scenario: str = "normal",
    zone_correlation: float = 0.0,
    original_global_shock_std: float | None = None,
    original_regional_shock_std: float | None = None,
    original_response_seed: int | None = None,
    profile_step: int | None = None,
    profile_steps: list[int] | None = None,
    signal_override: list[tuple[int, int]] | None = None,
    signal_overrides: list[list[tuple[int, int]]] | None = None,
    profile_steps_by_record: list[list[int]] | None = None,
    resource_response_sink: list[list[float]] | None = None,
    availability_overrides: list[list[float]] | np.ndarray | None = None,
    response_policy: Callable[[int, list[Any], np.ndarray, OriginalEPSAdapter], np.ndarray] | None = None,
    network_diagnostic_sink: list[dict[str, float | int]] | None = None,
    reset_each_step: bool = False,
    adapter_sink: list[Any] | None = None,
) -> SimulationTrace:
    if not records:
        raise ValueError("simulate_day requires at least one device-day")
    zones_cfg = config["zones"]["zones"]
    zone_ids = sorted(zones_cfg)
    network = IEEE33DistFlow(config["network"], config["control"])
    adapter = OriginalEPSAdapter(records, config, seed=seed)
    rng = np.random.default_rng(seed + 17)
    use_original_correlation = (
        original_global_shock_std is not None
        or original_regional_shock_std is not None
    )
    if use_original_correlation:
        if original_response_seed is not None:
            adapter.simulator.rng = np.random.default_rng(original_response_seed)
        global_std = float(original_global_shock_std or 0.0)
        regional_std = float(original_regional_shock_std or 0.0)
        adapter.simulator._region_perturbation = _sample_original_region_perturbations(
            adapter.simulator.rng,
            len(zone_ids),
            global_std,
            regional_std,
        )
    if profile_steps is None and profile_steps_by_record is None:
        steps = min(steps, len(records[0].load_kw))
    if profile_steps_by_record is not None:
        if len(profile_steps_by_record) < steps:
            raise ValueError("profile_steps_by_record must contain one row per simulated step")
        if any(len(row) != len(records) for row in profile_steps_by_record[:steps]):
            raise ValueError("profile_steps_by_record rows must match the resource count")
    if availability_overrides is not None:
        availability_array = np.asarray(availability_overrides, dtype=float)
        if availability_array.ndim != 2 or availability_array.shape[0] < steps:
            raise ValueError("availability_overrides must contain one row per simulated step")
        if availability_array.shape[1] != len(records):
            raise ValueError("availability_overrides rows must match the resource count")
    else:
        availability_array = None
    zone_records: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        zone_records[record.zone_id].append(index)
    zone_shocks = {
        zone: (
            rng.normal(0, np.sqrt(max(zone_correlation, 0.0)))
            if zone_correlation and not use_original_correlation
            else 0.0
        )
        for zone in zone_ids
    }
    fields: dict[str, list[Any]] = {key: [] for key in SimulationTrace.__annotations__ if key not in {"zone_control_kw", "bus_control_kw"}}
    fields["zone_control_kw"] = {zone: [] for zone in zone_ids}
    fields["bus_control_kw"] = {}

    for step in range(steps):
        if reset_each_step:
            adapter.reset_to_initial_state()
        if profile_steps_by_record is not None:
            profile_indices = [
                int(np.clip(profile_steps_by_record[step][index], 0, len(records[index].load_kw) - 1))
                for index in range(len(records))
            ]
        elif profile_steps is not None:
            if len(profile_steps) < steps:
                raise ValueError("profile_steps must contain one entry per simulated step")
            profile_index = int(profile_steps[step])
        elif profile_step is None:
            profile_index = step
        else:
                profile_index = int(profile_step) + step
        if profile_steps_by_record is None:
            profile_index = min(max(profile_index, 0), len(records[0].load_kw) - 1)
            profile_indices = [profile_index] * len(records)
        load_by_bus: dict[int, float] = defaultdict(float)
        input_by_bus: dict[int, float] = defaultdict(float)
        desired_by_bus: dict[int, float] = defaultdict(float)
        load = pv = energy_input = baseline_battery = 0.0
        zone_net_supply: dict[str, float] = {}
        zone_load: dict[str, float] = {}
        for zone in zone_ids:
            zcfg = zones_cfg[zone]
            indices = zone_records[zone]
            zload = sum(float(records[i].load_kw[profile_indices[i]]) * float(zcfg["load_multiplier"]) for i in indices)
            zinput = sum(_energy_input(records[i], profile_indices[i], config) * _input_multiplier(zcfg) for i in indices)
            zone_load[zone] = zload
            zone_net_supply[zone] = zinput - zload
            for i in indices:
                record = records[i]
                record_index = profile_indices[i]
                load_by_bus[record.bus_id] += float(record.load_kw[record_index]) * float(zcfg["load_multiplier"])
                input_by_bus[record.bus_id] += _energy_input(record, record_index, config) * _input_multiplier(zcfg)
                load += float(record.load_kw[record_index]) * float(zcfg["load_multiplier"])
                pv += float(record.pv_kw[record_index]) * _input_multiplier(zcfg)
                energy_input += _energy_input(record, record_index, config) * _input_multiplier(zcfg)
                baseline_battery += float(record.baseline_battery_kw[record_index])
        signals = []
        for zone_index, zone in enumerate(zone_ids):
            if signal_overrides is not None:
                if len(signal_overrides) < steps:
                    raise ValueError("signal_overrides must contain one entry per simulated step")
                direction, intensity = signal_overrides[step][zone_index]
            elif signal_override is not None:
                if len(signal_override) != len(zone_ids):
                    raise ValueError("signal_override must contain one pair per zone")
                direction, intensity = signal_override[zone_index]
            else:
                direction, intensity = _signal_values(zone_net_supply[zone], zone_load[zone], scenario, rng)
            signals.append(adapter.simulator._signal_generator.generate_signal(zone_index, direction, intensity, priority=10))

        availability = np.ones(len(records), dtype=float)
        for i, record in enumerate(records):
            availability[i] = float(zones_cfg[record.zone_id]["availability_multiplier"])
        if availability_array is not None:
            availability *= np.clip(availability_array[step], 0.0, 1.0)
        batch = adapter.step(signals, step, availability)
        desired_kw = np.asarray(batch.desired_kw, dtype=float)
        if response_policy is not None:
            desired_kw = np.asarray(
                response_policy(step, signals, desired_kw, adapter), dtype=float
            )
            if desired_kw.shape != batch.desired_kw.shape:
                raise ValueError("response_policy must return one power value per resource")
        for i, record in enumerate(records):
            zone_shock = 1.0 + zone_shocks[record.zone_id]
            desired_by_bus[record.bus_id] += desired_kw[i] * zone_shock
        network_feedback = bool(config["control"].get("network_feedback", True))
        baseline_network = network.evaluate(
            dict(load_by_bus), dict(input_by_bus), {}, apply_limits=False
        )
        if network_feedback:
            scale, network_after = network.constrained_dispatch_scale(
                dict(load_by_bus), dict(input_by_bus), dict(desired_by_bus)
            )
        else:
            scale = 1.0
            network_after = network.evaluate(
                dict(load_by_bus), dict(input_by_bus), dict(desired_by_bus), apply_limits=False
            )
        zone_factors = np.asarray([1.0 + zone_shocks[r.zone_id] for r in records])
        accepted = adapter.apply_dispatch(desired_kw * scale * zone_factors)
        if resource_response_sink is not None:
            resource_response_sink.append(accepted.astype(float).tolist())
        accepted_by_bus: dict[int, float] = defaultdict(float)
        zone_accepted: dict[str, float] = defaultdict(float)
        for i, record in enumerate(records):
            accepted_by_bus[record.bus_id] += float(accepted[i])
            zone_accepted[record.zone_id] += float(accepted[i])
        final_eval = network.evaluate(dict(load_by_bus), dict(input_by_bus), dict(accepted_by_bus), apply_limits=False)
        if network_diagnostic_sink is not None:
            network_diagnostic_sink.append({
                "network_scale": float(scale),
                "maximum_branch_loading": float(max(final_eval.branch_loading.values(), default=0.0)),
                "transformer_loading": float(final_eval.transformer_loading),
                "minimum_voltage_pu": float(min(final_eval.voltage_pu.values())),
                "maximum_voltage_pu": float(max(final_eval.voltage_pu.values())),
                "overloaded_branches": int(final_eval.overloaded_branches),
                "voltage_violations": int(final_eval.voltage_violations),
            })
        for key, value in accepted_by_bus.items():
            fields["bus_control_kw"].setdefault(str(key), []).append(float(value))
        for zone in zone_ids:
            fields["zone_control_kw"][zone].append(float(zone_accepted[zone]))
        dt_hours = float(config["simulation"]["time_step_seconds"]) / 3600.0
        direct_self = min(energy_input, load) * dt_hours
        eps_self = min(energy_input, load + max(float(np.sum(accepted)), 0.0)) * dt_hours
        fields["time_index"].append(step)
        fields["load_kw"].append(load)
        fields["pv_kw"].append(pv)
        fields["energy_input_kw"].append(energy_input)
        fields["baseline_battery_kw"].append(baseline_battery)
        fields["desired_control_kw"].append(float(np.sum(desired_kw)))
        fields["accepted_control_kw"].append(float(np.sum(accepted)))
        fields["net_grid_kw"].append(load - energy_input + float(np.sum(accepted)))
        fields["intensity"].append(float(np.mean([signal.intensity for signal in signals])))
        fields["direction"].append(int(round(np.mean([signal.supply_demand for signal in signals]))))
        fields["network_scale"].append(float(scale))
        fields["baseline_transformer_loading"].append(float(baseline_network.transformer_loading))
        fields["baseline_maximum_branch_loading"].append(float(max(
            baseline_network.branch_loading.values(), default=0.0
        )))
        fields["baseline_minimum_voltage_pu"].append(float(min(baseline_network.voltage_pu.values())))
        fields["baseline_maximum_voltage_pu"].append(float(max(baseline_network.voltage_pu.values())))
        fields["transformer_loading"].append(float(final_eval.transformer_loading))
        fields["minimum_voltage_pu"].append(float(min(final_eval.voltage_pu.values())))
        fields["maximum_voltage_pu"].append(float(max(final_eval.voltage_pu.values())))
        fields["overloaded_branches"].append(int(final_eval.overloaded_branches))
        fields["voltage_violations"].append(int(final_eval.voltage_violations))
        fields["self_consumption_baseline_kwh"].append(float(direct_self))
        fields["self_consumption_eps_kwh"].append(float(eps_self))
    if adapter_sink is not None:
        adapter_sink.append(adapter)
    return SimulationTrace(**fields)


TRACE_FEATURE_NAMES = [
    "intercept",
    "intensity_0_1",
    "shortage_direction",
    "signed_intensity",
    "signed_intensity_squared",
    "signed_intensity_cubed",
    "signed_hinge_0_25",
    "signed_hinge_0_50",
    "signed_hinge_0_75",
    "real_load_10mw",
    "real_energy_input_10mw",
    "signed_intensity_x_load",
    "signed_intensity_x_input",
    "baseline_transformer_loading",
    "baseline_maximum_branch_loading",
    "baseline_low_voltage_margin",
    "baseline_high_voltage_margin",
    "signed_intensity_x_branch_loading",
    "signed_intensity_x_low_voltage_margin",
]


def trace_features(trace: SimulationTrace) -> np.ndarray:
    intensity = np.asarray(trace.intensity, dtype=float) / 4095.0
    shortage = np.asarray(trace.direction) >= 8
    sign = np.where(shortage, -1.0, 1.0)
    signed_intensity = intensity * sign
    load = np.asarray(trace.load_kw, dtype=float) / 10000.0
    energy_input = np.asarray(trace.energy_input_kw, dtype=float) / 10000.0
    branch_loading = np.asarray(trace.baseline_maximum_branch_loading, dtype=float)
    low_voltage_margin = np.asarray(trace.baseline_minimum_voltage_pu, dtype=float) - 0.95
    high_voltage_margin = 1.05 - np.asarray(trace.baseline_maximum_voltage_pu, dtype=float)
    return np.column_stack([
        np.ones(len(trace.intensity)),
        intensity,
        shortage,
        signed_intensity,
        sign * intensity ** 2,
        sign * intensity ** 3,
        sign * np.maximum(intensity - 0.25, 0.0),
        sign * np.maximum(intensity - 0.50, 0.0),
        sign * np.maximum(intensity - 0.75, 0.0),
        load,
        energy_input,
        signed_intensity * load,
        signed_intensity * energy_input,
        np.asarray(trace.baseline_transformer_loading, dtype=float),
        branch_loading,
        low_voltage_margin,
        high_voltage_margin,
        signed_intensity * branch_loading,
        signed_intensity * low_voltage_margin,
    ]).astype(float)


def r2_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator) if denominator > 1e-12 else 0.0


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=True, indent=2)
