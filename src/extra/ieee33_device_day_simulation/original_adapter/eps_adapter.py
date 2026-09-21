from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.edge import BatteryState, create_residential_battery
from src.signal import EPSSignal
from src.simulation import EPSSimulator, SimulationConfig

from ..population.device_day_loader import DeviceDay


@dataclass
class ResponseBatch:
    desired_kw: np.ndarray
    accepted_kw: np.ndarray
    response_count: int
    signals: list[EPSSignal]


class OriginalEPSAdapter:

    def __init__(self, records: list[DeviceDay], config: dict[str, Any], seed: int):
        self.records = records
        self.config = config
        original = config["original_model"]
        sim_cfg = SimulationConfig(
            num_devices=len(records),
            duration_seconds=float(original["duration_seconds"]),
            time_step=float(original["time_step_seconds"]),
            random_seed=seed,
            num_regions=len(config["zones"]["zones"]),
            eps_packet_loss_rate=float(original["eps_packet_loss_rate"]),
            device_offline_rate=float(original["device_offline_rate"]),
            battery_capacity_range=tuple(original["battery_capacity_range_kwh"]),
            battery_c_rate_mean=float(original["battery_c_rate_mean"]),
            battery_c_rate_sigma=float(original["battery_c_rate_sigma"]),
            battery_soh_range=tuple(original["battery_soh_range"]),
            response_prob_bias=float(original.get("response_prob_bias", 1.0)),
            soc_noise_std=float(original.get("soc_noise_std", 0.0)),
        )
        self.simulator = EPSSimulator(sim_cfg).initialize()
        self._inject_population()
        self._initial_battery_state: list[tuple[float, float]] = []
        self._save_initial_battery_state()

    def _inject_population(self) -> None:
        population = self.simulator.population
        if population is None:
            raise RuntimeError("EPSSimulator did not initialize a population")
        population.batteries.clear()
        population.battery_states.clear()
        population.region_assignments.clear()
        zone_order = sorted(self.config["zones"]["zones"])
        constraints = self.config["device_constraints"]
        for index, record in enumerate(self.records):
            battery = create_residential_battery(
                capacity_kwh=record.capacity_kwh,
                c_rate=record.c_rate,
                initial_soh=float(record.initial_soh),
            )
            battery.params.soc_reserve = float(constraints["soc_min"])
            battery.params.soc_min = float(constraints["soc_min"])
            battery.params.soc_max = float(constraints["soc_max"])
            battery.soc = float(record.initial_soc)
            population.batteries.append(battery)
            zone_index = zone_order.index(record.zone_id)
            population.region_assignments[f"battery_{index}"] = zone_index
            population.battery_states.append(
                BatteryState(
                    device_id=record.device_id,
                    device_type="battery",
                    soc=record.initial_soc,
                    capacity_kwh=record.capacity_kwh,
                    max_charge_kw=record.peak_power_kw,
                    max_discharge_kw=record.peak_power_kw,
                )
            )

    def _save_initial_battery_state(self) -> None:
        population = self.simulator.population
        if population is None:
            raise RuntimeError("EPSSimulator did not initialize a population")
        self._initial_battery_state = [
            (float(battery.soc), float(battery.current_power_kw))
            for battery in population.batteries
        ]

    def reset_to_initial_state(self) -> None:
        population = self.simulator.population
        if population is None or len(population.batteries) != len(self._initial_battery_state):
            raise RuntimeError("battery population changed after initialization")
        for battery, (soc, current_power) in zip(population.batteries, self._initial_battery_state):
            battery.soc = soc
            battery.current_power_kw = current_power

    def step(self, signals: list[EPSSignal], step: int, availability: np.ndarray) -> ResponseBatch:
        if len(signals) != len(self.config["zones"]["zones"]):
            raise ValueError("one EPS signal is required per configured zone")
        desired = np.zeros(len(self.records), dtype=float)
        accepted = np.zeros(len(self.records), dtype=float)
        snapshots: list[tuple[float, float]] = []
        responses = 0
        for index, record in enumerate(self.records):
            battery = self.simulator.population.batteries[index]
            snapshots.append((float(battery.soc), float(battery.current_power_kw)))
            if self.simulator.rng.random() > float(np.clip(availability[index], 0.0, 1.0)):
                continue
            zone_index = sorted(self.config["zones"]["zones"]).index(record.zone_id)
            response = self.simulator._simulate_battery_response(
                battery,
                signals[zone_index],
                step,
                scenario="normal",
                device_id=f"battery_{index}",
            )
            if response["responded"]:
                desired[index] = float(response["power_kw"])
                responses += 1

        for index, (soc, current_power) in enumerate(snapshots):
            battery = self.simulator.population.batteries[index]
            battery.soc = soc
            battery.current_power_kw = current_power
        return ResponseBatch(desired_kw=desired, accepted_kw=accepted, response_count=responses, signals=signals)

    def apply_dispatch(self, powers_kw: np.ndarray) -> np.ndarray:
        actual = np.zeros(len(self.records), dtype=float)
        for index, power in enumerate(powers_kw):
            if abs(float(power)) < 1e-12:
                continue
            update = self.simulator.population.batteries[index].update(
                power_kw=float(power),
                dt_seconds=float(self.config["original_model"]["time_step_seconds"]),
            )
            actual[index] = float(update["actual_power_kw"])
        return actual

    def battery_states(self) -> np.ndarray:
        return np.asarray([battery.soc for battery in self.simulator.population.batteries], dtype=float)
