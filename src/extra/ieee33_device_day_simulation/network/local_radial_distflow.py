from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


TOLERANCE = 1e-9


@dataclass
class RadialEvaluation:
    branch_p_kw: dict[str, float]
    branch_q_kvar: dict[str, float]
    branch_loading: dict[str, float]
    voltage_pu: dict[int, float]
    transformer_loading: float
    loss_kw: float
    voltage_violations: int
    overloaded_branches: int


@dataclass
class LocalDispatch:
    accepted_kw: np.ndarray
    requested: RadialEvaluation
    executed: RadialEvaluation
    baseline: RadialEvaluation
    acceptance_ratio: float
    local_clip_fraction: float
    fallback_global: bool


def load_network_case(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class LocalRadialDistFlow:

    def __init__(
        self,
        config: dict[str, Any],
        *,
        capacity_multiplier: float = 1.0,
        transformer_multiplier: float = 1.0,
        branch_deratings: dict[str, float] | None = None,
    ) -> None:
        self.config = config
        self.name = str(config["name"])
        self.slack = int(config["slack_bus"])
        self.base_kv = float(config["base_kv"])
        self.vmin, self.vmax = map(float, config["voltage_limits_pu"])
        self.dispatch_limit = float(config.get("dispatch_limit_fraction", 0.995))
        self.voltage_margin = float(config.get("dispatch_voltage_margin_pu", 0.0002))
        self.default_power_factor = float(config.get("power_factor", 0.95))
        self.transformer_capacity = (
            float(config["transformer_capacity_kva"])
            * float(transformer_multiplier)
        )
        self.branches = [
            {
                "from": int(row[0]),
                "to": int(row[1]),
                "r": float(row[2]),
                "x": float(row[3]),
            }
            for row in config["branches"]
        ]
        self.parent = {branch["to"]: branch["from"] for branch in self.branches}
        self.branch_by_child = {branch["to"]: branch for branch in self.branches}
        self.children: dict[int, list[int]] = {}
        for branch in self.branches:
            self.children.setdefault(branch["from"], []).append(branch["to"])
        self.buses = tuple(sorted({self.slack} | set(self.parent)))
        self.depth = {self.slack: 0}
        self._assign_depth(self.slack)
        self.descendants = {bus: frozenset(self._descendants(bus)) for bus in self.buses}

        base_loads = config.get("base_loads", [])
        self.base_p_kw = {int(row[0]): float(row[1]) for row in base_loads}
        self.base_q_kvar = {int(row[0]): float(row[2]) for row in base_loads}
        tangent = np.tan(np.arccos(np.clip(self.default_power_factor, 0.01, 1.0)))
        self.q_per_p = {
            bus: (
                self.base_q_kvar.get(bus, 0.0) / self.base_p_kw[bus]
                if self.base_p_kw.get(bus, 0.0) > TOLERANCE
                else float(tangent)
            )
            for bus in self.buses
        }
        calibration = config["thermal_calibration"]
        reference_loading = float(calibration["reference_loading"])
        minimum_capacity = float(calibration["minimum_branch_capacity_kva"])
        base_p_flow = self._subtree_totals(self.base_p_kw)
        base_q_flow = self._subtree_totals(self.base_q_kvar)
        deratings = branch_deratings or {}
        self.branch_capacity: dict[str, float] = {}
        for branch in self.branches:
            child = branch["to"]
            key = self._key(branch)
            reference_flow = float(np.hypot(base_p_flow[child], base_q_flow[child]))
            derating = float(deratings.get(key, 1.0))
            if not 0.0 < derating <= 1.0:
                raise ValueError(f"{key}: the branch derating factor must lie in (0, 1]")
            self.branch_capacity[key] = (
                max(reference_flow / reference_loading, minimum_capacity)
                * float(capacity_multiplier)
                * derating
            )

    def _assign_depth(self, bus: int) -> None:
        for child in self.children.get(bus, []):
            self.depth[child] = self.depth[bus] + 1
            self._assign_depth(child)

    def _descendants(self, bus: int) -> set[int]:
        values = {bus}
        for child in self.children.get(bus, []):
            values.update(self._descendants(child))
        return values

    @staticmethod
    def _key(branch: dict[str, float]) -> str:
        return f"{int(branch['from'])}-{int(branch['to'])}"

    def _subtree_totals(self, values: dict[int, float]) -> dict[int, float]:
        totals: dict[int, float] = {}

        def visit(bus: int) -> float:
            total = float(values.get(bus, 0.0))
            for child in self.children.get(bus, []):
                total += visit(child)
            totals[bus] = total
            return total

        visit(self.slack)
        return totals

    def _reactive_load(self, load_kw: dict[int, float]) -> dict[int, float]:
        return {
            bus: max(float(power), 0.0) * self.q_per_p.get(bus, 0.0)
            for bus, power in load_kw.items()
        }

    def evaluate(
        self,
        load_kw: dict[int, float],
        input_kw: dict[int, float],
        control_kw: dict[int, float],
    ) -> RadialEvaluation:
        net_p = {
            bus: (
                float(load_kw.get(bus, 0.0))
                - float(input_kw.get(bus, 0.0))
                + float(control_kw.get(bus, 0.0))
            )
            for bus in self.buses
        }
        net_q = self._reactive_load(load_kw)
        p_total = self._subtree_totals(net_p)
        q_total = self._subtree_totals(net_q)
        branch_p: dict[str, float] = {}
        branch_q: dict[str, float] = {}
        branch_loading: dict[str, float] = {}
        voltage = {self.slack: 1.0}
        loss_kw = 0.0
        for branch in sorted(self.branches, key=lambda row: self.depth[row["to"]]):
            key = self._key(branch)
            child = branch["to"]
            p_flow = p_total[child]
            q_flow = q_total[child]
            branch_p[key] = p_flow
            branch_q[key] = q_flow
            branch_loading[key] = float(
                np.hypot(p_flow, q_flow) / max(self.branch_capacity[key], TOLERANCE)
            )
            drop = (
                branch["r"] * p_flow + branch["x"] * q_flow
            ) / max(1000.0 * self.base_kv**2, TOLERANCE)
            voltage[child] = voltage[branch["from"]] - drop
            loss_kw += (
                branch["r"] * (p_flow**2 + q_flow**2)
                / max(1000.0 * self.base_kv**2, TOLERANCE)
            )
        root_apparent = float(np.hypot(p_total[self.slack], q_total[self.slack]))
        transformer_loading = root_apparent / max(self.transformer_capacity, TOLERANCE)
        return RadialEvaluation(
            branch_p_kw=branch_p,
            branch_q_kvar=branch_q,
            branch_loading=branch_loading,
            voltage_pu=voltage,
            transformer_loading=transformer_loading,
            loss_kw=float(loss_kw),
            voltage_violations=sum(
                value < self.vmin - TOLERANCE or value > self.vmax + TOLERANCE
                for value in voltage.values()
            ),
            overloaded_branches=sum(
                value > 1.0 + TOLERANCE for value in branch_loading.values()
            ),
        )

    def _safe(self, current: RadialEvaluation, baseline: RadialEvaluation) -> bool:
        for key, loading in current.branch_loading.items():
            allowed = max(self.dispatch_limit, baseline.branch_loading[key])
            if loading > allowed + 1e-8:
                return False
        if current.transformer_loading > max(
            self.dispatch_limit, baseline.transformer_loading
        ) + 1e-8:
            return False
        for bus, value in current.voltage_pu.items():
            lower = min(self.vmin + self.voltage_margin, baseline.voltage_pu[bus])
            upper = max(self.vmax - self.voltage_margin, baseline.voltage_pu[bus])
            if value < lower - 1e-8 or value > upper + 1e-8:
                return False
        return True

    def _control_map(self, values: np.ndarray, device_buses: np.ndarray) -> dict[int, float]:
        totals = np.bincount(
            np.asarray(device_buses, dtype=int),
            weights=np.asarray(values, dtype=float),
            minlength=max(self.buses) + 1,
        )
        return {
            int(bus): float(totals[bus])
            for bus in np.flatnonzero(np.abs(totals) > TOLERANCE)
        }

    def _served_input(
        self,
        load_kw: dict[int, float],
        potential_input_kw: dict[int, float],
        control_kw: dict[int, float],
    ) -> dict[int, float]:
        return {
            bus: min(
                max(float(potential_input_kw.get(bus, 0.0)), 0.0),
                max(float(load_kw.get(bus, 0.0)), 0.0)
                + max(float(control_kw.get(bus, 0.0)), 0.0),
            )
            for bus in self.buses
        }

    def _evaluate_candidate(
        self,
        load_kw: dict[int, float],
        potential_input_kw: dict[int, float],
        control_kw: dict[int, float],
    ) -> RadialEvaluation:
        served_input = self._served_input(load_kw, potential_input_kw, control_kw)
        return self.evaluate(load_kw, served_input, control_kw)

    def _voltage_region(self, bus: int) -> frozenset[int]:
        current = int(bus)
        candidate = current
        while current != self.slack:
            parent = self.parent[current]
            candidate = current
            if len(self.children.get(parent, [])) > 1:
                break
            current = parent
        return self.descendants[candidate]

    def _bisect_region(
        self,
        accepted: np.ndarray,
        mask: np.ndarray,
        load_kw: dict[int, float],
        potential_input_kw: dict[int, float],
        baseline: RadialEvaluation,
    ) -> np.ndarray:
        if not np.any(mask):
            return accepted
        original = accepted.copy()
        low, high = 0.0, 1.0
        candidate = accepted.copy()
        candidate[mask] = 0.0
        zero_evaluation = self._evaluate_candidate(
            load_kw,
            potential_input_kw,
            self._control_map(candidate, self._device_buses),
        )
        if not self._safe(zero_evaluation, baseline):
            return candidate
        for _ in range(14):
            middle = (low + high) / 2.0
            candidate = accepted.copy()
            candidate[mask] = original[mask] * middle
            evaluation = self._evaluate_candidate(
                load_kw,
                potential_input_kw,
                self._control_map(candidate, self._device_buses),
            )
            if self._safe(evaluation, baseline):
                low = middle
            else:
                high = middle
        accepted[mask] = original[mask] * low
        return accepted

    def dispatch(
        self,
        load_kw: dict[int, float],
        potential_input_kw: dict[int, float],
        desired_kw: np.ndarray,
        device_buses: np.ndarray,
    ) -> LocalDispatch:
        desired = np.asarray(desired_kw, dtype=float)
        buses = np.asarray(device_buses, dtype=int)
        if desired.shape != buses.shape:
            raise ValueError("the device control vector and the bus mapping have different lengths")
        self._device_buses = buses
        baseline = self._evaluate_candidate(load_kw, potential_input_kw, {})
        requested = self._evaluate_candidate(
            load_kw, potential_input_kw, self._control_map(desired, buses)
        )
        accepted = desired.copy()
        fallback_global = False
        for _ in range(10):
            evaluation = self._evaluate_candidate(
                load_kw, potential_input_kw, self._control_map(accepted, buses)
            )
            if self._safe(evaluation, baseline):
                break
            overloaded = [
                key
                for key, value in evaluation.branch_loading.items()
                if value > max(1.0, baseline.branch_loading[key]) + 1e-8
            ]
            if overloaded:
                worst = max(overloaded, key=lambda key: evaluation.branch_loading[key])
                child = int(worst.split("-")[1])
                region = self.descendants[child]
                mask = np.isin(buses, tuple(region))
            elif evaluation.transformer_loading > max(1.0, baseline.transformer_loading) + 1e-8:
                mask = np.ones(len(accepted), dtype=bool)
            else:
                violating_buses = [
                    bus
                    for bus, value in evaluation.voltage_pu.items()
                    if value < min(self.vmin, baseline.voltage_pu[bus]) - 1e-8
                    or value > max(self.vmax, baseline.voltage_pu[bus]) + 1e-8
                ]
                if not violating_buses:
                    break
                worst_bus = max(
                    violating_buses,
                    key=lambda bus: abs(evaluation.voltage_pu[bus] - 1.0),
                )
                mask = np.isin(buses, tuple(self._voltage_region(worst_bus)))
            previous = accepted.copy()
            accepted = self._bisect_region(
                accepted, mask, load_kw, potential_input_kw, baseline
            )
            if np.allclose(previous, accepted, atol=1e-10, rtol=0.0):
                break
        executed = self._evaluate_candidate(
            load_kw, potential_input_kw, self._control_map(accepted, buses)
        )
        if not self._safe(executed, baseline):
            fallback_global = True
            low, high = 0.0, 1.0
            for _ in range(18):
                middle = (low + high) / 2.0
                candidate = desired * middle
                evaluation = self._evaluate_candidate(
                    load_kw,
                    potential_input_kw,
                    self._control_map(candidate, buses),
                )
                if self._safe(evaluation, baseline):
                    low = middle
                else:
                    high = middle
            accepted = desired * low
            executed = self._evaluate_candidate(
                load_kw,
                potential_input_kw,
                self._control_map(accepted, buses),
            )
        requested_abs = float(np.sum(np.abs(desired)))
        accepted_abs = float(np.sum(np.abs(accepted)))
        clipped_devices = np.abs(accepted - desired) > 1e-8
        return LocalDispatch(
            accepted_kw=accepted,
            requested=requested,
            executed=executed,
            baseline=baseline,
            acceptance_ratio=(accepted_abs / requested_abs if requested_abs > TOLERANCE else 1.0),
            local_clip_fraction=float(np.mean(clipped_devices)),
            fallback_global=fallback_global,
        )
