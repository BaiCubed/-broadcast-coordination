from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LIMIT_TOLERANCE = 1e-9


@dataclass
class NetworkEvaluation:
    bus_net_kw: dict[int, float]
    branch_flow_kw: dict[str, float]
    branch_loading: dict[str, float]
    voltage_pu: dict[int, float]
    transformer_loading: float
    global_scale: float
    voltage_violations: int
    overloaded_branches: int


class IEEE33DistFlow:

    def __init__(self, network_config: dict[str, Any], control_config: dict[str, Any]):
        self.config = network_config
        self.control = control_config
        self.slack = int(network_config["slack_bus"])
        self.vmin, self.vmax = map(float, network_config["voltage_limits_pu"])
        capacity_multiplier = float(control_config.get("network_capacity_multiplier", 1.0))
        if capacity_multiplier <= 0:
            raise ValueError("network_capacity_multiplier must be positive")
        self.capacity_multiplier = capacity_multiplier
        self.voltage_sensitivity = (
            float(control_config["voltage_sensitivity"]) / capacity_multiplier
        )
        self.transformer_capacity = float(network_config["transformer"]["capacity_kw"]) * capacity_multiplier
        self.branches = [
            {"from": int(row[0]), "to": int(row[1]), "r": float(row[2]), "x": float(row[3]), "capacity": float(row[4]) * capacity_multiplier}
            for row in network_config["branches"]
        ]
        self.parent = {branch["to"]: branch["from"] for branch in self.branches}
        self.branch_by_child = {branch["to"]: branch for branch in self.branches}
        self.children: dict[int, list[int]] = {}
        for branch in self.branches:
            self.children.setdefault(branch["from"], []).append(branch["to"])

    def _subtree_sum(self, bus: int, values: dict[int, float], totals: dict[int, float]) -> float:
        total = values.get(bus, 0.0)
        for child in self.children.get(bus, []):
            total += self._subtree_sum(child, values, totals)
        totals[bus] = total
        return total

    def evaluate(self, load_kw: dict[int, float], pv_kw: dict[int, float], control_kw: dict[int, float], apply_limits: bool = True) -> NetworkEvaluation:
        buses = set(load_kw) | set(pv_kw) | set(control_kw) | {self.slack}
        net = {bus: load_kw.get(bus, 0.0) - pv_kw.get(bus, 0.0) + control_kw.get(bus, 0.0) for bus in buses}
        totals: dict[int, float] = {}
        self._subtree_sum(self.slack, net, totals)
        flows = {f"{branch['from']}-{branch['to']}": totals.get(branch["to"], 0.0) for branch in self.branches}
        loading = {key: abs(value) / max(self._branch(key)["capacity"], 1e-9) for key, value in flows.items()}
        voltage = {self.slack: 1.0}
        for branch in self.branches:
            key = f"{branch['from']}-{branch['to']}"
            parent_v = voltage.get(branch["from"], 1.0)
            flow = flows[key]
            drop = self.voltage_sensitivity * (branch["r"] + 0.2 * branch["x"]) * flow / 100.0
            voltage[branch["to"]] = parent_v - drop
        transformer = abs(totals.get(self.slack, 0.0)) / max(self.transformer_capacity, 1e-9)
        scale = 1.0
        if apply_limits:
            if self.control.get("enforce_feeder_capacity", True):
                scale = min(scale, min((1.0 / max(value, 1e-9) for value in loading.values()), default=1.0))
            if self.control.get("enforce_transformer_limit", True):
                scale = min(scale, 1.0 / max(transformer, 1.0))
            violation = max([self.vmin - min(voltage.values()), max(voltage.values()) - self.vmax, 0.0])
            if self.control.get("enforce_voltage_limits", True) and violation > 0:
                scale = min(scale, max(0.0, 1.0 - 3.0 * violation))
        scale = float(np.clip(scale, 0.0, 1.0))
        return NetworkEvaluation(
            bus_net_kw=net,
            branch_flow_kw=flows,
            branch_loading=loading,
            voltage_pu=voltage,
            transformer_loading=float(transformer),
            global_scale=scale,
            voltage_violations=sum(
                value < self.vmin - LIMIT_TOLERANCE
                or value > self.vmax + LIMIT_TOLERANCE
                for value in voltage.values()
            ),
            overloaded_branches=sum(
                value > 1.0 + LIMIT_TOLERANCE for value in loading.values()
            ),
        )

    def _branch(self, key: str) -> dict[str, float]:
        left, right = key.split("-")
        return self.branch_by_child[int(right)]

    @staticmethod
    def _intersect_affine_bounds(
        lower_scale: float,
        upper_scale: float,
        *,
        offset: float,
        slope: float,
        lower_value: float,
        upper_value: float,
    ) -> tuple[float, float] | None:
        if abs(slope) <= 1e-12:
            if lower_value - 1e-12 <= offset <= upper_value + 1e-12:
                return lower_scale, upper_scale
            return None
        first = (lower_value - offset) / slope
        second = (upper_value - offset) / slope
        feasible_lower = min(first, second)
        feasible_upper = max(first, second)
        lower_scale = max(lower_scale, feasible_lower)
        upper_scale = min(upper_scale, feasible_upper)
        if lower_scale > upper_scale + 1e-12:
            return None
        return lower_scale, upper_scale

    def constrained_dispatch_scale(self, load_kw: dict[int, float], pv_kw: dict[int, float], desired_kw: dict[int, float]) -> tuple[float, NetworkEvaluation]:
        baseline = self.evaluate(load_kw, pv_kw, {}, apply_limits=False)
        requested = self.evaluate(load_kw, pv_kw, desired_kw, apply_limits=False)
        lower_scale = 0.0
        upper_scale = 1.0

        def intersect(
            offset: float,
            slope: float,
            lower_value: float,
            upper_value: float,
        ) -> bool:
            nonlocal lower_scale, upper_scale
            bounds = self._intersect_affine_bounds(
                lower_scale,
                upper_scale,
                offset=offset,
                slope=slope,
                lower_value=lower_value,
                upper_value=upper_value,
            )
            if bounds is None:
                return False
            lower_scale, upper_scale = bounds
            return True

        feasible = True
        if self.control.get("enforce_feeder_capacity", True):
            for key, base_flow in baseline.branch_flow_kw.items():
                requested_flow = requested.branch_flow_kw[key]
                capacity = float(self._branch(key)["capacity"])
                if not intersect(base_flow, requested_flow - base_flow, -capacity, capacity):
                    feasible = False
                    break

        if feasible and self.control.get("enforce_transformer_limit", True):
            base_flow = float(sum(baseline.bus_net_kw.values()))
            requested_flow = float(sum(requested.bus_net_kw.values()))
            feasible = intersect(
                base_flow,
                requested_flow - base_flow,
                -self.transformer_capacity,
                self.transformer_capacity,
            )

        if feasible and self.control.get("enforce_voltage_limits", True):
            for bus, base_voltage in baseline.voltage_pu.items():
                requested_voltage = requested.voltage_pu[bus]
                if not intersect(
                    base_voltage,
                    requested_voltage - base_voltage,
                    self.vmin,
                    self.vmax,
                ):
                    feasible = False
                    break

        scale = float(np.clip(upper_scale, 0.0, 1.0)) if feasible else 0.0
        after_control = {bus: value * scale for bus, value in desired_kw.items()}
        after = self.evaluate(load_kw, pv_kw, after_control, apply_limits=False)
        after.global_scale = scale
        return scale, after
