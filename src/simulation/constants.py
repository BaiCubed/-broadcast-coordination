from dataclasses import dataclass


@dataclass(frozen=True)
class EPSLatencyParams:
    base_ms: float = 33.0
    jitter_scale_ms: float = 8.0
    min_ms: float = 20.0
    max_ms: float = 100.0
    target_p99_ms: float = 70.0


@dataclass(frozen=True)
class BatteryResponseParams:
    soc_charge_max: float = 0.95
    soc_charge_range: float = 0.70
    soc_discharge_min: float = 0.20
    soc_discharge_range: float = 0.60
    prob_min: float = 0.05
    prob_max: float = 0.95


EPS_LATENCY = EPSLatencyParams()
BATTERY_RESPONSE = BatteryResponseParams()
