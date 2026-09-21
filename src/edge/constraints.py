from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum, auto

from .state_machine import DeviceState, BatteryState


class ConstraintViolation(Enum):
    MAX_POWER_EXCEEDED = auto()
    RAMP_RATE_EXCEEDED = auto()

    SOC_TOO_LOW = auto()
    SOC_TOO_HIGH = auto()
    DEPARTURE_SOC_RISK = auto()

    OVER_TEMPERATURE = auto()
    UNDER_TEMPERATURE = auto()

    MIN_RUN_TIME = auto()
    MIN_OFF_TIME = auto()
    DAILY_CYCLES_EXCEEDED = auto()

    DEVICE_FAULTED = auto()
    DEVICE_OFFLINE = auto()

    FREQUENCY_OUT_OF_RANGE = auto()
    FREQUENCY_CRITICAL = auto()
    VOLTAGE_OUT_OF_RANGE = auto()
    VOLTAGE_CRITICAL = auto()
    LVRT_VIOLATION = auto()
    HVRT_VIOLATION = auto()
    ANTI_ISLANDING_TRIGGERED = auto()
    HARMONIC_LIMIT_EXCEEDED = auto()


@dataclass
class ConstraintResult:
    violated: bool = False
    violations: List[ConstraintViolation] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)

    def add_violation(self, violation: ConstraintViolation, message: str = "") -> None:
        self.violated = True
        self.violations.append(violation)
        if message:
            self.messages.append(message)

    @property
    def is_safe(self) -> bool:
        safety_critical = {
            ConstraintViolation.OVER_TEMPERATURE,
            ConstraintViolation.DEVICE_FAULTED,
            ConstraintViolation.DEVICE_OFFLINE,
            ConstraintViolation.FREQUENCY_CRITICAL,
            ConstraintViolation.VOLTAGE_CRITICAL,
            ConstraintViolation.LVRT_VIOLATION,
            ConstraintViolation.HVRT_VIOLATION,
            ConstraintViolation.ANTI_ISLANDING_TRIGGERED,
        }
        return not any(v in safety_critical for v in self.violations)

    @property
    def has_grid_violation(self) -> bool:
        grid_violations = {
            ConstraintViolation.FREQUENCY_OUT_OF_RANGE,
            ConstraintViolation.FREQUENCY_CRITICAL,
            ConstraintViolation.VOLTAGE_OUT_OF_RANGE,
            ConstraintViolation.VOLTAGE_CRITICAL,
            ConstraintViolation.LVRT_VIOLATION,
            ConstraintViolation.HVRT_VIOLATION,
            ConstraintViolation.ANTI_ISLANDING_TRIGGERED,
            ConstraintViolation.HARMONIC_LIMIT_EXCEEDED,
        }
        return any(v in grid_violations for v in self.violations)


class GridStandard(Enum):
    STANDARD_A = auto()
    STANDARD_B = auto()
    CUSTOM = auto()


@dataclass
class GridConstraints:
    standard: GridStandard = GridStandard.STANDARD_B

    nominal_frequency: float = 50.0

    freq_normal_min: float = 49.5
    freq_normal_max: float = 50.2

    freq_warning_min: float = 48.5
    freq_warning_max: float = 50.5

    freq_critical_min: float = 47.5
    freq_critical_max: float = 51.5

    freq_warning_max_duration: float = 300.0
    freq_critical_max_duration: float = 0.16

    nominal_voltage_pu: float = 1.0

    voltage_normal_min_pu: float = 0.85
    voltage_normal_max_pu: float = 1.10

    voltage_warning_min_pu: float = 0.70
    voltage_warning_max_pu: float = 1.20

    voltage_critical_min_pu: float = 0.50
    voltage_critical_max_pu: float = 1.30

    lvrt_enabled: bool = True
    lvrt_threshold_20pct_duration: float = 0.625
    lvrt_threshold_50pct_duration: float = 2.0
    lvrt_threshold_85pct_duration: float = 10.0

    hvrt_enabled: bool = True
    hvrt_threshold_120pct_duration: float = 0.5
    hvrt_threshold_110pct_duration: float = 10.0

    anti_islanding_enabled: bool = True
    anti_islanding_detection_time: float = 2.0

    rocof_limit: float = 2.0

    @classmethod
    def standard_a(cls, category: int = 3) -> "GridConstraints":
        constraints = cls(
            standard=GridStandard.STANDARD_A,
            nominal_frequency=60.0,
            freq_normal_min=59.5,
            freq_normal_max=60.1,
            freq_warning_min=58.5 if category >= 2 else 59.0,
            freq_warning_max=61.2 if category >= 2 else 60.5,
            freq_critical_min=56.5,
            freq_critical_max=62.0,
            voltage_normal_min_pu=0.88,
            voltage_normal_max_pu=1.10,
            voltage_warning_min_pu=0.70,
            voltage_warning_max_pu=1.20,
            voltage_critical_min_pu=0.50,
            voltage_critical_max_pu=1.20,
            lvrt_threshold_20pct_duration=0.0 if category == 1 else 0.16,
            lvrt_threshold_50pct_duration=0.5 if category == 2 else (2.0 if category == 3 else 0.0),
            rocof_limit=3.0 if category == 1 else 2.0,
        )
        return constraints

    @classmethod
    def standard_b(cls) -> "GridConstraints":
        return cls(
            standard=GridStandard.STANDARD_B,
            nominal_frequency=50.0,
            freq_normal_min=49.5,
            freq_normal_max=50.2,
            freq_warning_min=48.5,
            freq_warning_max=50.5,
            freq_critical_min=47.5,
            freq_critical_max=51.5,
            voltage_normal_min_pu=0.85,
            voltage_normal_max_pu=1.10,
            voltage_warning_min_pu=0.70,
            voltage_warning_max_pu=1.20,
            voltage_critical_min_pu=0.20,
            voltage_critical_max_pu=1.30,
            lvrt_threshold_20pct_duration=0.625,
            lvrt_threshold_50pct_duration=2.0,
            lvrt_threshold_85pct_duration=10.0,
        )


@dataclass
class HarmonicConstraints:
    voltage_class_kv: float = 0.38

    thd_voltage_limit_038kv: float = 5.0
    thd_voltage_limit_10kv: float = 4.0
    thd_voltage_limit_66kv: float = 3.0
    thd_voltage_limit_110kv: float = 2.0

    ieee_thd_voltage_limit_1kv: float = 8.0
    ieee_thd_voltage_limit_69kv: float = 5.0
    ieee_thd_voltage_limit_161kv: float = 2.5
    ieee_thd_voltage_limit_hv: float = 1.5

    individual_harmonic_limit_1kv: float = 5.0
    individual_harmonic_limit_69kv: float = 3.0
    individual_harmonic_limit_161kv: float = 1.5
    individual_harmonic_limit_hv: float = 1.0


    tdd_limit_weak_grid: float = 5.0
    h2_11_limit_weak: float = 4.0
    h11_17_limit_weak: float = 2.0
    h17_23_limit_weak: float = 1.5
    h23_35_limit_weak: float = 0.6
    h35_50_limit_weak: float = 0.3

    tdd_limit_medium_grid: float = 8.0
    h2_11_limit_medium: float = 7.0
    h11_17_limit_medium: float = 3.5
    h17_23_limit_medium: float = 2.5
    h23_35_limit_medium: float = 1.0
    h35_50_limit_medium: float = 0.5

    tdd_limit_strong_grid: float = 12.0
    h2_11_limit_strong: float = 10.0
    h11_17_limit_strong: float = 4.5
    h17_23_limit_strong: float = 4.0
    h23_35_limit_strong: float = 1.5
    h35_50_limit_strong: float = 0.7

    tdd_limit_very_strong_grid: float = 15.0

    tdd_limit_extreme_grid: float = 20.0

    short_circuit_ratio: float = 50.0

    use_50hz_standard: bool = True

    def get_voltage_thd_limit(self) -> float:
        if self.use_50hz_standard:
            if self.voltage_class_kv <= 0.5:
                return self.thd_voltage_limit_038kv
            elif self.voltage_class_kv <= 10:
                return self.thd_voltage_limit_10kv
            elif self.voltage_class_kv <= 66:
                return self.thd_voltage_limit_66kv
            else:
                return self.thd_voltage_limit_110kv
        else:
            if self.voltage_class_kv <= 1:
                return self.ieee_thd_voltage_limit_1kv
            elif self.voltage_class_kv <= 69:
                return self.ieee_thd_voltage_limit_69kv
            elif self.voltage_class_kv <= 161:
                return self.ieee_thd_voltage_limit_161kv
            else:
                return self.ieee_thd_voltage_limit_hv

    def get_current_tdd_limit(self) -> float:
        isc_il = self.short_circuit_ratio
        if isc_il < 20:
            return self.tdd_limit_weak_grid
        elif isc_il < 50:
            return self.tdd_limit_medium_grid
        elif isc_il < 100:
            return self.tdd_limit_strong_grid
        elif isc_il < 1000:
            return self.tdd_limit_very_strong_grid
        else:
            return self.tdd_limit_extreme_grid

    def get_individual_harmonic_limit(self) -> float:
        if self.voltage_class_kv <= 1:
            return self.individual_harmonic_limit_1kv
        elif self.voltage_class_kv <= 69:
            return self.individual_harmonic_limit_69kv
        elif self.voltage_class_kv <= 161:
            return self.individual_harmonic_limit_161kv
        else:
            return self.individual_harmonic_limit_hv


@dataclass
class GridState:
    frequency: float = 50.0
    frequency_rate_of_change: float = 0.0

    voltage_pu: float = 1.0
    voltage_phase_a_pu: float = 1.0
    voltage_phase_b_pu: float = 1.0
    voltage_phase_c_pu: float = 1.0

    voltage_sag_duration: float = 0.0
    voltage_swell_duration: float = 0.0

    is_islanded: bool = False
    island_detection_time: float = 0.0

    thd_voltage: float = 0.0
    thd_current: float = 0.0
    tdd_current: float = 0.0

    individual_voltage_harmonics: dict = field(default_factory=dict)
    individual_current_harmonics: dict = field(default_factory=dict)

    max_individual_voltage_harmonic: float = 0.0
    max_individual_current_harmonic: float = 0.0
    max_harmonic_order: int = 0

    timestamp: float = 0.0


@dataclass
class PhysicalConstraints:
    max_charge_power: float = 10.0
    max_discharge_power: float = 10.0
    max_ramp_rate: float = 5.0

    min_soc: float = 0.1
    max_soc: float = 0.95
    safe_soc_range: tuple = (0.2, 0.8)

    max_temperature: float = 45.0
    min_temperature: float = 0.0

    min_run_time: float = 60.0
    min_off_time: float = 60.0
    max_daily_cycles: int = 10

class ConstraintChecker:

    def __init__(
        self,
        constraints: Optional[PhysicalConstraints] = None,
        grid_constraints: Optional[GridConstraints] = None,
        harmonic_constraints: Optional[HarmonicConstraints] = None
    ):
        self.constraints = constraints or PhysicalConstraints()
        self.grid_constraints = grid_constraints or GridConstraints()
        self.harmonic_constraints = harmonic_constraints or HarmonicConstraints()

    def check(
        self,
        state: DeviceState,
        target_power: Optional[float] = None,
        grid_state: Optional[GridState] = None
    ) -> ConstraintResult:
        result = ConstraintResult()

        if not state.is_healthy:
            result.add_violation(
                ConstraintViolation.DEVICE_FAULTED,
                f"Device faulted: {state.fault_code}"
            )

        if isinstance(state, BatteryState):
            self._check_battery(state, target_power, result)

        if target_power is not None:
            self._check_power_limits(state, target_power, result)

        if grid_state is not None:
            self._check_grid_requirements(grid_state, result)

        return result

    def check_grid_only(self, grid_state: GridState) -> ConstraintResult:
        result = ConstraintResult()
        self._check_grid_requirements(grid_state, result)
        return result

    def _check_grid_requirements(
        self,
        grid_state: GridState,
        result: ConstraintResult
    ) -> None:
        gc = self.grid_constraints

        freq = grid_state.frequency

        if freq < gc.freq_critical_min:
            result.add_violation(
                ConstraintViolation.FREQUENCY_CRITICAL,
                f"Grid frequency {freq:.2f}Hz critically low (< {gc.freq_critical_min:.1f}Hz), "
                f"must disconnect per {gc.standard.name}"
            )
        elif freq > gc.freq_critical_max:
            result.add_violation(
                ConstraintViolation.FREQUENCY_CRITICAL,
                f"Grid frequency {freq:.2f}Hz critically high (> {gc.freq_critical_max:.1f}Hz), "
                f"must disconnect per {gc.standard.name}"
            )
        elif freq < gc.freq_warning_min:
            result.add_violation(
                ConstraintViolation.FREQUENCY_OUT_OF_RANGE,
                f"Grid frequency {freq:.2f}Hz below normal ({gc.freq_warning_min:.1f}Hz), "
                f"limited duration operation (max {gc.freq_warning_max_duration:.0f}s)"
            )
        elif freq > gc.freq_warning_max:
            result.add_violation(
                ConstraintViolation.FREQUENCY_OUT_OF_RANGE,
                f"Grid frequency {freq:.2f}Hz above normal ({gc.freq_warning_max:.1f}Hz), "
                f"limited duration operation (max {gc.freq_warning_max_duration:.0f}s)"
            )
        elif freq < gc.freq_normal_min or freq > gc.freq_normal_max:
            result.add_violation(
                ConstraintViolation.FREQUENCY_OUT_OF_RANGE,
                f"Grid frequency {freq:.2f}Hz outside normal range "
                f"({gc.freq_normal_min:.1f}-{gc.freq_normal_max:.1f}Hz)"
            )

        rocof = abs(grid_state.frequency_rate_of_change)
        if rocof > gc.rocof_limit:
            result.add_violation(
                ConstraintViolation.FREQUENCY_CRITICAL,
                f"RoCoF {rocof:.2f}Hz/s exceeds limit {gc.rocof_limit:.1f}Hz/s, "
                f"possible islanding or major grid event"
            )

        voltage = grid_state.voltage_pu

        if voltage < gc.voltage_critical_min_pu:
            result.add_violation(
                ConstraintViolation.VOLTAGE_CRITICAL,
                f"Grid voltage {voltage:.2f}pu critically low (< {gc.voltage_critical_min_pu:.2f}pu), "
                f"momentary cessation required"
            )
        elif voltage > gc.voltage_critical_max_pu:
            result.add_violation(
                ConstraintViolation.VOLTAGE_CRITICAL,
                f"Grid voltage {voltage:.2f}pu critically high (> {gc.voltage_critical_max_pu:.2f}pu), "
                f"must trip"
            )
        elif voltage < gc.voltage_warning_min_pu:
            result.add_violation(
                ConstraintViolation.VOLTAGE_OUT_OF_RANGE,
                f"Grid voltage {voltage:.2f}pu below warning threshold ({gc.voltage_warning_min_pu:.2f}pu)"
            )
        elif voltage > gc.voltage_warning_max_pu:
            result.add_violation(
                ConstraintViolation.VOLTAGE_OUT_OF_RANGE,
                f"Grid voltage {voltage:.2f}pu above warning threshold ({gc.voltage_warning_max_pu:.2f}pu)"
            )
        elif voltage < gc.voltage_normal_min_pu or voltage > gc.voltage_normal_max_pu:
            result.add_violation(
                ConstraintViolation.VOLTAGE_OUT_OF_RANGE,
                f"Grid voltage {voltage:.2f}pu outside normal range "
                f"({gc.voltage_normal_min_pu:.2f}-{gc.voltage_normal_max_pu:.2f}pu)"
            )

        if gc.lvrt_enabled and grid_state.voltage_sag_duration > 0:
            self._check_lvrt(grid_state, result)

        if gc.hvrt_enabled and grid_state.voltage_swell_duration > 0:
            self._check_hvrt(grid_state, result)

        if gc.anti_islanding_enabled and grid_state.is_islanded:
            if grid_state.island_detection_time >= gc.anti_islanding_detection_time:
                result.add_violation(
                    ConstraintViolation.ANTI_ISLANDING_TRIGGERED,
                    f"Islanding detected for {grid_state.island_detection_time:.2f}s, "
                    f"exceeds detection limit {gc.anti_islanding_detection_time:.1f}s, must trip"
                )

        self._check_harmonics(grid_state, result)

    def _check_harmonics(self, grid_state: GridState, result: ConstraintResult) -> None:
        hc = self.harmonic_constraints

        thd_voltage_limit = hc.get_voltage_thd_limit()
        if grid_state.thd_voltage > thd_voltage_limit:
            standard_name = "50 Hz standard" if hc.use_50hz_standard else "60 Hz standard"
            result.add_violation(
                ConstraintViolation.HARMONIC_LIMIT_EXCEEDED,
                f"Voltage THD {grid_state.thd_voltage:.2f}% exceeds limit {thd_voltage_limit:.1f}% "
                f"({standard_name} at {hc.voltage_class_kv}kV)"
            )

        tdd_current_limit = hc.get_current_tdd_limit()
        if grid_state.tdd_current > tdd_current_limit:
            result.add_violation(
                ConstraintViolation.HARMONIC_LIMIT_EXCEEDED,
                f"Current TDD {grid_state.tdd_current:.2f}% exceeds limit {tdd_current_limit:.1f}% "
                f"(ISC/IL={hc.short_circuit_ratio:.0f})"
            )

        individual_limit = hc.get_individual_harmonic_limit()
        if grid_state.max_individual_voltage_harmonic > individual_limit:
            result.add_violation(
                ConstraintViolation.HARMONIC_LIMIT_EXCEEDED,
                f"Individual voltage harmonic (h{grid_state.max_harmonic_order}) "
                f"{grid_state.max_individual_voltage_harmonic:.2f}% exceeds limit {individual_limit:.1f}%"
            )

        if grid_state.individual_voltage_harmonics:
            self._check_individual_harmonics(
                grid_state.individual_voltage_harmonics,
                "voltage",
                result
            )

        if grid_state.individual_current_harmonics:
            self._check_individual_harmonics(
                grid_state.individual_current_harmonics,
                "current",
                result
            )

    def _check_individual_harmonics(
        self,
        harmonics: dict,
        harmonic_type: str,
        result: ConstraintResult
    ) -> None:
        hc = self.harmonic_constraints

        if harmonic_type == "voltage":
            base_limit = hc.get_individual_harmonic_limit()
            for order, magnitude in harmonics.items():
                if isinstance(order, int) and 2 <= order <= 50:
                    if magnitude > base_limit:
                        result.add_violation(
                            ConstraintViolation.HARMONIC_LIMIT_EXCEEDED,
                            f"Voltage harmonic h{order} = {magnitude:.2f}% exceeds limit {base_limit:.1f}%"
                        )
        else:
            isc_il = hc.short_circuit_ratio

            if isc_il < 20:
                limits = {
                    (2, 11): hc.h2_11_limit_weak,
                    (11, 17): hc.h11_17_limit_weak,
                    (17, 23): hc.h17_23_limit_weak,
                    (23, 35): hc.h23_35_limit_weak,
                    (35, 50): hc.h35_50_limit_weak,
                }
            elif isc_il < 50:
                limits = {
                    (2, 11): hc.h2_11_limit_medium,
                    (11, 17): hc.h11_17_limit_medium,
                    (17, 23): hc.h17_23_limit_medium,
                    (23, 35): hc.h23_35_limit_medium,
                    (35, 50): hc.h35_50_limit_medium,
                }
            else:
                limits = {
                    (2, 11): hc.h2_11_limit_strong,
                    (11, 17): hc.h11_17_limit_strong,
                    (17, 23): hc.h17_23_limit_strong,
                    (23, 35): hc.h23_35_limit_strong,
                    (35, 50): hc.h35_50_limit_strong,
                }

            for order, magnitude in harmonics.items():
                if isinstance(order, int) and 2 <= order <= 50:
                    limit = None
                    for (low, high), group_limit in limits.items():
                        if low <= order < high:
                            limit = group_limit
                            break

                    if limit is not None and magnitude > limit:
                        result.add_violation(
                            ConstraintViolation.HARMONIC_LIMIT_EXCEEDED,
                            f"Current harmonic h{order} = {magnitude:.2f}% exceeds limit {limit:.1f}%"
                        )

    def _check_lvrt(self, grid_state: GridState, result: ConstraintResult) -> None:
        gc = self.grid_constraints
        voltage = grid_state.voltage_pu
        duration = grid_state.voltage_sag_duration

        if voltage <= 0.20:
            max_duration = gc.lvrt_threshold_20pct_duration
            threshold_name = "20%"
        elif voltage <= 0.50:
            max_duration = gc.lvrt_threshold_50pct_duration
            threshold_name = "50%"
        elif voltage <= 0.85:
            max_duration = gc.lvrt_threshold_85pct_duration
            threshold_name = "85%"
        else:
            return

        if duration > max_duration:
            result.add_violation(
                ConstraintViolation.LVRT_VIOLATION,
                f"LVRT requirement violated: voltage {voltage:.1%} for {duration:.3f}s "
                f"exceeds {threshold_name} threshold limit of {max_duration:.3f}s"
            )

    def _check_hvrt(self, grid_state: GridState, result: ConstraintResult) -> None:
        gc = self.grid_constraints
        voltage = grid_state.voltage_pu
        duration = grid_state.voltage_swell_duration

        if voltage >= 1.20:
            max_duration = gc.hvrt_threshold_120pct_duration
            threshold_name = "120%"
        elif voltage >= 1.10:
            max_duration = gc.hvrt_threshold_110pct_duration
            threshold_name = "110%"
        else:
            return

        if duration > max_duration:
            result.add_violation(
                ConstraintViolation.HVRT_VIOLATION,
                f"HVRT requirement violated: voltage {voltage:.1%} for {duration:.3f}s "
                f"exceeds {threshold_name} threshold limit of {max_duration:.3f}s"
            )

    def _check_power_limits(
        self,
        state: DeviceState,
        target_power: float,
        result: ConstraintResult
    ) -> None:
        c = self.constraints

        if target_power > 0 and target_power > c.max_charge_power:
            result.add_violation(
                ConstraintViolation.MAX_POWER_EXCEEDED,
                f"Charge power {target_power:.1f}kW exceeds max {c.max_charge_power:.1f}kW"
            )
        elif target_power < 0 and abs(target_power) > c.max_discharge_power:
            result.add_violation(
                ConstraintViolation.MAX_POWER_EXCEEDED,
                f"Discharge power {abs(target_power):.1f}kW exceeds max {c.max_discharge_power:.1f}kW"
            )

        power_change = abs(target_power - state.power_current)
        if power_change > c.max_ramp_rate:
            result.add_violation(
                ConstraintViolation.RAMP_RATE_EXCEEDED,
                f"Power change {power_change:.1f}kW/s exceeds max ramp {c.max_ramp_rate:.1f}kW/s"
            )

    def _check_battery(
        self,
        state: BatteryState,
        target_power: Optional[float],
        result: ConstraintResult
    ) -> None:
        c = self.constraints

        if state.soc < c.min_soc:
            result.add_violation(
                ConstraintViolation.SOC_TOO_LOW,
                f"SOC {state.soc:.1%} below minimum {c.min_soc:.1%}"
            )
        elif state.soc > c.max_soc:
            result.add_violation(
                ConstraintViolation.SOC_TOO_HIGH,
                f"SOC {state.soc:.1%} above maximum {c.max_soc:.1%}"
            )

        if target_power is not None and target_power < 0:
            if state.soc <= c.safe_soc_range[0]:
                result.add_violation(
                    ConstraintViolation.SOC_TOO_LOW,
                    f"Cannot discharge: SOC {state.soc:.1%} at/below safe minimum {c.safe_soc_range[0]:.1%}"
                )

        if target_power is not None and target_power > 0:
            if state.soc >= c.safe_soc_range[1]:
                result.add_violation(
                    ConstraintViolation.SOC_TOO_HIGH,
                    f"Cannot charge: SOC {state.soc:.1%} at/above safe maximum {c.safe_soc_range[1]:.1%}"
                )

        if state.temperature > c.max_temperature:
            result.add_violation(
                ConstraintViolation.OVER_TEMPERATURE,
                f"Temperature {state.temperature:.1f}°C exceeds max {c.max_temperature:.1f}°C"
            )
        elif state.temperature < c.min_temperature:
            result.add_violation(
                ConstraintViolation.UNDER_TEMPERATURE,
                f"Temperature {state.temperature:.1f}°C below min {c.min_temperature:.1f}°C"
            )

        if state.daily_cycles >= c.max_daily_cycles:
            result.add_violation(
                ConstraintViolation.DAILY_CYCLES_EXCEEDED,
                f"Daily cycles {state.daily_cycles} at/above max {c.max_daily_cycles}"
            )

def check_discharge_feasibility(state: BatteryState, power_kw: float, duration_s: float) -> bool:
    energy_kwh = power_kw * (duration_s / 3600) / state.discharge_efficiency
    final_soc = state.soc - (energy_kwh / state.capacity_kwh)
    return final_soc >= 0.2


def check_charge_feasibility(state: BatteryState, power_kw: float, duration_s: float) -> bool:
    energy_kwh = power_kw * state.charge_efficiency * (duration_s / 3600)
    final_soc = state.soc + (energy_kwh / state.capacity_kwh)
    return final_soc <= 0.8
