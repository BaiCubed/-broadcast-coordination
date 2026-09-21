from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import math
from enum import Enum, auto


class BatteryChemistry(Enum):
    LFP = auto()


@dataclass
class BatteryParameters:
    nominal_capacity_kwh: float = 10.0
    usable_capacity_factor: float = 0.90

    max_charge_power_kw: float = 5.0
    max_discharge_power_kw: float = 5.0
    nominal_voltage_v: float = 400.0

    charge_efficiency_nominal: float = 0.95
    discharge_efficiency_nominal: float = 0.95
    inverter_efficiency: float = 0.97

    soc_min: float = 0.10
    soc_max: float = 0.95
    soc_reserve: float = 0.20

    temp_min_operating: float = 0.0
    temp_max_operating: float = 45.0
    temp_optimal_low: float = 15.0
    temp_optimal_high: float = 35.0
    temp_max_charging: float = 40.0

    max_charge_c_rate: float = 0.5
    max_discharge_c_rate: float = 0.5
    continuous_c_rate: float = 0.3

    max_daily_cycles: int = 2
    max_lifetime_cycles: int = 6000

    self_discharge_rate_per_day: float = 0.001

    chemistry: BatteryChemistry = BatteryChemistry.LFP

    ramp_rate_up: float = 2.0
    ramp_rate_down: float = 2.0


@dataclass
class BatteryDegradationState:
    total_cycles: float = 0.0

    calendar_age_days: float = 0.0

    cycle_capacity_fade: float = 0.0

    calendar_capacity_fade: float = 0.0

    @property
    def total_capacity_fade(self) -> float:
        return min(self.cycle_capacity_fade + self.calendar_capacity_fade, 0.30)

    @property
    def soh(self) -> float:
        return max(1.0 - self.total_capacity_fade, 0.70)


@dataclass
class BatteryThermalState:
    cell_temperature: float = 25.0
    ambient_temperature: float = 25.0
    coolant_temperature: float = 25.0

    cooling_active: bool = False
    heating_active: bool = False

    thermal_mass_kj_per_k: float = 50.0
    thermal_resistance_k_per_kw: float = 0.5


class BatteryModel:

    def __init__(
        self,
        params: Optional[BatteryParameters] = None,
        initial_soc: float = 0.5,
        initial_soh: float = 1.0,
    ):
        self.params = params or BatteryParameters()
        self.soc = initial_soc

        self.degradation = BatteryDegradationState()
        if initial_soh < 1.0:
            self.degradation.cycle_capacity_fade = (1.0 - initial_soh) * 0.7
            self.degradation.calendar_capacity_fade = (1.0 - initial_soh) * 0.3

        self.thermal = BatteryThermalState()

        self.current_power_kw: float = 0.0
        self.power_target_kw: float = 0.0

        self.energy_throughput_kwh: float = 0.0
        self.daily_cycle_count: float = 0.0
        self._daily_energy_start: float = 0.0

    @property
    def soh(self) -> float:
        return self.degradation.soh

    @property
    def available_capacity_kwh(self) -> float:
        return self.params.nominal_capacity_kwh * self.params.usable_capacity_factor * self.soh

    @property
    def stored_energy_kwh(self) -> float:
        return self.soc * self.available_capacity_kwh

    def get_discharge_efficiency(self, power_kw: float) -> float:
        base_efficiency = self.params.discharge_efficiency_nominal

        c_rate = abs(power_kw) / self.available_capacity_kwh
        c_rate_factor = 1.0 - 0.02 * max(0, c_rate - self.params.continuous_c_rate)

        temp = self.thermal.cell_temperature
        if temp < self.params.temp_optimal_low:
            temp_factor = 0.95 + 0.05 * (temp - self.params.temp_min_operating) / (self.params.temp_optimal_low - self.params.temp_min_operating)
        elif temp > self.params.temp_optimal_high:
            temp_factor = 0.95 + 0.05 * (self.params.temp_max_operating - temp) / (self.params.temp_max_operating - self.params.temp_optimal_high)
        else:
            temp_factor = 1.0

        if self.soc < 0.2:
            soc_factor = 0.95 + 0.05 * self.soc / 0.2
        else:
            soc_factor = 1.0

        inverter_eff = self.params.inverter_efficiency

        total_efficiency = base_efficiency * c_rate_factor * temp_factor * soc_factor * inverter_eff
        return max(0.85, min(0.98, total_efficiency))

    def get_charge_efficiency(self, power_kw: float) -> float:
        base_efficiency = self.params.charge_efficiency_nominal

        c_rate = abs(power_kw) / self.available_capacity_kwh
        c_rate_factor = 1.0 - 0.03 * max(0, c_rate - self.params.continuous_c_rate)

        temp = self.thermal.cell_temperature
        if temp < self.params.temp_optimal_low:
            temp_factor = 0.90 + 0.10 * (temp - self.params.temp_min_operating) / (self.params.temp_optimal_low - self.params.temp_min_operating)
        elif temp > self.params.temp_max_charging:
            temp_factor = 0.80
        elif temp > self.params.temp_optimal_high:
            temp_factor = 0.95 + 0.05 * (self.params.temp_max_charging - temp) / (self.params.temp_max_charging - self.params.temp_optimal_high)
        else:
            temp_factor = 1.0

        if self.soc > 0.9:
            soc_factor = 0.90 + 0.10 * (1.0 - self.soc) / 0.1
        else:
            soc_factor = 1.0

        inverter_eff = self.params.inverter_efficiency

        total_efficiency = base_efficiency * c_rate_factor * temp_factor * soc_factor * inverter_eff
        return max(0.85, min(0.98, total_efficiency))

    def get_max_discharge_power(self) -> float:
        max_power = self.params.max_discharge_power_kw

        c_rate_limit = self.available_capacity_kwh * self.params.max_discharge_c_rate
        max_power = min(max_power, c_rate_limit)

        if self.soc < 0.3:
            soc_factor = self.soc / 0.3
            max_power *= soc_factor

        temp = self.thermal.cell_temperature
        if temp < self.params.temp_min_operating:
            max_power = 0.0
        elif temp < self.params.temp_optimal_low:
            temp_factor = (temp - self.params.temp_min_operating) / (self.params.temp_optimal_low - self.params.temp_min_operating)
            max_power *= 0.5 + 0.5 * temp_factor
        elif temp > self.params.temp_max_operating:
            max_power = 0.0
        elif temp > self.params.temp_optimal_high:
            temp_factor = (self.params.temp_max_operating - temp) / (self.params.temp_max_operating - self.params.temp_optimal_high)
            max_power *= 0.7 + 0.3 * temp_factor

        return max(0.0, max_power)

    def get_max_charge_power(self) -> float:
        max_power = self.params.max_charge_power_kw

        c_rate_limit = self.available_capacity_kwh * self.params.max_charge_c_rate
        max_power = min(max_power, c_rate_limit)

        if self.soc > 0.8:
            soc_factor = (1.0 - self.soc) / 0.2
            max_power *= max(0.2, soc_factor)

        temp = self.thermal.cell_temperature
        if temp < self.params.temp_min_operating:
            max_power = 0.0
        elif temp < self.params.temp_optimal_low:
            temp_factor = (temp - self.params.temp_min_operating) / (self.params.temp_optimal_low - self.params.temp_min_operating)
            max_power *= 0.3 * temp_factor
        elif temp > self.params.temp_max_charging:
            max_power = 0.0
        elif temp > self.params.temp_optimal_high:
            temp_factor = (self.params.temp_max_charging - temp) / (self.params.temp_max_charging - self.params.temp_optimal_high)
            max_power *= 0.5 + 0.5 * temp_factor

        return max(0.0, max_power)

    def can_discharge(self, energy_kwh: float) -> bool:
        min_soc = self.params.soc_min
        available = (self.soc - min_soc) * self.available_capacity_kwh
        return available >= energy_kwh

    def can_charge(self, energy_kwh: float) -> bool:
        max_soc = self.params.soc_max
        headroom = (max_soc - self.soc) * self.available_capacity_kwh
        return headroom >= energy_kwh

    def update(self, power_kw: float, dt_seconds: float) -> Dict[str, float]:
        dt_hours = dt_seconds / 3600.0

        power_delta = power_kw - self.current_power_kw
        max_delta_up = self.params.ramp_rate_up * dt_seconds
        max_delta_down = self.params.ramp_rate_down * dt_seconds

        if power_delta > 0:
            power_delta = min(power_delta, max_delta_up)
        else:
            power_delta = max(power_delta, -max_delta_down)

        actual_power = self.current_power_kw + power_delta

        if actual_power > 0:
            actual_power = min(actual_power, self.get_max_charge_power())
            efficiency = self.get_charge_efficiency(actual_power)
            energy_change = actual_power * efficiency * dt_hours
        elif actual_power < 0:
            actual_power = max(actual_power, -self.get_max_discharge_power())
            efficiency = self.get_discharge_efficiency(-actual_power)
            energy_change = actual_power / efficiency * dt_hours
        else:
            energy_change = 0.0
            efficiency = 1.0

        self_discharge = self.params.self_discharge_rate_per_day * dt_hours / 24.0

        new_soc = self.soc + (energy_change - self_discharge * self.available_capacity_kwh) / self.available_capacity_kwh
        new_soc = max(self.params.soc_min, min(self.params.soc_max, new_soc))

        energy_throughput = abs(energy_change)
        self.energy_throughput_kwh += energy_throughput

        self.daily_cycle_count += energy_throughput / (2 * self.available_capacity_kwh)

        self._update_degradation(energy_throughput, dt_hours)

        self._update_thermal(actual_power, dt_seconds)

        old_soc = self.soc
        self.soc = new_soc
        self.current_power_kw = actual_power

        return {
            'soc_delta': new_soc - old_soc,
            'energy_kwh': energy_change,
            'efficiency': efficiency,
            'actual_power_kw': actual_power,
            'temperature_c': self.thermal.cell_temperature,
        }

    def _update_degradation(self, energy_kwh: float, dt_hours: float) -> None:
        cycle_fraction = energy_kwh / (2 * self.available_capacity_kwh)
        self.degradation.total_cycles += cycle_fraction

        k_cycle = 0.0005
        self.degradation.cycle_capacity_fade = k_cycle * math.sqrt(max(0, self.degradation.total_cycles))

        self.degradation.calendar_age_days += dt_hours / 24.0

        temp = self.thermal.cell_temperature
        soc_stress = 1.0 + 2.0 * (self.soc - 0.5) ** 2

        temp_diff = max(-50, min(50, temp - 25))
        temp_stress = math.exp(temp_diff / 15)

        k_calendar = 0.00001
        self.degradation.calendar_capacity_fade = min(
            0.3,
            k_calendar * self.degradation.calendar_age_days * soc_stress * temp_stress
        )

    def _update_thermal(self, power_kw: float, dt_seconds: float) -> None:
        heat_generation_kw = 0.02 * (power_kw ** 2) / (self.params.max_charge_power_kw ** 2)

        temp_diff = self.thermal.cell_temperature - self.thermal.ambient_temperature

        tau = self.thermal.thermal_resistance_k_per_kw * self.thermal.thermal_mass_kj_per_k

        active_temp_target = self.thermal.ambient_temperature
        if self.thermal.cooling_active:
            heat_generation_kw -= 1.0
        if self.thermal.heating_active:
            heat_generation_kw += 0.5

        heat_delta = heat_generation_kw * dt_seconds / self.thermal.thermal_mass_kj_per_k

        decay_factor = min(1.0, dt_seconds / tau)
        ambient_delta = -temp_diff * decay_factor

        delta_temp = heat_delta + ambient_delta
        delta_temp = max(-5.0, min(5.0, delta_temp))

        new_temp = self.thermal.cell_temperature + delta_temp

        new_temp = max(-40.0, min(80.0, new_temp))

        if new_temp > self.params.temp_optimal_high + 2:
            self.thermal.cooling_active = True
        elif new_temp < self.params.temp_optimal_high - 2:
            self.thermal.cooling_active = False

        if new_temp < self.params.temp_optimal_low - 2:
            self.thermal.heating_active = True
        elif new_temp > self.params.temp_optimal_low + 2:
            self.thermal.heating_active = False

        self.thermal.cell_temperature = new_temp

    def reset_daily_counters(self) -> None:
        self.daily_cycle_count = 0.0
        self._daily_energy_start = self.energy_throughput_kwh

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            'soc': self.soc,
            'soh': self.soh,
            'stored_energy_kwh': self.stored_energy_kwh,
            'available_capacity_kwh': self.available_capacity_kwh,
            'current_power_kw': self.current_power_kw,
            'max_discharge_power_kw': self.get_max_discharge_power(),
            'max_charge_power_kw': self.get_max_charge_power(),
            'cell_temperature_c': self.thermal.cell_temperature,
            'daily_cycles': self.daily_cycle_count,
            'total_cycles': self.degradation.total_cycles,
            'cooling_active': self.thermal.cooling_active,
            'heating_active': self.thermal.heating_active,
        }


def create_residential_battery(
    capacity_kwh: float = 10.0,
    c_rate: float = 0.5,
    chemistry: BatteryChemistry = BatteryChemistry.LFP,
    initial_soh: float = 1.0,
) -> BatteryModel:
    max_power = capacity_kwh * c_rate
    params = BatteryParameters(
        nominal_capacity_kwh=capacity_kwh,
        max_charge_power_kw=max_power,
        max_discharge_power_kw=max_power,
        chemistry=chemistry,
        max_charge_c_rate=c_rate,
        max_discharge_c_rate=c_rate,
        continuous_c_rate=c_rate * 0.6,
        max_daily_cycles=2,
    )
    return BatteryModel(params=params, initial_soc=0.5, initial_soh=initial_soh)


