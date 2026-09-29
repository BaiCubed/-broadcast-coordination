import pytest
import math
from src.edge.battery import (
    BatteryChemistry,
    BatteryParameters,
    BatteryDegradationState,
    BatteryThermalState,
    BatteryModel,
    create_residential_battery,
)


class TestBatteryParameters:

    def test_default_values(self):
        params = BatteryParameters()

        assert params.nominal_capacity_kwh == 10.0
        assert params.max_charge_power_kw == 5.0
        assert params.max_discharge_power_kw == 5.0
        assert params.soc_min == 0.10
        assert params.soc_max == 0.95

    def test_custom_values(self):
        params = BatteryParameters(
            nominal_capacity_kwh=20.0,
            max_charge_power_kw=10.0,
            chemistry=BatteryChemistry.LFP,
        )

        assert params.nominal_capacity_kwh == 20.0
        assert params.max_charge_power_kw == 10.0
        assert params.chemistry == BatteryChemistry.LFP

    def test_c_rate_limits(self):
        params = BatteryParameters()

        assert params.max_charge_c_rate == 0.5
        assert params.max_discharge_c_rate == 0.5
        assert params.continuous_c_rate == 0.3


class TestBatteryDegradationState:

    def test_initial_soh(self):
        degradation = BatteryDegradationState()

        assert degradation.soh == 1.0
        assert degradation.total_capacity_fade == 0.0

    def test_soh_with_cycle_fade(self):
        degradation = BatteryDegradationState(cycle_capacity_fade=0.10)

        assert degradation.soh == pytest.approx(0.90)

    def test_soh_with_calendar_fade(self):
        degradation = BatteryDegradationState(calendar_capacity_fade=0.05)

        assert degradation.soh == pytest.approx(0.95)

    def test_total_fade_cap(self):
        degradation = BatteryDegradationState(
            cycle_capacity_fade=0.25,
            calendar_capacity_fade=0.15,
        )

        assert degradation.total_capacity_fade == 0.30
        assert degradation.soh == pytest.approx(0.70)


class TestBatteryThermalState:

    def test_default_values(self):
        thermal = BatteryThermalState()

        assert thermal.cell_temperature == 25.0
        assert thermal.ambient_temperature == 25.0
        assert thermal.cooling_active is False
        assert thermal.heating_active is False


class TestBatteryModel:

    def test_initialization(self):
        model = BatteryModel(initial_soc=0.5)

        assert model.soc == 0.5
        assert model.soh == 1.0
        assert model.current_power_kw == 0.0

    def test_initialization_with_degraded_soh(self):
        model = BatteryModel(initial_soc=0.5, initial_soh=0.90)

        assert model.soh < 1.0

    def test_available_capacity(self):
        params = BatteryParameters(nominal_capacity_kwh=10.0, usable_capacity_factor=0.90)
        model = BatteryModel(params=params, initial_soc=0.5)

        assert model.available_capacity_kwh == pytest.approx(9.0)

    def test_stored_energy(self):
        params = BatteryParameters(nominal_capacity_kwh=10.0, usable_capacity_factor=0.90)
        model = BatteryModel(params=params, initial_soc=0.5)

        assert model.stored_energy_kwh == pytest.approx(4.5)


class TestBatteryEfficiency:

    def test_discharge_efficiency_nominal(self):
        model = BatteryModel(initial_soc=0.5)
        model.thermal.cell_temperature = 25.0

        efficiency = model.get_discharge_efficiency(2.5)

        assert 0.85 <= efficiency <= 0.98

    def test_discharge_efficiency_low_temperature(self):
        model = BatteryModel(initial_soc=0.5)
        model.thermal.cell_temperature = 5.0

        efficiency = model.get_discharge_efficiency(2.5)

        assert efficiency < 0.95

    def test_charge_efficiency_high_soc(self):
        model = BatteryModel(initial_soc=0.95)
        model.thermal.cell_temperature = 25.0

        efficiency = model.get_charge_efficiency(2.5)

        assert efficiency < 0.95


class TestBatteryPowerLimits:

    def test_max_discharge_power_normal(self):
        model = BatteryModel(initial_soc=0.5)
        model.thermal.cell_temperature = 25.0

        max_power = model.get_max_discharge_power()

        assert max_power > 0
        assert max_power <= model.params.max_discharge_power_kw

    def test_max_discharge_power_low_soc(self):
        model = BatteryModel(initial_soc=0.15)
        model.thermal.cell_temperature = 25.0

        max_power = model.get_max_discharge_power()

        assert max_power < model.params.max_discharge_power_kw

    def test_max_charge_power_high_soc(self):
        model = BatteryModel(initial_soc=0.90)
        model.thermal.cell_temperature = 25.0

        max_power = model.get_max_charge_power()

        assert max_power < model.params.max_charge_power_kw

    def test_max_charge_power_cold(self):
        model = BatteryModel(initial_soc=0.5)
        model.thermal.cell_temperature = 5.0

        max_power = model.get_max_charge_power()

        assert max_power < model.params.max_charge_power_kw * 0.5

    def test_no_discharge_below_min_temp(self):
        model = BatteryModel(initial_soc=0.5)
        model.thermal.cell_temperature = -5.0

        max_power = model.get_max_discharge_power()

        assert max_power == 0.0


class TestBatteryFeasibility:

    def test_can_discharge_sufficient_soc(self):
        model = BatteryModel(initial_soc=0.50)

        assert model.can_discharge(2.0)

    def test_cannot_discharge_insufficient_soc(self):
        model = BatteryModel(initial_soc=0.15)

        assert not model.can_discharge(2.0)

    def test_can_charge_sufficient_headroom(self):
        model = BatteryModel(initial_soc=0.50)

        assert model.can_charge(2.0)

    def test_cannot_charge_insufficient_headroom(self):
        model = BatteryModel(initial_soc=0.90)

        assert not model.can_charge(2.0)


class TestBatteryUpdate:

    def test_charging_increases_soc(self):
        model = BatteryModel(initial_soc=0.50)

        result = model.update(power_kw=5.0, dt_seconds=3600)

        assert result['soc_delta'] > 0
        assert model.soc > 0.50

    def test_discharging_decreases_soc(self):
        model = BatteryModel(initial_soc=0.50)

        result = model.update(power_kw=-5.0, dt_seconds=3600)

        assert result['soc_delta'] < 0
        assert model.soc < 0.50

    def test_ramp_rate_limiting(self):
        model = BatteryModel(initial_soc=0.50)
        model.current_power_kw = 0.0

        result = model.update(power_kw=5.0, dt_seconds=1.0)

        assert abs(result['actual_power_kw']) < 5.0

    def test_soc_bounds_respected(self):
        model = BatteryModel(initial_soc=0.95)

        model.update(power_kw=5.0, dt_seconds=3600)

        assert model.soc <= model.params.soc_max

    def test_energy_throughput_tracked(self):
        model = BatteryModel(initial_soc=0.50)
        initial_throughput = model.energy_throughput_kwh

        model.update(power_kw=5.0, dt_seconds=3600)

        assert model.energy_throughput_kwh > initial_throughput


class TestBatteryDegradation:

    def test_cycling_increases_cycle_count(self):
        model = BatteryModel(initial_soc=0.50)
        initial_cycles = model.degradation.total_cycles

        for _ in range(3):
            model.update(power_kw=3.0, dt_seconds=1800)
            model.update(power_kw=-3.0, dt_seconds=1800)

        assert model.degradation.total_cycles > initial_cycles

    def test_calendar_aging(self):
        model = BatteryModel(initial_soc=0.50)
        initial_age = model.degradation.calendar_age_days

        for _ in range(6):
            model.update(power_kw=0.0, dt_seconds=3600)

        assert model.degradation.calendar_age_days > initial_age


class TestBatteryThermal:

    def test_high_power_heats_battery(self):
        model = BatteryModel(initial_soc=0.50)
        model.thermal.cell_temperature = 20.0
        model.thermal.ambient_temperature = 25.0

        initial_temp = model.thermal.cell_temperature

        for _ in range(100):
            model.update(power_kw=5.0, dt_seconds=60)

        assert model.thermal.cell_temperature > initial_temp

    def test_cooling_activates(self):
        model = BatteryModel(initial_soc=0.50)
        model.thermal.cell_temperature = 38.0
        model.thermal.ambient_temperature = 38.0

        model.update(power_kw=5.0, dt_seconds=60)

        assert model.thermal.cooling_active


class TestBatteryDailyCounters:

    def test_reset_daily_counters(self):
        model = BatteryModel(initial_soc=0.50)

        model.update(power_kw=5.0, dt_seconds=3600)
        model.daily_cycle_count = 1.5

        model.reset_daily_counters()

        assert model.daily_cycle_count == 0.0


class TestBatteryStateSummary:

    def test_state_summary_fields(self):
        model = BatteryModel(initial_soc=0.50)

        summary = model.get_state_summary()

        assert 'soc' in summary
        assert 'soh' in summary
        assert 'stored_energy_kwh' in summary
        assert 'max_discharge_power_kw' in summary
        assert 'max_charge_power_kw' in summary
        assert 'cell_temperature_c' in summary


class TestBatteryFactoryFunctions:

    def test_create_residential_battery(self):
        model = create_residential_battery(capacity_kwh=10.0)

        assert model.params.nominal_capacity_kwh == 10.0
        assert model.params.chemistry == BatteryChemistry.LFP

