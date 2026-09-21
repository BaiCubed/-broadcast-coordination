import pytest
import time
from src.edge.constraints import (
    ConstraintChecker,
    ConstraintResult,
    ConstraintViolation,
    PhysicalConstraints,
    GridStandard,
    GridConstraints,
    GridState,
    HarmonicConstraints,
    check_discharge_feasibility,
    check_charge_feasibility,
)
from src.edge.state_machine import BatteryState, DeviceState


class TestConstraintResult:

    def test_default_not_violated(self):
        result = ConstraintResult()
        assert not result.violated
        assert len(result.violations) == 0
        assert len(result.messages) == 0

    def test_add_violation(self):
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.MAX_POWER_EXCEEDED, "Test message")

        assert result.violated
        assert ConstraintViolation.MAX_POWER_EXCEEDED in result.violations
        assert "Test message" in result.messages

    def test_add_violation_without_message(self):
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.SOC_TOO_LOW)

        assert result.violated
        assert len(result.messages) == 0

    def test_is_safe_with_no_violations(self):
        result = ConstraintResult()
        assert result.is_safe

    def test_is_safe_with_non_critical_violation(self):
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.SOC_TOO_LOW)

        assert result.violated
        assert result.is_safe

    def test_is_not_safe_with_critical_violation(self):
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.OVER_TEMPERATURE)

        assert result.violated
        assert not result.is_safe

    def test_is_not_safe_with_device_faulted(self):
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.DEVICE_FAULTED)

        assert not result.is_safe

    def test_is_not_safe_with_device_offline(self):
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.DEVICE_OFFLINE)

        assert not result.is_safe


class TestPhysicalConstraints:

    def test_default_values(self):
        c = PhysicalConstraints()

        assert c.max_charge_power == 10.0
        assert c.max_discharge_power == 10.0
        assert c.max_ramp_rate == 5.0
        assert c.min_soc == 0.1
        assert c.max_soc == 0.95
        assert c.safe_soc_range == (0.2, 0.8)
        assert c.max_temperature == 45.0
        assert c.min_temperature == 0.0
        assert c.max_daily_cycles == 10


class TestConstraintCheckerBattery:

    def setup_method(self):
        self.checker = ConstraintChecker()
        self.state = BatteryState(
            device_id='bat_001',
            device_type='battery',
            soc=0.5,
            temperature=25.0,
        )

    def test_healthy_battery_passes(self):
        result = self.checker.check(self.state)
        assert not result.violated

    def test_faulted_device_fails(self):
        self.state.is_healthy = False
        self.state.fault_code = 'test_fault'

        result = self.checker.check(self.state)
        assert result.violated
        assert ConstraintViolation.DEVICE_FAULTED in result.violations
        assert not result.is_safe

    def test_soc_too_low(self):
        self.state.soc = 0.05

        result = self.checker.check(self.state)
        assert result.violated
        assert ConstraintViolation.SOC_TOO_LOW in result.violations

    def test_soc_too_high(self):
        self.state.soc = 0.98

        result = self.checker.check(self.state)
        assert result.violated
        assert ConstraintViolation.SOC_TOO_HIGH in result.violations

    def test_over_temperature(self):
        self.state.temperature = 50.0

        result = self.checker.check(self.state)
        assert result.violated
        assert ConstraintViolation.OVER_TEMPERATURE in result.violations
        assert not result.is_safe

    def test_under_temperature(self):
        self.state.temperature = -5.0

        result = self.checker.check(self.state)
        assert result.violated
        assert ConstraintViolation.UNDER_TEMPERATURE in result.violations

    def test_daily_cycles_exceeded(self):
        self.state.daily_cycles = 15

        result = self.checker.check(self.state)
        assert result.violated
        assert ConstraintViolation.DAILY_CYCLES_EXCEEDED in result.violations

    def test_discharge_at_low_soc_blocked(self):
        self.state.soc = 0.2

        result = self.checker.check(self.state, target_power=-5.0)
        assert result.violated
        assert ConstraintViolation.SOC_TOO_LOW in result.violations

    def test_charge_at_high_soc_blocked(self):
        self.state.soc = 0.8

        result = self.checker.check(self.state, target_power=5.0)
        assert result.violated
        assert ConstraintViolation.SOC_TOO_HIGH in result.violations

    def test_max_charge_power_exceeded(self):
        result = self.checker.check(self.state, target_power=15.0)
        assert result.violated
        assert ConstraintViolation.MAX_POWER_EXCEEDED in result.violations

    def test_max_discharge_power_exceeded(self):
        result = self.checker.check(self.state, target_power=-15.0)
        assert result.violated
        assert ConstraintViolation.MAX_POWER_EXCEEDED in result.violations

    def test_ramp_rate_exceeded(self):
        self.state.power_current = 0.0
        result = self.checker.check(self.state, target_power=8.0)
        assert result.violated
        assert ConstraintViolation.RAMP_RATE_EXCEEDED in result.violations


class TestFeasibilityFunctions:

    def setup_method(self):
        self.state = BatteryState(
            device_id='bat_001',
            device_type='battery',
            soc=0.5,
            capacity_kwh=10.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
        )

    def test_discharge_feasibility_ok(self):
        result = check_discharge_feasibility(self.state, 5.0, 3600)
        assert not result

        result = check_discharge_feasibility(self.state, 1.0, 3600)
        assert result

    def test_discharge_feasibility_blocked(self):
        self.state.soc = 0.3
        result = check_discharge_feasibility(self.state, 5.0, 3600)
        assert not result

    def test_charge_feasibility_ok(self):
        result = check_charge_feasibility(self.state, 5.0, 3600)
        assert not result

        result = check_charge_feasibility(self.state, 1.0, 3600)
        assert result

    def test_charge_feasibility_blocked(self):
        self.state.soc = 0.7
        result = check_charge_feasibility(self.state, 5.0, 3600)
        assert not result


class TestConstraintViolationEnum:

    def test_all_violations_defined(self):
        violations = list(ConstraintViolation)

        assert ConstraintViolation.MAX_POWER_EXCEEDED in violations
        assert ConstraintViolation.RAMP_RATE_EXCEEDED in violations
        assert ConstraintViolation.SOC_TOO_LOW in violations
        assert ConstraintViolation.SOC_TOO_HIGH in violations
        assert ConstraintViolation.DEPARTURE_SOC_RISK in violations
        assert ConstraintViolation.OVER_TEMPERATURE in violations
        assert ConstraintViolation.UNDER_TEMPERATURE in violations
        assert ConstraintViolation.MIN_RUN_TIME in violations
        assert ConstraintViolation.DEVICE_FAULTED in violations
        assert ConstraintViolation.DEVICE_OFFLINE in violations


class TestCustomConstraints:

    def test_custom_power_limits(self):
        constraints = PhysicalConstraints(
            max_charge_power=5.0,
            max_discharge_power=3.0,
        )
        checker = ConstraintChecker(constraints)
        state = BatteryState(device_id='bat_001', device_type='battery', soc=0.5)

        result = checker.check(state, target_power=6.0)
        assert result.violated
        assert ConstraintViolation.MAX_POWER_EXCEEDED in result.violations

        result = checker.check(state, target_power=-4.0)
        assert result.violated
        assert ConstraintViolation.MAX_POWER_EXCEEDED in result.violations

    def test_custom_soc_limits(self):
        constraints = PhysicalConstraints(
            min_soc=0.2,
            max_soc=0.9,
        )
        checker = ConstraintChecker(constraints)
        state = BatteryState(device_id='bat_001', device_type='battery', soc=0.15)

        result = checker.check(state)
        assert result.violated
        assert ConstraintViolation.SOC_TOO_LOW in result.violations


class TestGridStandard:
    pass


class TestGridConstraints:

    def test_default_values_gb_t(self):
        gc = GridConstraints()

        assert gc.standard == GridStandard.STANDARD_B
        assert gc.nominal_frequency == 50.0
        assert gc.freq_normal_min == 49.5
        assert gc.freq_normal_max == 50.2
        assert gc.freq_warning_min == 48.5
        assert gc.freq_warning_max == 50.5
        assert gc.freq_critical_min == 47.5
        assert gc.freq_critical_max == 51.5
        assert gc.voltage_normal_min_pu == 0.85
        assert gc.voltage_normal_max_pu == 1.10
        assert gc.lvrt_enabled is True
        assert gc.hvrt_enabled is True
        assert gc.anti_islanding_enabled is True

    def test_standard_a_category_3(self):
        gc = GridConstraints.standard_a(category=3)

        assert gc.standard == GridStandard.STANDARD_A
        assert gc.nominal_frequency == 60.0
        assert gc.freq_normal_min == 59.5
        assert gc.freq_normal_max == 60.1
        assert gc.freq_warning_min == 58.5
        assert gc.freq_critical_min == 56.5
        assert gc.freq_critical_max == 62.0
        assert gc.voltage_normal_min_pu == 0.88
        assert gc.voltage_normal_max_pu == 1.10
        assert gc.lvrt_threshold_50pct_duration == 2.0
        assert gc.rocof_limit == 2.0

    def test_standard_a_category_1(self):
        gc = GridConstraints.standard_a(category=1)

        assert gc.freq_warning_min == 59.0
        assert gc.lvrt_threshold_20pct_duration == 0.0
        assert gc.rocof_limit == 3.0

    def test_standard_b_factory(self):
        gc = GridConstraints.standard_b()

        assert gc.standard == GridStandard.STANDARD_B
        assert gc.nominal_frequency == 50.0
        assert gc.freq_normal_min == 49.5
        assert gc.freq_normal_max == 50.2
        assert gc.voltage_critical_min_pu == 0.20
        assert gc.lvrt_threshold_20pct_duration == 0.625


class TestGridState:

    def test_default_values(self):
        gs = GridState()

        assert gs.frequency == 50.0
        assert gs.frequency_rate_of_change == 0.0
        assert gs.voltage_pu == 1.0
        assert gs.voltage_sag_duration == 0.0
        assert gs.voltage_swell_duration == 0.0
        assert gs.is_islanded is False
        assert gs.thd_voltage == 0.0
        assert gs.thd_current == 0.0

    def test_custom_state(self):
        gs = GridState(
            frequency=49.8,
            voltage_pu=0.95,
            frequency_rate_of_change=-0.5,
        )

        assert gs.frequency == 49.8
        assert gs.voltage_pu == 0.95
        assert gs.frequency_rate_of_change == -0.5


class TestConstraintResultGridViolation:

    def test_has_grid_violation_false_by_default(self):
        result = ConstraintResult()
        assert not result.has_grid_violation

    def test_has_grid_violation_frequency(self):
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.FREQUENCY_OUT_OF_RANGE)
        assert result.has_grid_violation

    def test_has_grid_violation_voltage(self):
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.VOLTAGE_CRITICAL)
        assert result.has_grid_violation

    def test_has_grid_violation_lvrt(self):
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.LVRT_VIOLATION)
        assert result.has_grid_violation

    def test_has_grid_violation_anti_islanding(self):
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.ANTI_ISLANDING_TRIGGERED)
        assert result.has_grid_violation

    def test_is_safe_with_frequency_critical(self):
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.FREQUENCY_CRITICAL)
        assert not result.is_safe

    def test_is_safe_with_voltage_critical(self):
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.VOLTAGE_CRITICAL)
        assert not result.is_safe

    def test_is_safe_with_lvrt_violation(self):
        result = ConstraintResult()
        result.add_violation(ConstraintViolation.LVRT_VIOLATION)
        assert not result.is_safe


class TestGridConstraintChecker:

    def setup_method(self):
        self.checker = ConstraintChecker()
        self.battery_state = BatteryState(
            device_id='bat_001',
            device_type='battery',
            soc=0.5,
            temperature=25.0,
        )

    def test_check_with_no_grid_state(self):
        result = self.checker.check(self.battery_state)
        assert not result.has_grid_violation

    def test_check_with_nominal_grid_state(self):
        grid_state = GridState()
        result = self.checker.check(self.battery_state, grid_state=grid_state)
        assert not result.has_grid_violation

    def test_check_grid_only(self):
        grid_state = GridState(frequency=47.0)
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.FREQUENCY_CRITICAL in result.violations


class TestGridFrequencyConstraints:

    def setup_method(self):
        self.checker = ConstraintChecker()

    def test_frequency_normal_passes(self):
        grid_state = GridState(frequency=50.0)
        result = self.checker.check_grid_only(grid_state)
        assert not result.has_grid_violation

        grid_state = GridState(frequency=49.5)
        result = self.checker.check_grid_only(grid_state)
        assert not result.has_grid_violation

        grid_state = GridState(frequency=50.2)
        result = self.checker.check_grid_only(grid_state)
        assert not result.has_grid_violation

    def test_frequency_warning_low(self):
        grid_state = GridState(frequency=49.0)
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.FREQUENCY_OUT_OF_RANGE in result.violations
        assert ConstraintViolation.FREQUENCY_CRITICAL not in result.violations

    def test_frequency_warning_high(self):
        grid_state = GridState(frequency=50.4)
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.FREQUENCY_OUT_OF_RANGE in result.violations

    def test_frequency_critical_low(self):
        grid_state = GridState(frequency=47.0)
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.FREQUENCY_CRITICAL in result.violations
        assert not result.is_safe

    def test_frequency_critical_high(self):
        grid_state = GridState(frequency=52.0)
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.FREQUENCY_CRITICAL in result.violations
        assert not result.is_safe

    def test_rocof_exceeded(self):
        grid_state = GridState(
            frequency=50.0,
            frequency_rate_of_change=2.5
        )
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.FREQUENCY_CRITICAL in result.violations

    def test_rocof_within_limit(self):
        grid_state = GridState(
            frequency=50.0,
            frequency_rate_of_change=1.5
        )
        result = self.checker.check_grid_only(grid_state)
        assert not result.has_grid_violation


class TestGridVoltageConstraints:

    def setup_method(self):
        self.checker = ConstraintChecker()

    def test_voltage_normal_passes(self):
        grid_state = GridState(voltage_pu=1.0)
        result = self.checker.check_grid_only(grid_state)
        assert not result.has_grid_violation

        grid_state = GridState(voltage_pu=0.85)
        result = self.checker.check_grid_only(grid_state)
        assert not result.has_grid_violation

        grid_state = GridState(voltage_pu=1.10)
        result = self.checker.check_grid_only(grid_state)
        assert not result.has_grid_violation

    def test_voltage_warning_low(self):
        grid_state = GridState(voltage_pu=0.80)
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.VOLTAGE_OUT_OF_RANGE in result.violations

    def test_voltage_warning_high(self):
        grid_state = GridState(voltage_pu=1.15)
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.VOLTAGE_OUT_OF_RANGE in result.violations

    def test_voltage_critical_low(self):
        grid_state = GridState(voltage_pu=0.40)
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.VOLTAGE_CRITICAL in result.violations
        assert not result.is_safe

    def test_voltage_critical_high(self):
        grid_state = GridState(voltage_pu=1.35)
        result = self.checker.check_grid_only(grid_state)
        assert result.has_grid_violation
        assert ConstraintViolation.VOLTAGE_CRITICAL in result.violations
        assert not result.is_safe


class TestLVRTConstraints:

    def setup_method(self):
        self.checker = ConstraintChecker()

    def test_lvrt_20pct_within_time(self):
        grid_state = GridState(
            voltage_pu=0.20,
            voltage_sag_duration=0.5
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.LVRT_VIOLATION not in result.violations

    def test_lvrt_20pct_exceeded(self):
        grid_state = GridState(
            voltage_pu=0.20,
            voltage_sag_duration=0.7
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.LVRT_VIOLATION in result.violations
        assert not result.is_safe

    def test_lvrt_50pct_within_time(self):
        grid_state = GridState(
            voltage_pu=0.50,
            voltage_sag_duration=1.5
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.LVRT_VIOLATION not in result.violations

    def test_lvrt_50pct_exceeded(self):
        grid_state = GridState(
            voltage_pu=0.40,
            voltage_sag_duration=2.5
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.LVRT_VIOLATION in result.violations

    def test_lvrt_85pct_within_time(self):
        grid_state = GridState(
            voltage_pu=0.80,
            voltage_sag_duration=8.0
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.LVRT_VIOLATION not in result.violations

    def test_lvrt_85pct_exceeded(self):
        grid_state = GridState(
            voltage_pu=0.70,
            voltage_sag_duration=12.0
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.LVRT_VIOLATION in result.violations

    def test_lvrt_disabled(self):
        gc = GridConstraints(lvrt_enabled=False)
        checker = ConstraintChecker(grid_constraints=gc)

        grid_state = GridState(
            voltage_pu=0.20,
            voltage_sag_duration=10.0
        )
        result = checker.check_grid_only(grid_state)
        assert ConstraintViolation.LVRT_VIOLATION not in result.violations


class TestHVRTConstraints:

    def setup_method(self):
        self.checker = ConstraintChecker()

    def test_hvrt_120pct_within_time(self):
        grid_state = GridState(
            voltage_pu=1.20,
            voltage_swell_duration=0.3
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HVRT_VIOLATION not in result.violations

    def test_hvrt_120pct_exceeded(self):
        grid_state = GridState(
            voltage_pu=1.25,
            voltage_swell_duration=0.7
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HVRT_VIOLATION in result.violations
        assert not result.is_safe

    def test_hvrt_110pct_within_time(self):
        grid_state = GridState(
            voltage_pu=1.15,
            voltage_swell_duration=8.0
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HVRT_VIOLATION not in result.violations

    def test_hvrt_110pct_exceeded(self):
        grid_state = GridState(
            voltage_pu=1.12,
            voltage_swell_duration=12.0
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HVRT_VIOLATION in result.violations

    def test_hvrt_disabled(self):
        gc = GridConstraints(hvrt_enabled=False)
        checker = ConstraintChecker(grid_constraints=gc)

        grid_state = GridState(
            voltage_pu=1.25,
            voltage_swell_duration=5.0
        )
        result = checker.check_grid_only(grid_state)
        assert ConstraintViolation.HVRT_VIOLATION not in result.violations


class TestAntiIslandingConstraints:

    def setup_method(self):
        self.checker = ConstraintChecker()

    def test_no_islanding_passes(self):
        grid_state = GridState(is_islanded=False)
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.ANTI_ISLANDING_TRIGGERED not in result.violations

    def test_islanding_within_detection_time(self):
        grid_state = GridState(
            is_islanded=True,
            island_detection_time=1.5
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.ANTI_ISLANDING_TRIGGERED not in result.violations

    def test_islanding_exceeded_detection_time(self):
        grid_state = GridState(
            is_islanded=True,
            island_detection_time=2.5
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.ANTI_ISLANDING_TRIGGERED in result.violations
        assert not result.is_safe

    def test_anti_islanding_disabled(self):
        gc = GridConstraints(anti_islanding_enabled=False)
        checker = ConstraintChecker(grid_constraints=gc)

        grid_state = GridState(
            is_islanded=True,
            island_detection_time=10.0
        )
        result = checker.check_grid_only(grid_state)
        assert ConstraintViolation.ANTI_ISLANDING_TRIGGERED not in result.violations


class TestCombinedConstraints:

    def setup_method(self):
        self.checker = ConstraintChecker()
        self.battery_state = BatteryState(
            device_id='bat_001',
            device_type='battery',
            soc=0.5,
            temperature=25.0,
        )

    def test_both_device_and_grid_violations(self):
        self.battery_state.temperature = 50.0

        grid_state = GridState(frequency=47.0)

        result = self.checker.check(self.battery_state, grid_state=grid_state)

        assert ConstraintViolation.OVER_TEMPERATURE in result.violations
        assert ConstraintViolation.FREQUENCY_CRITICAL in result.violations
        assert not result.is_safe

    def test_device_ok_grid_violation(self):
        grid_state = GridState(voltage_pu=1.35)

        result = self.checker.check(self.battery_state, grid_state=grid_state)

        assert ConstraintViolation.VOLTAGE_CRITICAL in result.violations
        assert ConstraintViolation.SOC_TOO_LOW not in result.violations
        assert not result.is_safe

    def test_device_violation_grid_ok(self):
        self.battery_state.soc = 0.05

        grid_state = GridState()

        result = self.checker.check(self.battery_state, grid_state=grid_state)

        assert ConstraintViolation.SOC_TOO_LOW in result.violations
        assert not result.has_grid_violation
        assert result.is_safe


class TestHarmonicConstraints:

    def test_default_values(self):
        hc = HarmonicConstraints()

        assert hc.voltage_class_kv == 0.38
        assert hc.use_50hz_standard is True
        assert hc.short_circuit_ratio == 50.0
        assert hc.thd_voltage_limit_038kv == 5.0
        assert hc.tdd_limit_medium_grid == 8.0

    def test_voltage_thd_limit_50hz_038kv(self):
        hc = HarmonicConstraints(voltage_class_kv=0.38)
        assert hc.get_voltage_thd_limit() == 5.0

    def test_voltage_thd_limit_50hz_10kv(self):
        hc = HarmonicConstraints(voltage_class_kv=10)
        assert hc.get_voltage_thd_limit() == 4.0

    def test_voltage_thd_limit_50hz_35kv(self):
        hc = HarmonicConstraints(voltage_class_kv=35)
        assert hc.get_voltage_thd_limit() == 3.0

    def test_voltage_thd_limit_50hz_110kv(self):
        hc = HarmonicConstraints(voltage_class_kv=110)
        assert hc.get_voltage_thd_limit() == 2.0

    def test_voltage_thd_limit_ieee_1kv(self):
        hc = HarmonicConstraints(use_50hz_standard=False, voltage_class_kv=0.4)
        assert hc.get_voltage_thd_limit() == 8.0

    def test_voltage_thd_limit_ieee_69kv(self):
        hc = HarmonicConstraints(use_50hz_standard=False, voltage_class_kv=35)
        assert hc.get_voltage_thd_limit() == 5.0

    def test_voltage_thd_limit_ieee_161kv(self):
        hc = HarmonicConstraints(use_50hz_standard=False, voltage_class_kv=110)
        assert hc.get_voltage_thd_limit() == 2.5

    def test_voltage_thd_limit_ieee_hv(self):
        hc = HarmonicConstraints(use_50hz_standard=False, voltage_class_kv=220)
        assert hc.get_voltage_thd_limit() == 1.5

    def test_current_tdd_limit_weak_grid(self):
        hc = HarmonicConstraints(short_circuit_ratio=10)
        assert hc.get_current_tdd_limit() == 5.0

    def test_current_tdd_limit_medium_grid(self):
        hc = HarmonicConstraints(short_circuit_ratio=30)
        assert hc.get_current_tdd_limit() == 8.0

    def test_current_tdd_limit_strong_grid(self):
        hc = HarmonicConstraints(short_circuit_ratio=80)
        assert hc.get_current_tdd_limit() == 12.0

    def test_current_tdd_limit_very_strong_grid(self):
        hc = HarmonicConstraints(short_circuit_ratio=500)
        assert hc.get_current_tdd_limit() == 15.0

    def test_current_tdd_limit_extreme_grid(self):
        hc = HarmonicConstraints(short_circuit_ratio=2000)
        assert hc.get_current_tdd_limit() == 20.0

    def test_individual_harmonic_limit_by_voltage(self):
        assert HarmonicConstraints(voltage_class_kv=0.4).get_individual_harmonic_limit() == 5.0
        assert HarmonicConstraints(voltage_class_kv=35).get_individual_harmonic_limit() == 3.0
        assert HarmonicConstraints(voltage_class_kv=110).get_individual_harmonic_limit() == 1.5
        assert HarmonicConstraints(voltage_class_kv=220).get_individual_harmonic_limit() == 1.0


class TestGridStateHarmonics:

    def test_default_harmonic_values(self):
        gs = GridState()
        assert gs.thd_voltage == 0.0
        assert gs.thd_current == 0.0
        assert gs.tdd_current == 0.0
        assert gs.max_individual_voltage_harmonic == 0.0

    def test_harmonic_measurements(self):
        gs = GridState(
            thd_voltage=3.5,
            thd_current=6.2,
            tdd_current=7.8,
            max_individual_voltage_harmonic=2.1,
            max_harmonic_order=5
        )
        assert gs.thd_voltage == 3.5
        assert gs.thd_current == 6.2
        assert gs.tdd_current == 7.8
        assert gs.max_individual_voltage_harmonic == 2.1
        assert gs.max_harmonic_order == 5

    def test_individual_harmonics_dict(self):
        gs = GridState(
            individual_voltage_harmonics={3: 2.5, 5: 4.0, 7: 1.5},
            individual_current_harmonics={5: 3.0, 7: 2.0}
        )
        assert gs.individual_voltage_harmonics[5] == 4.0
        assert len(gs.individual_current_harmonics) == 2


class TestHarmonicConstraintChecker:

    def setup_method(self):
        self.checker = ConstraintChecker()

    def test_voltage_thd_normal_passes(self):
        grid_state = GridState(thd_voltage=3.0)
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED not in result.violations

    def test_voltage_thd_exceeds_limit(self):
        grid_state = GridState(thd_voltage=6.0)
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations
        assert result.has_grid_violation
        assert "Voltage THD" in result.messages[0]

    def test_current_tdd_normal_passes(self):
        grid_state = GridState(tdd_current=5.0)
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED not in result.violations

    def test_current_tdd_exceeds_limit(self):
        grid_state = GridState(tdd_current=15.0)
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations
        assert "Current TDD" in result.messages[0]

    def test_individual_harmonic_exceeds_limit(self):
        grid_state = GridState(
            max_individual_voltage_harmonic=6.0,
            max_harmonic_order=5
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations
        assert "Individual voltage harmonic" in result.messages[0]

    def test_multiple_harmonic_violations(self):
        grid_state = GridState(
            thd_voltage=8.0,
            tdd_current=18.0,
            max_individual_voltage_harmonic=7.0,
            max_harmonic_order=3
        )
        result = self.checker.check_grid_only(grid_state)
        harmonic_violations = [v for v in result.violations
                               if v == ConstraintViolation.HARMONIC_LIMIT_EXCEEDED]
        assert len(harmonic_violations) == 3

    def test_different_voltage_class_limits(self):
        hc_10kv = HarmonicConstraints(voltage_class_kv=10)
        checker_10kv = ConstraintChecker(harmonic_constraints=hc_10kv)

        grid_state = GridState(thd_voltage=4.5)

        result = checker_10kv.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations

    def test_ieee_vs_50hz_standard(self):
        hc_ieee = HarmonicConstraints(use_50hz_standard=False)
        checker_ieee = ConstraintChecker(harmonic_constraints=hc_ieee)

        grid_state = GridState(thd_voltage=6.0)

        result = checker_ieee.check_grid_only(grid_state)
        thd_violations = [m for m in result.messages if "Voltage THD" in m]
        assert len(thd_violations) == 0

    def test_weak_grid_tdd_limit(self):
        hc_weak = HarmonicConstraints(short_circuit_ratio=10)
        checker_weak = ConstraintChecker(harmonic_constraints=hc_weak)

        grid_state = GridState(tdd_current=6.0)

        result = checker_weak.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations


class TestIndividualHarmonicsCheck:

    def setup_method(self):
        self.checker = ConstraintChecker()

    def test_individual_voltage_harmonic_violation(self):
        grid_state = GridState(
            individual_voltage_harmonics={
                3: 2.0,
                5: 6.5,
                7: 1.5
            }
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations
        assert "Voltage harmonic h5" in str(result.messages)

    def test_individual_current_harmonic_violation(self):
        grid_state = GridState(
            individual_current_harmonics={
                5: 5.0,
                7: 12.0,
                11: 3.0
            }
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations
        assert "Current harmonic h7" in str(result.messages)

    def test_high_order_current_harmonic_stricter_limit(self):
        hc = HarmonicConstraints(short_circuit_ratio=80)
        checker = ConstraintChecker(harmonic_constraints=hc)

        grid_state = GridState(
            individual_current_harmonics={
                37: 1.0,
            }
        )
        result = checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations

    def test_all_harmonics_within_limits(self):
        grid_state = GridState(
            thd_voltage=3.0,
            tdd_current=5.0,
            individual_voltage_harmonics={3: 2.0, 5: 3.0, 7: 1.5},
            individual_current_harmonics={3: 3.0, 5: 4.0, 7: 2.0}
        )
        result = self.checker.check_grid_only(grid_state)
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED not in result.violations


class TestHarmonicWithOtherGridConstraints:

    def test_harmonic_and_frequency_violation(self):
        checker = ConstraintChecker()
        grid_state = GridState(
            frequency=47.0,
            thd_voltage=7.0
        )
        result = checker.check_grid_only(grid_state)

        assert ConstraintViolation.FREQUENCY_CRITICAL in result.violations
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations

    def test_harmonic_and_voltage_violation(self):
        checker = ConstraintChecker()
        grid_state = GridState(
            voltage_pu=0.45,
            tdd_current=15.0
        )
        result = checker.check_grid_only(grid_state)

        assert ConstraintViolation.VOLTAGE_CRITICAL in result.violations
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED in result.violations

    def test_normal_harmonics_with_other_violations(self):
        checker = ConstraintChecker()
        grid_state = GridState(
            frequency=47.0,
            voltage_pu=1.35,
            thd_voltage=2.0,
            tdd_current=3.0
        )
        result = checker.check_grid_only(grid_state)

        assert ConstraintViolation.FREQUENCY_CRITICAL in result.violations
        assert ConstraintViolation.VOLTAGE_CRITICAL in result.violations
        assert ConstraintViolation.HARMONIC_LIMIT_EXCEEDED not in result.violations
