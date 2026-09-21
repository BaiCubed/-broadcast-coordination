import pytest
from src.signal.validator import (
    EPSSignalValidator,
    ValidationResult,
    ValidationLevel,
    ValidationIssue,
    quick_validate,
)
from src.signal.encoder import EPSSignal


class TestQuickValidate:

    def test_valid_signal(self):
        signal = EPSSignal(
            version=1,
            timestamp_seq=5,
            region_id=100,
            supply_demand=8,
            intensity=2048,
            price=1000,
            priority=5,
        )

        assert quick_validate(signal) is True

    def test_invalid_version_negative(self):
        signal = EPSSignal(
            version=-1,
            timestamp_seq=0,
            region_id=0,
            supply_demand=0,
            intensity=0,
            price=0,
            priority=0,
        )

        assert quick_validate(signal) is False

    def test_invalid_version_too_large(self):
        signal = EPSSignal(
            version=16,
            timestamp_seq=0,
            region_id=0,
            supply_demand=0,
            intensity=0,
            price=0,
            priority=0,
        )

        assert quick_validate(signal) is False

    def test_invalid_region_id(self):
        signal = EPSSignal(
            version=1,
            timestamp_seq=0,
            region_id=5000,
            supply_demand=0,
            intensity=0,
            price=0,
            priority=0,
        )

        assert quick_validate(signal) is False

    def test_invalid_supply_demand(self):
        signal = EPSSignal(
            version=1,
            timestamp_seq=0,
            region_id=0,
            supply_demand=20,
            intensity=0,
            price=0,
            priority=0,
        )

        assert quick_validate(signal) is False

    def test_invalid_intensity(self):
        signal = EPSSignal(
            version=1,
            timestamp_seq=0,
            region_id=0,
            supply_demand=0,
            intensity=5000,
            price=0,
            priority=0,
        )

        assert quick_validate(signal) is False

    def test_invalid_price(self):
        signal = EPSSignal(
            version=1,
            timestamp_seq=0,
            region_id=0,
            supply_demand=0,
            intensity=0,
            price=-100,
            priority=0,
        )

        assert quick_validate(signal) is False

    def test_invalid_priority(self):
        signal = EPSSignal(
            version=1,
            timestamp_seq=0,
            region_id=0,
            supply_demand=0,
            intensity=0,
            price=0,
            priority=16,
        )

        assert quick_validate(signal) is False

    def test_boundary_values_max(self):
        signal = EPSSignal(
            version=15,
            timestamp_seq=15,
            region_id=4095,
            supply_demand=15,
            intensity=4095,
            price=4095,
            priority=15,
        )

        assert quick_validate(signal) is True

    def test_boundary_values_min(self):
        signal = EPSSignal(
            version=0,
            timestamp_seq=0,
            region_id=0,
            supply_demand=0,
            intensity=0,
            price=0,
            priority=0,
        )

        assert quick_validate(signal) is True


class TestEPSSignalValidator:

    def setup_method(self):
        self.validator = EPSSignalValidator()

    def test_validate_valid_signal(self):
        signal = EPSSignal(
            version=1,
            timestamp_seq=5,
            region_id=100,
            supply_demand=8,
            intensity=2048,
            price=1000,
            priority=5,
        )

        result = self.validator.validate(signal)
        assert result.is_valid

    def test_validate_invalid_version_range(self):
        signal = EPSSignal(
            version=16,
            timestamp_seq=0,
            region_id=100,
            supply_demand=8,
            intensity=2048,
            price=1000,
            priority=5,
        )

        result = self.validator.validate(signal)
        assert not result.is_valid
        assert any('version' in e.field.lower() for e in result.errors if e.field)

    def test_validate_invalid_region_range(self):
        signal = EPSSignal(
            version=1,
            timestamp_seq=0,
            region_id=5000,
            supply_demand=8,
            intensity=2048,
            price=1000,
            priority=5,
        )

        result = self.validator.validate(signal)
        assert not result.is_valid
        assert any('region' in e.field.lower() for e in result.errors if e.field)

    def test_validate_unsupported_version(self):
        signal = EPSSignal(
            version=2,
            timestamp_seq=0,
            region_id=100,
            supply_demand=8,
            intensity=2048,
            price=1000,
            priority=5,
        )

        result = self.validator.validate(signal)
        assert not result.is_valid
        assert any('UNSUPPORTED_VERSION' in e.code for e in result.errors)

    def test_validate_with_valid_regions_allowed(self):
        validator = EPSSignalValidator(config={'valid_regions': [100, 200, 300]})

        signal = EPSSignal(
            version=1,
            timestamp_seq=0,
            region_id=100,
            supply_demand=8,
            intensity=2048,
            price=1000,
            priority=5,
        )
        result = validator.validate(signal)
        assert result.is_valid

    def test_validate_with_valid_regions_disallowed(self):
        validator = EPSSignalValidator(config={'valid_regions': [100, 200, 300]})

        signal = EPSSignal(
            version=1,
            timestamp_seq=0,
            region_id=500,
            supply_demand=8,
            intensity=2048,
            price=1000,
            priority=5,
        )
        result = validator.validate(signal)
        assert not result.is_valid
        assert any('INVALID_REGION' in e.code for e in result.errors)

    def test_validate_high_intensity_warning(self):
        signal = EPSSignal(
            version=1,
            timestamp_seq=0,
            region_id=100,
            supply_demand=8,
            intensity=4000,
            price=1000,
            priority=5,
        )

        result = self.validator.validate(signal)
        assert result.is_valid
        assert any('HIGH_INTENSITY' in w.code for w in result.warnings)

    def test_validate_shortage_no_intensity_warning(self):
        signal = EPSSignal(
            version=1,
            timestamp_seq=0,
            region_id=100,
            supply_demand=12,
            intensity=0,
            price=1000,
            priority=5,
        )

        result = self.validator.validate(signal)
        assert result.is_valid
        assert any('SHORTAGE_NO_INTENSITY' in w.code for w in result.warnings)


class TestValidationResult:

    def test_valid_result(self):
        result = ValidationResult(is_valid=True)
        assert result.is_valid
        assert len(result.issues) == 0

    def test_add_error(self):
        result = ValidationResult(is_valid=True)
        result.add_error("TEST_ERROR", "Test error message", "field")

        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].code == "TEST_ERROR"

    def test_add_warning(self):
        result = ValidationResult(is_valid=True)
        result.add_warning("TEST_WARNING", "Test warning message", "field")

        assert result.is_valid
        assert len(result.warnings) == 1
        assert result.warnings[0].code == "TEST_WARNING"

    def test_add_info(self):
        result = ValidationResult(is_valid=True)
        result.add_info("TEST_INFO", "Test info message", "field")

        assert result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0].level == ValidationLevel.INFO

    def test_errors_property(self):
        result = ValidationResult(is_valid=True)
        result.add_error("ERR1", "Error 1")
        result.add_warning("WARN1", "Warning 1")
        result.add_error("ERR2", "Error 2")

        assert len(result.errors) == 2
        assert all(e.level == ValidationLevel.ERROR for e in result.errors)

    def test_warnings_property(self):
        result = ValidationResult(is_valid=True)
        result.add_error("ERR1", "Error 1")
        result.add_warning("WARN1", "Warning 1")
        result.add_warning("WARN2", "Warning 2")

        assert len(result.warnings) == 2
        assert all(w.level == ValidationLevel.WARNING for w in result.warnings)


class TestValidationIssue:

    def test_creation(self):
        issue = ValidationIssue(
            level=ValidationLevel.ERROR,
            code="TEST_CODE",
            message="Test message",
            field="test_field"
        )

        assert issue.level == ValidationLevel.ERROR
        assert issue.code == "TEST_CODE"
        assert issue.message == "Test message"
        assert issue.field == "test_field"

    def test_creation_without_field(self):
        issue = ValidationIssue(
            level=ValidationLevel.WARNING,
            code="TEST_CODE",
            message="Test message"
        )

        assert issue.field is None


class TestValidationLevel:

    def test_all_levels_defined(self):
        levels = list(ValidationLevel)

        assert ValidationLevel.ERROR in levels
        assert ValidationLevel.WARNING in levels
        assert ValidationLevel.INFO in levels

    def test_level_values(self):
        assert ValidationLevel.ERROR.value == "error"
        assert ValidationLevel.WARNING.value == "warning"
        assert ValidationLevel.INFO.value == "info"
