from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import numpy as np
from datetime import datetime, timedelta


class ScenarioType(Enum):
    NORMAL = "normal"

    PEAK_SHAVING = "peak_shaving"
    VALLEY_FILLING = "valley_filling"

    EMERGENCY_GRID_STABILITY = "emergency_grid_stability"
    EMERGENCY_SUPPLY_SHORTAGE = "emergency_supply_shortage"

    EMERGENCY = "emergency"

    FREQUENCY_REGULATION = "frequency_regulation"


@dataclass
class LoadProfile:
    hourly_factors: np.ndarray = field(
        default_factory=lambda: np.ones(24)
    )

    base_load_mw: float = 1000.0

    peak_hour: int = 19

    valley_hour: int = 4

    def get_load(self, hour: float) -> float:
        hour_int = int(hour) % 24
        hour_frac = hour - int(hour)

        next_hour = (hour_int + 1) % 24
        factor = (1 - hour_frac) * self.hourly_factors[hour_int] + \
                 hour_frac * self.hourly_factors[next_hour]

        return self.base_load_mw * factor

    @classmethod
    def typical_residential(cls) -> 'LoadProfile':
        factors = np.array([
            0.6, 0.55, 0.5, 0.5, 0.5, 0.55,
            0.7, 0.85, 0.9, 0.8, 0.75, 0.8,
            0.85, 0.8, 0.75, 0.75, 0.8, 0.9,
            1.0, 1.0, 0.95, 0.85, 0.75, 0.65,
        ])
        return cls(hourly_factors=factors, base_load_mw=1000.0, peak_hour=19, valley_hour=4)

    @classmethod
    def typical_commercial(cls) -> 'LoadProfile':
        factors = np.array([
            0.4, 0.35, 0.35, 0.35, 0.35, 0.4,
            0.5, 0.7, 0.85, 1.0, 1.0, 1.0,
            0.95, 1.0, 1.0, 1.0, 0.95, 0.85,
            0.6, 0.5, 0.45, 0.4, 0.4, 0.4,
        ])
        return cls(hourly_factors=factors, base_load_mw=500.0, peak_hour=10, valley_hour=3)

    @classmethod
    def high_demand(cls) -> 'LoadProfile':
        factors = np.array([
            0.7, 0.65, 0.6, 0.6, 0.6, 0.65,
            0.75, 0.85, 0.95, 1.0, 1.0, 1.0,
            1.0, 1.05, 1.1, 1.1, 1.05, 1.0,
            0.95, 0.9, 0.85, 0.8, 0.75, 0.7,
        ])
        return cls(hourly_factors=factors, base_load_mw=1200.0, peak_hour=15, valley_hour=3)


@dataclass
class RenewableProfile:
    solar_factors: np.ndarray = field(
        default_factory=lambda: np.zeros(24)
    )

    wind_factors: np.ndarray = field(
        default_factory=lambda: np.ones(24) * 0.3
    )

    solar_capacity_mw: float = 500.0
    wind_capacity_mw: float = 200.0

    def get_solar(self, hour: float) -> float:
        hour_int = int(hour) % 24
        return self.solar_capacity_mw * self.solar_factors[hour_int]

    def get_wind(self, hour: float) -> float:
        hour_int = int(hour) % 24
        return self.wind_capacity_mw * self.wind_factors[hour_int]

    def get_total(self, hour: float) -> float:
        return self.get_solar(hour) + self.get_wind(hour)

    @classmethod
    def typical_renewable(cls) -> 'RenewableProfile':
        solar = np.array([
            0, 0, 0, 0, 0, 0.05,
            0.15, 0.35, 0.55, 0.75, 0.9, 0.95,
            1.0, 0.95, 0.85, 0.7, 0.5, 0.3,
            0.1, 0.02, 0, 0, 0, 0,
        ])
        wind = np.array([
            0.4, 0.45, 0.5, 0.5, 0.45, 0.35,
            0.25, 0.2, 0.2, 0.25, 0.3, 0.35,
            0.35, 0.3, 0.25, 0.2, 0.2, 0.25,
            0.3, 0.35, 0.4, 0.45, 0.45, 0.4,
        ])
        return cls(solar_factors=solar, wind_factors=wind)

    @classmethod
    def high_variability(cls) -> 'RenewableProfile':
        solar = np.array([
            0, 0, 0, 0, 0, 0.05,
            0.15, 0.35, 0.6, 0.4, 0.9, 0.5,
            1.0, 0.6, 0.85, 0.4, 0.5, 0.3,
            0.1, 0.02, 0, 0, 0, 0,
        ])
        wind = np.array([
            0.4, 0.6, 0.3, 0.5, 0.45, 0.25,
            0.25, 0.2, 0.2, 0.25, 0.3, 0.35,
            0.35, 0.3, 0.25, 0.2, 0.2, 0.25,
            0.3, 0.5, 0.3, 0.6, 0.4, 0.5,
        ])
        return cls(solar_factors=solar, wind_factors=wind)


@dataclass
class ScenarioConfig:
    name: str
    scenario_type: ScenarioType
    description: str

    paper_description: str = ""

    perspective: str = "supply"

    duration_hours: float = 24.0

    load_profile: LoadProfile = field(default_factory=LoadProfile.typical_residential)
    renewable_profile: Optional[RenewableProfile] = None

    signal_intensity_range: tuple = (1000, 3000)

    target_response_rate: float = 0.5
    target_peak_reduction: float = 0.15
    max_latency_ms: float = 100.0

    emergency_duration_minutes: float = 30.0
    emergency_intensity: int = 4000

    supply_deficit_mw: float = 0.0
    shortage_cause: str = ""


class ScenarioGenerator:


    @staticmethod
    def peak_shaving() -> ScenarioConfig:
        return ScenarioConfig(
            name="Peak Shaving",
            scenario_type=ScenarioType.PEAK_SHAVING,
            description="Reduce evening peak demand through coordinated discharge",
            paper_description=(
                "Peak shaving scenario evaluates EPS capability to reduce "
                "evening demand peaks (17:00-21:00) when solar generation drops. "
                "Primary metric: Response magnitude (MW)."
            ),
            perspective="supply",
            load_profile=LoadProfile.typical_residential(),
            renewable_profile=RenewableProfile.typical_renewable(),
            signal_intensity_range=(2000, 4000),

            target_peak_reduction=0.20,
            target_response_rate=0.5,
        )


    @staticmethod
    def valley_filling() -> ScenarioConfig:
        return ScenarioConfig(
            name="Valley Filling",
            scenario_type=ScenarioType.VALLEY_FILLING,
            description="Absorb renewable surplus through flexible charging",
            paper_description=(
                "Valley filling scenario evaluates EPS capability to absorb "
                "excess renewable generation during low-demand periods. "
                "Primary metric: Absorption capacity (MW), Curtailment reduction (%)."
            ),
            perspective="supply",
            load_profile=LoadProfile.typical_residential(),
            renewable_profile=RenewableProfile.typical_renewable(),
            signal_intensity_range=(2000, 4000),

            target_response_rate=0.6,
        )


    @staticmethod
    def emergency_grid_stability() -> ScenarioConfig:
        return ScenarioConfig(
            name="Emergency - Grid Stability",
            scenario_type=ScenarioType.EMERGENCY_GRID_STABILITY,
            description="Rapid frequency/voltage stabilization response",
            paper_description=(
                "Grid stability scenario evaluates EPS emergency response capability "
                "for supply-side contingencies (generator trip, renewable fluctuation). "
                "Critical metric: P99 latency < 100ms for frequency regulation."
            ),
            perspective="supply",
            renewable_profile=RenewableProfile.high_variability(),
            signal_intensity_range=(3500, 4095),

            target_response_rate=0.8,
            max_latency_ms=100.0,
            emergency_duration_minutes=15.0,
            emergency_intensity=4000,
        )

    @staticmethod
    def emergency_supply_shortage() -> ScenarioConfig:
        return ScenarioConfig(
            name="Emergency - Supply Shortage",
            scenario_type=ScenarioType.EMERGENCY_SUPPLY_SHORTAGE,
            description="Sustained response to urban power deficit",
            paper_description=(
                "Supply shortage scenario evaluates EPS capability for demand-side "
                "power deficit events (extreme weather, equipment failure). "
                "Demonstrates cross-regional power mutual support. "
                "Primary metric: Coverage rate (%), sustained response duration."
            ),
            perspective="demand",
            load_profile=LoadProfile.high_demand(),
            renewable_profile=RenewableProfile.typical_renewable(),
            signal_intensity_range=(3000, 4095),

            target_response_rate=0.7,
            max_latency_ms=300.0,
            emergency_duration_minutes=120.0,
            emergency_intensity=3500,
            supply_deficit_mw=500.0,
            shortage_cause="supply_deficit",
        )


    @staticmethod
    def emergency_response() -> ScenarioConfig:
        config = ScenarioGenerator.emergency_grid_stability()
        config.name = "Emergency Response"
        config.scenario_type = ScenarioType.EMERGENCY
        return config


    @classmethod
    def get_all_scenarios(cls) -> List[ScenarioConfig]:
        return [
            cls.peak_shaving(),
            cls.valley_filling(),
            cls.emergency_grid_stability(),
            cls.emergency_supply_shortage(),
        ]

    @classmethod
    def get_primary_scenarios(cls) -> List[ScenarioConfig]:
        return [
            cls.peak_shaving(),
            cls.valley_filling(),
            cls.emergency_response(),
        ]

    @classmethod
    def get_scenario_by_name(cls, name: str) -> Optional[ScenarioConfig]:
        name_lower = name.lower().replace("_", " ").replace("-", " ")

        mapping = {
            "peak shaving": cls.peak_shaving,
            "peak_shaving": cls.peak_shaving,
            "valley filling": cls.valley_filling,
            "valley_filling": cls.valley_filling,
            "emergency": cls.emergency_response,
            "emergency response": cls.emergency_response,
            "emergency_response": cls.emergency_response,
            "emergency grid stability": cls.emergency_grid_stability,
            "emergency_grid_stability": cls.emergency_grid_stability,
            "grid stability": cls.emergency_grid_stability,
            "emergency supply shortage": cls.emergency_supply_shortage,
            "emergency_supply_shortage": cls.emergency_supply_shortage,
            "supply shortage": cls.emergency_supply_shortage,
        }

        if name_lower in mapping:
            return mapping[name_lower]()
        return None


@dataclass
class ScenarioMetrics:
    total_responses: int = 0
    response_rate: float = 0.0

    total_energy_kwh: float = 0.0
    peak_reduction_mw: float = 0.0
    peak_reduction_percent: float = 0.0

    absorption_mw: float = 0.0
    curtailment_reduction_percent: float = 0.0

    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    total_incentive: float = 0.0
    cost_per_kwh: float = 0.0

    frequency_deviation_hz: float = 0.0
    voltage_deviation_percent: float = 0.0

    coverage_rate: float = 0.0
    sustained_duration_minutes: float = 0.0

    met_response_target: bool = False
    met_latency_target: bool = False
    met_peak_reduction_target: bool = False

    def evaluate(self, config: ScenarioConfig) -> Dict[str, Any]:
        self.met_response_target = self.response_rate >= config.target_response_rate
        self.met_latency_target = self.p99_latency_ms <= config.max_latency_ms
        self.met_peak_reduction_target = self.peak_reduction_percent >= config.target_peak_reduction

        return {
            'overall_success': all([
                self.met_response_target,
                self.met_latency_target,
            ]),
            'response_rate': {
                'actual': self.response_rate,
                'target': config.target_response_rate,
                'met': self.met_response_target,
            },
            'latency': {
                'p99_ms': self.p99_latency_ms,
                'target_ms': config.max_latency_ms,
                'met': self.met_latency_target,
            },
            'peak_reduction': {
                'percent': self.peak_reduction_percent,
                'target': config.target_peak_reduction,
                'met': self.met_peak_reduction_target,
            },
            'perspective': config.perspective,
            'scenario_type': config.scenario_type.value,
        }


