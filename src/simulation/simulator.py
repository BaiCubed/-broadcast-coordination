from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import logging

from ..signal import EPSSignal, EPSSignalEncoder, EPSSignalDecoder
from ..edge import (
    BatteryModel,
    create_residential_battery,
    BatteryState,
)
from .constants import (
    EPS_LATENCY, BATTERY_RESPONSE,
)

logger = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    enable_correlation: bool = False

    global_shock_std: float = 0.15

    regional_shock_std: float = 0.08

    battery_regional_soc_std: float = 0.15
    battery_within_cluster_std: float = 0.08


class SimulationLevel(Enum):
    LEVEL1_AGENT = 1


@dataclass
class SimulationConfig:
    level: SimulationLevel = SimulationLevel.LEVEL1_AGENT
    num_devices: int = 5000
    duration_seconds: float = 86400.0
    time_step: float = 60.0
    random_seed: Optional[int] = 42

    battery_fraction: float = 1.0

    num_workers: int = 4
    chunk_size: int = 1000

    num_regions: int = 5

    broadcast_interval: float = 60.0

    eps_packet_loss_rate: float = 0.001

    battery_capacity_range: Tuple[float, float] = (5.0, 20.0)

    battery_c_rate_mean: float = 0.45
    battery_c_rate_sigma: float = 0.35
    battery_soh_range: Tuple[float, float] = (0.82, 1.0)
    device_offline_rate: float = 0.015

    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)

    response_prob_bias: float = 1.0
    soc_noise_std: float = 0.0

    continuous_response: bool = False
    sigmoid_response: bool = False
    sigmoid_steepness: float = 5.0

    def __post_init__(self):
        if abs(self.battery_fraction - 1.0) > 1e-6:
            raise ValueError(f"battery_fraction must be 1.0, got {self.battery_fraction}")


@dataclass
class DevicePopulation:
    batteries: List[BatteryModel] = field(default_factory=list)

    battery_states: List[BatteryState] = field(default_factory=list)

    region_assignments: Dict[str, int] = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        return len(self.batteries)

    def get_devices_by_region(self, region_id: int) -> Dict[str, List]:
        result = {'batteries': []}
        for i, battery in enumerate(self.batteries):
            if self.region_assignments.get(f'battery_{i}') == region_id:
                result['batteries'].append((i, battery))
        return result


@dataclass
class TimeStep:
    step_index: int
    timestamp: datetime
    elapsed_seconds: float

    signal: Optional[EPSSignal] = None

    total_response_kw: float = 0.0
    responding_devices: int = 0
    latency_samples: List[float] = field(default_factory=list)


@dataclass
class SimulationResult:
    total_responses: int = 0
    total_energy_kwh: float = 0.0
    average_latency_ms: float = 0.0
    response_rate: float = 0.0

    time_steps: List[TimeStep] = field(default_factory=list)

    region_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)

    battery_response_kwh: float = 0.0

    charge_energy_kwh: float = 0.0
    discharge_energy_kwh: float = 0.0
    net_energy_kwh: float = 0.0
    load_reduction_kwh: float = 0.0
    load_increase_kwh: float = 0.0

    latency_percentiles: Dict[str, float] = field(default_factory=dict)
    response_distribution: Dict[str, float] = field(default_factory=dict)

    n_responding_devices: int = 0

    devices_per_second: float = 0.0
    memory_usage_mb: float = 0.0

    metrics: Dict[str, Any] = field(default_factory=dict)

    def compute_latency_percentiles(self):
        all_latencies = []
        for ts in self.time_steps:
            all_latencies.extend(ts.latency_samples)

        if all_latencies:
            arr = np.array(all_latencies)
            self.latency_percentiles = {
                'p50': float(np.percentile(arr, 50)),
                'p90': float(np.percentile(arr, 90)),
                'p95': float(np.percentile(arr, 95)),
                'p99': float(np.percentile(arr, 99)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
            }
            self.average_latency_ms = self.latency_percentiles['mean']


class PopulationGenerator:

    def __init__(self, config: SimulationConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    def generate(self) -> DevicePopulation:
        pop = DevicePopulation()
        corr = self.config.correlation

        n_batteries = self.config.num_devices

        if corr.enable_correlation:
            region_soc_centers = {}
            for r in range(self.config.num_regions):
                region_soc_centers[r] = np.clip(
                    self.rng.normal(0.5, corr.battery_regional_soc_std), 0.15, 0.85
                )

        for i in range(n_batteries):
            capacity = self.rng.uniform(*self.config.battery_capacity_range)

            region = int(self.rng.integers(0, self.config.num_regions))
            pop.region_assignments[f'battery_{i}'] = region

            if corr.enable_correlation:
                initial_soc = np.clip(
                    self.rng.normal(region_soc_centers[region], corr.battery_within_cluster_std),
                    0.1, 0.95
                )
            else:
                initial_soc = self.rng.uniform(0.2, 0.9)

            mu_c = self.config.battery_c_rate_mean
            sigma_c = self.config.battery_c_rate_sigma
            c_rate = float(np.clip(self.rng.lognormal(
                mean=np.log(mu_c) - sigma_c**2 / 2,
                sigma=sigma_c,
            ), 0.2, 1.0))

            initial_soh = float(self.rng.uniform(*self.config.battery_soh_range))

            battery = create_residential_battery(
                capacity_kwh=capacity,
                c_rate=c_rate,
                initial_soh=initial_soh,
            )
            battery.params.soc_reserve = float(self.rng.uniform(0.1, 0.2))
            battery.params.soc_max = float(self.rng.uniform(0.8, 0.95))
            battery._soc = initial_soc
            pop.batteries.append(battery)

            max_power = capacity * c_rate
            state = BatteryState(
                device_id=f'battery_{i}',
                device_type='battery',
                soc=initial_soc,
                capacity_kwh=capacity,
                max_charge_kw=max_power,
                max_discharge_kw=max_power,
            )
            pop.battery_states.append(state)

        return pop


class SignalGenerator:

    def __init__(self, config: SimulationConfig, rng: np.random.Generator, start_hour: float = 0.0):
        self.config = config
        self.rng = rng
        self.encoder = EPSSignalEncoder()
        self.start_hour = start_hour

        self.base_solar_capacity = 500.0
        self.base_wind_capacity = 150.0
        self.base_load = 400.0

        self._override_signal: Optional[EPSSignal] = None

    def set_override_signal(
        self,
        intensity: Optional[int] = None,
        price_value: Optional[float] = None,
        supply_demand: Optional[int] = None,
    ) -> None:
        if intensity is not None or supply_demand is not None:
            self._override_signal = {
                'intensity': intensity,
                'supply_demand': supply_demand,
            }
        else:
            self._override_signal = None

    def clear_override_signal(self) -> None:
        self._override_signal = None

    def set_start_hour(self, hour: float) -> None:
        self.start_hour = hour % 24

    def get_solar_generation(self, hour: float) -> float:
        if hour < 6 or hour > 18:
            return 0.0

        phase = (hour - 6) / 12 * np.pi
        generation = self.base_solar_capacity * np.sin(phase)

        noise = self.rng.normal(1.0, 0.1)
        return max(0, generation * noise)

    def get_wind_generation(self, hour: float) -> float:
        if 22 <= hour or hour <= 6:
            base_factor = 0.7
        elif 10 <= hour <= 16:
            base_factor = 0.3
        else:
            base_factor = 0.5

        noise = self.rng.normal(1.0, 0.3)
        return max(0, self.base_wind_capacity * base_factor * noise)

    def get_base_load(self, hour: float) -> float:
        if 0 <= hour < 5:
            factor = 0.6
        elif 5 <= hour < 7:
            factor = 0.7
        elif 7 <= hour < 9:
            factor = 1.0
        elif 9 <= hour < 11:
            factor = 0.9
        elif 11 <= hour < 14:
            factor = 0.95
        elif 14 <= hour < 17:
            factor = 0.85
        elif 17 <= hour < 21:
            factor = 1.2
        elif 21 <= hour < 23:
            factor = 0.9
        else:
            factor = 0.7

        noise = self.rng.normal(1.0, 0.05)
        return self.base_load * factor * noise

    def compute_supply_demand_state(self, hour: float) -> dict:
        solar = self.get_solar_generation(hour)
        wind = self.get_wind_generation(hour)
        load = self.get_base_load(hour)

        total_supply = solar + wind
        net_balance = total_supply - load

        if net_balance > 200:
            supply_demand = 0
            scenario = 'valley_filling'
            is_surplus = True
        elif net_balance > 100:
            supply_demand = 2
            scenario = 'valley_filling'
            is_surplus = True
        elif net_balance > 50:
            supply_demand = 4
            scenario = 'valley_filling'
            is_surplus = True
        elif net_balance > 0:
            supply_demand = 6
            scenario = 'normal'
            is_surplus = True
        elif net_balance > -50:
            supply_demand = 7
            scenario = 'normal'
            is_surplus = False
        elif net_balance > -100:
            supply_demand = 9
            scenario = 'peak_shaving'
            is_surplus = False
        elif net_balance > -200:
            supply_demand = 11
            scenario = 'peak_shaving'
            is_surplus = False
        elif net_balance > -300:
            supply_demand = 13
            scenario = 'peak_shaving'
            is_surplus = False
        else:
            supply_demand = 15
            scenario = 'emergency'
            is_surplus = False

        r = total_supply / max(load, 1e-6)

        return {
            'supply_demand': supply_demand,
            'supply_demand_ratio': r,
            'scenario': scenario,
            'is_surplus': is_surplus,
            'net_balance': net_balance,
            'solar_mw': solar,
            'wind_mw': wind,
            'load_mw': load,
            'total_supply_mw': total_supply,
        }

    def generate_signal(
        self,
        region_id: int,
        supply_demand: int,
        intensity: int,
        priority: int = 8,
    ) -> EPSSignal:
        return EPSSignal(
            version=1,
            timestamp_seq=int(datetime.now().timestamp()) % 16,
            region_id=region_id,
            supply_demand=supply_demand,
            intensity=intensity,
            price=0,
            priority=priority,
        )

    def generate_scenario_signals(
        self,
        scenario: str,
        time_step: int,
        num_regions: int,
    ) -> List[EPSSignal]:
        signals = []
        hour = (self.start_hour + time_step * self.config.time_step / 3600) % 24

        state = self.compute_supply_demand_state(hour)
        net_balance = state['net_balance']
        imbalance = abs(net_balance)

        if scenario == 'peak_shaving':
            if net_balance < -100:
                supply_demand = min(15, 10 + int(imbalance / 50))
                intensity = int(self.rng.uniform(3000, 4095))
                priority = 12
            elif net_balance < -50:
                supply_demand = 10
                intensity = int(self.rng.uniform(2000, 3000))
                priority = 10
            elif net_balance < 0:
                supply_demand = 8
                intensity = int(self.rng.uniform(1500, 2000))
                priority = 8
            else:
                supply_demand = max(3, 7 - int(net_balance / 30))
                intensity = int(self.rng.uniform(1000, 1500))
                priority = 6

        elif scenario == 'valley_filling':
            if net_balance > 100:
                supply_demand = max(0, 3 - int(net_balance / 50))
                intensity = int(self.rng.uniform(3000, 4095))
                priority = 12
            elif net_balance > 50:
                supply_demand = 4
                intensity = int(self.rng.uniform(2000, 3000))
                priority = 10
            elif net_balance > 0:
                supply_demand = 6
                intensity = int(self.rng.uniform(1500, 2000))
                priority = 8
            else:
                supply_demand = min(12, 8 + int(imbalance / 30))
                intensity = int(self.rng.uniform(1000, 1500))
                priority = 6

        elif scenario in ['emergency', 'emergency_grid_stability']:
            intensity = int(self.rng.uniform(3500, 4095))
            supply_demand = 15
            priority = 15

        elif scenario == 'emergency_supply_shortage':
            if imbalance > 200:
                intensity = int(self.rng.uniform(3000, 4095))
                supply_demand = 14
                priority = 14
            elif imbalance > 100:
                intensity = int(self.rng.uniform(2500, 3500))
                supply_demand = 12
                priority = 12
            else:
                intensity = int(self.rng.uniform(2000, 3000))
                supply_demand = 10
                priority = 10

        else:
            supply_demand = state['supply_demand']

            if imbalance > 200:
                intensity = int(self.rng.uniform(3000, 4095))
                priority = 12
            elif imbalance > 100:
                intensity = int(self.rng.uniform(2000, 3000))
                priority = 10
            elif imbalance > 50:
                intensity = int(self.rng.uniform(1500, 2000))
                priority = 8
            else:
                intensity = int(self.rng.uniform(800, 1500))
                priority = 6

        if self._override_signal is not None:
            if self._override_signal.get('intensity') is not None:
                intensity = self._override_signal['intensity']
            if self._override_signal.get('supply_demand') is not None:
                supply_demand = self._override_signal['supply_demand']

        for region_id in range(num_regions):
            regional_intensity = max(0, min(4095,
                intensity + int(self.rng.normal(0, 200))))
            regional_sd = max(0, min(15,
                supply_demand + int(self.rng.normal(0, 1))))

            signal = self.generate_signal(
                region_id=region_id,
                supply_demand=regional_sd,
                intensity=regional_intensity,
                priority=priority,
            )
            signals.append(signal)

        return signals


class EPSSimulator:

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

        self._population: Optional[DevicePopulation] = None
        self._signal_generator: Optional[SignalGenerator] = None
        self._is_initialized = False

        self._region_perturbation: Optional[Dict[int, float]] = None

        self._step_metrics: List[Dict] = []

    def initialize(self) -> 'EPSSimulator':
        logger.info(f"Initializing simulator with {self.config.num_devices} devices")

        pop_gen = PopulationGenerator(self.config, self.rng)
        self._population = pop_gen.generate()

        self._signal_generator = SignalGenerator(self.config, self.rng)

        self._is_initialized = True
        logger.info(f"Initialization complete: {self._population.total_count} devices")
        return self

    def set_override_signal(
        self,
        intensity: Optional[int] = None,
        price_value: Optional[float] = None,
        supply_demand: Optional[int] = None,
    ) -> None:
        if not self._is_initialized:
            self.initialize()
        self._signal_generator.set_override_signal(intensity, price_value, supply_demand)

    def clear_override_signal(self) -> None:
        if self._signal_generator:
            self._signal_generator.clear_override_signal()

    def set_start_hour(self, hour: float) -> None:
        if not self._is_initialized:
            self.initialize()
        self._signal_generator.set_start_hour(hour)

    def run(
        self,
        num_steps: Optional[int] = None,
        scenario: str = 'normal',
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SimulationResult:
        if not self._is_initialized:
            self.initialize()

        if num_steps is None:
            num_steps = int(self.config.duration_seconds / self.config.time_step)

        result = SimulationResult()
        start_time = datetime.now()

        result = self._run_level1_agent(num_steps, scenario, progress_callback)

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        result.devices_per_second = (self.config.num_devices * num_steps) / max(elapsed, 0.001)
        result.compute_latency_percentiles()

        return result

    def _run_level1_agent(
        self,
        num_steps: int,
        scenario: str,
        progress_callback: Optional[Callable],
    ) -> SimulationResult:
        result = SimulationResult()

        corr = self.config.correlation
        if corr.enable_correlation:
            persistent_global_shock = self.rng.normal(0, corr.global_shock_std)
            self._region_perturbation = {}
            for r in range(self.config.num_regions):
                self._region_perturbation[r] = (
                    persistent_global_shock
                    + self.rng.normal(0, corr.regional_shock_std)
                )
        else:
            self._region_perturbation = None

        for step in range(num_steps):
            ts = TimeStep(
                step_index=step,
                timestamp=datetime.now() + timedelta(seconds=step * self.config.time_step),
                elapsed_seconds=step * self.config.time_step,
            )

            signals = self._signal_generator.generate_scenario_signals(
                scenario, step, self.config.num_regions
            )

            if signals:
                ts.signal = signals[0]

            for region_id, signal in enumerate(signals):
                region_response = self._process_region_level1(
                    region_id, signal, step, scenario, sim_time=ts.timestamp
                )
                ts.total_response_kw += region_response['total_kw']
                ts.responding_devices += region_response['responding']
                ts.latency_samples.extend(region_response['latencies'])

                result.battery_response_kwh += region_response.get('battery_kwh', 0)

            result.time_steps.append(ts)
            result.total_responses += ts.responding_devices

            energy_delta_kwh = ts.total_response_kw * (self.config.time_step / 3600)

            if energy_delta_kwh > 0:
                result.charge_energy_kwh += energy_delta_kwh
                result.load_increase_kwh += energy_delta_kwh
            else:
                result.discharge_energy_kwh += abs(energy_delta_kwh)
                result.load_reduction_kwh += abs(energy_delta_kwh)

            result.net_energy_kwh += energy_delta_kwh
            result.total_energy_kwh = result.net_energy_kwh

            if progress_callback:
                progress_callback(step + 1, num_steps)

        total_possible = self.config.num_devices * num_steps
        result.response_rate = result.total_responses / max(total_possible, 1)
        result.n_responding_devices = result.total_responses // max(num_steps, 1)

        return result

    def _process_region_level1(
        self,
        region_id: int,
        signal: EPSSignal,
        step: int,
        scenario: str = 'normal',
        sim_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        result = {
            'total_kw': 0.0,
            'responding': 0,
            'latencies': [],
            'battery_kwh': 0.0,
        }

        devices = self._population.get_devices_by_region(region_id)

        for idx, battery in devices['batteries']:
            response = self._simulate_battery_response(
                battery, signal, step, scenario, device_id=f'battery_{idx}')
            if response['responded']:
                result['responding'] += 1
                result['total_kw'] += response['power_kw']
                result['latencies'].append(response['latency_ms'])
                result['battery_kwh'] += abs(response['power_kw']) * (self.config.time_step / 3600)

        return result


    def _simulate_battery_response(
        self,
        battery: BatteryModel,
        signal: EPSSignal,
        step: int,
        scenario: str = 'normal',
        device_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.rng.random() < self.config.eps_packet_loss_rate:
            return {'responded': False, 'power_kw': 0, 'latency_ms': 0,
                    'reason': 'packet_loss'}

        if self.rng.random() < self.config.device_offline_rate:
            return {'responded': False, 'power_kw': 0, 'latency_ms': 0,
                    'reason': 'device_offline'}

        soc = battery.soc
        soc_observed = soc
        if self.config.soc_noise_std > 0:
            soc_observed = float(np.clip(
                soc + self.rng.normal(0, self.config.soc_noise_std), 0.0, 1.0))
        is_discharge = signal.supply_demand >= 8

        soc_reserve = getattr(battery.params, 'soc_reserve', 0.2)
        soc_max = getattr(battery.params, 'soc_max', 0.95)

        if is_discharge:
            qualified = soc_observed > soc_reserve
        else:
            qualified = soc_observed < soc_max

        if not qualified:
            return {'responded': False, 'power_kw': 0, 'latency_ms': 0,
                    'reason': 'eligibility_fail'}

        dt_hours = self.config.time_step / 3600.0
        e_available = battery.available_capacity_kwh

        if is_discharge:
            p_rated_eff = battery.get_max_discharge_power()
            e_headroom_kw = max(0, (soc_observed - soc_reserve) * e_available / dt_hours)
        else:
            p_rated_eff = battery.get_max_charge_power()
            e_headroom_kw = max(0, (soc_max - soc_observed) * e_available / dt_hours)

        c_avail = min(p_rated_eff, e_headroom_kw)

        if c_avail < 0.01:
            return {'responded': False, 'power_kw': 0, 'latency_ms': 0,
                    'reason': 'insufficient_capacity'}

        score = signal.intensity / 4095.0

        soc_range = soc_max - soc_reserve
        if is_discharge:
            w_soc = max(0, min(1, (soc_observed - soc_reserve) / max(soc_range, 0.1)))
        else:
            w_soc = max(0, min(1, (soc_max - soc_observed) / max(soc_range, 0.1)))

        prob = score * w_soc

        if self.config.sigmoid_response:
            k = self.config.sigmoid_steepness
            prob = float(1.0 / (1.0 + np.exp(-k * (prob - 0.5))))

        if self.config.response_prob_bias != 1.0:
            prob = float(np.clip(prob * self.config.response_prob_bias, 0.0, 1.0))

        if self.config.continuous_response:
            inverter_err = self.rng.normal(0, 0.025)
            voltage_err = self.rng.normal(0, 0.030)
            eta_delivery = 1.0 + inverter_err + voltage_err
            power_kw = prob * c_avail * eta_delivery * (-1 if is_discharge else 1)

            battery.update(power_kw=power_kw, dt_seconds=self.config.time_step)

            latency_ms = EPS_LATENCY.base_ms + self.rng.exponential(
                EPS_LATENCY.jitter_scale_ms)
            return {'responded': True, 'power_kw': power_kw, 'latency_ms': latency_ms}

        responded = self.rng.random() < prob

        if not responded:
            return {'responded': False, 'power_kw': 0, 'latency_ms': 0,
                    'reason': 'bernoulli_reject'}

        inverter_err = self.rng.normal(0, 0.025)
        voltage_err = self.rng.normal(0, 0.030)
        eta_delivery = 1.0 + inverter_err + voltage_err

        if is_discharge:
            power_kw = -c_avail * eta_delivery
        else:
            power_kw = c_avail * eta_delivery

        if self._region_perturbation is not None and device_id is not None:
            region = self._population.region_assignments.get(device_id)
            if region is not None:
                perturbation = self._region_perturbation.get(region, 0.0)
                power_kw *= (1.0 + perturbation * 0.15)

        battery.update(power_kw=power_kw, dt_seconds=self.config.time_step)

        latency_ms = EPS_LATENCY.base_ms + self.rng.exponential(
            EPS_LATENCY.jitter_scale_ms)

        return {'responded': True, 'power_kw': power_kw, 'latency_ms': latency_ms}


    def run_with_mask(
        self,
        signal_override: Dict[str, Any],
        device_mask: Dict[str, bool],
        num_steps: Optional[int] = None,
        scenario: str = 'normal',
    ) -> SimulationResult:
        if not self._is_initialized:
            self.initialize()

        if num_steps is None:
            num_steps = int(self.config.duration_seconds / self.config.time_step)

        self.set_override_signal(
            intensity=signal_override.get('intensity'),
            price_value=signal_override.get('price_value') or signal_override.get('price'),
            supply_demand=signal_override.get('supply_demand'),
        )

        result = SimulationResult()

        corr = self.config.correlation
        if corr.enable_correlation:
            persistent_global_shock = self.rng.normal(0, corr.global_shock_std)
            self._region_perturbation = {}
            for r in range(self.config.num_regions):
                self._region_perturbation[r] = (
                    persistent_global_shock
                    + self.rng.normal(0, corr.regional_shock_std)
                )
        else:
            self._region_perturbation = None

        for step in range(num_steps):
            ts = TimeStep(
                step_index=step,
                timestamp=datetime.now() + timedelta(seconds=step * self.config.time_step),
                elapsed_seconds=step * self.config.time_step,
            )

            signals = self._signal_generator.generate_scenario_signals(
                scenario, step, self.config.num_regions
            )

            if signals:
                ts.signal = signals[0]

            for region_id, signal in enumerate(signals):
                devices = self._population.get_devices_by_region(region_id)

                for idx, battery in devices['batteries']:
                    did = f'battery_{idx}'
                    if device_mask.get(did, False):
                        response = self._simulate_battery_response(
                            battery, signal, step, scenario, device_id=did)
                        if response['responded']:
                            ts.responding_devices += 1
                            ts.total_response_kw += response['power_kw']
                            ts.latency_samples.append(response['latency_ms'])
                            result.battery_response_kwh += abs(response['power_kw']) * (self.config.time_step / 3600)
                    else:
                        if battery._soc < 0.5 and self.rng.random() < 0.3:
                            charge_kw = battery.get_max_charge_power() * 0.3
                            battery.update(power_kw=charge_kw, dt_seconds=self.config.time_step)

            result.time_steps.append(ts)
            result.total_responses += ts.responding_devices

            energy_delta_kwh = ts.total_response_kw * (self.config.time_step / 3600)
            if energy_delta_kwh > 0:
                result.charge_energy_kwh += energy_delta_kwh
                result.load_increase_kwh += energy_delta_kwh
            else:
                result.discharge_energy_kwh += abs(energy_delta_kwh)
                result.load_reduction_kwh += abs(energy_delta_kwh)
            result.net_energy_kwh += energy_delta_kwh
            result.total_energy_kwh = result.net_energy_kwh

        self.clear_override_signal()

        n_masked = sum(1 for v in device_mask.values() if v)
        total_possible = n_masked * num_steps
        result.response_rate = result.total_responses / max(total_possible, 1)
        result.compute_latency_percentiles()

        return result

    @property
    def population(self) -> Optional[DevicePopulation]:
        return self._population

    def reset(self) -> None:
        self._population = None
        self._signal_generator = None
        self._step_metrics = []
        self._region_perturbation = None
        self._is_initialized = False

    def get_population_summary(self) -> Dict[str, Any]:
        if not self._population:
            return {}

        return {
            'total_devices': self._population.total_count,
            'batteries': len(self._population.batteries),
            'regions': self.config.num_regions,
        }
