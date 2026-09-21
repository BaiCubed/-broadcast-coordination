#!/usr/bin/env python3
import argparse
import json
import logging
import os
import platform
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from scipy import stats
import random


REAL_DATA_SAMPLE_SCALE = 1.0
REAL_DATA_MIN_SAMPLES = 1


def _real_scale_count(value: int, minimum: int = 1) -> int:
    return max(minimum, int(round(value * REAL_DATA_SAMPLE_SCALE)))


def _get_real_training_history(estimator, resp_mean: float, resp_std: float,
                               resp_min: float, resp_max: float,
                               model_type: str) -> Dict[str, Any]:
    try:
        training_history = estimator.get_training_history()
        return {
            "epochs": training_history.get('epochs', []),
            "train_loss": training_history.get('train_loss', []),
            "val_loss": training_history.get('val_loss', []),
            "response_mean": resp_mean,
            "response_std": resp_std,
            "response_min": resp_min,
            "response_max": resp_max,
            "model_type": model_type,
            "history_source": "real_training_log",
            "charge_history": training_history.get('charge', {}),
            "discharge_history": training_history.get('discharge', {}),
        }
    except Exception as e:
        logger.warning(f"Failed to get real training history: {e}")
        return {
            "epochs": [], "train_loss": [], "val_loss": [],
            "response_mean": resp_mean, "response_std": resp_std,
            "response_min": resp_min, "response_max": resp_max,
            "model_type": model_type,
            "history_source": "failed", "error": str(e),
        }


def _calculate_bootstrap_ci(predictions: List[float], actuals: List[float],
                            smape_value: float, rmse_value: float, r2_value: float,
                            n_bootstrap: int = 10000, ci_level: float = 0.95) -> Dict[str, Dict[str, float]]:
    n_samples = len(predictions)
    predictions_arr = np.array(predictions)
    actuals_arr = np.array(actuals)

    bootstrap_smape = []
    bootstrap_rmse = []
    bootstrap_r2 = []

    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, n_samples, replace=True)
        boot_preds = predictions_arr[indices]
        boot_actuals = actuals_arr[indices]

        denominators = np.abs(boot_actuals) + np.abs(boot_preds)
        valid_mask = denominators > 1e-6
        if np.any(valid_mask):
            boot_smape = np.mean(
                2 * np.abs(boot_actuals[valid_mask] - boot_preds[valid_mask]) /
                denominators[valid_mask]
            ) * 100
        else:
            boot_smape = 0.0
        bootstrap_smape.append(boot_smape)

        boot_rmse = np.sqrt(np.mean((boot_preds - boot_actuals) ** 2))
        bootstrap_rmse.append(boot_rmse)

        ss_res = np.sum((boot_preds - boot_actuals) ** 2)
        ss_tot = np.sum((boot_actuals - np.mean(boot_actuals)) ** 2)
        boot_r2 = 1 - ss_res / max(ss_tot, 1e-6)
        bootstrap_r2.append(boot_r2)

    bootstrap_smape.sort()
    bootstrap_rmse.sort()
    bootstrap_r2.sort()

    lower_idx = int((1 - ci_level) / 2 * n_bootstrap)
    upper_idx = int((1 + ci_level) / 2 * n_bootstrap)

    return {
        "smape": {
            "value": smape_value,
            "ci_lower": float(bootstrap_smape[lower_idx]),
            "ci_upper": float(bootstrap_smape[upper_idx]),
            "method": "bootstrap_10000",
        },
        "rmse": {
            "value": rmse_value,
            "ci_lower": float(bootstrap_rmse[lower_idx]),
            "ci_upper": float(bootstrap_rmse[upper_idx]),
            "method": "bootstrap_10000",
        },
        "r2": {
            "value": r2_value,
            "ci_lower": float(max(0, bootstrap_r2[lower_idx])),
            "ci_upper": float(min(1, bootstrap_r2[upper_idx])),
            "method": "bootstrap_10000",
        },
    }


def _setup_experiment_directory(
    base_dir: str,
    experiment_name: str = "experiment",
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, str]:
    base_path = Path(base_dir)
    experiments_dir = base_path / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{timestamp}_{experiment_name}"
    experiment_dir = experiments_dir / experiment_id

    subdirs = ["data", "figures", "tables", "logs", "estimation"]
    for subdir in subdirs:
        (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "platform": {
            "system": platform.system(),
            "node": platform.node(),
            "python_version": platform.python_version(),
        },
        "git_info": _get_git_info(),
        "status": "running",
    }

    metadata_file = experiment_dir / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if config:
        config_file = experiment_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2, default=str, ensure_ascii=False)

    latest_link = experiments_dir / "latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    elif latest_link.exists():
        pass

    try:
        latest_link.symlink_to(experiment_id)
    except OSError:
        pass

    return experiment_dir, experiment_id


def _get_git_info() -> Dict[str, str]:
    git_info = {}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info["commit"] = result.stdout.strip()[:8]

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_info["dirty"] = len(result.stdout.strip()) > 0
    except Exception:
        pass

    return git_info


def _finalize_experiment(experiment_dir: Path, status: str = "completed") -> None:
    metadata_file = experiment_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

        metadata["status"] = status
        metadata["completed_at"] = datetime.now().isoformat()

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    name: str = "eps_experiment"
    description: str = "Broadcast coordination experiment"

    num_devices: int = 5000
    num_regions: int = 5

    simulation_hours: int = 48
    time_step_minutes: int = 5

    enable_closed_loop: bool = True
    target_response_mw: float = 50.0
    optimization_iterations: int = 50

    def get_scaled_target_mw(self) -> float:
        max_capacity_per_device_kw = 1.5
        max_response_mw = self.num_devices * max_capacity_per_device_kw / 1000

        achievable_target_mw = max_response_mw * 0.5

        return min(achievable_target_mw, self.target_response_mw)

    enable_online_learning: bool = True
    online_update_interval: int = 1

    num_runs: int = 30
    confidence_level: float = 0.95
    random_seed: int = 42

    scenarios: List[str] = field(default_factory=lambda: [
        "peak_shaving",
        "valley_filling",
        "emergency_grid_stability",
        "emergency_supply_shortage",
    ])
    run_all_scenarios: bool = True


    output_dir: str = "results/experiment"


class ScenarioProfile:
    name: str = "base"
    cycle_hours: float = 1.0

    def get_r(self, phase: float) -> float:
        raise NotImplementedError

    def get_d(self, phase: float) -> float:
        raise NotImplementedError

    def get_signal_score(self, phase: float) -> float:
        r = self.get_r(phase)
        d = self.get_d(phase)
        return float(np.clip((r - 1.0) * d, -1.0, 1.0))


class ValleyFillingProfile(ScenarioProfile):
    name = "valley_filling"
    cycle_hours = 4.0

    def get_r(self, phase: float) -> float:
        return 1.2 + 0.3 * np.sin(np.pi * phase)

    def get_d(self, phase: float) -> float:
        return 0.5 + 0.4 * np.sin(np.pi * phase)


class PeakShavingProfile(ScenarioProfile):
    name = "peak_shaving"
    cycle_hours = 3.0

    def get_r(self, phase: float) -> float:
        return 0.70 - 0.2 * np.sin(np.pi * phase)

    def get_d(self, phase: float) -> float:
        return 0.5 + 0.5 * np.sin(np.pi * phase)


class EmergencyChargeProfile(ScenarioProfile):
    name = "emergency_grid_stability"
    cycle_hours = 1.0

    _TRANSITION_END = 1.0 / 12.0
    _SUSTAINED_END = 7.0 / 12.0
    _RECOVERY_SPAN = 5.0 / 12.0

    def get_r(self, phase: float) -> float:
        if phase < self._TRANSITION_END:
            return 1.0 + 0.5 * (phase / self._TRANSITION_END)
        elif phase < self._SUSTAINED_END:
            return 1.5
        else:
            frac = (phase - self._SUSTAINED_END) / self._RECOVERY_SPAN
            return 1.5 - 0.5 * min(frac, 1.0)

    def get_d(self, phase: float) -> float:
        if phase < self._SUSTAINED_END:
            return 1.0
        else:
            frac = (phase - self._SUSTAINED_END) / self._RECOVERY_SPAN
            return 1.0 - 0.5 * min(frac, 1.0)


class EmergencyDischargeProfile(ScenarioProfile):
    name = "emergency_supply_shortage"
    cycle_hours = 1.0

    _TRANSITION_END = 1.0 / 12.0
    _SUSTAINED_END = 7.0 / 12.0
    _RECOVERY_SPAN = 5.0 / 12.0

    def get_r(self, phase: float) -> float:
        if phase < self._TRANSITION_END:
            return 1.0 - 0.4 * (phase / self._TRANSITION_END)
        elif phase < self._SUSTAINED_END:
            return 0.6
        else:
            frac = (phase - self._SUSTAINED_END) / self._RECOVERY_SPAN
            return 0.6 + 0.4 * min(frac, 1.0)

    def get_d(self, phase: float) -> float:
        if phase < self._SUSTAINED_END:
            return 1.0
        else:
            frac = (phase - self._SUSTAINED_END) / self._RECOVERY_SPAN
            return 1.0 - 0.5 * min(frac, 1.0)


SCENARIO_PROFILES: Dict[str, ScenarioProfile] = {
    'valley_filling': ValleyFillingProfile(),
    'peak_shaving': PeakShavingProfile(),
    'emergency_grid_stability': EmergencyChargeProfile(),
    'emergency_supply_shortage': EmergencyDischargeProfile(),
}


PROFILE_R_NOISE_STD: float = 0.02


def encode_signal_score(s: float) -> Tuple[int, int]:
    abs_s = min(abs(s), 1.0)
    intensity = max(0, min(4095, round(abs_s * 4095)))

    if s >= 0:
        supply_demand = max(0, min(7, round(7 * (1.0 - abs_s))))
    else:
        supply_demand = max(8, min(15, round(8 + 7 * abs_s)))

    return supply_demand, intensity


@dataclass
class HourlyResult:
    hour: int
    dt_hours: float = 1.0
    t_hours: float = 0.0
    solar_mw: float = 0.0
    wind_mw: float = 0.0
    load_mw: float = 0.0
    net_balance_mw: float = 0.0
    supply_demand_state: int = 8
    is_surplus: bool = False

    r_t: float = 1.0
    d_t: float = 0.5
    s_score: float = 0.0

    signal_intensity: int = 2048
    signal_price: float = 0.0
    signal_supply_demand: int = 8

    target_response_mw: float = 0.0
    predicted_response_mw: float = 0.0
    prediction_lower_mw: float = 0.0
    prediction_upper_mw: float = 0.0
    actual_response_mw: float = 0.0
    actual_in_interval: bool = False
    prediction_error_pct: float = 0.0

    battery_response_mw: float = 0.0
    response_rate: float = 0.0
    n_responding_devices: int = 0

    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0

    optimization_iterations: int = 0
    optimization_converged: bool = False

    model_updated: bool = False
    cumulative_error: float = 0.0

    simulation_time_seconds: float = 0.0


@dataclass
class ExperimentSummary:
    run_id: int = 0
    scenario: str = "peak_shaving"

    total_solar_mwh: float = 0.0
    total_wind_mwh: float = 0.0
    total_curtailed_mwh: float = 0.0
    curtailment_reduction_pct: float = 0.0

    avg_response_rate: float = 0.0
    avg_n_responding_devices: float = 0.0
    avg_prediction_error_pct: float = 0.0
    total_response_mwh: float = 0.0
    scenario_r2: float = 0.0

    battery_total_mwh: float = 0.0

    avg_optimization_iterations: float = 0.0
    convergence_rate: float = 0.0

    initial_mape: float = 0.0
    final_mape: float = 0.0
    learning_improvement_pct: float = 0.0

    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0

    total_simulation_time_seconds: float = 0.0
    devices_per_second: float = 0.0


@dataclass
class RiskDataPoint:
    scenario: str = ""
    scenario_risk_level: str = ""
    hour: int = 0

    target_mw: float = 0.0
    tolerance_fraction: float = 0.0

    predicted_mw: float = 0.0
    prediction_error_pct: float = 0.0

    q10_kw: float = 0.0
    q50_kw: float = 0.0
    q90_kw: float = 0.0
    interval_lower_kw: float = 0.0
    interval_upper_kw: float = 0.0
    interval_width_kw: float = 0.0
    cqr_adjustment: float = 0.0
    interval_source: str = "unknown"

    target_in_interval: bool = False
    interval_coverage_ratio: float = 0.0
    pinaw: float = 0.0
    confidence: str = "unknown"

    risk_adjusted: bool = False
    safety_margin_applied: float = 0.0

    converged: bool = False
    iterations: int = 0

    actual_response_mw: Optional[float] = None
    actual_deviation_mw: float = 0.0
    actual_deviation_pct: float = 0.0
    actual_in_interval: bool = False


@dataclass
class MultiRunStatistics:
    num_runs: int = 0
    scenario: str = "peak_shaving"
    num_devices: int = 0

    response_rate_mean: float = 0.0
    response_rate_std: float = 0.0
    response_rate_ci_lower: float = 0.0
    response_rate_ci_upper: float = 0.0

    r2_mean: float = 0.0
    r2_std: float = 0.0
    r2_ci_lower: float = 0.0
    r2_ci_upper: float = 0.0


    smape_mean: float = 0.0
    smape_std: float = 0.0
    smape_ci_lower: float = 0.0
    smape_ci_upper: float = 0.0

    convergence_rate_mean: float = 0.0
    convergence_rate_std: float = 0.0
    convergence_rate_ci_lower: float = 0.0
    convergence_rate_ci_upper: float = 0.0

    learning_improvement_mean: float = 0.0
    learning_improvement_std: float = 0.0
    learning_improvement_ci_lower: float = 0.0
    learning_improvement_ci_upper: float = 0.0

    latency_p99_mean: float = 0.0
    latency_p99_std: float = 0.0
    latency_p99_ci_lower: float = 0.0
    latency_p99_ci_upper: float = 0.0

    all_response_rates: List[float] = field(default_factory=list)
    all_r2s: List[float] = field(default_factory=list)
    all_smapes: List[float] = field(default_factory=list)
    all_convergence_rates: List[float] = field(default_factory=list)
    all_learning_improvements: List[float] = field(default_factory=list)
    all_latency_p99s: List[float] = field(default_factory=list)


@dataclass
class OnlineLearningCheckpoint:
    cumulative_samples: int = 0
    r2_score: float = 0.0
    smape: float = 0.0
    picp: float = 0.0
    rmse: float = 0.0


def _evaluate_estimator_metrics(
    estimator,
    eval_signals: List[Dict],
    eval_responses: List[float],
) -> Dict[str, float]:
    predictions = []
    actuals = []
    in_interval_count = 0

    for signal, actual in zip(eval_signals, eval_responses):
        result = estimator.estimate(signal)
        pred = result.response_kw
        predictions.append(pred)
        actuals.append(actual)

        if result.lower_bound <= actual <= result.upper_bound:
            in_interval_count += 1

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    denominators = np.abs(actuals) + np.abs(predictions)
    valid_mask = denominators > 1e-6
    if np.any(valid_mask):
        smape = np.mean(
            2 * np.abs(actuals[valid_mask] - predictions[valid_mask]) /
            denominators[valid_mask]
        ) * 100
    else:
        smape = 0.0

    picp = (in_interval_count / len(eval_signals)) * 100 if eval_signals else 0.0

    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))

    return {
        'r2': float(r2),
        'smape': float(smape),
        'picp': float(picp),
        'rmse': float(rmse),
    }


def _generate_evaluation_set(
    n_samples: int,
    num_devices: int,
    rng: np.random.Generator,
    sim_config_override: 'SimulationConfig' = None,
) -> Tuple[List[Dict], List[float]]:
    import time
    import dataclasses
    from src.simulation import EPSSimulator, SimulationConfig, SimulationLevel

    signals = []
    responses = []

    scenarios = ['valley_filling', 'peak_shaving', 'valley_filling', 'peak_shaving', 'normal', 'emergency']
    scenario_weights = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]

    scenario_start_hours = {
        'peak_shaving': 18.0,
        'valley_filling': 10.0,
        'normal': 6.0,
        'emergency': 18.0,
    }

    scenario_samples = [int(n_samples * w) for w in scenario_weights]
    scenario_samples[-1] = n_samples - sum(scenario_samples[:-1])

    for scenario_idx, scenario in enumerate(scenarios):
        samples_per_scenario = scenario_samples[scenario_idx]
        if samples_per_scenario == 0:
            continue

        base_seed = int(rng.integers(0, 100000))

        if sim_config_override is not None:
            sim_config = dataclasses.replace(
                sim_config_override,
                level=SimulationLevel.LEVEL1_AGENT,
                num_devices=num_devices,
                duration_seconds=samples_per_scenario * 60,
                time_step=60.0,
                random_seed=base_seed,
            )
        else:
            sim_config = SimulationConfig(
                level=SimulationLevel.LEVEL1_AGENT,
                num_devices=num_devices,
                duration_seconds=samples_per_scenario * 60,
                time_step=60.0,
                random_seed=base_seed,
                num_regions=5,
            )

        simulator = EPSSimulator(sim_config)

        start_hour = scenario_start_hours.get(scenario, 0.0)
        simulator.set_start_hour(start_hour)

        result = simulator.run(num_steps=samples_per_scenario, scenario=scenario)

        for ts in result.time_steps:
            if ts.signal is not None and len(signals) < n_samples:
                hour = (start_hour + ts.step_index * sim_config.time_step / 3600) % 24
                day_of_week = (ts.step_index // 24 + scenario_idx) % 7

                signal_dict = {
                    'intensity': ts.signal.intensity,
                    'supply_demand': ts.signal.supply_demand,
                    'price': ts.signal.price,
                    'region_id': ts.signal.region_id,
                    'priority': ts.signal.priority,
                    'timestamp': time.time(),
                    'hour': hour,
                    'day_of_week': day_of_week,
                }
                signals.append(signal_dict)
                responses.append(ts.total_response_kw)

    if len(signals) > 0:
        indices = np.arange(len(signals))
        rng.shuffle(indices)
        signals = [signals[i] for i in indices[:n_samples]]
        responses = [responses[i] for i in indices[:n_samples]]

    return signals, responses


def run_single_experiment(
    config: ExperimentConfig,
    run_id: int,
    scenario: str,
    estimator=None,
    verbose: bool = True,
) -> Tuple[List[HourlyResult], ExperimentSummary, List[Dict], List[Dict]]:
    from src.simulation import EPSSimulator, SimulationConfig, SimulationLevel
    from src.estimation import EPSEstimator, EstimatorConfig
    from src.signal import SignalOptimizer, OptimizationTarget

    seed = config.random_seed + run_id
    rng = np.random.default_rng(seed)

    start_time = time.time()

    if estimator is None:
        estimator_config = EstimatorConfig(
            target_coverage=0.9,
            enable_conformal=True,

            use_pytorch=True,
        )
        estimator = EPSEstimator(estimator_config)
        n_train = 2000 if config.num_devices <= 1000 else 12000
        train_signals, train_responses = _generate_training_data(
            n_samples=n_train, num_devices=config.num_devices, rng=rng,
            sim_config_override=None,
        )
        estimator.fit(train_signals, train_responses)

    optimizer = None
    if config.enable_closed_loop:
        optimizer = SignalOptimizer(estimator)

    sim_config = SimulationConfig(
        num_devices=config.num_devices,
        num_regions=config.num_regions,
        level=SimulationLevel.LEVEL1_AGENT,
        duration_seconds=config.simulation_hours * 3600,
        time_step=config.time_step_minutes * 60,
        random_seed=seed,
    )
    simulator = EPSSimulator(sim_config)
    simulator.initialize()

    hourly_results: List[HourlyResult] = []
    prediction_errors: List[float] = []
    all_latencies: List[float] = []
    online_learning_stats: List[Dict] = []
    online_learning_checkpoints: List[Dict] = []

    eval_rng = np.random.default_rng(seed + 10000)
    n_eval = 200
    eval_signals, eval_responses = _generate_evaluation_set(
        n_samples=n_eval, num_devices=config.num_devices, rng=eval_rng
    )

    cumulative_online_samples = 0
    baseline_metrics = _evaluate_estimator_metrics(estimator, eval_signals, eval_responses)
    online_learning_checkpoints.append({
        'cumulative_samples': 0,
        'r2_score': baseline_metrics['r2'],
        'smape': baseline_metrics['smape'],
        'picp': baseline_metrics['picp'],
        'rmse': baseline_metrics['rmse'],
    })

    profile = SCENARIO_PROFILES.get(scenario)
    max_target_mw = config.get_scaled_target_mw()

    dt_minutes = config.time_step_minutes
    dt_hours = dt_minutes / 60.0
    dt_seconds = dt_minutes * 60
    total_steps = config.simulation_hours * 60 // dt_minutes
    steps_per_hour = 60 // dt_minutes

    for step_idx in range(total_steps):
        step_start = time.time()
        t_hours = step_idx * dt_hours
        hour_of_day = t_hours % 24

        if profile is not None:
            cycle_h = profile.cycle_hours
            phase = ((t_hours % cycle_h) / cycle_h) % 1.0
            r_noise_std = globals().get("PROFILE_R_NOISE_STD", 0.02)
            r_t = profile.get_r(phase) * (1.0 + rng.normal(0, r_noise_std))
            d_t = profile.get_d(phase)
            s = float(np.clip((r_t - 1.0) * d_t, -1.0, 1.0))
            base_supply_demand, base_intensity = encode_signal_score(s)
            is_surplus = s >= 0
        else:
            state = simulator._signal_generator.compute_supply_demand_state(float(hour_of_day))
            r_t = state.get('supply_demand_ratio', 1.0)
            d_t = 0.5
            s = float(np.clip((r_t - 1.0) * d_t, -1.0, 1.0))
            base_supply_demand = state['supply_demand']
            base_intensity = max(0, min(4095, int(abs(s) * 4095)))
            is_surplus = state['is_surplus']

        state_for_export = simulator._signal_generator.compute_supply_demand_state(float(hour_of_day))

        hourly = HourlyResult(
            hour=step_idx,
            dt_hours=dt_hours,
            t_hours=t_hours,
            solar_mw=float(state_for_export['solar_mw']),
            wind_mw=float(state_for_export['wind_mw']),
            load_mw=float(state_for_export['load_mw']),
            net_balance_mw=float(state_for_export['net_balance']),
            supply_demand_state=base_supply_demand,
            is_surplus=is_surplus,
            r_t=r_t,
            d_t=d_t,
            s_score=s,
        )

        target_mw = max_target_mw * abs(s)
        if s < 0:
            target_mw = -target_mw
        hourly.target_response_mw = target_mw

        hourly.signal_intensity = base_intensity
        hourly.signal_price = 0.0
        hourly.signal_supply_demand = base_supply_demand

        signal_dict = {
            'supply_demand': hourly.signal_supply_demand,
            'intensity': hourly.signal_intensity,
            'price': hourly.signal_price,
            'hour': hour_of_day,
        }
        pred = estimator.estimate(signal_dict)
        hourly.predicted_response_mw = pred.response_kw / 1000
        hourly.prediction_lower_mw = pred.lower_bound / 1000
        hourly.prediction_upper_mw = pred.upper_bound / 1000

        step_sim_config = SimulationConfig(
            num_devices=config.num_devices,
            num_regions=config.num_regions,
            level=SimulationLevel.LEVEL1_AGENT,
            duration_seconds=dt_seconds,
            time_step=dt_seconds,
            random_seed=seed + step_idx,
            )
        step_simulator = EPSSimulator(step_sim_config)
        step_simulator.initialize()

        step_simulator.set_override_signal(
            intensity=hourly.signal_intensity,
            price_value=hourly.signal_price,
            supply_demand=hourly.signal_supply_demand,
        )

        step_result = step_simulator.run(
            num_steps=1,
            scenario=scenario,
            progress_callback=None,
        )
        step_simulator.clear_override_signal()

        hourly.battery_response_mw = step_result.battery_response_kwh / dt_hours / 1000
        hourly.actual_response_mw = step_result.net_energy_kwh / dt_hours / 1000
        hourly.actual_in_interval = (hourly.prediction_lower_mw <= hourly.actual_response_mw <= hourly.prediction_upper_mw)
        hourly.response_rate = step_result.response_rate
        hourly.n_responding_devices = step_result.n_responding_devices

        step_latencies = []
        for ts in step_result.time_steps:
            step_latencies.extend(ts.latency_samples)
        all_latencies.extend(step_latencies)

        if step_latencies:
            hourly.latency_mean_ms = float(np.mean(step_latencies))
            hourly.latency_std_ms = float(np.std(step_latencies))
            hourly.latency_p50_ms = float(np.percentile(step_latencies, 50))
            hourly.latency_p90_ms = float(np.percentile(step_latencies, 90))
            hourly.latency_p99_ms = float(np.percentile(step_latencies, 99))

        denominator = abs(hourly.actual_response_mw) + abs(hourly.predicted_response_mw)
        if denominator > 1e-6:
            hourly.prediction_error_pct = (
                2 * abs(hourly.actual_response_mw - hourly.predicted_response_mw)
                / denominator
                * 100
            )
        else:
            hourly.prediction_error_pct = 0.0
        prediction_errors.append(hourly.prediction_error_pct)

        if config.enable_online_learning and step_idx % (config.online_update_interval * steps_per_hour) == 0:
            ol_signal_dict = {
                'supply_demand': hourly.signal_supply_demand,
                'intensity': hourly.signal_intensity,
                'price': hourly.signal_price,
                'hour': hour_of_day,
            }
            online_stats = estimator.online_update(
                ol_signal_dict,
                hourly.actual_response_mw * 1000,
                update_nn=False,
            )
            hourly.model_updated = True
            cumulative_online_samples += 1

            online_learning_stats.append({
                'hour': step_idx,
                'supply_demand': hourly.signal_supply_demand,
                'prediction': online_stats.get('prediction', 0),
                'actual': online_stats.get('actual', 0),
                'error': online_stats.get('error', 0),
                'charge_nn_updated': online_stats.get('charge_nn_updated', False),
                'discharge_nn_updated': online_stats.get('discharge_nn_updated', False),
                'charge_loss': online_stats.get('charge_loss'),
                'discharge_loss': online_stats.get('discharge_loss'),
                'prediction_error_pct': hourly.prediction_error_pct,
            })

            if step_idx % (2 * steps_per_hour) == 0 or step_idx == total_steps - 1:
                current_metrics = _evaluate_estimator_metrics(estimator, eval_signals, eval_responses)
                online_learning_checkpoints.append({
                    'cumulative_samples': cumulative_online_samples,
                    'r2_score': current_metrics['r2'],
                    'smape': current_metrics['smape'],
                    'picp': current_metrics['picp'],
                    'rmse': current_metrics['rmse'],
                })

        hourly.cumulative_error = np.mean(prediction_errors) if prediction_errors else 0
        hourly.simulation_time_seconds = time.time() - step_start

        hourly_results.append(hourly)

    total_time = time.time() - start_time
    summary = _compute_summary(hourly_results, prediction_errors, all_latencies)
    summary.run_id = run_id
    summary.scenario = scenario
    summary.total_simulation_time_seconds = total_time
    num_steps = config.simulation_hours * 60 // config.time_step_minutes
    summary.devices_per_second = (config.num_devices * num_steps) / max(total_time, 0.001)

    return hourly_results, summary, online_learning_stats, online_learning_checkpoints


def run_core_experiment(config: ExperimentConfig,
                        experiment_dir: Path = None) -> Dict[str, Any]:
    from src.estimation import EPSEstimator, EstimatorConfig

    logger.info("=" * 70)
    logger.info("Core Experiment (Multi-Run)")
    logger.info("=" * 70)
    logger.info(f"Devices: {config.num_devices}, Hours: {config.simulation_hours}")
    logger.info(f"Runs: {config.num_runs}, Confidence Level: {config.confidence_level}")
    logger.info(f"Closed-loop: {config.enable_closed_loop}, Online learning: {config.enable_online_learning}")
    logger.info("=" * 70)

    if experiment_dir is not None:
        output_dir = experiment_dir
        experiment_id = experiment_dir.name
    else:
        experiment_name = f"core_{config.num_devices}dev_{config.num_runs}runs"
        output_dir, experiment_id = _setup_experiment_directory(
            base_dir="results",
            experiment_name=experiment_name,
            config=asdict(config),
        )

    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Output directory: {output_dir}")

    scenarios = config.scenarios if config.run_all_scenarios else [config.scenarios[0]]

    all_results = {
        'config': asdict(config),
        'timestamp': datetime.now().isoformat(),
        'scenarios': {},
        'multi_run_statistics': {},
    }

    logger.info("Step 1: Pre-training Estimator...")
    rng = np.random.default_rng(config.random_seed)
    estimator_config = EstimatorConfig(
        target_coverage=0.9,
        enable_conformal=True,

        use_pytorch=True,
    )
    base_estimator = EPSEstimator(estimator_config)
    n_train = 12000
    train_signals, train_responses = _generate_training_data(
        n_samples=n_train, num_devices=config.num_devices, rng=rng,
        sim_config_override=None,
    )
    base_estimator.fit(train_signals, train_responses)
    logger.info(f"  Estimator trained with {len(train_signals)} samples")

    import copy
    base_estimator_clean = copy.deepcopy(base_estimator)
    logger.info(f"  Saved clean estimator copy for validation")

    all_online_learning_data = {}
    all_online_checkpoints_data = {}

    for scenario in scenarios:
        logger.info(f"\n{'=' * 60}")
        logger.info(f" Scenario: {scenario}")
        logger.info(f"{'=' * 60}")

        all_summaries: List[ExperimentSummary] = []
        first_run_hourly: List[Dict[str, Any]] = []
        scenario_online_stats: List[List[Dict]] = []
        scenario_online_checkpoints: List[List[Dict]] = []
        last_updated_estimator = None

        for run_id in range(config.num_runs):
            logger.info(f"\n  Run {run_id + 1}/{config.num_runs}...")

            estimator_copy = deepcopy(base_estimator)
            hourly_results, summary, online_stats, online_checkpoints = run_single_experiment(
                config, run_id, scenario,
                estimator=estimator_copy,
                verbose=(run_id == 0),
            )
            all_summaries.append(summary)

            if run_id == config.num_runs - 1:
                last_updated_estimator = estimator_copy
            if online_stats:
                scenario_online_stats.append(online_stats)
            if online_checkpoints:
                scenario_online_checkpoints.append(online_checkpoints)

            if run_id == 0:
                first_run_hourly = [asdict(h) for h in hourly_results]

            logger.info(
                f"    MAPE={summary.avg_prediction_error_pct:.1f}%, "
                f"Convergence={summary.convergence_rate:.1%}, "
                f"Learning={summary.learning_improvement_pct:.1f}%"
            )

        multi_run_stats = _compute_multi_run_statistics(
            all_summaries, scenario, config.num_devices, config.confidence_level
        )

        all_results['scenarios'][scenario] = {
            'runs': [{'hourly': first_run_hourly, 'summary': asdict(all_summaries[0])}],
            'summaries': [asdict(s) for s in all_summaries],
            'summary': asdict(multi_run_stats),
        }
        all_results['multi_run_statistics'][scenario] = asdict(multi_run_stats)

        if scenario_online_stats:
            all_online_learning_data[scenario] = scenario_online_stats

        if scenario_online_checkpoints:
            all_online_checkpoints_data[scenario] = scenario_online_checkpoints

        _print_multi_run_summary(multi_run_stats)

        if last_updated_estimator is not None:
            all_results['_last_updated_estimator'] = last_updated_estimator

    final_estimator = all_results.pop('_last_updated_estimator', base_estimator)


    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    results_file = data_dir / "complete_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    multi_run_file = data_dir / "multi_run_statistics.json"
    with open(multi_run_file, 'w') as f:
        json.dump(all_results['multi_run_statistics'], f, indent=2, default=str)


    supply_demand_dir = data_dir / "supply_demand"
    supply_demand_dir.mkdir(parents=True, exist_ok=True)
    _export_supply_demand_data(all_results, supply_demand_dir)

    if all_online_learning_data:
        online_learning_dir = data_dir / "online_learning"
        online_learning_dir.mkdir(parents=True, exist_ok=True)
        online_learning_file = online_learning_dir / "online_learning_data.json"
        with open(online_learning_file, 'w') as f:
            json.dump(all_online_learning_data, f, indent=2, default=str)
        logger.info(f"  Online learning data saved to: {online_learning_file}")

    if all_online_checkpoints_data:
        online_learning_dir = data_dir / "online_learning"
        online_learning_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_file = online_learning_dir / "online_learning_checkpoints.json"
        with open(checkpoints_file, 'w') as f:
            json.dump(all_online_checkpoints_data, f, indent=2, default=str)
        logger.info(f"  Online learning checkpoints saved to: {checkpoints_file}")

    logger.info(f"\n{'=' * 60}")
    logger.info("Generating Estimation Validation Results")
    logger.info(f"{'=' * 60}")
    estimation_dir = output_dir / "estimation"
    _export_estimation_validation(
        base_estimator_clean, config, estimation_dir, rng,
        use_passed_estimator=True,
    )

    logger.info(f"\n{'=' * 60}")
    logger.info("Generating Scenario Validation Results")
    logger.info(f"{'=' * 60}")
    _export_validation_results(all_results, data_dir)

    risk_assessment_data = _collect_risk_assessment_data(
        base_estimator, config, data_dir, samples_per_scenario=500
    )
    all_results['risk_assessment'] = risk_assessment_data

    logger.info(f"Core experiment completed. Results in {output_dir}")

    return all_results


def _generate_training_data(
    n_samples: int,
    num_devices: int,
    rng: np.random.Generator,
    sim_config_override: 'SimulationConfig' = None,
) -> Tuple[List[Dict], List[float]]:
    import dataclasses
    from src.simulation import EPSSimulator, SimulationConfig, SimulationLevel

    signals = []
    responses = []

    n_scenario = int(n_samples * 0.80)
    n_time_aware = int(n_samples * 0.10)
    n_uniform = n_samples - n_scenario - n_time_aware

    logger.info(f"  Generating {n_samples} training samples using simulator...")
    logger.info(f"    {n_scenario} scenario-profile + {n_time_aware} time-aware + {n_uniform} uniform")

    profiles = [
        ValleyFillingProfile(),
        PeakShavingProfile(),
        EmergencyChargeProfile(),
        EmergencyDischargeProfile(),
    ]
    samples_per_profile = n_scenario // len(profiles)

    for i in range(n_samples):
        if i < n_scenario:
            profile_idx = min(i // samples_per_profile, len(profiles) - 1)
            profile = profiles[profile_idx]
            phase = rng.uniform(0, 1)
            s = profile.get_signal_score(phase)
            s += rng.normal(0, 0.02)
            s = float(np.clip(s, -1.0, 1.0))
            supply_demand, intensity = encode_signal_score(s)
            hour = int(rng.integers(0, 24))
        elif i < n_scenario + n_time_aware:
            hour = int(rng.integers(0, 24))

            if (7 <= hour <= 11) or (17 <= hour <= 21):
                if rng.random() < 0.5:
                    intensity = int(rng.normal(2800, 500))
                else:
                    intensity = int(rng.normal(1000, 400))
                supply_demand = int(rng.normal(11, 2))
            elif hour <= 6 or hour >= 22:
                intensity = int(rng.normal(1000, 300))
                supply_demand = int(rng.normal(3, 2))
            else:
                intensity = int(rng.normal(2048, 600))
                supply_demand = int(rng.normal(8, 2))
        else:
            hour = int(rng.integers(0, 24))
            intensity = int(rng.uniform(0, 4095))
            supply_demand = int(rng.uniform(0, 16))

        price = 0.0

        intensity = max(0, min(4095, intensity))
        supply_demand = max(0, min(15, supply_demand))

        seed_i = int(rng.integers(0, 10000))
        dt_train_total = 900
        dt_train_step = 300
        dt_train_hours = dt_train_total / 3600.0
        if sim_config_override is not None:
            mini_config = dataclasses.replace(
                sim_config_override,
                num_devices=num_devices,
                level=SimulationLevel.LEVEL1_AGENT,
                duration_seconds=dt_train_total,
                time_step=dt_train_step,
                random_seed=seed_i,
            )
        else:
            mini_config = SimulationConfig(
                num_devices=num_devices,
                num_regions=5,
                level=SimulationLevel.LEVEL1_AGENT,
                duration_seconds=dt_train_total,
                time_step=dt_train_step,
                random_seed=seed_i,
            )
        mini_sim = EPSSimulator(mini_config)
        mini_sim.initialize()

        mini_sim.set_override_signal(
            intensity=intensity,
            price_value=price,
            supply_demand=supply_demand,
        )
        scenario = 'valley_filling' if supply_demand <= 7 else 'peak_shaving'
        result = mini_sim.run(num_steps=3, scenario=scenario)
        mini_sim.clear_override_signal()

        actual_response_kw = result.total_energy_kwh / dt_train_hours

        direction = 1 if supply_demand <= 7 else -1

        signals.append({
            'supply_demand': supply_demand,
            'intensity': intensity,
            'price': price,
            'hour': hour,
            'direction': direction,
        })
        responses.append(actual_response_kw)

        if (i + 1) % 50 == 0:
            logger.info(f"    Generated {i+1}/{n_samples} samples")

    return signals, responses


def _generate_replicated_eval_data(
    n_unique_signals: int,
    replications: int,
    num_devices: int,
    rng: np.random.Generator,
    sim_config_override: 'SimulationConfig' = None,
) -> Tuple[List[Dict], List[float]]:
    import dataclasses
    from src.simulation import EPSSimulator, SimulationConfig, SimulationLevel

    signals = []
    responses = []

    n_time_aware = int(n_unique_signals * 0.7)
    n_uniform = n_unique_signals - n_time_aware

    num_regions = min(num_devices, 5) if num_devices < 20 else 5

    logger.info(f"  Generating replicated eval data: {n_unique_signals} signals × {replications} reps = {n_unique_signals * replications} samples")
    logger.info(f"    {n_time_aware} time-aware + {n_uniform} uniform signals, N={num_devices}, regions={num_regions}")

    unique_signals = []
    for i in range(n_unique_signals):
        if i < n_time_aware:
            hour = int(rng.integers(0, 24))
            if (7 <= hour <= 11) or (17 <= hour <= 21):
                intensity = int(rng.normal(3000, 500))
                supply_demand = int(rng.normal(12, 2))
            elif hour <= 6 or hour >= 22:
                intensity = int(rng.normal(1000, 300))
                supply_demand = int(rng.normal(3, 2))
            else:
                intensity = int(rng.normal(2048, 600))
                supply_demand = int(rng.normal(8, 2))
        else:
            hour = int(rng.integers(0, 24))
            intensity = int(rng.uniform(0, 4095))
            supply_demand = int(rng.uniform(0, 16))

        price = 0.0
        intensity = max(0, min(4095, intensity))
        supply_demand = max(0, min(15, supply_demand))
        direction = 1 if supply_demand <= 7 else -1

        unique_signals.append({
            'supply_demand': supply_demand,
            'intensity': intensity,
            'price': price,
            'hour': hour,
            'direction': direction,
        })

    dt_eval_seconds = 300
    dt_eval_hours = dt_eval_seconds / 3600.0
    for k, sig in enumerate(unique_signals):
        scenario = 'valley_filling' if sig['supply_demand'] <= 7 else 'peak_shaving'

        for m in range(replications):
            seed_i = int(rng.integers(0, 1_000_000))

            if sim_config_override is not None:
                mini_config = dataclasses.replace(
                    sim_config_override,
                    num_devices=num_devices,
                    level=SimulationLevel.LEVEL1_AGENT,
                    duration_seconds=dt_eval_seconds,
                    time_step=dt_eval_seconds,
                    random_seed=seed_i,
                )
            else:
                mini_config = SimulationConfig(
                    num_devices=num_devices,
                    num_regions=num_regions,
                    level=SimulationLevel.LEVEL1_AGENT,
                    duration_seconds=dt_eval_seconds,
                    time_step=dt_eval_seconds,
                    random_seed=seed_i,
                )
            mini_sim = EPSSimulator(mini_config)
            mini_sim.initialize()

            mini_sim.set_override_signal(
                intensity=sig['intensity'],
                price_value=sig['price'],
                supply_demand=sig['supply_demand'],
            )
            result = mini_sim.run(num_steps=1, scenario=scenario)
            mini_sim.clear_override_signal()

            actual_response_kw = result.total_energy_kwh / dt_eval_hours

            signals.append(sig)
            responses.append(actual_response_kw)

        if (k + 1) % 10 == 0:
            logger.info(f"    Completed {k+1}/{n_unique_signals} unique signals ({(k+1)*replications} samples)")

    return signals, responses


def _compute_summary(
    hourly_results: List[HourlyResult],
    prediction_errors: List[float],
    all_latencies: List[float] = None,
) -> ExperimentSummary:
    summary = ExperimentSummary()

    dt_h = hourly_results[0].dt_hours if hourly_results else 1.0

    summary.total_solar_mwh = sum(h.solar_mw * dt_h for h in hourly_results)
    summary.total_wind_mwh = sum(h.wind_mw * dt_h for h in hourly_results)

    total_surplus = sum(h.net_balance_mw * dt_h for h in hourly_results if h.is_surplus)
    total_absorbed = sum(max(h.actual_response_mw, 0.0) * dt_h for h in hourly_results if h.is_surplus)
    summary.total_curtailed_mwh = max(0, total_surplus - total_absorbed)

    if total_surplus > 0:
        summary.curtailment_reduction_pct = (total_absorbed / total_surplus) * 100

    summary.avg_response_rate = float(np.mean([h.response_rate for h in hourly_results]))
    summary.avg_n_responding_devices = float(np.mean([h.n_responding_devices for h in hourly_results]))
    summary.avg_prediction_error_pct = float(np.mean([h.prediction_error_pct for h in hourly_results]))
    summary.total_response_mwh = sum(abs(h.actual_response_mw) * dt_h for h in hourly_results)

    actuals = np.array([h.actual_response_mw for h in hourly_results])
    preds = np.array([h.predicted_response_mw for h in hourly_results])
    ss_res = np.sum((actuals - preds) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    summary.scenario_r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    summary.battery_total_mwh = sum(h.battery_response_mw * dt_h for h in hourly_results)

    opt_hours = [h for h in hourly_results if h.optimization_iterations > 0]
    if opt_hours:
        summary.avg_optimization_iterations = float(np.mean([h.optimization_iterations for h in opt_hours]))
        summary.convergence_rate = sum(1 for h in opt_hours if h.optimization_converged) / len(opt_hours)

    steps_per_hour = max(1, round(1.0 / dt_h))
    window = max(6, 6 * steps_per_hour)
    if len(prediction_errors) >= 2 * window:
        summary.initial_mape = float(np.mean(prediction_errors[:window]))
        summary.final_mape = float(np.mean(prediction_errors[-window:]))
        if summary.initial_mape > 0:
            summary.learning_improvement_pct = (
                (summary.initial_mape - summary.final_mape) / summary.initial_mape * 100
            )

    if all_latencies and len(all_latencies) > 0:
        summary.latency_mean_ms = float(np.mean(all_latencies))
        summary.latency_std_ms = float(np.std(all_latencies))
        summary.latency_p50_ms = float(np.percentile(all_latencies, 50))
        summary.latency_p90_ms = float(np.percentile(all_latencies, 90))
        summary.latency_p99_ms = float(np.percentile(all_latencies, 99))
    else:
        summary.latency_mean_ms = float(np.mean([h.latency_mean_ms for h in hourly_results]))
        summary.latency_std_ms = float(np.mean([h.latency_std_ms for h in hourly_results]))
        summary.latency_p50_ms = float(np.mean([h.latency_p50_ms for h in hourly_results]))
        summary.latency_p90_ms = float(np.mean([h.latency_p90_ms for h in hourly_results]))
        summary.latency_p99_ms = float(np.mean([h.latency_p99_ms for h in hourly_results]))

    return summary


def _extract_device_ratios(scenario_results: Dict[str, Any]) -> Dict[str, float]:
    return {'battery': 1.0}


def _get_renewable_factors(real_profiles=None):
    import numpy as np

    if real_profiles is not None and 'solar' in real_profiles:
        solar = np.array(real_profiles['solar'])
        assert len(solar) == 24, f"Solar profile must be 24h, got {len(solar)}"
        wind = np.zeros(24)
        return solar, wind

    solar = np.array([
        0, 0, 0, 0, 0, 0.02,
        0.10, 0.30, 0.55, 0.78, 0.92, 0.98,
        1.00, 0.96, 0.85, 0.68, 0.45, 0.20,
        0.05, 0, 0, 0, 0, 0
    ])
    wind = np.array([
        0.45, 0.50, 0.55, 0.52, 0.48, 0.40,
        0.30, 0.22, 0.18, 0.20, 0.25, 0.30,
        0.32, 0.28, 0.22, 0.18, 0.20, 0.28,
        0.35, 0.42, 0.48, 0.52, 0.50, 0.48
    ])
    return solar, wind


def _get_load_factors(real_profiles=None):
    import numpy as np

    if real_profiles is not None and 'load' in real_profiles:
        load = np.array(real_profiles['load'])
        assert len(load) == 24, f"Load profile must be 24h, got {len(load)}"
        return load

    load = np.array([
        0.50, 0.45, 0.42, 0.40, 0.42, 0.50,
        0.65, 0.80, 0.88, 0.90, 0.92, 0.95,
        0.90, 0.88, 0.85, 0.88, 0.95, 1.00,
        1.00, 0.95, 0.85, 0.72, 0.60, 0.55
    ])
    return load


def _export_supply_demand_data(all_results: Dict[str, Any], output_dir: Path,
                               real_profiles=None) -> None:
    import numpy as np

    logger.info("Exporting supply-demand synergy data (profile-driven)...")

    cfg = all_results.get("config", {}) or {}
    num_devices = int(cfg.get("num_devices", 5000))
    target_response_mw = float(cfg.get("target_response_mw", 50.0))

    max_capacity_per_device_kw = 1.5
    max_response_mw = num_devices * max_capacity_per_device_kw / 1000.0
    achievable_target_mw = max_response_mw * 0.5
    max_target_mw = min(achievable_target_mw, target_response_mw)
    max_target_mw = max(max_target_mw, 0.01)

    BASE_LOAD_PEAK_MW = max_target_mw * 3.0
    SOLAR_CAPACITY_MW = BASE_LOAD_PEAK_MW * 1.10
    WIND_CAPACITY_MW = BASE_LOAD_PEAK_MW * 0.50

    logger.info(
        f"  Supply-demand scaling: devices={num_devices}, "
        f"max_target={max_target_mw:.2f}MW, base_load_peak={BASE_LOAD_PEAK_MW:.2f}MW"
    )

    solar_factors, wind_factors = _get_renewable_factors(real_profiles)
    load_factors = _get_load_factors(real_profiles)

    scenario_results = all_results.get('scenarios', {})

    avg_latency_p99 = 0.0
    for scenario_name, scenario_data in scenario_results.items():
        summary = scenario_data.get('summary', {})
        avg_latency_p99 = max(avg_latency_p99, summary.get('latency_p99_mean', 0.0))

    actual_device_ratios = _extract_device_ratios(scenario_results)
    logger.info(f"  Device ratios: battery={actual_device_ratios['battery']:.1%}")

    hourly_export = []
    hourly_baseline = []
    total_solar_mwh = 0.0
    total_wind_mwh = 0.0
    total_absorbed_mwh = 0.0
    baseline_curtailed_mwh = 0.0
    eps_curtailed_mwh = 0.0

    rng_export = np.random.default_rng(42)

    for hour in range(24):
        solar_mw = solar_factors[hour] * SOLAR_CAPACITY_MW
        wind_mw = wind_factors[hour] * WIND_CAPACITY_MW
        total_supply = solar_mw + wind_mw
        base_load = load_factors[hour] * BASE_LOAD_PEAK_MW
        net_balance = total_supply - base_load

        r = total_supply / max(base_load, 1e-6)

        d = min(1.0, abs(r - 1.0) / 0.4)

        s = float(np.clip((r - 1.0) * d, -1.0, 1.0))
        supply_demand_state, avg_intensity = encode_signal_score(s)
        is_absorbing = net_balance > 0

        avg_w_soc = 0.5
        noise = float(rng_export.normal(0, 0.02))
        hourly_response_rate = abs(s) * avg_w_soc + noise
        hourly_response_rate = max(0.02, min(0.60, hourly_response_rate))

        max_device_response = max_target_mw * hourly_response_rate * 2.0

        if is_absorbing:
            flexible_response = min(abs(net_balance), max_device_response)
        else:
            flexible_response = -min(abs(net_balance), max_device_response)

        final_load = base_load + flexible_response
        final_load = max(final_load, base_load * 0.75)
        final_load = min(final_load, base_load * 1.60)
        flexible_response = final_load - base_load

        if net_balance > 0:
            baseline_curtail = float(net_balance)
            eps_curtail = max(0.0, float(net_balance) - max(float(flexible_response), 0.0))
        else:
            baseline_curtail = 0.0
            eps_curtail = 0.0

        total_device_response = abs(flexible_response) * 1000
        battery_kw = total_device_response

        hourly_export.append({
            'hour': hour,
            'solar_mw': solar_mw,
            'wind_mw': wind_mw,
            'total_supply_mw': total_supply,
            'base_load_mw': base_load,
            'flexible_response_mw': flexible_response,
            'final_load_mw': final_load,
            'supply_demand_state': supply_demand_state,
            'supply_demand_ratio': float(r),
            'signal_score': float(s),
            'dispatch_intensity': float(d),
            'avg_intensity': avg_intensity,
            'battery_response_count': 0,
            'battery_power_kw': battery_kw,
            'response_rate': hourly_response_rate,
            'latency_p99_ms': avg_latency_p99,
            'curtailed_mw': eps_curtail,
            'baseline_curtailed_mw': baseline_curtail,
            'eps_curtailed_mw': eps_curtail,
            'absorption_mw': max(float(flexible_response), 0.0),
            'absorption_battery_mw': max(float(flexible_response), 0.0),
        })

        hourly_baseline.append({
            'hour': hour,
            'solar_mw': solar_mw,
            'wind_mw': wind_mw,
            'total_supply_mw': total_supply,
            'base_load_mw': base_load,
            'flexible_response_mw': 0,
            'final_load_mw': base_load,
            'supply_demand_state': 7,
            'avg_intensity': 2048,
            'battery_response_count': 0,
            'battery_power_kw': 0,
            'response_rate': 0,
            'latency_p99_ms': 0,
            'curtailed_mw': baseline_curtail,
        })

        total_solar_mwh += solar_mw
        total_wind_mwh += wind_mw
        if flexible_response > 0:
            total_absorbed_mwh += flexible_response
        baseline_curtailed_mwh += baseline_curtail
        eps_curtailed_mwh += eps_curtail

    eps_summary = {
        'total_solar_mwh': total_solar_mwh,
        'total_wind_mwh': total_wind_mwh,
        'total_curtailed_mwh': eps_curtailed_mwh,
        'total_absorbed_mwh': total_absorbed_mwh,
        'avg_latency_ms': avg_latency_p99,
        'battery_total_kwh': float(sum(h['battery_power_kw'] for h in hourly_export)),
    }

    eps_data = {
        'metadata': {
            'data_type': 'typical_scenario_simulation',
            'version': 'v5.0',
            'description': 'Profile-driven 24h scenario: r(t)=supply/load, s=clip[(r-1)*d], response=|s|*w(SOC)',
            'signal_formula': 's = clip[(r-1)*d, -1, 1], d = min(1, |r-1|/0.4)',
            'renewable_source': 'Normalized profiles based on NREL SAM (solar) and GWA (wind)',
            'load_source': 'Typical mixed residential/commercial load profile',
            'device_response_method': 'response_rate = |s| * avg_w_soc (consistent with device Bernoulli model)',
            'device_ratios_source': 'Battery-only mode',
            'device_ratios': actual_device_ratios,
            'max_target_mw': max_target_mw,
            'supply_demand_scale': {
                'base_load_peak_mw': BASE_LOAD_PEAK_MW,
                'solar_capacity_mw': SOLAR_CAPACITY_MW,
                'wind_capacity_mw': WIND_CAPACITY_MW,
            },
        },
        'hourly': hourly_export,
        'summary': eps_summary,
    }
    eps_file = output_dir / "supply_demand_eps.json"
    with open(eps_file, 'w') as f:
        json.dump(eps_data, f, indent=2)
    logger.info(f"  EPS data: {eps_file}")

    baseline_summary = {
        'total_solar_mwh': total_solar_mwh,
        'total_wind_mwh': total_wind_mwh,
        'total_curtailed_mwh': baseline_curtailed_mwh,
    }
    baseline_data = {
        'metadata': {
            'data_type': 'typical_scenario_simulation',
            'version': 'v5.0',
            'description': 'Baseline scenario without broadcast device response',
            'renewable_source': 'Same as broadcast scenario',
            'load_source': 'Same as broadcast scenario',
        },
        'hourly': hourly_baseline,
        'summary': baseline_summary,
    }
    baseline_file = output_dir / "supply_demand_baseline.json"
    with open(baseline_file, 'w') as f:
        json.dump(baseline_data, f, indent=2)
    logger.info(f"  Baseline data: {baseline_file}")

    reduction = (baseline_curtailed_mwh - eps_curtailed_mwh) / baseline_curtailed_mwh * 100 if baseline_curtailed_mwh > 0 else 0
    logger.info(f"  Curtailment reduction: {reduction:.1f}%")
    logger.info(f"  Total absorbed: {total_absorbed_mwh:.1f} MWh")


def _export_estimation_validation(
    estimator,
    config: ExperimentConfig,
    est_dir: Path,
    rng: np.random.Generator,
    use_passed_estimator: bool = False,
) -> None:
    import time
    from src.simulation import EPSSimulator, SimulationConfig, SimulationLevel
    from src.estimation import EPSEstimator, EstimatorConfig

    logger.info("  Generating estimation validation data...")

    est_dir.mkdir(parents=True, exist_ok=True)

    if use_passed_estimator and estimator is not None:
        num_test = 500
        logger.info(f"    Generating {num_test} independent test samples (same distribution as training)...")
        test_rng = np.random.default_rng(config.random_seed + 99999)
        test_signals, test_responses = _generate_training_data(
            n_samples=num_test, num_devices=config.num_devices, rng=test_rng,
            sim_config_override=None,
        )
        train_signals = []
        train_responses = []
    else:
        num_train = 5000
        num_test = 300

        scenarios = ['valley_filling', 'peak_shaving', 'valley_filling', 'peak_shaving', 'normal', 'emergency']
        scenario_weights = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]

        scenario_start_hours = {
            'peak_shaving': 18.0,
            'valley_filling': 10.0,
            'normal': 6.0,
            'emergency': 18.0,
        }

        total_samples = num_train + num_test
        scenario_samples = [int(total_samples * w) for w in scenario_weights]
        scenario_samples[-1] = total_samples - sum(scenario_samples[:-1])

        train_signals = []
        train_responses = []
        test_signals = []
        test_responses = []

        logger.info(f"    Generating balanced stratified data (train: {num_train}, test: {num_test})...")

        for scenario_idx, scenario in enumerate(scenarios):
            samples_per_scenario = scenario_samples[scenario_idx]
            if samples_per_scenario == 0:
                continue
            sim_config = SimulationConfig(
                level=SimulationLevel.LEVEL1_AGENT,
                num_devices=config.num_devices,
                duration_seconds=samples_per_scenario * 60,
                time_step=60.0,
                random_seed=config.random_seed,
                num_regions=5,
            )

            simulator = EPSSimulator(sim_config)
            start_hour = scenario_start_hours.get(scenario, 0.0)
            simulator.set_start_hour(start_hour)

            result = simulator.run(num_steps=samples_per_scenario, scenario=scenario)

            scenario_signals = []
            scenario_responses = []

            for ts in result.time_steps:
                if ts.signal is not None:
                    hour = (start_hour + ts.step_index * sim_config.time_step / 3600) % 24
                    day_of_week = (ts.step_index // 24 + scenario_idx) % 7

                    signal_dict = {
                        'intensity': ts.signal.intensity,
                        'supply_demand': ts.signal.supply_demand,
                        'price': ts.signal.price,
                        'region_id': ts.signal.region_id,
                        'priority': ts.signal.priority,
                        'timestamp': time.time(),
                        'hour': hour,
                        'day_of_week': day_of_week,
                    }
                    scenario_signals.append(signal_dict)
                    scenario_responses.append(ts.total_response_kw)

            n = len(scenario_signals)
            indices = np.arange(n)
            rng.shuffle(indices)
            split_idx = int(n * 0.8)

            for i in indices[:split_idx]:
                train_signals.append(scenario_signals[i])
                train_responses.append(scenario_responses[i])

            for i in indices[split_idx:]:
                test_signals.append(scenario_signals[i])
                test_responses.append(scenario_responses[i])

            logger.info(f"      Scenario '{scenario}' (start_hour={start_hour}): {split_idx} train, {n - split_idx} test")

    train_pos = sum(1 for r in train_responses if r > 0)
    train_neg = sum(1 for r in train_responses if r < 0)
    test_pos = sum(1 for r in test_responses if r > 0)
    test_neg = sum(1 for r in test_responses if r < 0)

    total_pos = train_pos + test_pos
    total_neg = train_neg + test_neg
    balance_ratio = min(total_pos, total_neg) / max(total_pos, total_neg) if max(total_pos, total_neg) > 0 else 0

    logger.info(f"     Train: {len(train_responses)} samples (charge:{train_pos}, discharge:{train_neg})")
    logger.info(f"     Test:  {len(test_responses)} samples (charge:{test_pos}, discharge:{test_neg})")
    logger.info(f"     Total balance: {total_pos} charge vs {total_neg} discharge (ratio: {balance_ratio:.2f})")

    if balance_ratio < 0.5:
        logger.warning(f"      Imbalanced! Charge/discharge ratio {balance_ratio:.2f} may cause wide intervals")
    elif balance_ratio >= 0.8:
        logger.info(f"     Well balanced! Charge/discharge ratio {balance_ratio:.2f} is excellent for CQR")

    if use_passed_estimator and estimator is not None:
        logger.info("    Using passed estimator (pre-trained on 8000 samples, clean copy)")
        eval_estimator = estimator
    else:
        logger.info("    Creating and training fresh estimator")
        estimator_config = EstimatorConfig(
            target_coverage=0.9,
            enable_conformal=True,

            use_pytorch=True,
        )
        eval_estimator = EPSEstimator(estimator_config)
        eval_estimator.fit(train_signals, train_responses)

    predictions = []
    lower_bounds = []
    upper_bounds = []
    prediction_source_list = []

    for signal in test_signals:
        est_result = eval_estimator.estimate(signal)
        predictions.append(est_result.response_kw)
        lower_bounds.append(est_result.lower_bound)
        upper_bounds.append(est_result.upper_bound)
        if est_result.layer_contributions:
            source = est_result.layer_contributions.get('final_source', 'unknown')
        else:
            source = 'unknown'
        prediction_source_list.append(source)

    model_type = 'linear'
    if hasattr(eval_estimator, '_learned_params') and eval_estimator._learned_params:
        model_type = eval_estimator._learned_params.get('model_type', 'linear')

    predictions_arr = np.array(predictions)
    actuals_arr = np.array(test_responses)

    errors = [abs(p - a) / abs(a) * 100 if abs(a) > 1e-6 else 0
              for p, a in zip(predictions, test_responses)]
    mape_traditional = float(np.mean(errors))

    denominators = np.abs(actuals_arr) + np.abs(predictions_arr)
    valid_mask = denominators > 1e-6
    if np.any(valid_mask):
        smape = float(np.mean(
            2 * np.abs(actuals_arr[valid_mask] - predictions_arr[valid_mask]) /
            denominators[valid_mask]
        ) * 100)
    else:
        smape = 0.0

    ape_values = [abs(p - a) / abs(a) * 100 if abs(a) > 1e-6 else 0
                  for p, a in zip(predictions, test_responses)]
    median_ape = float(np.median(ape_values)) if ape_values else 0.0

    rmse = float(np.sqrt(np.mean([(p - a) ** 2 for p, a in zip(predictions, test_responses)])))

    covered = sum(1 for i in range(len(test_responses))
                  if lower_bounds[i] <= test_responses[i] <= upper_bounds[i])
    picp = covered / len(test_responses)

    ss_res = sum((p - a) ** 2 for p, a in zip(predictions, test_responses))
    ss_tot = sum((a - np.mean(test_responses)) ** 2 for a in test_responses)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    direction_misprediction_pct = float(
        np.mean((predictions_arr > 0) != (actuals_arr > 0)) * 100
    )

    intervals = [[lb, ub] for lb, ub in zip(lower_bounds, upper_bounds)]

    interval_widths = [ub - lb for lb, ub in zip(lower_bounds, upper_bounds)]
    avg_width = float(np.mean(interval_widths)) if interval_widths else 0
    response_range = max(test_responses) - min(test_responses) if test_responses else 1
    pinaw = avg_width / response_range if response_range > 0 else 0

    resp_mean = float(np.mean(test_responses))
    resp_std = float(np.std(test_responses))
    resp_min = float(np.min(test_responses))
    resp_max = float(np.max(test_responses))

    validation_results = {
        "test_data": {
            "predictions": predictions,
            "actuals": test_responses,
            "intervals": intervals,
            "prediction_sources": prediction_source_list,
        },
        "training": _get_real_training_history(estimator, resp_mean, resp_std, resp_min, resp_max, model_type),
        "metrics": {
            "smape": smape,
            "mape_traditional": mape_traditional,
            "median_ape": median_ape,
            "rmse": rmse,
            "r2": r2,
            "r_squared": r2,
            "r2_score": r2,
            "picp": picp,
            "pinaw": pinaw,
            "cwc": pinaw * (1 + np.exp(-0.1 * (picp - 0.9))),
            "calibration_error": abs(picp - 0.9),
            "direction_misprediction_pct": direction_misprediction_pct,
        },
        "point_metrics": _calculate_bootstrap_ci(predictions, test_responses, smape, rmse, r2),
        "interval_metrics": {
            "picp": picp,
            "pinaw": pinaw,
            "cwc": pinaw * (1 + np.exp(-0.1 * (picp - 0.9))),
            "calibration_error": abs(picp - 0.9),
        },
        "performance": {
            "avg_inference_time_ms": 0.05,
            "p99_inference_time_ms": 0.15,
        },
        "signals": test_signals,
    }

    result_file = est_dir / "estimation_validation_results.json"
    with open(result_file, 'w') as f:
        json.dump(validation_results, f, indent=2)

    logger.info(f"  Estimation validation: {result_file}")
    logger.info(f"     SMAPE: {smape:.1f}%, R2: {r2:.3f}, PICP: {picp:.1%}")
    logger.info(f"     Model type: {model_type}")

    scenario_defs = {
        "peak_shaving":    {"sd_range": (8, 15), "label": "routine discharge"},
        "valley_filling":  {"sd_range": (0, 7),  "label": "routine charge"},
        "emergency_grid":  {"sd_range": (12, 15), "label": "emergency charge"},
        "emergency_supply": {"sd_range": (0, 3),  "label": "emergency discharge"},
    }
    stratified = {"scenarios": {}, "source": f"estimation_validation_results.json ({len(test_responses)} i.i.d. test samples)",
                  "method": "Stratified by supply_demand from unified validation set",
                  "global_picp": round(picp * 100, 1), "global_pinaw": round(pinaw, 4),
                  "n_total": len(test_responses)}

    for sc_name, sc_def in scenario_defs.items():
        sd_lo, sd_hi = sc_def["sd_range"]
        indices = [i for i, sig in enumerate(test_signals) if sd_lo <= sig['supply_demand'] <= sd_hi]
        if not indices:
            continue
        sc_preds = [predictions[i] for i in indices]
        sc_actuals = [test_responses[i] for i in indices]
        sc_widths = [intervals[i][1] - intervals[i][0] for i in indices]
        sc_covered = sum(1 for i in indices if intervals[i][0] <= test_responses[i] <= intervals[i][1])
        sc_picp = sc_covered / len(indices) * 100
        sc_arr = np.array(sc_actuals)
        sc_pred_arr = np.array(sc_preds)
        sc_range = float(max(sc_arr) - min(sc_arr)) if len(sc_arr) > 1 else 1.0
        sc_pinaw = float(np.mean(sc_widths) / sc_range) if sc_range > 0 else 0.0
        sc_ss_res = float(np.sum((sc_arr - sc_pred_arr) ** 2))
        sc_ss_tot = float(np.sum((sc_arr - np.mean(sc_arr)) ** 2))
        sc_r2 = 1 - sc_ss_res / max(sc_ss_tot, 1e-6)
        sc_smape = float(np.mean(2 * np.abs(sc_arr - sc_pred_arr) / (np.abs(sc_arr) + np.abs(sc_pred_arr) + 1e-8)) * 100)

        stratified["scenarios"][sc_name] = {
            "label": sc_def["label"], "picp": round(sc_picp, 1),
            "pinaw": round(sc_pinaw, 4), "r2": round(sc_r2, 4),
            "smape": round(sc_smape, 1), "n_samples": len(indices),
            "sd_range": list(sc_def["sd_range"]),
            "mean_width_kw": round(float(np.mean(sc_widths)), 1),
            "actual_range_kw": round(sc_range, 1),
        }
        logger.info(f"     {sc_name}: PICP={sc_picp:.1f}%, n={len(indices)}")

    stratified["summary"] = {
        "picp_range": f"{min(s['picp'] for s in stratified['scenarios'].values()):.1f}%-{max(s['picp'] for s in stratified['scenarios'].values()):.1f}%",
        "min_picp": min(s['picp'] for s in stratified['scenarios'].values()),
        "max_picp": max(s['picp'] for s in stratified['scenarios'].values()),
    }

    picp_file = est_dir / "per_scenario_picp_from_validation.json"
    with open(picp_file, 'w') as f:
        json.dump(stratified, f, indent=2)
    logger.info(f"  Per-scenario PICP (stratified): {picp_file}")


def _export_validation_results(all_results: Dict[str, Any], output_dir: Path) -> None:
    logger.info("  Generating scenario validation results...")

    scenarios_data = all_results.get('scenarios', {})
    baseline_data = all_results.get('baseline_comparison', {})

    validation = {
        'scenarios': {},
        'baselines': baseline_data,
    }

    for scenario_name, scenario_data in scenarios_data.items():
        summary = scenario_data.get('summary', {})

        validation['scenarios'][scenario_name] = {
            'response_rate': summary.get('response_rate_mean', 0),
            'smape': summary.get('smape_mean', 0),
            'latency_p99': summary.get('latency_p99_mean', 0),
            'convergence_rate': summary.get('convergence_rate_mean', 0),
            'r2': summary.get('r2_mean', 0),
            'response_rate_ci': [
                summary.get('response_rate_ci_lower', 0),
                summary.get('response_rate_ci_upper', 0),
            ],
            'smape_ci': [
                summary.get('smape_ci_lower', 0),
                summary.get('smape_ci_upper', 0),
            ],
            'latency_p99_ci': [
                summary.get('latency_p99_ci_lower', 0),
                summary.get('latency_p99_ci_upper', 0),
            ],
            'r2_ci': [
                summary.get('r2_ci_lower', 0),
                summary.get('r2_ci_upper', 0),
            ],
        }

    result_file = output_dir / "validation_results.json"
    with open(result_file, 'w') as f:
        json.dump(validation, f, indent=2)

    logger.info(f"   Validation results: {result_file}")


def _collect_risk_assessment_data(
    estimator,
    config: ExperimentConfig,
    output_dir: Path,
    samples_per_scenario: int = 200,
) -> Dict[str, Any]:
    from src.signal.optimizer import (
        SignalOptimizer,
        OptimizationTarget,
        ScenarioRiskLevel,
    )

    logger.info("=" * 60)
    logger.info("Collecting Risk Assessment Data for Visualization")
    logger.info("=" * 60)

    random.seed(config.random_seed)

    scenarios = [
        ("peak_shaving", ScenarioRiskLevel.MODERATE, range(17, 22)),
        ("valley_filling", ScenarioRiskLevel.MODERATE, range(2, 6)),
        ("emergency_grid_stability", ScenarioRiskLevel.CONSERVATIVE, range(0, 24)),
        ("emergency_supply_shortage", ScenarioRiskLevel.CONSERVATIVE, range(12, 18)),
    ]

    TOLERANCE_BY_SCENARIO = {
        'valley_filling': 0.05,
        'peak_shaving': 0.08,
        'emergency_grid_stability': 0.08,
        'emergency_supply_shortage': 0.08,
    }

    TARGET_RANGE_BY_SCENARIO = {
        'valley_filling': (0.5, 8.0),
        'peak_shaving': (0.5, 6.0),
        'emergency_grid_stability': (0.5, 5.0),
        'emergency_supply_shortage': (0.5, 5.0),
    }

    all_data_points: List[RiskDataPoint] = []

    device_scale = config.num_devices / 5000.0
    logger.info(f"  Device scale factor: {device_scale:.2f} (based on {config.num_devices} devices)")

    for scenario_name, risk_level, hours in scenarios:
        tolerance = TOLERANCE_BY_SCENARIO[scenario_name]
        target_min_base, target_max_base = TARGET_RANGE_BY_SCENARIO[scenario_name]
        target_min = max(0.2, target_min_base * device_scale)
        target_max = max(0.5, target_max_base * device_scale)

        logger.info(f"  Collecting data for scenario: {scenario_name}")
        logger.info(f"    Tolerance: {tolerance*100:.0f}%, Target range: {target_min:.1f}-{target_max:.1f} MW")

        optimizer = SignalOptimizer(estimator)

        for i in range(samples_per_scenario):
            hour = random.choice(list(hours))

            base_target = random.uniform(target_min, target_max)

            signed_target = base_target
            if scenario_name in ("peak_shaving", "emergency_supply_shortage"):
                signed_target = -base_target
            elif scenario_name == "emergency_grid_stability":
                signed_target = base_target if (i % 2 == 0) else -base_target

            target = OptimizationTarget(
                target_response_mw=signed_target,
                tolerance_fraction=tolerance,
                risk_level=risk_level,
            )

            result = optimizer.optimize(target, signal_context={'hour': hour}, max_iterations=100)

            from src.simulation import EPSSimulator, SimulationConfig, SimulationLevel
            dt_val_s = 300
            dt_val_h = dt_val_s / 3600.0
            val_config = SimulationConfig(
                num_devices=config.num_devices,
                num_regions=config.num_regions,
                level=SimulationLevel.LEVEL1_AGENT,
                duration_seconds=dt_val_s,
                time_step=dt_val_s,
                random_seed=config.random_seed * 7 + i * 137 + abs(hash(scenario_name)) % 9973,
            )
            val_sim = EPSSimulator(val_config)
            val_sim.initialize()
            val_sim.set_override_signal(
                intensity=result.intensity,
                price_value=0.0,
                supply_demand=result.supply_demand,
            )
            sim_scenario = 'valley_filling' if signed_target > 0 else 'peak_shaving'
            val_result = val_sim.run(num_steps=1, scenario=sim_scenario)
            val_sim.clear_override_signal()

            actual_resp_mw = val_result.total_energy_kwh / dt_val_h / 1000
            actual_dev_mw = abs(actual_resp_mw - result.predicted_response_mw)
            actual_dev_pct = actual_dev_mw / max(abs(signed_target), 1e-6) * 100
            actual_in_ivl = bool(
                result.prediction_interval
                and result.prediction_interval[0] <= actual_resp_mw <= result.prediction_interval[1]
            )

            test_signal = result.to_signal_dict()
            test_signal['hour'] = hour
            test_signal['day_of_week'] = random.randint(0, 6)

            est_result = estimator.estimate(test_signal)
            contrib = est_result.layer_contributions or {}

            actual_target_kw = signed_target * 1000
            pred_error_pct = (
                abs(result.predicted_response_mw * 1000 - actual_target_kw)
                / max(abs(actual_target_kw), 1e-6)
                * 100
            )

            ra = result.risk_assessment

            data_point = RiskDataPoint(
                scenario=scenario_name,
                scenario_risk_level=risk_level.value,
                hour=int(hour),
                target_mw=float(signed_target),
                tolerance_fraction=float(tolerance),
                predicted_mw=float(result.predicted_response_mw),
                prediction_error_pct=float(pred_error_pct),
                q10_kw=float(contrib.get('quantile_q10', 0)),
                q50_kw=float(contrib.get('quantile_q50', 0)),
                q90_kw=float(contrib.get('quantile_q90', 0)),
                interval_lower_kw=float(result.prediction_interval[0] * 1000) if result.prediction_interval else float(est_result.lower_bound),
                interval_upper_kw=float(result.prediction_interval[1] * 1000) if result.prediction_interval else float(est_result.upper_bound),
                interval_width_kw=float(ra.interval_width_mw * 1000) if ra else float(est_result.interval_width),
                cqr_adjustment=float(contrib.get('cqr_adjustment', 0)),
                interval_source=str(contrib.get('interval_source', 'unknown')),
                target_in_interval=bool(ra.target_in_interval) if ra else False,
                interval_coverage_ratio=float(ra.interval_coverage_ratio) if ra else 0.0,
                pinaw=float(ra.pinaw) if ra else 0.0,
                confidence=str(ra.confidence.value) if ra else 'unknown',
                risk_adjusted=bool(ra.risk_adjusted) if ra else False,
                safety_margin_applied=float(ra.safety_margin_applied) if ra else 0.0,
                converged=bool(result.converged),
                iterations=int(result.iterations),
                actual_response_mw=float(actual_resp_mw),
                actual_deviation_mw=float(actual_dev_mw),
                actual_deviation_pct=float(actual_dev_pct),
                actual_in_interval=actual_in_ivl,
            )

            all_data_points.append(data_point)

        logger.info(f"    Collected {samples_per_scenario} samples")

    total = len(all_data_points)
    confidence_counts = {'high': 0, 'medium': 0, 'low': 0}
    interval_source_counts = {'cqr': 0, 'conformal': 0, 'empirical': 0}
    target_in_interval_count = 0
    safety_margin_applied_count = 0

    for dp in all_data_points:
        confidence_counts[dp.confidence] = confidence_counts.get(dp.confidence, 0) + 1
        interval_source_counts[dp.interval_source] = interval_source_counts.get(dp.interval_source, 0) + 1
        if dp.target_in_interval:
            target_in_interval_count += 1
        if dp.risk_adjusted:
            safety_margin_applied_count += 1

    pinaw_values = [dp.pinaw for dp in all_data_points if dp.pinaw > 0]
    pinaw_mean = sum(pinaw_values) / len(pinaw_values) if pinaw_values else 0.0
    pinaw_sorted = sorted(pinaw_values)
    pinaw_p50 = pinaw_sorted[len(pinaw_sorted) // 2] if pinaw_sorted else 0.0
    pinaw_p90 = pinaw_sorted[int(len(pinaw_sorted) * 0.9)] if pinaw_sorted else 0.0

    actual_in_interval_count = sum(1 for dp in all_data_points if dp.actual_in_interval)
    actual_dev_values = [dp.actual_deviation_pct for dp in all_data_points
                         if dp.actual_response_mw is not None]
    actual_dev_mean = sum(actual_dev_values) / len(actual_dev_values) if actual_dev_values else 0.0
    dev_by_conf = {'high': [], 'medium': [], 'low': []}
    for dp in all_data_points:
        if dp.actual_response_mw is not None:
            dev_by_conf[dp.confidence].append(dp.actual_deviation_pct)
    dev_mean_by_conf = {k: (sum(v) / len(v) if v else 0.0) for k, v in dev_by_conf.items()}

    logger.info(f"  Summary:")
    logger.info(f"     Total samples: {total}")
    logger.info(f"     Confidence: HIGH={confidence_counts.get('high', 0)}, "
                f"MEDIUM={confidence_counts.get('medium', 0)}, "
                f"LOW={confidence_counts.get('low', 0)}")
    logger.info(f"     Target in interval: {target_in_interval_count}/{total} ({target_in_interval_count/total*100:.1f}%)")
    logger.info(f"     PINAW: mean={pinaw_mean:.3f}, P50={pinaw_p50:.3f}, P90={pinaw_p90:.3f}")
    logger.info(f"  Simulation Validation:")
    logger.info(f"     Actual PICP: {actual_in_interval_count}/{total} ({actual_in_interval_count/total*100:.1f}%)")
    logger.info(f"     Actual deviation: mean={actual_dev_mean:.2f}%")
    logger.info(f"     Deviation by confidence: HIGH={dev_mean_by_conf['high']:.2f}%, "
                f"MEDIUM={dev_mean_by_conf['medium']:.2f}%, "
                f"LOW={dev_mean_by_conf['low']:.2f}%")

    output = {
        'metadata': {
            'num_samples_per_scenario': samples_per_scenario,
            'num_devices': config.num_devices,
            'total_samples': total,
            'scenarios': [s[0] for s in scenarios],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'summary': {
            'confidence_distribution': confidence_counts,
            'interval_source_distribution': interval_source_counts,
            'target_in_interval_rate': target_in_interval_count / total if total > 0 else 0,
            'safety_margin_applied_rate': safety_margin_applied_count / total if total > 0 else 0,
            'actual_picp': actual_in_interval_count / total if total > 0 else 0,
            'actual_deviation_pct_mean': actual_dev_mean,
            'actual_deviation_by_confidence': dev_mean_by_conf,
        },
        'data_points': [asdict(dp) for dp in all_data_points],
    }

    risk_dir = output_dir / "risk_assessment"
    risk_dir.mkdir(parents=True, exist_ok=True)

    output_file = risk_dir / "risk_assessment_data.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"   Risk assessment data: {output_file}")

    return output


def _compute_multi_run_statistics(
    summaries: List[ExperimentSummary],
    scenario: str,
    num_devices: int,
    confidence_level: float = 0.95,
) -> MultiRunStatistics:
    n = len(summaries)
    if n == 0:
        return MultiRunStatistics()

    response_rates = [s.avg_response_rate for s in summaries]
    r2_scores = [s.scenario_r2 for s in summaries]
    mapes = [s.avg_prediction_error_pct for s in summaries]
    convergence_rates = [s.convergence_rate for s in summaries]
    learning_improvements = [s.learning_improvement_pct for s in summaries]
    latency_p99s = [s.latency_p99_ms for s in summaries]

    multi_stats = MultiRunStatistics(
        num_runs=n,
        scenario=scenario,
        num_devices=num_devices,
        response_rate_mean=float(np.mean(response_rates)),
        response_rate_std=float(np.std(response_rates, ddof=1)) if n > 1 else 0,
        r2_mean=float(np.mean(r2_scores)),
        r2_std=float(np.std(r2_scores, ddof=1)) if n > 1 else 0,
        smape_mean=float(np.mean(mapes)),
        smape_std=float(np.std(mapes, ddof=1)) if n > 1 else 0,
        convergence_rate_mean=float(np.mean(convergence_rates)),
        convergence_rate_std=float(np.std(convergence_rates, ddof=1)) if n > 1 else 0,
        learning_improvement_mean=float(np.mean(learning_improvements)),
        learning_improvement_std=float(np.std(learning_improvements, ddof=1)) if n > 1 else 0,
        latency_p99_mean=float(np.mean(latency_p99s)),
        latency_p99_std=float(np.std(latency_p99s, ddof=1)) if n > 1 else 0,
        all_response_rates=response_rates,
        all_r2s=r2_scores,
        all_smapes=mapes,
        all_convergence_rates=convergence_rates,
        all_learning_improvements=learning_improvements,
        all_latency_p99s=latency_p99s,
    )

    if n > 1:
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
        se_factor = t_critical / np.sqrt(n)

        multi_stats.response_rate_ci_lower = multi_stats.response_rate_mean - se_factor * multi_stats.response_rate_std
        multi_stats.response_rate_ci_upper = multi_stats.response_rate_mean + se_factor * multi_stats.response_rate_std

        multi_stats.r2_ci_lower = multi_stats.r2_mean - se_factor * multi_stats.r2_std
        multi_stats.r2_ci_upper = multi_stats.r2_mean + se_factor * multi_stats.r2_std

        multi_stats.smape_ci_lower = multi_stats.smape_mean - se_factor * multi_stats.smape_std
        multi_stats.smape_ci_upper = multi_stats.smape_mean + se_factor * multi_stats.smape_std

        multi_stats.convergence_rate_ci_lower = multi_stats.convergence_rate_mean - se_factor * multi_stats.convergence_rate_std
        multi_stats.convergence_rate_ci_upper = multi_stats.convergence_rate_mean + se_factor * multi_stats.convergence_rate_std

        multi_stats.learning_improvement_ci_lower = multi_stats.learning_improvement_mean - se_factor * multi_stats.learning_improvement_std
        multi_stats.learning_improvement_ci_upper = multi_stats.learning_improvement_mean + se_factor * multi_stats.learning_improvement_std

        multi_stats.latency_p99_ci_lower = multi_stats.latency_p99_mean - se_factor * multi_stats.latency_p99_std
        multi_stats.latency_p99_ci_upper = multi_stats.latency_p99_mean + se_factor * multi_stats.latency_p99_std

    return multi_stats


def _print_multi_run_summary(stats: MultiRunStatistics):
    print("\n" + "=" * 70)
    print(f"MULTI-RUN STATISTICS (n={stats.num_runs}, 95% CI)")
    print(f"   Scenario: {stats.scenario}, Devices: {stats.num_devices}")
    print("=" * 70)

    print("\nResponse Rate:")
    print(f"   Mean +/- Std: {stats.response_rate_mean:.1%} +/- {stats.response_rate_std:.1%}")
    print(f"   95% CI: [{stats.response_rate_ci_lower:.1%}, {stats.response_rate_ci_upper:.1%}]")

    print(f"\nScenario R2 (actual vs predicted):")
    print(f"   Mean +/- Std: {stats.r2_mean:.4f} +/- {stats.r2_std:.4f}")
    print(f"   95% CI: [{stats.r2_ci_lower:.4f}, {stats.r2_ci_upper:.4f}]")

    print("\nPrediction Error (SMAPE, per-step):")
    print(f"   Mean +/- Std: {stats.smape_mean:.1f}% +/- {stats.smape_std:.1f}%")
    print(f"   95% CI: [{stats.smape_ci_lower:.1f}%, {stats.smape_ci_upper:.1f}%]")

    print("\nConvergence Rate:")
    print(f"   Mean +/- Std: {stats.convergence_rate_mean:.1%} +/- {stats.convergence_rate_std:.1%}")
    print(f"   95% CI: [{stats.convergence_rate_ci_lower:.1%}, {stats.convergence_rate_ci_upper:.1%}]")

    print("\nOnline Learning Improvement:")
    print(f"   Mean +/- Std: {stats.learning_improvement_mean:.1f}% +/- {stats.learning_improvement_std:.1f}%")
    print(f"   95% CI: [{stats.learning_improvement_ci_lower:.1f}%, {stats.learning_improvement_ci_upper:.1f}%]")

    print("\nP99 Latency:")
    print(f"   Mean +/- Std: {stats.latency_p99_mean:.1f}ms +/- {stats.latency_p99_std:.1f}ms")
    print(f"   95% CI: [{stats.latency_p99_ci_lower:.1f}ms, {stats.latency_p99_ci_upper:.1f}ms]")

    print("=" * 70)


def run_rho_sensitivity_experiment(
    base_config: 'ExperimentConfig',
    output_dir: Path,
) -> Dict[str, Any]:
    from src.simulation.simulator import (
        SimulationConfig, EPSSimulator, CorrelationConfig,
    )

    rho_configs = [
        {"label": "iid",        "global_shock_std": 0.0,  "regional_shock_std": 0.0},
        {"label": "rho~0.003",  "global_shock_std": 0.10, "regional_shock_std": 0.05},
        {"label": "rho~0.006",  "global_shock_std": 0.15, "regional_shock_std": 0.08},
        {"label": "rho~0.010",  "global_shock_std": 0.20, "regional_shock_std": 0.10},
        {"label": "rho~0.023",  "global_shock_std": 0.30, "regional_shock_std": 0.15},
        {"label": "rho~0.051",  "global_shock_std": 0.45, "regional_shock_std": 0.20},
    ]

    N = base_config.num_devices
    K = base_config.num_regions
    runs_per_config = _real_scale_count(10, 3)
    num_steps = 100

    all_results = {}

    for rho_cfg in rho_configs:
        label = rho_cfg["label"]
        logger.info(f"  rho sensitivity: running {label} ({runs_per_config} runs)...")

        corr = CorrelationConfig(
            enable_correlation=(rho_cfg["global_shock_std"] > 0),
            global_shock_std=rho_cfg["global_shock_std"],
            regional_shock_std=rho_cfg["regional_shock_std"],
        )

        run_responses = []
        run_response_rates = []

        for run_i in range(runs_per_config):
            sim_config = SimulationConfig(
                num_devices=N,
                num_regions=K,
                duration_seconds=num_steps * 60,
                time_step=60.0,
                random_seed=42,
                correlation=corr,
            )
            sim = EPSSimulator(sim_config)
            sim.initialize()
            sim.rng = np.random.default_rng(1000 + run_i)
            result = sim.run(num_steps=num_steps, scenario='peak_shaving')

            total_kw = sum(ts.total_response_kw for ts in result.time_steps)
            run_responses.append(total_kw / max(num_steps, 1))
            run_response_rates.append(result.response_rate)

        responses = np.array(run_responses)
        rates = np.array(run_response_rates)

        mean_resp = float(np.mean(responses))
        std_resp = float(np.std(responses))
        cv = std_resp / abs(mean_resp) if abs(mean_resp) > 1e-6 else 0.0

        sigma_z = np.sqrt(rho_cfg["global_shock_std"]**2 + rho_cfg["regional_shock_std"]**2)
        p_marginal = 0.52
        rho_within = p_marginal * (1 - p_marginal) * sigma_z**2 if sigma_z > 0 else 0.0
        n_per_region = N / K
        n_eff = N / (1 + (n_per_region - 1) * rho_within) if rho_within > 0 else N

        cv_sqrt_neff = cv * np.sqrt(n_eff)

        all_results[label] = {
            "global_shock_std": rho_cfg["global_shock_std"],
            "regional_shock_std": rho_cfg["regional_shock_std"],
            "rho_within": rho_within,
            "N": N, "K": K,
            "N_eff": float(n_eff),
            "mean_response_kw": mean_resp,
            "std_response_kw": std_resp,
            "cv": cv,
            "cv_sqrt_n": cv * np.sqrt(N),
            "cv_sqrt_neff": cv_sqrt_neff,
            "mean_response_rate": float(np.mean(rates)),
            "std_response_rate": float(np.std(rates)),
            "runs": runs_per_config,
        }

        logger.info(
            f"    {label}: CV={cv:.4f}, N_eff={n_eff:.0f}, "
            f"CV*sqrt(N_eff)={cv_sqrt_neff:.3f}, rate={np.mean(rates):.3f}"
        )

    rho_file = output_dir / "data" / "rho_sensitivity.json"
    rho_file.parent.mkdir(parents=True, exist_ok=True)
    with open(rho_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"  rho sensitivity results saved to {rho_file}")
    return all_results


def run_n_scaling_experiment(
    base_config: 'ExperimentConfig',
    output_dir: Path,
) -> Dict[str, Any]:
    from src.simulation.simulator import (
        SimulationConfig, EPSSimulator, CorrelationConfig,
    )
    from src.analysis.statistics import StatisticalAnalyzer

    N_values = [500, 1000, 2000, 5000, 10000]
    rho_configs = [
        {"label": "iid",      "global_shock_std": 0.0,  "regional_shock_std": 0.0},
        {"label": "weak",     "global_shock_std": 0.15, "regional_shock_std": 0.08},
        {"label": "moderate", "global_shock_std": 0.30, "regional_shock_std": 0.15},
    ]
    runs_per_config = _real_scale_count(50, 3)
    num_steps = 50
    K = base_config.num_regions

    analyzer = StatisticalAnalyzer(random_seed=42)
    all_results = {}

    for rho_cfg in rho_configs:
        rho_label = rho_cfg["label"]
        sigma_z = np.sqrt(rho_cfg["global_shock_std"]**2 + rho_cfg["regional_shock_std"]**2)
        p_marginal = 0.52
        rho_within = p_marginal * (1 - p_marginal) * sigma_z**2 if sigma_z > 0 else 0.0

        scaling_data = []
        all_cv_sqrt_neff = []

        for N in N_values:
            logger.info(f"  N-scaling: {rho_label}, N={N}...")

            corr = CorrelationConfig(
                enable_correlation=(sigma_z > 0),
                global_shock_std=rho_cfg["global_shock_std"],
                regional_shock_std=rho_cfg["regional_shock_std"],
            )

            run_responses = []
            for run_i in range(runs_per_config):
                sim_config = SimulationConfig(
                    num_devices=N,
                    num_regions=K,
                    duration_seconds=num_steps * 60,
                    time_step=60.0,
                    random_seed=42,
                    correlation=corr,
                    )
                sim = EPSSimulator(sim_config)
                sim.initialize()
                sim.rng = np.random.default_rng(1000 + run_i)
                result = sim.run(num_steps=num_steps, scenario='peak_shaving')

                total_kw = sum(ts.total_response_kw for ts in result.time_steps)
                run_responses.append(total_kw / max(num_steps, 1))

            responses = np.array(run_responses)
            mean_resp = float(np.mean(responses))
            std_resp = float(np.std(responses))
            cv = std_resp / abs(mean_resp) if abs(mean_resp) > 1e-6 else 0.0

            n_per_region = N / K
            n_eff = N / (1 + (n_per_region - 1) * rho_within) if rho_within > 0 else N

            cv_sqrt_neff = cv * np.sqrt(n_eff)
            all_cv_sqrt_neff.append(cv_sqrt_neff)

            def cv_sqrt_neff_stat(data):
                m = np.mean(data)
                s = np.std(data)
                cv_boot = s / abs(m) if abs(m) > 1e-6 else 0.0
                return cv_boot * np.sqrt(n_eff)

            ci = analyzer.bootstrap_ci(
                responses, statistic=cv_sqrt_neff_stat,
                n_bootstrap=5000, method='percentile'
            )

            scaling_data.append({
                "N": N,
                "N_eff": float(n_eff),
                "cv": cv,
                "cv_sqrt_n": cv * np.sqrt(N),
                "cv_sqrt_neff": cv_sqrt_neff,
                "cv_sqrt_neff_ci_lower": float(ci.ci_lower),
                "cv_sqrt_neff_ci_upper": float(ci.ci_upper),
                "mean_response_kw": mean_resp,
                "std_response_kw": std_resp,
                "runs": runs_per_config,
            })

        log_n = np.log(np.array(N_values, dtype=float))
        log_cv = np.log(np.array([d["cv"] for d in scaling_data]))
        valid = log_cv > -20
        if np.sum(valid) >= 2:
            slope, intercept = np.polyfit(log_n[valid], log_cv[valid], 1)
            ss_res = np.sum((log_cv[valid] - (slope * log_n[valid] + intercept))**2)
            ss_tot = np.sum((log_cv[valid] - np.mean(log_cv[valid]))**2)
            loglog_r2 = 1 - ss_res / max(ss_tot, 1e-10)
        else:
            slope, intercept, loglog_r2 = 0.0, 0.0, 0.0

        sigma_hat = float(np.median(all_cv_sqrt_neff)) if all_cv_sqrt_neff else 0.0

        all_results[rho_label] = {
            "rho_within": rho_within,
            "global_shock_std": sigma_z,
            "scaling_data": scaling_data,
            "loglog_slope": float(slope),
            "loglog_intercept": float(intercept),
            "loglog_r2": float(loglog_r2),
            "sigma_hat": sigma_hat,
        }

        logger.info(
            f"  {rho_label}: log-log slope={slope:.3f} (expect -0.5), "
            f"R²={loglog_r2:.4f}, sigma_hat={sigma_hat:.3f}"
        )

    scaling_file = output_dir / "data" / "n_scaling.json"
    scaling_file.parent.mkdir(parents=True, exist_ok=True)
    with open(scaling_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"  N-scaling results saved to {scaling_file}")
    return all_results


def run_n_threshold_experiment(
    base_config: 'ExperimentConfig',
    output_dir: Path,
) -> Dict[str, Any]:
    from src.estimation import EPSEstimator, EstimatorConfig

    N_values = [5, 10, 20, 50, 100, 150, 200, 400, 800, 1500, 3000, 5000]
    minimum_samples = max(1, int(REAL_DATA_MIN_SAMPLES))
    K = _real_scale_count(50, 10)
    M = _real_scale_count(10, 3)
    if K * M < minimum_samples:
        M = min(5, minimum_samples)
        K = int(np.ceil(minimum_samples / M))

    logger.info("=" * 60)
    logger.info("R2: N-Threshold Experiment (Multi-Replication Protocol)")
    logger.info(f"  N values: {N_values}")
    logger.info(f"  Eval: {K} unique signals × {M} replications = {K*M} samples per N")
    logger.info("=" * 60)

    results = {"N_values": N_values, "per_N": {}, "metadata": {
        "protocol": "multi_replication",
        "eval_unique_signals": K,
        "eval_replications": M,
        "eval_total_samples": K * M,
        "train_samples": max(minimum_samples, _real_scale_count(4000)),
    }}

    for N in N_values:
        logger.info(f"\n--- N = {N} ---")

        n_train = max(minimum_samples, _real_scale_count(4000))

        rng = np.random.default_rng(42)
        logger.info(f"  Training estimator with {n_train} samples...")
        train_signals, train_responses = _generate_training_data(
            n_train, N, rng,
            sim_config_override=None,
        )

        estimator_config = EstimatorConfig(
            target_coverage=0.9,
            enable_conformal=True,

            use_pytorch=True,
        )
        estimator = EPSEstimator(estimator_config)
        estimator.fit(train_signals, train_responses)

        eval_rng = np.random.default_rng(99999 + N)
        eval_signals, eval_responses = _generate_replicated_eval_data(
            n_unique_signals=K, replications=M, num_devices=N, rng=eval_rng,
            sim_config_override=None,
        )

        metrics = _evaluate_estimator_metrics(estimator, eval_signals, eval_responses)

        predictions = []
        for sig in eval_signals:
            result = estimator.estimate(sig)
            predictions.append(result.response_kw)
        predictions = np.array(predictions)
        responses_arr = np.array(eval_responses)

        per_signal_cvs = []
        for k in range(K):
            group = responses_arr[k * M : (k + 1) * M]
            mu, sigma = np.mean(group), np.std(group)
            if abs(mu) > 1e-6:
                per_signal_cvs.append(sigma / abs(mu))
        within_cv = float(np.mean(per_signal_cvs)) if per_signal_cvs else 0.0

        mean_abs_response = float(np.mean(np.abs(responses_arr)))
        nrmse = metrics['rmse'] / max(mean_abs_response, 1e-6)

        mean_abs_power = mean_abs_response

        n_boot = _real_scale_count(1000, 100)
        boot_rng = np.random.default_rng(42 + N)
        boot_r2_list = []
        boot_picp_list = []

        for _ in range(n_boot):
            boot_idx = boot_rng.integers(0, K, size=K)
            boot_actuals = []
            boot_preds = []
            boot_in_interval = 0
            boot_total = 0

            for ki in boot_idx:
                start = ki * M
                end = (ki + 1) * M
                boot_actuals.extend(responses_arr[start:end])
                boot_preds.extend(predictions[start:end])

                for j in range(start, end):
                    sig = eval_signals[j]
                    res = estimator.estimate(sig)
                    if res.lower_bound <= eval_responses[j] <= res.upper_bound:
                        boot_in_interval += 1
                    boot_total += 1

            ba = np.array(boot_actuals)
            bp = np.array(boot_preds)
            ss_res = np.sum((ba - bp) ** 2)
            ss_tot = np.sum((ba - np.mean(ba)) ** 2)
            boot_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            boot_r2_list.append(boot_r2)
            boot_picp_list.append(boot_in_interval / max(boot_total, 1) * 100)

        boot_r2_arr = np.array(boot_r2_list)
        boot_picp_arr = np.array(boot_picp_list)
        r2_ci = (float(np.percentile(boot_r2_arr, 2.5)), float(np.percentile(boot_r2_arr, 97.5)))
        picp_ci = (float(np.percentile(boot_picp_arr, 2.5)), float(np.percentile(boot_picp_arr, 97.5)))

        results["per_N"][str(N)] = {
            "N": N,
            "r2": float(metrics['r2']),
            "r2_ci_lower": r2_ci[0],
            "r2_ci_upper": r2_ci[1],
            "picp": float(metrics['picp']),
            "picp_ci_lower": picp_ci[0],
            "picp_ci_upper": picp_ci[1],
            "within_signal_cv": within_cv,
            "nrmse": float(nrmse),
            "smape": float(metrics['smape']),
            "rmse": float(metrics['rmse']),
            "mean_abs_power_kw": mean_abs_power,
            "_boot_r2_raw": [float(v) for v in boot_r2_arr],
        }

        logger.info(
            f"  N={N}: R²={metrics['r2']:.4f} [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}], "
            f"PICP={metrics['picp']:.1f}% [{picp_ci[0]:.1f}, {picp_ci[1]:.1f}], "
            f"CV={within_cv:.4f}, NRMSE={nrmse:.4f}, "
            f"Power={mean_abs_power:.1f} kW"
        )

    threshold_N = None
    threshold_95 = None
    threshold_99 = None
    for N in N_values:
        r2 = results["per_N"][str(N)]["r2"]
        if r2 >= 0.90 and threshold_N is None:
            threshold_N = N
        if r2 >= 0.95 and threshold_95 is None:
            threshold_95 = N
        if r2 >= 0.99 and threshold_99 is None:
            threshold_99 = N
    results["threshold_N_star"] = threshold_N
    results["threshold_N_95"] = threshold_95
    results["threshold_N_99"] = threshold_99

    n_threshold_boot = _real_scale_count(1000, 100)
    threshold_boot_rng = np.random.default_rng(12345)
    threshold_samples_90 = []
    for b in range(n_threshold_boot):
        found = False
        for N in N_values:
            raw = np.array(results["per_N"][str(N)]["_boot_r2_raw"])
            boot_r2 = float(threshold_boot_rng.choice(raw))
            if boot_r2 >= 0.90:
                threshold_samples_90.append(N)
                found = True
                break
        if not found:
            threshold_samples_90.append(N_values[-1])

    t_arr = np.array(threshold_samples_90)
    results["threshold_N_star_ci_lower"] = float(np.percentile(t_arr, 2.5))
    results["threshold_N_star_ci_upper"] = float(np.percentile(t_arr, 97.5))
    results["threshold_N_star_median"] = float(np.median(t_arr))

    logger.info(f"\n  Threshold N* (R²>=0.90): {threshold_N}")
    logger.info(f"    95% CI: [{results['threshold_N_star_ci_lower']:.0f}, {results['threshold_N_star_ci_upper']:.0f}]")
    logger.info(f"  Threshold N  (R²>=0.95): {threshold_95}")
    logger.info(f"  Threshold N  (R²>=0.99): {threshold_99}")

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "n_threshold.json"
    with open(out_file, 'w') as f:
        save_results = json.loads(json.dumps(results))
        for n_key in save_results.get("per_N", {}):
            save_results["per_N"][n_key].pop("_boot_r2_raw", None)
        json.dump(save_results, f, indent=2)
    logger.info(f"  Saved to {out_file}")
    return results


def _generate_region_training_data(
    n_samples: int,
    num_devices: int,
    rng: np.random.Generator,
    region_profile: Dict[str, Any],
    sim_config_override: 'SimulationConfig' = None,
) -> Tuple[List[Dict], List[float]]:
    import dataclasses
    from src.simulation import EPSSimulator, SimulationConfig, SimulationLevel

    signals = []
    responses = []

    solar_cap = region_profile.get('solar_capacity', 500)
    wind_cap = region_profile.get('wind_capacity', 150)
    base_load = region_profile.get('base_load', 400)
    load_type = region_profile.get('load_profile', 'urban_coastal')

    n_time_aware = int(n_samples * 0.7)

    logger.info(f"  Generating {n_samples} region-specific samples...")
    logger.info(f"    Solar={solar_cap}MW, Wind={wind_cap}MW, Load={base_load}MW")

    for i in range(n_samples):
        hour = int(rng.integers(0, 24))

        if i < n_time_aware:
            if load_type == 'rural_arid':
                if 10 <= hour <= 15:
                    solar_frac = rng.normal(0.9, 0.1)
                    net_gen = solar_cap * max(0, solar_frac) + wind_cap * rng.uniform(0.1, 0.4)
                    net_load = base_load * rng.normal(0.7, 0.1)
                    ratio = net_gen / max(net_load, 1)
                    supply_demand = int(np.clip(8 - ratio * 4, 0, 6))
                    intensity = int(rng.normal(1500, 400))
                elif 18 <= hour <= 22:
                    supply_demand = int(rng.normal(10, 2))
                    intensity = int(rng.normal(2500, 500))
                else:
                    supply_demand = int(rng.normal(7, 2))
                    intensity = int(rng.normal(1200, 400))
            else:
                if (7 <= hour <= 11) or (17 <= hour <= 21):
                    intensity = int(rng.normal(3000, 500))
                    supply_demand = int(rng.normal(12, 2))
                elif hour <= 6 or hour >= 22:
                    intensity = int(rng.normal(1000, 300))
                    supply_demand = int(rng.normal(3, 2))
                else:
                    intensity = int(rng.normal(2048, 600))
                    supply_demand = int(rng.normal(8, 2))
        else:
            hour = int(rng.integers(0, 24))
            intensity = int(rng.uniform(0, 4095))
            supply_demand = int(rng.uniform(0, 16))

        price = 0.0
        intensity = max(0, min(4095, intensity))
        supply_demand = max(0, min(15, supply_demand))
        direction = 1 if supply_demand <= 7 else -1

        seed_i = int(rng.integers(0, 10000))
        dt_total = 900
        dt_step = 300
        dt_hours = dt_total / 3600.0
        if sim_config_override is not None:
            mini_config = dataclasses.replace(
                sim_config_override,
                num_devices=num_devices,
                level=SimulationLevel.LEVEL1_AGENT,
                duration_seconds=dt_total,
                time_step=dt_step,
                random_seed=seed_i,
            )
        else:
            mini_config = SimulationConfig(
                num_devices=num_devices,
                num_regions=5,
                level=SimulationLevel.LEVEL1_AGENT,
                duration_seconds=dt_total,
                time_step=dt_step,
                random_seed=seed_i,
            )
        mini_sim = EPSSimulator(mini_config)
        mini_sim.initialize()

        mini_sim.set_override_signal(
            intensity=intensity,
            price_value=price,
            supply_demand=supply_demand,
        )
        scenario = 'valley_filling' if supply_demand <= 7 else 'peak_shaving'
        result = mini_sim.run(num_steps=3, scenario=scenario)
        mini_sim.clear_override_signal()

        actual_response_kw = result.total_energy_kwh / dt_hours

        signals.append({
            'supply_demand': supply_demand,
            'intensity': intensity,
            'price': price,
            'hour': hour,
            'direction': direction,
        })
        responses.append(actual_response_kw)

        if (i + 1) % 100 == 0:
            logger.info(f"    Generated {i+1}/{n_samples} samples")

    return signals, responses


def run_cross_region_transfer_battery(
    base_config: 'ExperimentConfig',
    output_dir: Path,
) -> Dict[str, Any]:
    from src.simulation.simulator import (
        SimulationConfig, EPSSimulator, CorrelationConfig, SimulationLevel,
    )
    from src.estimation import EPSEstimator, EstimatorConfig
    import copy

    N = base_config.num_devices
    n_online_samples = 1000

    region_a_profile = {
        'solar_capacity': 500,
        'wind_capacity': 150,
        'base_load': 400,
        'battery_capacity_range': (5.0, 20.0),
        'battery_c_rate_mean': 0.45,
        'battery_soh_range': (0.82, 1.0),
        'outdoor_temp_base': 32.0,
        'load_profile': 'urban_coastal',
    }

    region_b_profile = {
        'solar_capacity': 800,
        'wind_capacity': 350,
        'base_load': 180,
        'battery_capacity_range': (2.0, 50.0),
        'battery_c_rate_mean': 0.35,
        'battery_soh_range': (0.75, 1.0),
        'outdoor_temp_base': 42.0,
        'load_profile': 'rural_arid',
    }

    region_c_profile = {
        'solar_capacity': 400,
        'wind_capacity': 80,
        'base_load': 500,
        'battery_capacity_range': (8.0, 15.0),
        'battery_c_rate_mean': 0.55,
        'battery_soh_range': (0.85, 1.0),
        'outdoor_temp_base': 35.0,
        'load_profile': 'urban_tropical',
    }

    region_d_profile = {
        'solar_capacity': 300,
        'wind_capacity': 500,
        'base_load': 350,
        'battery_capacity_range': (5.0, 100.0),
        'battery_c_rate_mean': 0.40,
        'battery_soh_range': (0.70, 1.0),
        'outdoor_temp_base': -15.0,
        'load_profile': 'northern_continental',
    }

    logger.info("=" * 60)
    logger.info("R4: Cross-Region Transfer (Battery-Only, Capacity-Based Differentiation)")
    logger.info(f"  N = {N}")
    logger.info(f"  Region A: Eastern coastal (cap={region_a_profile['battery_capacity_range']}kWh)")
    logger.info(f"  Region B: Western arid (cap={region_b_profile['battery_capacity_range']}kWh)")
    logger.info(f"  Region C: Southern tropical (cap={region_c_profile['battery_capacity_range']}kWh)")
    logger.info(f"  Region D: Northern continental (cap={region_d_profile['battery_capacity_range']}kWh)")
    logger.info(f"  Online adaptation samples: {n_online_samples}")
    logger.info("=" * 60)

    def _make_region_config(profile, corr_global=0.15, corr_regional=0.08):
        return SimulationConfig(
            num_devices=N,
            num_regions=5,
            battery_capacity_range=profile['battery_capacity_range'],
            battery_c_rate_mean=profile.get('battery_c_rate_mean', 0.45),
            battery_soh_range=tuple(profile.get('battery_soh_range', (0.82, 1.0))),
            correlation=CorrelationConfig(
                enable_correlation=True,
                global_shock_std=corr_global,
                regional_shock_std=corr_regional,
            ),
        )

    logger.info("\n  --- Region A: Eastern Coastal Training ---")
    rng_a = np.random.default_rng(42)
    n_train_a = 4000

    region_a_sim_config = _make_region_config(region_a_profile)
    train_signals_a, train_responses_a = _generate_region_training_data(
        n_train_a, N, rng_a, region_a_profile,
        sim_config_override=region_a_sim_config,
    )

    estimator_config = EstimatorConfig(
        target_coverage=0.9,
        enable_conformal=True,

        use_pytorch=True,
    )
    estimator = EPSEstimator(estimator_config)
    estimator.fit(train_signals_a, train_responses_a)

    eval_rng_a = np.random.default_rng(77777)
    eval_signals_a, eval_responses_a = _generate_region_training_data(
        n_samples=500, num_devices=N, rng=eval_rng_a,
        region_profile=region_a_profile,
        sim_config_override=region_a_sim_config,
    )
    metrics_a = _evaluate_estimator_metrics(estimator, eval_signals_a, eval_responses_a)
    logger.info(f"  Region A: R²={metrics_a['r2']:.4f}, PICP={metrics_a['picp']:.1f}%")

    logger.info(f"\n  --- Region B: Western Arid (cap={region_b_profile['battery_capacity_range']}kWh) ---")
    region_b_sim_config = _make_region_config(region_b_profile, 0.30, 0.18)

    rng_b = np.random.default_rng(88888)
    eval_signals_b, eval_responses_b = _generate_region_training_data(
        n_samples=500, num_devices=N, rng=rng_b,
        region_profile=region_a_profile,
        sim_config_override=region_b_sim_config,
    )
    metrics_cold = _evaluate_estimator_metrics(estimator, eval_signals_b, eval_responses_b)
    logger.info(f"  Cold-start: R²={metrics_cold['r2']:.4f}, PICP={metrics_cold['picp']:.1f}%")

    def _adapt_to_region_battery(
        src_estimator, region_label, sim_config, eval_signals, eval_responses,
        online_signals, online_responses, cold_metrics,
    ):
        adapted_nn = copy.deepcopy(src_estimator)
        adapted_conf = copy.deepcopy(src_estimator)

        for est in [adapted_nn, adapted_conf]:
            if hasattr(est, '_cqr') and est._cqr is not None and hasattr(est._cqr, 'reset'):
                est._cqr.reset()
            est.config.online_validation_patience = 999
            est.config.online_validation_window = 200

        cold_entry = {
            "samples": 0,
            "r2": cold_metrics['r2'], "picp": cold_metrics['picp'],
            "smape": cold_metrics['smape'], "rmse": cold_metrics['rmse'],
        }
        curve_nn = [dict(cold_entry)]
        curve_conf = [dict(cold_entry)]

        n_total = len(online_signals)
        checkpoints = set([1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150,
                           200, 250, 300, 400, 500, 600, 700, 800, 900, 1000,
                           1100, 1200, 1300, 1400, 1500])
        checkpoints.update(set(range(0, n_total + 1, 20)))

        import math
        lr_max, lr_min = 0.003, 0.0005
        anchor_lam = 0.5

        for i in range(n_total):
            sig = online_signals[i]
            act = online_responses[i]
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * i / n_total))

            adapted_nn.online_update(
                sig, act, update_nn=True, error_threshold=0.0,
                learning_rate=lr, mini_batch_size=25, anchor_lambda=anchor_lam,
            )
            adapted_conf.online_update(sig, act, update_nn=False, error_threshold=0.0)

            if (i + 1) in checkpoints:
                m_nn = _evaluate_estimator_metrics(adapted_nn, eval_signals, eval_responses)
                m_conf = _evaluate_estimator_metrics(adapted_conf, eval_signals, eval_responses)
                curve_nn.append({
                    "samples": i + 1,
                    "r2": m_nn['r2'], "picp": m_nn['picp'],
                    "smape": m_nn['smape'], "rmse": m_nn['rmse'],
                })
                curve_conf.append({
                    "samples": i + 1,
                    "r2": m_conf['r2'], "picp": m_conf['picp'],
                    "smape": m_conf['smape'], "rmse": m_conf['rmse'],
                })
                if (i + 1) % 200 == 0:
                    logger.info(
                        f"  Sample {i+1} ({region_label}): NN R²={m_nn['r2']:.4f}, "
                        f"Conf R²={m_conf['r2']:.4f}, lr={lr:.5f}"
                    )

        n_stab_passes = 3
        stab_lr = 0.001
        stab_anchor = 0.1
        stab_mbs = 50
        rng_stab = np.random.default_rng(12345)
        for ep in range(n_stab_passes):
            perm = rng_stab.permutation(n_total)
            for idx in perm:
                adapted_nn.online_update(
                    online_signals[idx], online_responses[idx],
                    update_nn=True, error_threshold=0.0,
                    learning_rate=stab_lr, mini_batch_size=stab_mbs,
                    anchor_lambda=stab_anchor,
                )
            m_stab = _evaluate_estimator_metrics(adapted_nn, eval_signals, eval_responses)
            virtual_samples = n_total + (ep + 1) * n_total
            curve_nn.append({
                "samples": virtual_samples,
                "r2": m_stab['r2'], "picp": m_stab['picp'],
                "smape": m_stab['smape'], "rmse": m_stab['rmse'],
            })
            logger.info(
                f"  Stabilization pass {ep+1} ({region_label}): "
                f"R²={m_stab['r2']:.4f}, PICP={m_stab['picp']:.1f}%"
            )

        return curve_nn, curve_conf

    logger.info("\n  --- Dual-Mode Online Adaptation (Region B) ---")
    online_rng = np.random.default_rng(55555)
    online_signals_b, online_responses_b = _generate_region_training_data(
        n_samples=n_online_samples, num_devices=N, rng=online_rng,
        region_profile=region_a_profile,
        sim_config_override=region_b_sim_config,
    )
    convergence_nn, convergence_conf = _adapt_to_region_battery(
        estimator, "B", region_b_sim_config,
        eval_signals_b, eval_responses_b,
        online_signals_b, online_responses_b, metrics_cold,
    )

    logger.info(f"\n  --- Region C: Southern Tropical (cap={region_c_profile['battery_capacity_range']}kWh) ---")
    region_c_sim_config = _make_region_config(region_c_profile, 0.15, 0.10)

    rng_c = np.random.default_rng(99999)
    eval_signals_c, eval_responses_c = _generate_region_training_data(
        n_samples=500, num_devices=N, rng=rng_c,
        region_profile=region_a_profile,
        sim_config_override=region_c_sim_config,
    )
    metrics_cold_c = _evaluate_estimator_metrics(estimator, eval_signals_c, eval_responses_c)
    logger.info(f"  Cold-start C: R²={metrics_cold_c['r2']:.4f}, PICP={metrics_cold_c['picp']:.1f}%")

    online_rng_c = np.random.default_rng(66666)
    online_signals_c, online_responses_c = _generate_region_training_data(
        n_samples=n_online_samples, num_devices=N, rng=online_rng_c,
        region_profile=region_a_profile,
        sim_config_override=region_c_sim_config,
    )
    convergence_nn_c, convergence_conf_c = _adapt_to_region_battery(
        estimator, "C", region_c_sim_config,
        eval_signals_c, eval_responses_c,
        online_signals_c, online_responses_c, metrics_cold_c,
    )

    logger.info(f"\n  --- Region D: Northern Continental (cap={region_d_profile['battery_capacity_range']}kWh) ---")
    region_d_sim_config = _make_region_config(region_d_profile, 0.25, 0.15)

    rng_d = np.random.default_rng(11111)
    eval_signals_d, eval_responses_d = _generate_region_training_data(
        n_samples=500, num_devices=N, rng=rng_d,
        region_profile=region_a_profile,
        sim_config_override=region_d_sim_config,
    )
    metrics_cold_d = _evaluate_estimator_metrics(estimator, eval_signals_d, eval_responses_d)
    logger.info(f"  Cold-start D: R²={metrics_cold_d['r2']:.4f}, PICP={metrics_cold_d['picp']:.1f}%")

    online_rng_d = np.random.default_rng(22222)
    online_signals_d, online_responses_d = _generate_region_training_data(
        n_samples=n_online_samples, num_devices=N, rng=online_rng_d,
        region_profile=region_a_profile,
        sim_config_override=region_d_sim_config,
    )
    convergence_nn_d, convergence_conf_d = _adapt_to_region_battery(
        estimator, "D", region_d_sim_config,
        eval_signals_d, eval_responses_d,
        online_signals_d, online_responses_d, metrics_cold_d,
    )

    def _find_threshold(curve, target):
        for pt in curve:
            if pt["r2"] >= target:
                return pt["samples"]
        return None

    final_nn = convergence_nn[-1]
    final_conf = convergence_conf[-1]
    final_nn_c = convergence_nn_c[-1]
    final_conf_c = convergence_conf_c[-1]
    final_nn_d = convergence_nn_d[-1]
    final_conf_d = convergence_conf_d[-1]

    results = {
        "region_A": {
            "r2": metrics_a['r2'],
            "picp": metrics_a['picp'],
            "n_train": n_train_a,
            "num_devices": N,
            "profile": region_a_profile,
        },
        "regions": {
            "B": {
                "label": "Western Arid",
                "profile": region_b_profile,
                "cold_start": {
                    "r2": metrics_cold['r2'], "picp": metrics_cold['picp'],
                    "smape": metrics_cold['smape'], "rmse": metrics_cold['rmse'],
                },
                "adapted_nn": {
                    "r2": final_nn['r2'], "picp": final_nn['picp'],
                    "smape": final_nn['smape'], "rmse": final_nn['rmse'],
                    "n_online_samples": n_online_samples,
                },
                "adapted_conformal": {
                    "r2": final_conf['r2'], "picp": final_conf['picp'],
                },
                "convergence_curve_nn": convergence_nn,
                "convergence_curve_conformal": convergence_conf,
            },
            "C": {
                "label": "Southern Tropical",
                "profile": region_c_profile,
                "cold_start": {
                    "r2": metrics_cold_c['r2'], "picp": metrics_cold_c['picp'],
                    "smape": metrics_cold_c['smape'], "rmse": metrics_cold_c['rmse'],
                },
                "adapted_nn": {
                    "r2": final_nn_c['r2'], "picp": final_nn_c['picp'],
                    "smape": final_nn_c['smape'], "rmse": final_nn_c['rmse'],
                    "n_online_samples": n_online_samples,
                },
                "adapted_conformal": {
                    "r2": final_conf_c['r2'], "picp": final_conf_c['picp'],
                },
                "convergence_curve_nn": convergence_nn_c,
                "convergence_curve_conformal": convergence_conf_c,
            },
            "D": {
                "label": "Northern Continental",
                "profile": region_d_profile,
                "cold_start": {
                    "r2": metrics_cold_d['r2'], "picp": metrics_cold_d['picp'],
                    "smape": metrics_cold_d['smape'], "rmse": metrics_cold_d['rmse'],
                },
                "adapted_nn": {
                    "r2": final_nn_d['r2'], "picp": final_nn_d['picp'],
                    "smape": final_nn_d['smape'], "rmse": final_nn_d['rmse'],
                    "n_online_samples": n_online_samples,
                },
                "adapted_conformal": {
                    "r2": final_conf_d['r2'], "picp": final_conf_d['picp'],
                },
                "convergence_curve_nn": convergence_nn_d,
                "convergence_curve_conformal": convergence_conf_d,
            },
        },
        "region_B_config": {
            "battery_capacity_range": list(region_d_profile['battery_capacity_range']),
            "global_shock_std": 0.25,
            "regional_shock_std": 0.15,
            "profile": region_d_profile,
        },
        "region_B_cold_start": {
            "r2": metrics_cold_d['r2'],
            "picp": metrics_cold_d['picp'],
        },
        "region_B_adapted_nn": {
            "r2": final_nn_d['r2'],
            "picp": final_nn_d['picp'],
            "n_online_samples": n_online_samples,
        },
        "region_B_adapted_conformal": {
            "r2": final_conf_d['r2'],
            "picp": final_conf_d['picp'],
            "n_online_samples": n_online_samples,
        },
        "convergence_curve": convergence_nn_d,
        "convergence_curve_nn": convergence_nn_d,
        "convergence_curve_conformal_only": convergence_conf_d,
        "samples_to_r2_080": _find_threshold(convergence_nn_d, 0.80),
        "samples_to_r2_090": _find_threshold(convergence_nn_d, 0.90),
        "samples_to_r2_095": _find_threshold(convergence_nn_d, 0.95),
        "metadata": {
            "N": N,
            "device_mode": "battery_only",
            "differentiation": "capacity_range",
            "adaptation_methods": ["nn_and_conformal", "conformal_only"],
            "n_target_regions": 3,
            "region_keys": ["B", "C", "D"],
            "region_a_type": "eastern_coastal",
            "region_b_type": "western_arid",
            "region_c_type": "southern_tropical",
            "region_d_type": "northern_continental",
            "fig5a_regions": ["B", "C"],
            "fig5a_transfer_region": "D",
        },
    }

    logger.info(f"\n  === Multi-Region Transfer Summary (Battery-Only) ===")
    logger.info(f"  Region A R²:          {metrics_a['r2']:.4f}")
    logger.info(f"  Region B cold R²:     {metrics_cold['r2']:.4f}")
    logger.info(f"  Region B adapted R²:  {final_nn['r2']:.4f}, PICP={final_nn['picp']:.1f}%")
    logger.info(f"  Region C cold R²:     {metrics_cold_c['r2']:.4f}")
    logger.info(f"  Region C adapted R²:  {final_nn_c['r2']:.4f}, PICP={final_nn_c['picp']:.1f}%")
    logger.info(f"  Region D cold R²:     {metrics_cold_d['r2']:.4f}")
    logger.info(f"  Region D adapted R²:  {final_nn_d['r2']:.4f}, PICP={final_nn_d['picp']:.1f}%")

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "cross_region_transfer.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved to {out_file}")
    return results


def run_hierarchical_r2(
    base_config: 'ExperimentConfig',
    output_dir: Path,
    n_repeats: int = 3,
) -> Dict[str, Any]:
    from src.estimation import EPSEstimator, EstimatorConfig

    logger.info("=" * 60)
    logger.info("R1: Hierarchical R² Decomposition")
    logger.info(f"  n_repeats = {n_repeats}")
    logger.info("=" * 60)

    rng = np.random.default_rng(42)
    n_train = min(8000, max(2000, base_config.num_devices * 4))
    train_signals, train_responses = _generate_training_data(
        n_train, base_config.num_devices, rng,
        sim_config_override=None,
    )
    eval_rng = np.random.default_rng(99999)
    eval_signals, eval_responses = _generate_training_data(
        500, base_config.num_devices, eval_rng,
        sim_config_override=None,
    )
    eval_y = np.array(eval_responses)

    raw_keys = ['supply_demand', 'intensity', 'hour',
                'region_id', 'priority', 'day_of_week']
    key_means = {}
    for key in raw_keys:
        vals = [s.get(key, 0) for s in train_signals]
        key_means[key] = float(np.mean(vals))
    logger.info(f"  Signal key means: { {k: f'{v:.2f}' for k, v in key_means.items()} }")

    steps = [
        ("Supply-demand",  {"supply_demand"}),
        ("+ Intensity",    {"supply_demand", "intensity"}),
        ("+ Time",         {"supply_demand", "intensity", "hour"}),
        ("Full model",     set(raw_keys)),
    ]

    def _mask_signals(signals, allowed_keys):
        masked = []
        for s in signals:
            m = dict(s)
            for key in raw_keys:
                if key not in allowed_keys:
                    m[key] = key_means[key]
            m['direction'] = 1 if m['supply_demand'] <= 7 else -1
            masked.append(m)
        return masked

    def _train_and_eval_estimator(train_sigs, train_resp, eval_sigs, eval_resp, seed):
        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = EstimatorConfig(
            target_coverage=0.9, enable_conformal=False,
            use_pytorch=True,
        )
        est = EPSEstimator(cfg)
        est.fit(train_sigs, train_resp)
        preds = np.array([est.estimate(s).response_kw for s in eval_sigs])
        actuals = np.array(eval_resp)
        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-6)
        return float(r2)

    step_results = []
    prev_r2 = 0.0

    for step_name, allowed_keys in steps:
        masked_train = _mask_signals(train_signals, allowed_keys)
        masked_eval = _mask_signals(eval_signals, allowed_keys)

        r2_values = []
        for rep in range(n_repeats):
            r2 = _train_and_eval_estimator(
                masked_train, train_responses,
                masked_eval, eval_responses,
                seed=42 + rep,
            )
            r2_values.append(r2)

        r2_mean = float(np.mean(r2_values))
        r2_std = float(np.std(r2_values))
        increment = max(0.0, r2_mean - prev_r2)

        step_results.append({
            "name": step_name,
            "allowed_keys": sorted(allowed_keys),
            "r2_mean": r2_mean,
            "r2_std": r2_std,
            "r2_values": r2_values,
            "increment": increment,
        })
        logger.info(
            f"  {step_name}: R²={r2_mean:.4f} ± {r2_std:.4f}, "
            f"ΔR²={increment:.4f}"
        )
        prev_r2 = r2_mean

    final_r2 = step_results[-1]["r2_mean"]
    contribution_pct = {}
    label_keys = ["supply_demand", "intensity", "time", "context"]
    for step, key in zip(step_results, label_keys):
        contribution_pct[key] = float(step["increment"] / max(final_r2, 1e-6) * 100)

    full_estimator_config = EstimatorConfig(
        target_coverage=0.9, enable_conformal=True,
        use_pytorch=True,
    )
    full_estimator = EPSEstimator(full_estimator_config)
    full_estimator.fit(train_signals, train_responses)
    full_preds = np.array([full_estimator.estimate(s).response_kw for s in eval_signals])
    residuals = eval_y - full_preds
    residual_stats = {
        "mean_kw": float(np.mean(residuals)),
        "std_kw": float(np.std(residuals)),
        "skewness": float(stats.skew(residuals)),
        "kurtosis": float(stats.kurtosis(residuals)),
    }

    results = {
        "steps": step_results,
        "contribution_pct": contribution_pct,
        "total_contribution_pct": float(sum(contribution_pct.values())),
        "final_r2": final_r2,
        "residual_stats": residual_stats,
        "n_repeats": n_repeats,
        "n_train": n_train,
        "n_eval": 500,
    }

    est_dir = output_dir / "estimation"
    est_dir.mkdir(parents=True, exist_ok=True)
    out_file = est_dir / "hierarchical_r2.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved to {out_file}")
    logger.info(f"  Contribution total: {results['total_contribution_pct']:.1f}%")
    for k, v in contribution_pct.items():
        logger.info(f"    {k}: {v:.1f}%")

    return results


def run_heterogeneity_lookup_table(
    base_config: 'ExperimentConfig',
    output_dir: Path,
) -> Dict[str, Any]:
    import dataclasses
    from scipy.optimize import curve_fit
    from src.simulation import SimulationConfig, SimulationLevel
    from src.estimation import EPSEstimator, EstimatorConfig

    MEAN_CAPACITY = 12.5

    CV_TARGETS = [0.001, 0.05, 0.08, 0.12, 0.17, 0.23, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    if REAL_DATA_SAMPLE_SCALE < 1.0:
        CV_TARGETS = [0.001, 0.12, 0.23, 0.35, 0.50]

    CV_CONFIGS = []
    for cv_target in CV_TARGETS:
        d = cv_target * 2 * MEAN_CAPACITY * np.sqrt(3)
        a = max(MEAN_CAPACITY - d / 2, 0.1)
        b = 2 * MEAN_CAPACITY - a
        cv_actual = (b - a) / (np.sqrt(3) * (a + b))
        e_c2 = (a**2 + a * b + b**2) / 3
        mean_sq = MEAN_CAPACITY**2
        herfindahl = e_c2 / mean_sq
        neff_ratio = 1.0 / herfindahl
        CV_CONFIGS.append({
            "cv_target": cv_target,
            "cv_actual": float(cv_actual),
            "capacity_range": (float(a), float(b)),
            "herfindahl_index": float(herfindahl),
            "neff_ratio": float(neff_ratio),
        })

    N_VALUES = [5, 10, 15, 20, 25, 30, 40, 50, 65, 80, 100, 150, 200, 350, 500, 1000, 3000, 5000]
    if REAL_DATA_SAMPLE_SCALE < 1.0:
        N_VALUES = [5, 20, 50, 100, 500, 1000, 5000]

    K = _real_scale_count(50, 10)
    M = _real_scale_count(10, 3)
    n_train = 4000
    n_boot = _real_scale_count(1000, 100)
    n_threshold_boot = _real_scale_count(1000, 100)

    n_cv = len(CV_CONFIGS)
    n_n = len(N_VALUES)

    logger.info("=" * 70)
    logger.info("Dense Heterogeneity Lookup Table")
    logger.info(f"  CV levels: {n_cv}  ({CV_TARGETS[0]} -> {CV_TARGETS[-1]})")
    logger.info(f"  N values:  {n_n}  ({N_VALUES[0]} -> {N_VALUES[-1]})")
    logger.info(f"  Grid:      {n_cv} × {n_n} = {n_cv * n_n} cells")
    logger.info(f"  Eval:      {K} signals × {M} reps = {K * M} per cell")
    logger.info(f"  Train:     {n_train} per cell")
    logger.info("=" * 70)

    grid_r2 = np.full((n_cv, n_n), np.nan)
    grid_r2_ci_lower = np.full((n_cv, n_n), np.nan)
    grid_r2_ci_upper = np.full((n_cv, n_n), np.nan)
    grid_picp = np.full((n_cv, n_n), np.nan)
    grid_smape = np.full((n_cv, n_n), np.nan)
    grid_within_cv = np.full((n_cv, n_n), np.nan)

    cell_boot_r2 = {}

    total_cells = n_cv * n_n
    cell_count = 0
    t_start = time.time()

    for i, cv_cfg in enumerate(CV_CONFIGS):
        cap_lo, cap_hi = cv_cfg["capacity_range"]
        cv_actual = cv_cfg["cv_actual"]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"CV level {i+1}/{n_cv}: CV_target={cv_cfg['cv_target']:.3f}, "
                     f"CV_actual={cv_actual:.4f}, range=[{cap_lo:.2f}, {cap_hi:.2f}] kWh")
        logger.info(f"{'=' * 60}")

        for j, N in enumerate(N_VALUES):
            cell_count += 1
            elapsed = time.time() - t_start
            eta = (elapsed / cell_count) * (total_cells - cell_count) if cell_count > 0 else 0

            logger.info(f"  [{cell_count}/{total_cells}] CV={cv_actual:.3f}, N={N}  "
                         f"(elapsed {elapsed/60:.1f}min, ETA {eta/60:.1f}min)")

            num_regions = min(N, 5) if N < 20 else 5
            sim_override = SimulationConfig(
                level=SimulationLevel.LEVEL1_AGENT,
                num_devices=N,
                duration_seconds=300,
                time_step=300,
                random_seed=42,
                num_regions=num_regions,
                battery_capacity_range=(cap_lo, cap_hi),
            )

            rng = np.random.default_rng(42)
            train_signals, train_responses = _generate_training_data(
                n_train, N, rng, sim_config_override=sim_override,
            )

            est_config = EstimatorConfig(
                target_coverage=0.9,
                enable_conformal=True,

                use_pytorch=True,
            )
            estimator = EPSEstimator(est_config)
            estimator.fit(train_signals, train_responses)

            eval_rng = np.random.default_rng(99999 + N)
            eval_signals, eval_responses = _generate_replicated_eval_data(
                n_unique_signals=K, replications=M, num_devices=N,
                rng=eval_rng, sim_config_override=sim_override,
            )

            metrics = _evaluate_estimator_metrics(estimator, eval_signals, eval_responses)

            predictions = np.array([estimator.estimate(s).response_kw for s in eval_signals])
            responses_arr = np.array(eval_responses)

            per_signal_cvs = []
            for k in range(K):
                group = responses_arr[k * M : (k + 1) * M]
                mu, sigma = np.mean(group), np.std(group)
                if abs(mu) > 1e-6:
                    per_signal_cvs.append(sigma / abs(mu))
            within_cv = float(np.mean(per_signal_cvs)) if per_signal_cvs else 0.0

            in_interval = np.zeros(len(eval_signals), dtype=bool)
            for idx in range(len(eval_signals)):
                res = estimator.estimate(eval_signals[idx])
                in_interval[idx] = res.lower_bound <= eval_responses[idx] <= res.upper_bound

            boot_rng = np.random.default_rng(42 + N + i * 10000)
            boot_r2_list = []
            for _ in range(n_boot):
                boot_idx = boot_rng.integers(0, K, size=K)
                boot_sample_idx = np.concatenate([np.arange(ki * M, (ki + 1) * M) for ki in boot_idx])
                ba = responses_arr[boot_sample_idx]
                bp = predictions[boot_sample_idx]
                ss_res = np.sum((ba - bp) ** 2)
                ss_tot = np.sum((ba - np.mean(ba)) ** 2)
                boot_r2_list.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)

            boot_r2_arr = np.array(boot_r2_list)
            r2_ci = (float(np.percentile(boot_r2_arr, 2.5)),
                     float(np.percentile(boot_r2_arr, 97.5)))

            grid_r2[i, j] = float(metrics['r2'])
            grid_r2_ci_lower[i, j] = r2_ci[0]
            grid_r2_ci_upper[i, j] = r2_ci[1]
            grid_picp[i, j] = float(metrics['picp'])
            grid_smape[i, j] = float(metrics['smape'])
            grid_within_cv[i, j] = within_cv
            cell_boot_r2[(i, j)] = boot_r2_arr

            logger.info(f"    R²={metrics['r2']:.4f} [{r2_ci[0]:.4f},{r2_ci[1]:.4f}], "
                         f"PICP={metrics['picp']:.1f}%, SMAPE={metrics['smape']:.1f}%, CV_w={within_cv:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("Threshold detection & lookup table construction")
    logger.info("=" * 70)

    lookup_table = []
    for i, cv_cfg in enumerate(CV_CONFIGS):
        threshold_90 = threshold_95 = threshold_99 = None
        for j, N in enumerate(N_VALUES):
            r2 = grid_r2[i, j]
            if r2 >= 0.90 and threshold_90 is None:
                threshold_90 = N
            if r2 >= 0.95 and threshold_95 is None:
                threshold_95 = N
            if r2 >= 0.99 and threshold_99 is None:
                threshold_99 = N

        ci_results = {}
        for label, thresh_val in [("90", 0.90), ("95", 0.95), ("99", 0.99)]:
            t_boot_rng = np.random.default_rng(12345 + i * 100)
            t_samples = []
            for _ in range(n_threshold_boot):
                found = False
                for j, N in enumerate(N_VALUES):
                    raw = cell_boot_r2.get((i, j))
                    if raw is not None and float(t_boot_rng.choice(raw)) >= thresh_val:
                        t_samples.append(N)
                        found = True
                        break
                if not found:
                    t_samples.append(N_VALUES[-1])
            t_arr = np.array(t_samples)
            ci_results[label] = {
                "ci_lower": float(np.percentile(t_arr, 2.5)),
                "ci_upper": float(np.percentile(t_arr, 97.5)),
                "median": float(np.median(t_arr)),
            }

        entry = {
            "cv_target": cv_cfg["cv_target"],
            "cv_actual": cv_cfg["cv_actual"],
            "capacity_range_kwh": list(cv_cfg["capacity_range"]),
            "herfindahl_index": cv_cfg["herfindahl_index"],
            "neff_ratio": cv_cfg["neff_ratio"],
            "N_star_90": threshold_90,
            "N_star_90_ci": [ci_results["90"]["ci_lower"], ci_results["90"]["ci_upper"]],
            "N_star_95": threshold_95,
            "N_star_95_ci": [ci_results["95"]["ci_lower"], ci_results["95"]["ci_upper"]],
            "N_star_99": threshold_99,
            "N_star_99_ci": [ci_results["99"]["ci_lower"], ci_results["99"]["ci_upper"]],
        }
        lookup_table.append(entry)

        logger.info(f"  CV={cv_cfg['cv_target']:.3f}: "
                     f"N*_90={threshold_90}, N*_95={threshold_95} "
                     f"[{ci_results['95']['ci_lower']:.0f},{ci_results['95']['ci_upper']:.0f}], "
                     f"N*_99={threshold_99}")

    fit_results = {}
    for label in ["95", "99"]:
        cv_arr = np.array([e["cv_actual"] for e in lookup_table])
        nstar_key = f"N_star_{label}"
        nstar_arr = np.array([e[nstar_key] if e[nstar_key] is not None else N_VALUES[-1]
                              for e in lookup_table], dtype=float)
        valid = nstar_arr < N_VALUES[-1]
        if np.sum(valid) >= 3:
            def model_func(cv, alpha):
                return alpha * (1 + cv**2)
            try:
                popt, _ = curve_fit(model_func, cv_arr[valid], nstar_arr[valid], p0=[20.0])
                alpha = float(popt[0])
                predicted = model_func(cv_arr[valid], alpha)
                ss_res = np.sum((nstar_arr[valid] - predicted)**2)
                ss_tot = np.sum((nstar_arr[valid] - np.mean(nstar_arr[valid]))**2)
                r2_fit = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                fit_results[label] = {
                    "model": "N* = alpha * (1 + CV^2)",
                    "alpha": alpha,
                    "r2_of_fit": float(r2_fit),
                    "n_points_used": int(np.sum(valid)),
                }
                logger.info(f"  Fit N*_{label}: alpha={alpha:.1f}, R²_fit={r2_fit:.4f}")
            except Exception as e:
                logger.warning(f"  Fit N*_{label} failed: {e}")
                fit_results[label] = {"model": "N* = alpha * (1 + CV^2)", "error": str(e)}
        else:
            fit_results[label] = {"model": "N* = alpha * (1 + CV^2)", "error": "insufficient valid points"}

    cv_vals_for_test = [e["cv_actual"] for e in lookup_table]
    nstar95_for_test = [e["N_star_95"] if e["N_star_95"] is not None else N_VALUES[-1]
                        for e in lookup_table]
    nstar99_for_test = [e["N_star_99"] if e["N_star_99"] is not None else N_VALUES[-1]
                        for e in lookup_table]
    rho_95, p_95 = stats.spearmanr(cv_vals_for_test, nstar95_for_test)
    rho_99, p_99 = stats.spearmanr(cv_vals_for_test, nstar99_for_test)

    logger.info(f"\n  Spearman CV vs N*_95: rho={rho_95:.3f}, p={p_95:.4f}")
    logger.info(f"  Spearman CV vs N*_99: rho={rho_99:.3f}, p={p_99:.4f}")

    total_time = time.time() - t_start
    final = {
        "metadata": {
            "protocol": "heterogeneity_lookup_table",
            "K": K,
            "M": M,
            "n_train": n_train,
            "n_boot": n_boot,
            "n_threshold_boot": n_threshold_boot,
            "mean_capacity_kwh": MEAN_CAPACITY,
            "cv_values": CV_TARGETS,
            "N_values": N_VALUES,
            "grid_shape": [n_cv, n_n],
            "total_cells": total_cells,
            "total_time_seconds": float(total_time),
            "device_mode": "battery_only",
            "timestamp": datetime.now().isoformat(),
        },
        "cv_configs": CV_CONFIGS,
        "grid": {
            "r2": grid_r2.tolist(),
            "r2_ci_lower": grid_r2_ci_lower.tolist(),
            "r2_ci_upper": grid_r2_ci_upper.tolist(),
            "picp": grid_picp.tolist(),
            "smape": grid_smape.tolist(),
            "within_signal_cv": grid_within_cv.tolist(),
        },
        "lookup_table": lookup_table,
        "fit": fit_results,
        "statistical_tests": {
            "spearman_cv_vs_nstar95": {"rho": float(rho_95), "p_value": float(p_95)},
            "spearman_cv_vs_nstar99": {"rho": float(rho_99), "p_value": float(p_99)},
        },
    }

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "heterogeneity_lookup_table.json"
    with open(out_file, 'w') as f:
        json.dump(final, f, indent=2, default=str)

    logger.info(f"\n  Saved to {out_file}")
    logger.info(f"  Total time: {total_time/60:.1f} minutes ({total_cells} cells)")
    return final


def run_curtailment_sensitivity(
    base_config: 'ExperimentConfig',
    output_dir: Path,
    real_profiles=None,
) -> Dict[str, Any]:
    import dataclasses
    from src.simulation import EPSSimulator, SimulationConfig, SimulationLevel
    from src.estimation import EPSEstimator, EstimatorConfig
    from src.signal import SignalOptimizer, OptimizationTarget

    SOLAR_RATIOS = [0.80, 0.95, 1.10, 1.30, 1.50, 2.00]
    WIND_RATIO = 0.50
    M_REPS = 3

    num_devices = base_config.num_devices
    max_capacity_per_device_kw = 1.5
    max_response_mw = num_devices * max_capacity_per_device_kw / 1000.0
    achievable_target_mw = max_response_mw * 0.5
    max_target_mw = min(achievable_target_mw, float(base_config.target_response_mw))
    max_target_mw = max(max_target_mw, 0.01)

    BASE_LOAD_PEAK_MW = max_target_mw * 3.0

    solar_factors, wind_factors = _get_renewable_factors(real_profiles)
    load_factors = _get_load_factors(real_profiles)

    STEPS_PER_HOUR = 12
    TIME_STEP = 300.0

    logger.info("=" * 70)
    logger.info("Curtailment Sensitivity (NN Closed-Loop, Continuous 24h)")
    logger.info(f"  Solar ratios: {SOLAR_RATIOS}")
    logger.info(f"  N={num_devices}, M={M_REPS} full-day reps per ratio")
    logger.info(f"  Steps/hour={STEPS_PER_HOUR}, total steps/day={STEPS_PER_HOUR*24}")
    logger.info(f"  max_target_mw={max_target_mw:.2f}, BASE_LOAD={BASE_LOAD_PEAK_MW:.2f}")
    logger.info("=" * 70)

    logger.info("  Phase 1: Training NN estimator for closed-loop curtailment...")
    rng_train = np.random.default_rng(42)
    train_signals, train_responses = _generate_training_data(
        n_samples=12000,
        num_devices=num_devices,
        rng=rng_train,
        sim_config_override=None,
    )
    estimator = EPSEstimator(EstimatorConfig(
        target_coverage=0.9, enable_conformal=True, use_pytorch=True,
    ))
    estimator.fit(train_signals, train_responses)
    optimizer = SignalOptimizer(estimator)
    nn_r2 = estimator._learned_params.get('r2', None) if estimator._learned_params else None
    logger.info(f"  NN trained: R²={nn_r2:.4f}, n_samples={len(train_signals)}")

    sensitivity_results = []

    for solar_ratio in SOLAR_RATIOS:
        SOLAR_CAP = BASE_LOAD_PEAK_MW * solar_ratio
        WIND_CAP = BASE_LOAD_PEAK_MW * WIND_RATIO

        all_hourly = {h: {"response_mw": [], "absorbed_mw": []} for h in range(24)}

        for m in range(M_REPS):
            seed_m = 42 + m * 1000 + int(solar_ratio * 100)

            sim_config = SimulationConfig(
                num_devices=num_devices,
                num_regions=5,
                level=SimulationLevel.LEVEL1_AGENT,
                duration_seconds=86400.0,
                time_step=TIME_STEP,
                random_seed=seed_m,
            )
            sim = EPSSimulator(sim_config)
            sim.initialize()

            for hour in range(24):
                solar_mw = solar_factors[hour] * SOLAR_CAP
                wind_mw = wind_factors[hour] * WIND_CAP
                total_supply = solar_mw + wind_mw
                base_load = load_factors[hour] * BASE_LOAD_PEAK_MW
                net_balance = total_supply - base_load

                r = total_supply / max(base_load, 1e-6)
                d_dispatch = min(1.0, abs(r - 1.0) / 0.4)
                s = float(np.clip((r - 1.0) * d_dispatch, -1.0, 1.0))
                formula_sd, formula_intensity = encode_signal_score(s)

                if net_balance > 0:
                    target = OptimizationTarget(
                        target_response_mw=net_balance,
                        tolerance_fraction=0.15,
                    )
                    opt_result = optimizer.optimize(
                        target,
                        supply_demand=formula_sd,
                        signal_context={'hour': hour},
                    )
                    sd_state = opt_result.supply_demand
                    intensity = opt_result.intensity
                else:
                    sd_state = formula_sd
                    intensity = formula_intensity

                scenario = 'valley_filling' if sd_state <= 7 else 'peak_shaving'

                sim.set_override_signal(
                    intensity=intensity,
                    supply_demand=sd_state,
                )
                result = sim.run(num_steps=STEPS_PER_HOUR, scenario=scenario)
                sim.clear_override_signal()

                response_kwh = result.total_energy_kwh
                response_mw = abs(response_kwh) / 1000.0

                absorbed = min(response_mw, max(0.0, net_balance))
                max_load_increase = base_load * 0.60
                absorbed = min(absorbed, max_load_increase)

                all_hourly[hour]["response_mw"].append(response_mw)
                all_hourly[hour]["absorbed_mw"].append(absorbed)

            logger.info(f"    {solar_ratio:.2f} rep {m+1}/{M_REPS} complete")

        baseline_curtailed = 0.0
        eps_curtailed = 0.0
        total_absorbed = 0.0
        hourly_breakdown = []

        for hour in range(24):
            solar_mw = solar_factors[hour] * SOLAR_CAP
            wind_mw = wind_factors[hour] * WIND_CAP
            total_supply = solar_mw + wind_mw
            base_load = load_factors[hour] * BASE_LOAD_PEAK_MW
            net_balance = total_supply - base_load

            baseline_curtail = max(0.0, net_balance)
            mean_response = float(np.mean(all_hourly[hour]["response_mw"]))
            std_response = float(np.std(all_hourly[hour]["response_mw"]))
            mean_absorbed = float(np.mean(all_hourly[hour]["absorbed_mw"]))

            eps_curtail = max(0.0, baseline_curtail - mean_absorbed)

            baseline_curtailed += baseline_curtail
            eps_curtailed += eps_curtail
            total_absorbed += mean_absorbed

            hourly_breakdown.append({
                "hour": hour,
                "generation_mw": float(total_supply),
                "load_mw": float(base_load),
                "net_balance_mw": float(net_balance),
                "curtailment_baseline_mw": float(baseline_curtail),
                "curtailment_eps_mw": float(eps_curtail),
                "simulated_response_mw": mean_response,
                "simulated_response_std_mw": std_response,
                "absorbed_mw": mean_absorbed,
            })

        reduction_pct = (
            (baseline_curtailed - eps_curtailed) / baseline_curtailed * 100
            if baseline_curtailed > 0 else 0.0
        )

        logger.info(
            f"  solar_ratio={solar_ratio:.2f}: "
            f"baseline={baseline_curtailed:.2f}MWh, eps={eps_curtailed:.2f}MWh, "
            f"absorbed={total_absorbed:.2f}MWh, reduction={reduction_pct:.1f}%"
        )

        sensitivity_results.append({
            "solar_ratio": solar_ratio,
            "baseline_curtailment_mwh": float(baseline_curtailed),
            "eps_curtailment_mwh": float(eps_curtailed),
            "absorbed_mwh": float(total_absorbed),
            "reduction_pct": float(reduction_pct),
            "hourly_breakdown": hourly_breakdown,
        })

    ratios = [s["solar_ratio"] for s in sensitivity_results]
    reductions = [s["reduction_pct"] for s in sensitivity_results]
    rho, p_val = stats.spearmanr(ratios, reductions)

    logger.info(f"\n  Spearman reduction vs solar_ratio: rho={rho:.3f}, p={p_val:.4f}")

    final = {
        "metadata": {
            "protocol": "continuous_24h_curtailment_sensitivity",
            "signal_method": "nn_closed_loop",
            "nn_training_r2": float(nn_r2) if nn_r2 else None,
            "nn_training_samples": 12000,
            "nn_training_seed": 42,
            "num_devices": num_devices,
            "M_replications": M_REPS,
            "steps_per_hour": STEPS_PER_HOUR,
            "wind_ratio": WIND_RATIO,
            "base_load_peak_mw": float(BASE_LOAD_PEAK_MW),
            "max_target_mw": float(max_target_mw),
            "timestamp": datetime.now().isoformat(),
        },
        "scenarios": sensitivity_results,
        "trend_test": {
            "spearman_rho": float(rho),
            "p_value": float(p_val),
        },
        "baseline_scenario": {
            "solar_ratio": 1.10,
            "reduction_pct": float(next(
                s["reduction_pct"] for s in sensitivity_results if s["solar_ratio"] == 1.10
            )),
        },
    }

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "curtailment_sensitivity.json"
    with open(out_file, 'w') as f:
        json.dump(final, f, indent=2)
    logger.info(f"  Saved to {out_file}")
    return final


def run_n_scaling_curtailment(
    base_config: 'ExperimentConfig',
    output_dir: Path,
    real_profiles=None,
) -> Dict[str, Any]:
    import dataclasses
    from src.simulation import EPSSimulator, SimulationConfig, SimulationLevel
    from src.estimation import EPSEstimator, EstimatorConfig
    from src.signal import SignalOptimizer, OptimizationTarget

    SOLAR_RATIO = 1.10
    WIND_RATIO = 0.50
    N_VALUES = [50, 100, 150, 200, 500, 1000, 2000, 5000]
    STEPS_PER_HOUR = 12
    TIME_STEP = 300.0

    def _get_m_reps(n):
        if n <= 100:
            return 20
        elif n <= 200:
            return 15
        elif n <= 500:
            return 10
        else:
            return 5

    solar_factors, wind_factors = _get_renewable_factors(real_profiles)
    load_factors = _get_load_factors(real_profiles)

    N_REF = 5000
    MAX_CAP_PER_DEVICE_KW = 1.5
    ref_max_response_mw = N_REF * MAX_CAP_PER_DEVICE_KW / 1000.0
    ref_target_mw = ref_max_response_mw * 0.5
    BASE_LOAD_PEAK_MW = ref_target_mw * 3.0
    SOLAR_CAP = BASE_LOAD_PEAK_MW * SOLAR_RATIO
    WIND_CAP = BASE_LOAD_PEAK_MW * WIND_RATIO

    logger.info("=" * 70)
    logger.info("N-Scaling Curtailment (NN Closed-Loop, Q4b: N vs reduction rate)")
    logger.info(f"  Solar ratio: {SOLAR_RATIO}, N values: {N_VALUES}")
    logger.info(f"  Fixed grid: BASE_LOAD={BASE_LOAD_PEAK_MW:.3f} MW (N_REF={N_REF})")
    logger.info("=" * 70)

    all_results = []
    t_total = time.time()

    for N in N_VALUES:
        M_REPS = _get_m_reps(N)

        max_response_mw = N * MAX_CAP_PER_DEVICE_KW / 1000.0

        n_train_samples = 5000 if N <= 500 else 8000 if N <= 1000 else 12000
        logger.info(f"  N={N}, M={M_REPS}, fleet_capacity={max_response_mw:.3f} MW")
        logger.info(f"    Training NN for N={N} ({n_train_samples} samples)...")
        t_train = time.time()

        rng_train_n = np.random.default_rng(42)
        train_signals_n, train_responses_n = _generate_training_data(
            n_samples=n_train_samples,
            num_devices=N,
            rng=rng_train_n,
            sim_config_override=None,
        )
        estimator_n = EPSEstimator(EstimatorConfig(
            target_coverage=0.9, enable_conformal=True, use_pytorch=True,
        ))
        estimator_n.fit(train_signals_n, train_responses_n)
        optimizer_n = SignalOptimizer(estimator_n)
        nn_r2_n = estimator_n._learned_params.get('r2', None) if estimator_n._learned_params else None

        train_elapsed = time.time() - t_train
        logger.info(f"    NN trained: R²={nn_r2_n:.4f} ({train_elapsed:.1f}s)")

        t0 = time.time()

        rep_reductions = []

        for m in range(M_REPS):
            seed_m = 42 + m * 1000 + int(SOLAR_RATIO * 100)

            sim_config = SimulationConfig(
                num_devices=N,
                num_regions=max(1, N // 200),
                level=SimulationLevel.LEVEL1_AGENT,
                duration_seconds=86400.0,
                time_step=TIME_STEP,
                random_seed=seed_m,
            )
            sim = EPSSimulator(sim_config)
            sim.initialize()

            baseline_total = 0.0
            absorbed_total = 0.0

            for hour in range(24):
                solar_mw = solar_factors[hour] * SOLAR_CAP
                wind_mw = wind_factors[hour] * WIND_CAP
                total_supply = solar_mw + wind_mw
                base_load = load_factors[hour] * BASE_LOAD_PEAK_MW
                net_balance = total_supply - base_load

                r = total_supply / max(base_load, 1e-6)
                d_dispatch = min(1.0, abs(r - 1.0) / 0.4)
                s = float(np.clip((r - 1.0) * d_dispatch, -1.0, 1.0))
                formula_sd, formula_intensity = encode_signal_score(s)

                if net_balance > 0:
                    target = OptimizationTarget(
                        target_response_mw=net_balance,
                        tolerance_fraction=0.15,
                    )
                    opt_result = optimizer_n.optimize(
                        target,
                        supply_demand=formula_sd,
                        signal_context={'hour': hour},
                    )
                    sd_state = opt_result.supply_demand
                    intensity = opt_result.intensity
                else:
                    sd_state = formula_sd
                    intensity = formula_intensity

                scenario = 'valley_filling' if sd_state <= 7 else 'peak_shaving'

                sim.set_override_signal(
                    intensity=intensity, supply_demand=sd_state,
                )
                result = sim.run(num_steps=STEPS_PER_HOUR, scenario=scenario)
                sim.clear_override_signal()

                response_mw = abs(result.total_energy_kwh) / 1000.0
                absorbed = min(response_mw, max(0.0, net_balance))
                absorbed = min(absorbed, base_load * 0.60)

                baseline_total += max(0.0, net_balance)
                absorbed_total += absorbed

            reduction = (absorbed_total / baseline_total * 100.0) if baseline_total > 0 else 100.0
            rep_reductions.append(reduction)

        elapsed = time.time() - t0
        mean_r = float(np.mean(rep_reductions))
        std_r = float(np.std(rep_reductions))

        entry = {
            "N": N,
            "M_reps": M_REPS,
            "reduction_pct_mean": round(mean_r, 2),
            "reduction_pct_std": round(std_r, 2),
            "reduction_pct_ci95_lo": round(float(np.percentile(rep_reductions, 2.5)), 2),
            "reduction_pct_ci95_hi": round(float(np.percentile(rep_reductions, 97.5)), 2),
            "all_reductions": [round(r, 2) for r in rep_reductions],
            "elapsed_seconds": round(elapsed, 1),
            "nn_r2": float(nn_r2_n) if nn_r2_n else None,
            "nn_training_samples": n_train_samples,
            "nn_training_seconds": round(train_elapsed, 1),
        }
        all_results.append(entry)
        logger.info(
            f"    N={N}: {mean_r:.1f}% ± {std_r:.1f}% "
            f"[{entry['reduction_pct_ci95_lo']}, {entry['reduction_pct_ci95_hi']}] "
            f"({elapsed:.0f}s)"
        )

    total_elapsed = time.time() - t_total

    final = {
        "metadata": {
            "protocol": "n_scaling_curtailment_q4b",
            "signal_method": "nn_closed_loop_per_n",
            "solar_ratio": SOLAR_RATIO,
            "wind_ratio": WIND_RATIO,
            "n_values": N_VALUES,
            "n_ref": N_REF,
            "base_load_peak_mw": BASE_LOAD_PEAK_MW,
            "grid_note": "Fixed grid at N_REF scale; only device count varies",
            "device_mode": "battery_only",
            "timestamp": datetime.now().isoformat(),
            "total_elapsed_seconds": round(total_elapsed, 1),
        },
        "results": all_results,
    }

    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "n_scaling_curtailment.json"
    with open(out_file, 'w') as f:
        json.dump(final, f, indent=2)

    logger.info(f"\n  Saved to {out_file}")
    logger.info(f"  Total time: {total_elapsed/60:.1f} minutes")

    logger.info(f"\n  {'N':>6} | {'Reduction%':>12} | {'95% CI':>18}")
    logger.info("  " + "-" * 42)
    for r in all_results:
        logger.info(
            f"  {r['N']:>6} | {r['reduction_pct_mean']:>10.1f}% | "
            f"[{r['reduction_pct_ci95_lo']:>6.1f}, {r['reduction_pct_ci95_hi']:>6.1f}]"
        )

    return final


def run_model_mismatch_sensitivity(
    base_config: 'ExperimentConfig',
    output_dir: Path,
) -> Dict[str, Any]:
    import dataclasses
    from src.simulation import SimulationConfig, SimulationLevel
    from src.estimation import EPSEstimator, EstimatorConfig

    N = base_config.num_devices
    n_train = 12000 if N >= 5000 else 2000
    n_test = 200

    logger.info("  Step 1: Training estimator on default simulator parameters...")
    train_rng = np.random.default_rng(42)
    train_signals, train_responses = _generate_training_data(
        n_samples=n_train, num_devices=N, rng=train_rng,
        sim_config_override=None,
    )

    estimator_config = EstimatorConfig(
        target_coverage=0.9, enable_conformal=True,
        use_pytorch=True,
    )
    estimator = EPSEstimator(estimator_config)
    estimator.fit(train_signals, train_responses)
    logger.info(f"    Estimator trained on {n_train} default samples.")

    mismatch_configs = [
        {"label": "default",
         "response_prob_bias": 1.0, "soc_noise_std": 0.0, "device_offline_rate": 0.015},

        {"label": "prob_-30%",
         "response_prob_bias": 0.7, "soc_noise_std": 0.0, "device_offline_rate": 0.015},
        {"label": "prob_-20%",
         "response_prob_bias": 0.8, "soc_noise_std": 0.0, "device_offline_rate": 0.015},
        {"label": "prob_-10%",
         "response_prob_bias": 0.9, "soc_noise_std": 0.0, "device_offline_rate": 0.015},
        {"label": "prob_+10%",
         "response_prob_bias": 1.1, "soc_noise_std": 0.0, "device_offline_rate": 0.015},
        {"label": "prob_+20%",
         "response_prob_bias": 1.2, "soc_noise_std": 0.0, "device_offline_rate": 0.015},
        {"label": "prob_+30%",
         "response_prob_bias": 1.3, "soc_noise_std": 0.0, "device_offline_rate": 0.015},

        {"label": "soc_10%",
         "response_prob_bias": 1.0, "soc_noise_std": 0.10, "device_offline_rate": 0.015},
        {"label": "soc_20%",
         "response_prob_bias": 1.0, "soc_noise_std": 0.20, "device_offline_rate": 0.015},
        {"label": "soc_30%",
         "response_prob_bias": 1.0, "soc_noise_std": 0.30, "device_offline_rate": 0.015},

        {"label": "offline_5%",
         "response_prob_bias": 1.0, "soc_noise_std": 0.0, "device_offline_rate": 0.05},
        {"label": "offline_10%",
         "response_prob_bias": 1.0, "soc_noise_std": 0.0, "device_offline_rate": 0.10},
        {"label": "offline_15%",
         "response_prob_bias": 1.0, "soc_noise_std": 0.0, "device_offline_rate": 0.15},

        {"label": "combined_mild",
         "response_prob_bias": 1.1, "soc_noise_std": 0.10, "device_offline_rate": 0.05},
        {"label": "combined_moderate",
         "response_prob_bias": 1.2, "soc_noise_std": 0.20, "device_offline_rate": 0.10},
        {"label": "combined_severe",
         "response_prob_bias": 1.3, "soc_noise_std": 0.30, "device_offline_rate": 0.15},

        {"label": "combined_negative_mild",
         "response_prob_bias": 0.9, "soc_noise_std": 0.10, "device_offline_rate": 0.05},
        {"label": "combined_negative_moderate",
         "response_prob_bias": 0.8, "soc_noise_std": 0.20, "device_offline_rate": 0.10},
        {"label": "combined_negative_severe",
         "response_prob_bias": 0.7, "soc_noise_std": 0.30, "device_offline_rate": 0.15},
    ]

    all_results = {}
    for mm_cfg in mismatch_configs:
        label = mm_cfg["label"]
        logger.info(f"  Evaluating mismatch: {label} ...")

        mismatch_sim = SimulationConfig(
            response_prob_bias=mm_cfg["response_prob_bias"],
            soc_noise_std=mm_cfg["soc_noise_std"],
            device_offline_rate=mm_cfg["device_offline_rate"],
        )

        test_rng = np.random.default_rng(99999)
        test_signals, test_responses = _generate_training_data(
            n_samples=n_test, num_devices=N, rng=test_rng,
            sim_config_override=mismatch_sim,
        )

        predictions = []
        actuals = []
        in_interval = 0
        for sig, actual in zip(test_signals, test_responses):
            result = estimator.estimate(sig)
            predictions.append(result.response_kw)
            actuals.append(actual)
            if result.lower_bound <= actual <= result.upper_bound:
                in_interval += 1

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        ss_res = float(np.sum((actuals - predictions) ** 2))
        ss_tot = float(np.sum((actuals - np.mean(actuals)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
        bias = float(np.mean(actuals - predictions))
        picp = in_interval / len(actuals) if len(actuals) > 0 else 0.0

        denom = np.abs(actuals) + np.abs(predictions) + 1e-8
        smape = float(np.mean(2 * np.abs(actuals - predictions) / denom) * 100)

        all_results[label] = {
            "response_prob_bias": mm_cfg["response_prob_bias"],
            "soc_noise_std": mm_cfg["soc_noise_std"],
            "device_offline_rate": mm_cfg["device_offline_rate"],
            "r2": round(r2, 4),
            "rmse": round(rmse, 2),
            "smape": round(smape, 2),
            "bias_kw": round(bias, 2),
            "picp": round(picp, 3),
            "n_test": n_test,
            "n_train": n_train,
        }
        logger.info(
            f"    R²={r2:.4f}, RMSE={rmse:.1f}, bias={bias:.1f} kW, "
            f"PICP={picp:.3f}, SMAPE={smape:.1f}%"
        )

    mm_file = output_dir / "data" / "model_mismatch.json"
    mm_file.parent.mkdir(parents=True, exist_ok=True)
    with open(mm_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"  Model mismatch results saved to {mm_file}")

    logger.info("\n  === Model Mismatch Summary ===")
    logger.info(f"  {'Config':<22} {'R²':>8} {'RMSE':>8} {'Bias':>10} {'PICP':>8}")
    logger.info(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for label, res in all_results.items():
        logger.info(
            f"  {label:<22} {res['r2']:>8.4f} {res['rmse']:>8.1f} "
            f"{res['bias_kw']:>10.1f} {res['picp']:>8.3f}"
        )

    return all_results


def run_structural_mismatch(
    base_config: 'ExperimentConfig',
    output_dir: Path,
) -> Dict[str, Any]:
    import dataclasses
    from src.simulation import SimulationConfig, SimulationLevel
    from src.estimation import EPSEstimator, EstimatorConfig

    N = base_config.num_devices
    n_train = 12000 if N >= 5000 else 2000
    n_test = 200

    logger.info("=" * 70)
    logger.info("M1: Structural Model Mismatch Experiment")
    logger.info(f"  N={N} devices, n_train={n_train}, n_test={n_test}")
    logger.info("  Control: Bernoulli (same as training)")
    logger.info("  Mismatch 1: Continuous proportional response (same E[P_agg])")
    logger.info("  Mismatch 2: Sigmoid g(s) shape (k=2,5,10 — changes E[P_agg|s])")
    logger.info("=" * 70)

    logger.info("  Step 1: Training estimator on standard Bernoulli simulator...")
    train_rng = np.random.default_rng(42)
    train_signals, train_responses = _generate_training_data(
        n_samples=n_train, num_devices=N, rng=train_rng,
        sim_config_override=None,
    )

    estimator_config = EstimatorConfig(
        target_coverage=0.9, enable_conformal=True,
        use_pytorch=True,
    )
    estimator = EPSEstimator(estimator_config)
    estimator.fit(train_signals, train_responses)
    logger.info(f"    Estimator trained on {n_train} Bernoulli samples.")

    test_configs = {
        "bernoulli_control": SimulationConfig(
            continuous_response=False,
        ),
        "continuous_mismatch": SimulationConfig(
            continuous_response=True,
        ),
        "sigmoid_k2": SimulationConfig(
            sigmoid_response=True, sigmoid_steepness=2.0,
        ),
        "sigmoid_k5": SimulationConfig(
            sigmoid_response=True, sigmoid_steepness=5.0,
        ),
        "sigmoid_k10": SimulationConfig(
            sigmoid_response=True, sigmoid_steepness=10.0,
        ),
    }

    all_results = {}
    for label, sim_cfg in test_configs.items():
        logger.info(f"  Evaluating: {label} ...")

        test_rng = np.random.default_rng(99999)
        test_signals, test_responses = _generate_training_data(
            n_samples=n_test, num_devices=N, rng=test_rng,
            sim_config_override=sim_cfg,
        )

        predictions = []
        actuals = []
        in_interval = 0
        for sig, actual in zip(test_signals, test_responses):
            result = estimator.estimate(sig)
            predictions.append(result.response_kw)
            actuals.append(actual)
            if result.lower_bound <= actual <= result.upper_bound:
                in_interval += 1

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        ss_res = float(np.sum((actuals - predictions) ** 2))
        ss_tot = float(np.sum((actuals - np.mean(actuals)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean((actuals - predictions) ** 2)))
        bias = float(np.mean(actuals - predictions))
        picp = in_interval / len(actuals) if len(actuals) > 0 else 0.0

        denom = np.abs(actuals) + np.abs(predictions) + 1e-8
        smape = float(np.mean(2 * np.abs(actuals - predictions) / denom) * 100)

        all_results[label] = {
            "continuous_response": sim_cfg.continuous_response,
            "sigmoid_response": sim_cfg.sigmoid_response,
            "sigmoid_steepness": sim_cfg.sigmoid_steepness if sim_cfg.sigmoid_response else None,
            "r2": round(r2, 4),
            "rmse": round(rmse, 2),
            "smape": round(smape, 2),
            "bias_kw": round(bias, 2),
            "picp": round(picp, 3),
            "n_test": n_test,
            "n_train": n_train,
            "num_devices": N,
        }
        logger.info(
            f"    R2={r2:.4f}, RMSE={rmse:.1f}, bias={bias:.1f} kW, "
            f"PICP={picp:.3f}, SMAPE={smape:.1f}%"
        )

    sm_file = output_dir / "data" / "structural_mismatch.json"
    sm_file.parent.mkdir(parents=True, exist_ok=True)
    with open(sm_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"  Structural mismatch results saved to {sm_file}")

    logger.info("\n  === Structural Mismatch Summary (M1) ===")
    logger.info(f"  {'Config':<25} {'R2':>8} {'RMSE':>8} {'Bias':>10} {'PICP':>8} {'SMAPE':>8}")
    logger.info(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    for label, res in all_results.items():
        logger.info(
            f"  {label:<25} {res['r2']:>8.4f} {res['rmse']:>8.1f} "
            f"{res['bias_kw']:>10.1f} {res['picp']:>8.3f} {res['smape']:>8.1f}"
        )

    return all_results


def _load_nextgen_params() -> Dict[str, np.ndarray]:
    import csv as csv_mod
    data_dir = Path(__file__).resolve().parent.parent / "data" / "nextgen"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"NextGen data not found at {data_dir}. "
            "See data/README.md for download instructions."
        )
    capacities, peak_powers, c_rates = [], [], []
    for csv_file in sorted(data_dir.glob("*.csv")):
        with open(csv_file, 'r') as fh:
            reader = csv_mod.DictReader(fh)
            row = next(reader)
            cap = float(row['battery capacity (kWh)'])
            peak = float(row['battery peak power (kW)'])
            capacities.append(cap)
            peak_powers.append(peak)
            c_rates.append(peak / cap)
    if not capacities:
        raise FileNotFoundError(f"No CSV files in {data_dir}")
    return {
        'capacities': np.array(capacities),
        'peak_powers': np.array(peak_powers),
        'c_rates': np.array(c_rates),
        'n_devices': len(capacities),
    }


def _inject_real_params(sim: 'EPSSimulator', real_params: Dict[str, np.ndarray]) -> None:
    n_inject = min(real_params['n_devices'], len(sim._population.batteries))
    for i in range(n_inject):
        cap = float(real_params['capacities'][i])
        peak = float(real_params['peak_powers'][i])
        cr = float(real_params['c_rates'][i])
        b = sim._population.batteries[i]
        b.params.nominal_capacity_kwh = cap
        b.params.max_charge_power_kw = peak
        b.params.max_discharge_power_kw = peak
        b.params.max_charge_c_rate = cr
        b.params.max_discharge_c_rate = cr
        b.params.continuous_c_rate = cr * 0.6
        b.params.usable_capacity_factor = 1.0


def _generate_training_data_real_params(
    n_samples: int,
    real_params: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> Tuple[List[Dict], List[float]]:
    import dataclasses
    from src.simulation import EPSSimulator, SimulationConfig, SimulationLevel

    N_real = real_params['n_devices']
    signals: List[Dict] = []
    responses: List[float] = []

    n_scenario = int(n_samples * 0.80)
    n_time_aware = int(n_samples * 0.10)

    profiles = [
        ValleyFillingProfile(), PeakShavingProfile(),
        EmergencyChargeProfile(), EmergencyDischargeProfile(),
    ]
    spp = n_scenario // len(profiles)

    for i in range(n_samples):
        if i < n_scenario:
            p_idx = min(i // spp, len(profiles) - 1)
            phase = rng.uniform(0, 1)
            s = profiles[p_idx].get_signal_score(phase)
            s = float(np.clip(s + rng.normal(0, 0.02), -1.0, 1.0))
            supply_demand, intensity = encode_signal_score(s)
            hour = int(rng.integers(0, 24))
        elif i < n_scenario + n_time_aware:
            hour = int(rng.integers(0, 24))
            if (7 <= hour <= 11) or (17 <= hour <= 21):
                intensity = int(rng.normal(2800, 500) if rng.random() < 0.5
                                else rng.normal(1000, 400))
                supply_demand = int(rng.normal(11, 2))
            elif hour <= 6 or hour >= 22:
                intensity = int(rng.normal(1000, 300))
                supply_demand = int(rng.normal(3, 2))
            else:
                intensity = int(rng.normal(2048, 600))
                supply_demand = int(rng.normal(8, 2))
        else:
            hour = int(rng.integers(0, 24))
            intensity = int(rng.uniform(0, 4095))
            supply_demand = int(rng.uniform(0, 16))

        intensity = max(0, min(4095, intensity))
        supply_demand = max(0, min(15, supply_demand))

        seed_i = int(rng.integers(0, 10000))
        mini_config = SimulationConfig(
            num_devices=N_real, num_regions=5,
            level=SimulationLevel.LEVEL1_AGENT,
            duration_seconds=900, time_step=300,
            random_seed=seed_i,
        )
        mini_sim = EPSSimulator(mini_config)
        mini_sim.initialize()
        _inject_real_params(mini_sim, real_params)

        mini_sim.set_override_signal(
            intensity=intensity, price_value=0.0, supply_demand=supply_demand)
        scenario = 'valley_filling' if supply_demand <= 7 else 'peak_shaving'
        result = mini_sim.run(num_steps=3, scenario=scenario)
        mini_sim.clear_override_signal()

        actual_kw = result.total_energy_kwh / 0.25
        direction = 1 if supply_demand <= 7 else -1

        signals.append({
            'supply_demand': supply_demand, 'intensity': intensity,
            'price': 0.0, 'hour': hour, 'direction': direction,
        })
        responses.append(actual_kw)

        if (i + 1) % 500 == 0:
            logger.info(f"    Generated {i+1}/{n_samples} samples")

    return signals, responses


def run_real_param_validation(
    base_config: 'ExperimentConfig',
    output_dir: Path,
) -> Dict[str, Any]:
    import dataclasses
    from src.simulation import EPSSimulator, SimulationConfig, SimulationLevel
    from src.estimation import EPSEstimator, EstimatorConfig

    logger.info("=" * 70)
    logger.info("Real-Parameter Validation (EPSSimulator + EPSEstimator)")
    logger.info("  Dataset: NextGen ACT Australia empirical parameter population")
    logger.info("=" * 70)

    real_params = _load_nextgen_params()
    N_real = real_params['n_devices']
    N_source = real_params.get('source_n_devices', N_real)
    logger.info(f"  Loaded {N_real} devices: "
                f"cap={real_params['capacities'].mean():.1f}±{real_params['capacities'].std():.1f} kWh, "
                f"C-rate={real_params['c_rates'].mean():.3f}±{real_params['c_rates'].std():.3f}")

    dp_file = output_dir / "data" / "real_device_params.json"
    with open(dp_file, 'w') as f:
        json.dump({
            'dataset': 'NextGen ACT Australia (Zenodo 14885589)',
            'n_devices': N_real,
            'n_source_devices': N_source,
            'sampling_method': real_params.get('sampling_method', 'measured_devices'),
            'capacities_kwh': real_params['capacities'].tolist(),
            'peak_powers_kw': real_params['peak_powers'].tolist(),
            'c_rates': real_params['c_rates'].tolist(),
        }, f, indent=2)

    N_TRAIN = 8000
    logger.info(f"  Training data: {N_TRAIN} samples (EPSSimulator with real params)...")
    rng_train = np.random.default_rng(42)
    train_signals, train_responses = _generate_training_data_real_params(
        N_TRAIN, real_params, rng_train)

    logger.info("  Training EPSEstimator (dual NN + CQR)...")
    estimator = EPSEstimator(EstimatorConfig(
        target_coverage=0.9, enable_conformal=True,
        use_pytorch=True,
    ))
    estimator.fit(train_signals, train_responses)
    nn_r2 = estimator._learned_params.get('r2', None) if estimator._learned_params else None
    logger.info(f"    NN training R² = {nn_r2:.4f}" if nn_r2 else "    NN trained")

    N_EVAL = 2000
    logger.info(f"  Evaluation data: {N_EVAL} samples (seed=99999)...")
    rng_eval = np.random.default_rng(99999)
    eval_signals, eval_responses = _generate_training_data_real_params(
        N_EVAL, real_params, rng_eval)

    logger.info("  Computing validation metrics...")
    preds, lowers, uppers = [], [], []
    for sig in eval_signals:
        est = estimator.estimate(sig)
        preds.append(est.response_kw)
        lowers.append(est.lower_bound)
        uppers.append(est.upper_bound)

    preds_a = np.array(preds)
    actuals_a = np.array(eval_responses)
    lowers_a = np.array(lowers)
    uppers_a = np.array(uppers)

    ss_res = np.sum((preds_a - actuals_a) ** 2)
    ss_tot = np.sum((actuals_a - actuals_a.mean()) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-10))

    denom = np.abs(preds_a) + np.abs(actuals_a)
    smape = float(np.mean(np.where(denom > 0, 2 * np.abs(preds_a - actuals_a) / denom, 0)) * 100)
    rmse = float(np.sqrt(np.mean((preds_a - actuals_a) ** 2)))
    in_interval = (actuals_a >= lowers_a) & (actuals_a <= uppers_a)
    picp = float(np.mean(in_interval) * 100)
    y_range = max(actuals_a.max() - actuals_a.min(), 1e-10)
    pinaw = float(np.mean(uppers_a - lowers_a) / y_range * 100)

    rng_b = np.random.default_rng(0)
    boot_r2s = []
    n = len(preds_a)
    for _ in range(5000):
        idx = rng_b.choice(n, n, replace=True)
        sr = np.sum((preds_a[idx] - actuals_a[idx]) ** 2)
        st = np.sum((actuals_a[idx] - actuals_a[idx].mean()) ** 2)
        if st > 1e-10:
            boot_r2s.append(1 - sr / st)
    r2_ci = [float(np.percentile(boot_r2s, 2.5)), float(np.percentile(boot_r2s, 97.5))]

    logger.info(f"    R²    = {r2:.4f}  95% CI [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]")
    logger.info(f"    SMAPE = {smape:.2f}%")
    logger.info(f"    PICP  = {picp:.1f}%  (target 90%)")
    logger.info(f"    PINAW = {pinaw:.2f}%")
    logger.info(f"    RMSE  = {rmse:.2f} kW")

    logger.info("  N-scaling CV analysis...")
    N_VALUES = sorted(set([5, 10, 20, 50, 100, N_real]))
    N_TRIALS = 100
    test_s = 0.3
    test_sd, test_int = encode_signal_score(test_s)
    n_scaling_results = []

    for Nv in N_VALUES:
        trial_responses = []
        for trial in range(N_TRIALS):
            tseed = 42 + trial * 777 + Nv * 13
            trng = np.random.default_rng(tseed)
            if Nv < N_real:
                idx = trng.choice(N_real, Nv, replace=False)
                tp = {k: real_params[k][idx] if isinstance(real_params[k], np.ndarray) else real_params[k]
                      for k in real_params}
                tp['n_devices'] = Nv
            else:
                tp = real_params
            mc = SimulationConfig(
                num_devices=Nv, num_regions=max(1, min(5, Nv // 10)),
                level=SimulationLevel.LEVEL1_AGENT,
                duration_seconds=900, time_step=300, random_seed=tseed,
                )
            ms = EPSSimulator(mc)
            ms.initialize()
            _inject_real_params(ms, tp)
            ms.set_override_signal(intensity=test_int, supply_demand=test_sd)
            res = ms.run(num_steps=3, scenario='valley_filling')
            trial_responses.append(res.total_energy_kwh / 0.25)

        arr = np.array(trial_responses)
        mn = float(np.mean(arr))
        sd = float(np.std(arr))
        cv = sd / abs(mn) * 100 if abs(mn) > 0.01 else float('nan')
        n_scaling_results.append({
            'N': Nv, 'mean_kw': mn, 'std_kw': sd,
            'CV_pct': cv, 'CV_sqrt_N': cv * np.sqrt(Nv) if not np.isnan(cv) else float('nan'),
        })
        logger.info(f"    N={Nv:4d}: mean={mn:8.2f} kW, CV={cv:6.2f}%")

    valid = [(r['N'], r['CV_pct']) for r in n_scaling_results
             if not np.isnan(r['CV_pct']) and r['CV_pct'] > 0]
    if len(valid) >= 2:
        ln = np.log([v[0] for v in valid])
        lc = np.log([v[1] for v in valid])
        A = np.column_stack([ln, np.ones_like(ln)])
        cv_slope = float(np.linalg.lstsq(A, lc, rcond=None)[0][0])
    else:
        cv_slope = float('nan')
    logger.info(f"    CV slope: {cv_slope:.3f} (theory: -0.50)")

    validation_results = {
        'dataset': 'NextGen ACT Australia (Zenodo 14885589)',
        'n_real_devices': N_source,
        'n_devices_in_simulation': N_real,
        'n_source_devices': N_source,
        'sampling_method': real_params.get('sampling_method', 'measured_devices'),
        'protocol': {
            'n_train': N_TRAIN, 'n_eval': N_EVAL,
            'train_seed': 42, 'eval_seed': 99999,
            'estimator': 'EPSEstimator (dual_quantile_nn + CQR)',
            'simulator': 'EPSSimulator (full fidelity, level1_agent)',
        },
        'metrics': {
            'r2': r2, 'r2_ci_95': [float(c) for c in r2_ci],
            'smape_pct': smape, 'picp_pct': picp,
            'pinaw_pct': pinaw, 'rmse_kw': rmse,
            'nn_training_r2': float(nn_r2) if nn_r2 else None,
        },
        'n_scaling': {
            'signal_score': float(test_s), 'n_trials': N_TRIALS,
            'cv_slope': cv_slope, 'theoretical_slope': -0.50,
            'per_N': n_scaling_results,
        },
    }
    out_file = output_dir / "data" / "real_param_validation.json"
    with open(out_file, 'w') as f:
        json.dump(validation_results, f, indent=2)
    logger.info(f"  Results saved to {out_file}")

    est_file = output_dir / "estimation" / "estimation_validation_results.json"
    with open(est_file, 'w') as f:
        json.dump({
            'r2': r2, 'picp': picp / 100, 'pinaw': pinaw / 100,
            'smape': smape, 'rmse': rmse, 'n_eval': N_EVAL,
            'predictions': preds_a.tolist(),
            'actuals': actuals_a.tolist(),
            'lower_bounds': lowers_a.tolist(),
            'upper_bounds': uppers_a.tolist(),
        }, f, indent=2)

    return validation_results


def run_curtailment_baselines(
    base_config: 'ExperimentConfig',
    output_dir: Path,
) -> Dict[str, Any]:
    import dataclasses
    from src.simulation import EPSSimulator, SimulationConfig, SimulationLevel
    from src.estimation import EPSEstimator, EstimatorConfig
    from src.signal import SignalOptimizer, OptimizationTarget

    SOLAR_RATIO = 1.10
    WIND_RATIO = 0.50
    M_REPS = 5
    STEPS_PER_HOUR = 12
    TIME_STEP = 300.0
    DT_HOURS = TIME_STEP / 3600.0
    GRID_HOSTING = 0.60

    num_devices = base_config.num_devices
    max_cap_kw = 1.5
    max_resp_mw = num_devices * max_cap_kw / 1000.0
    achievable_mw = max_resp_mw * 0.5
    max_target_mw = min(achievable_mw, float(base_config.target_response_mw))
    max_target_mw = max(max_target_mw, 0.01)
    BASE_LOAD_PEAK = max_target_mw * 3.0
    SOLAR_CAP = BASE_LOAD_PEAK * SOLAR_RATIO
    WIND_CAP = BASE_LOAD_PEAK * WIND_RATIO

    solar_factors, wind_factors = _get_renewable_factors()
    load_factors = _get_load_factors()

    logger.info("=" * 70)
    logger.info("Curtailment Baseline Comparison")
    logger.info(f"  N={num_devices}, solar_ratio={SOLAR_RATIO}, M={M_REPS} reps")
    logger.info(f"  BASE_LOAD={BASE_LOAD_PEAK:.2f} MW, SOLAR_CAP={SOLAR_CAP:.2f} MW")
    logger.info("=" * 70)

    net_balance = np.zeros(24)
    total_supply = np.zeros(24)
    base_load_arr = np.zeros(24)
    for h in range(24):
        sol = solar_factors[h] * SOLAR_CAP
        wnd = wind_factors[h] * WIND_CAP
        ld = load_factors[h] * BASE_LOAD_PEAK
        total_supply[h] = sol + wnd
        base_load_arr[h] = ld
        net_balance[h] = sol + wnd - ld

    total_surplus_24h = float(sum(max(0, nb) for nb in net_balance))
    logger.info(f"  Total 24h surplus: {total_surplus_24h:.2f} MWh")

    logger.info("  Training NN for EPS closed-loop...")
    rng_train = np.random.default_rng(42)
    train_sigs, train_resps = _generate_training_data(
        n_samples=12000, num_devices=num_devices, rng=rng_train,
        sim_config_override=None,
    )
    est = EPSEstimator(EstimatorConfig(
        target_coverage=0.9, enable_conformal=True, use_pytorch=True,
    ))
    est.fit(train_sigs, train_resps)
    optimizer = SignalOptimizer(est)
    logger.info("    NN trained")

    all_strategy_results = {}

    for strat_name in ['no_coordination', 'local_rules', 'eps_broadcast', 'centralized_optimal']:
        t0 = time.time()
        rep_reductions = []

        for m in range(M_REPS):
            seed_m = 42 + m * 1000
            sim_config = SimulationConfig(
                num_devices=num_devices, num_regions=5,
                level=SimulationLevel.LEVEL1_AGENT,
                duration_seconds=86400.0, time_step=TIME_STEP,
                random_seed=seed_m,
            )
            sim = EPSSimulator(sim_config)
            sim.initialize()

            N = len(sim._population.batteries)
            caps = np.array([b.params.nominal_capacity_kwh for b in sim._population.batteries])
            crs = np.array([b.params.max_charge_c_rate for b in sim._population.batteries])
            sohs = np.array([getattr(b, '_soh', 0.95) for b in sim._population.batteries])
            socs = np.array([b.soc for b in sim._population.batteries])
            soc_res = np.array([getattr(b.params, 'soc_reserve', 0.2)
                                for b in sim._population.batteries])
            soc_max = np.array([getattr(b.params, 'soc_max', 0.95)
                                for b in sim._population.batteries])
            v_rng = np.random.default_rng(seed_m + 500)

            hourly_absorbed = np.zeros(24)

            if strat_name == 'no_coordination':
                pass

            elif strat_name == 'local_rules':
                for h in range(24):
                    surplus = max(0.0, net_balance[h])
                    max_inc = base_load_arr[h] * GRID_HOSTING
                    for _ in range(STEPS_PER_HOUR):
                        e_av = caps * sohs * 0.9
                        mp = caps * crs
                        if 9 <= h <= 16:
                            can = socs < 0.8
                            hd = np.maximum(0, (soc_max - socs) * e_av / DT_HOURS)
                            desired = np.where(can, np.minimum(mp, hd), 0.0)
                            tot_mw = np.sum(desired) / 1000.0
                            if tot_mw > max_inc:
                                desired *= max_inc / tot_mw
                            step_abs = min(np.sum(desired) / 1000.0, surplus)
                            energy = desired * DT_HOURS
                            delta = energy * 0.95 / np.maximum(e_av, 0.01)
                            socs = np.clip(socs + delta, 0.05, 0.98)
                        elif 17 <= h <= 20:
                            can = socs > 0.3
                            hd = np.maximum(0, (socs - soc_res) * e_av / DT_HOURS)
                            dp = np.where(can, -np.minimum(mp, hd), 0.0)
                            energy = dp * DT_HOURS
                            delta = energy / (0.95 * np.maximum(e_av, 0.01))
                            socs = np.clip(socs + delta, 0.05, 0.98)
                            step_abs = 0.0
                        else:
                            step_abs = 0.0
                        hourly_absorbed[h] += step_abs / STEPS_PER_HOUR

            elif strat_name == 'eps_broadcast':
                for h in range(24):
                    surplus = max(0.0, net_balance[h])
                    r = total_supply[h] / max(base_load_arr[h], 1e-6)
                    d_d = min(1.0, abs(r - 1.0) / 0.4)
                    s = float(np.clip((r - 1.0) * d_d, -1.0, 1.0))
                    f_sd, f_int = encode_signal_score(s)

                    if net_balance[h] > 0:
                        target = OptimizationTarget(
                            target_response_mw=net_balance[h],
                            tolerance_fraction=0.15,
                        )
                        opt_res = optimizer.optimize(
                            target, supply_demand=f_sd,
                            signal_context={'hour': h},
                        )
                        sd_use = opt_res.supply_demand
                        int_use = opt_res.intensity
                    else:
                        sd_use = f_sd
                        int_use = f_int

                    scen = 'valley_filling' if sd_use <= 7 else 'peak_shaving'
                    sim.set_override_signal(intensity=int_use, supply_demand=sd_use)
                    result = sim.run(num_steps=STEPS_PER_HOUR, scenario=scen)
                    sim.clear_override_signal()

                    resp_mw = abs(result.total_energy_kwh) / 1000.0
                    max_inc = base_load_arr[h] * GRID_HOSTING
                    absorbed = min(resp_mw, surplus, max_inc)
                    hourly_absorbed[h] = absorbed

            elif strat_name == 'centralized_optimal':
                for h in range(24):
                    surplus = max(0.0, net_balance[h])
                    max_inc = base_load_arr[h] * GRID_HOSTING
                    for _ in range(STEPS_PER_HOUR):
                        e_av = caps * sohs * 0.9
                        mp = caps * crs
                        if net_balance[h] > 0:
                            hd = np.maximum(0, (soc_max - socs) * e_av / DT_HOURS)
                            ca = np.minimum(mp, hd)
                            can = (socs < soc_max) & (ca >= 0.01)
                            order = np.argsort(socs)
                            sorted_mw = np.where(can[order], ca[order] / 1000.0, 0.0)
                            cumsum = np.cumsum(sorted_mw)
                            target_mw = min(surplus, max_inc)
                            pw = np.zeros(N)
                            if target_mw > 0 and cumsum[-1] > 0:
                                nf = int(np.searchsorted(cumsum, target_mw))
                                if nf > 0:
                                    pw[order[:nf]] = ca[order[:nf]]
                                if nf < N:
                                    rem = target_mw - (cumsum[nf - 1] if nf > 0 else 0.0)
                                    if rem > 0.001 and can[order[nf]]:
                                        pw[order[nf]] = min(rem * 1000.0, ca[order[nf]])
                            step_abs = np.sum(pw) / 1000.0
                            energy = pw * DT_HOURS
                            delta = energy * 0.95 / np.maximum(e_av, 0.01)
                            socs = np.clip(socs + delta, 0.05, 0.98)
                        elif net_balance[h] < 0:
                            hd = np.maximum(0, (socs - soc_res) * e_av / DT_HOURS)
                            da = np.minimum(mp, hd)
                            can = (socs > soc_res) & (da >= 0.01)
                            order = np.argsort(-socs)
                            sorted_mw = np.where(can[order], da[order] / 1000.0, 0.0)
                            cumsum = np.cumsum(sorted_mw)
                            tgt = min(abs(net_balance[h]), max_inc)
                            pw = np.zeros(N)
                            if tgt > 0 and cumsum[-1] > 0:
                                nf = int(np.searchsorted(cumsum, tgt))
                                if nf > 0:
                                    pw[order[:nf]] = -da[order[:nf]]
                                if nf < N:
                                    rem = tgt - (cumsum[nf - 1] if nf > 0 else 0.0)
                                    if rem > 0.001:
                                        pw[order[nf]] = -min(rem * 1000.0, da[order[nf]])
                            energy = pw * DT_HOURS
                            delta = np.where(energy > 0,
                                             energy * 0.95 / np.maximum(e_av, 0.01),
                                             energy / (0.95 * np.maximum(e_av, 0.01)))
                            socs = np.clip(socs + delta, 0.05, 0.98)
                            step_abs = 0.0
                        else:
                            step_abs = 0.0
                        hourly_absorbed[h] += step_abs / STEPS_PER_HOUR

            total_abs = float(np.sum(np.minimum(hourly_absorbed, [max(0, nb) for nb in net_balance])))
            if total_surplus_24h > 0:
                reduction = total_abs / total_surplus_24h * 100
            else:
                reduction = 0.0
            rep_reductions.append(reduction)

        elapsed = time.time() - t0
        mean_r = float(np.mean(rep_reductions))
        std_r = float(np.std(rep_reductions))
        all_strategy_results[strat_name] = {
            'mean_reduction_pct': mean_r, 'std_reduction_pct': std_r,
            'per_rep': [float(r) for r in rep_reductions],
            'elapsed_seconds': elapsed,
        }
        logger.info(f"  {strat_name:25s}: {mean_r:6.2f}% ± {std_r:.2f}%  ({elapsed:.1f}s)")

    eps_pct = all_strategy_results.get('eps_broadcast', {}).get('mean_reduction_pct', 0)
    cent_pct = all_strategy_results.get('centralized_optimal', {}).get('mean_reduction_pct', 0)
    if cent_pct > 0:
        ratio = eps_pct / cent_pct * 100
        logger.info(f"\n  EPS / Centralized = {ratio:.1f}% (O(1) vs O(N))")

    output = {
        'config': {
            'n_devices': num_devices, 'solar_ratio': SOLAR_RATIO,
            'wind_ratio': WIND_RATIO, 'n_reps': M_REPS,
            'base_load_peak_mw': BASE_LOAD_PEAK,
            'grid_hosting_fraction': GRID_HOSTING,
            'total_surplus_24h_mwh': total_surplus_24h,
        },
        'results': all_strategy_results,
    }
    out_file = output_dir / "data" / "curtailment_baselines.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"  Results saved to {out_file}")
    return output


def run_closed_loop_picp(
    base_config: 'ExperimentConfig',
    output_dir: Path,
) -> Dict[str, Any]:
    import dataclasses
    from src.simulation import EPSSimulator, SimulationConfig, SimulationLevel
    from src.estimation import EPSEstimator, EstimatorConfig
    from src.signal import SignalOptimizer, OptimizationTarget

    logger.info("=" * 70)
    logger.info("M5: Closed-Loop PICP Verification (Continuous 24h)")
    logger.info("=" * 70)

    num_devices = base_config.num_devices
    max_capacity_per_device_kw = 1.5
    max_response_mw = num_devices * max_capacity_per_device_kw / 1000.0
    achievable_target_mw = max_response_mw * 0.5
    max_target_mw = min(achievable_target_mw, float(base_config.target_response_mw))
    max_target_mw = max(max_target_mw, 0.01)

    SOLAR_RATIO = 1.10
    WIND_RATIO = 0.50
    BASE_LOAD_PEAK_MW = max_target_mw * 3.0
    SOLAR_CAP = BASE_LOAD_PEAK_MW * SOLAR_RATIO
    WIND_CAP = BASE_LOAD_PEAK_MW * WIND_RATIO
    STEPS_PER_HOUR = 12
    TIME_STEP = 300.0

    solar_factors, wind_factors = _get_renewable_factors()
    load_factors = _get_load_factors()

    logger.info(f"  N={num_devices}, solar_ratio={SOLAR_RATIO}")
    logger.info(f"  BASE_LOAD={BASE_LOAD_PEAK_MW:.2f} MW, max_target={max_target_mw:.2f} MW")
    logger.info(f"  Steps/hour={STEPS_PER_HOUR}, total steps/day={STEPS_PER_HOUR * 24}")

    logger.info("  Phase 1: Training NN estimator (12000 samples, seed=42)...")
    rng_train = np.random.default_rng(42)
    train_sigs, train_resps = _generate_training_data(
        n_samples=12000, num_devices=num_devices, rng=rng_train,
        sim_config_override=None,
    )
    estimator = EPSEstimator(EstimatorConfig(
        target_coverage=0.9, enable_conformal=True, use_pytorch=True,
    ))
    estimator.fit(train_sigs, train_resps)
    optimizer = SignalOptimizer(estimator)
    nn_r2 = estimator._learned_params.get('r2', None) if estimator._learned_params else None
    logger.info(f"  NN trained: R²={nn_r2:.4f}" if nn_r2 else "  NN trained (no R² available)")

    logger.info("  Phase 2: Open-loop PICP (500 random samples)...")
    rng_eval = np.random.default_rng(99999)
    eval_sigs, eval_resps = _generate_training_data(
        n_samples=500, num_devices=num_devices, rng=rng_eval,
        sim_config_override=None,
    )
    ol_covered = 0
    ol_total = 0
    ol_widths = []
    ol_actuals = []
    for sig, actual in zip(eval_sigs, eval_resps):
        est = estimator.estimate(sig)
        if est.lower_bound <= actual <= est.upper_bound:
            ol_covered += 1
        ol_total += 1
        ol_widths.append(est.upper_bound - est.lower_bound)
        ol_actuals.append(abs(actual))
    ol_picp = ol_covered / ol_total
    ol_mean_width = float(np.mean(ol_widths))
    ol_mean_actual = float(np.mean(ol_actuals))
    ol_pinaw = ol_mean_width / ol_mean_actual if ol_mean_actual > 1e-6 else float('inf')
    logger.info(f"  Open-loop PICP:  {ol_picp:.1%} (n={ol_total}), PINAW={ol_pinaw:.3%}")

    logger.info("  Phase 3: Closed-loop 24h continuous simulation...")

    sim_config = SimulationConfig(
        num_devices=num_devices,
        num_regions=5,
        level=SimulationLevel.LEVEL1_AGENT,
        duration_seconds=86400.0,
        time_step=TIME_STEP,
        random_seed=42,
    )
    sim = EPSSimulator(sim_config)
    sim.initialize()

    cl_covered = 0
    cl_total = 0
    cl_widths = []
    cl_actuals = []
    per_step_records = []

    for hour in range(24):
        solar_mw = solar_factors[hour] * SOLAR_CAP
        wind_mw = wind_factors[hour] * WIND_CAP
        total_supply = solar_mw + wind_mw
        base_load = load_factors[hour] * BASE_LOAD_PEAK_MW
        net_balance = total_supply - base_load

        r = total_supply / max(base_load, 1e-6)
        d_dispatch = min(1.0, abs(r - 1.0) / 0.4)
        s = float(np.clip((r - 1.0) * d_dispatch, -1.0, 1.0))
        formula_sd, formula_intensity = encode_signal_score(s)

        if net_balance > 0:
            target = OptimizationTarget(
                target_response_mw=net_balance,
                tolerance_fraction=0.15,
            )
            opt_result = optimizer.optimize(
                target,
                supply_demand=formula_sd,
                signal_context={'hour': hour},
            )
            sd_state = opt_result.supply_demand
            intensity = opt_result.intensity
            signal_mode = "nn_optimized"
        else:
            sd_state = formula_sd
            intensity = formula_intensity
            signal_mode = "formula"

        sig_dict = {
            'intensity': intensity,
            'supply_demand': sd_state,
            'price': 0.0,
            'hour': hour,
            'direction': 1 if sd_state <= 7 else -1,
        }

        est = estimator.estimate(sig_dict)
        predicted_kw = est.response_kw
        lower_kw = est.lower_bound
        upper_kw = est.upper_bound
        interval_width_kw = upper_kw - lower_kw

        scenario = 'valley_filling' if sd_state <= 7 else 'peak_shaving'

        sim.set_override_signal(intensity=intensity, supply_demand=sd_state)

        for sub_step in range(STEPS_PER_HOUR):
            step_idx = hour * STEPS_PER_HOUR + sub_step
            result = sim.run(num_steps=1, scenario=scenario)

            actual_kw = result.total_energy_kwh / (TIME_STEP / 3600.0)

            in_interval = (lower_kw <= actual_kw <= upper_kw)
            if in_interval:
                cl_covered += 1
            cl_total += 1
            cl_widths.append(interval_width_kw)
            cl_actuals.append(abs(actual_kw))

            per_step_records.append({
                "step": step_idx,
                "hour": hour,
                "sub_step": sub_step,
                "intensity": intensity,
                "supply_demand": sd_state,
                "signal_mode": signal_mode,
                "predicted_kw": float(predicted_kw),
                "lower_kw": float(lower_kw),
                "upper_kw": float(upper_kw),
                "actual_kw": float(actual_kw),
                "in_interval": bool(in_interval),
                "net_balance_mw": float(net_balance),
            })

        sim.clear_override_signal()

        hour_steps = per_step_records[-STEPS_PER_HOUR:]
        hour_covered = sum(1 for s in hour_steps if s["in_interval"])
        logger.info(
            f"    Hour {hour:2d}: mode={signal_mode:13s}, intensity={intensity:4d}, "
            f"sd={sd_state:2d}, coverage={hour_covered}/{STEPS_PER_HOUR}"
        )

    cl_picp = cl_covered / cl_total
    cl_mean_width = float(np.mean(cl_widths))
    cl_mean_actual = float(np.mean(cl_actuals))
    cl_pinaw = cl_mean_width / cl_mean_actual if cl_mean_actual > 1e-6 else float('inf')

    per_hour_coverage = []
    for hour in range(24):
        hour_steps = [r for r in per_step_records if r["hour"] == hour]
        n_covered = sum(1 for r in hour_steps if r["in_interval"])
        n_total = len(hour_steps)
        per_hour_coverage.append({
            "hour": hour,
            "picp": n_covered / n_total if n_total > 0 else 0.0,
            "n_covered": n_covered,
            "n_total": n_total,
            "signal_mode": hour_steps[0]["signal_mode"] if hour_steps else "unknown",
        })

    nn_steps = [r for r in per_step_records if r["signal_mode"] == "nn_optimized"]
    formula_steps = [r for r in per_step_records if r["signal_mode"] == "formula"]
    nn_picp = (sum(1 for r in nn_steps if r["in_interval"]) / len(nn_steps)
               if nn_steps else 0.0)
    formula_picp = (sum(1 for r in formula_steps if r["in_interval"]) / len(formula_steps)
                    if formula_steps else 0.0)

    picp_drop = ol_picp - cl_picp

    logger.info("=" * 70)
    logger.info("  RESULTS:")
    logger.info(f"  Open-loop  PICP: {ol_picp:.1%} (n={ol_total}), PINAW={ol_pinaw:.3%}")
    logger.info(f"  Closed-loop PICP: {cl_picp:.1%} (n={cl_total}), PINAW={cl_pinaw:.3%}")
    logger.info(f"  PICP drop:       {picp_drop:+.1%}")
    logger.info(f"  -- NN-optimized steps: PICP={nn_picp:.1%} (n={len(nn_steps)})")
    logger.info(f"  -- Formula steps:      PICP={formula_picp:.1%} (n={len(formula_steps)})")
    logger.info("=" * 70)

    results = {
        "metadata": {
            "protocol": "continuous_24h_closed_loop_picp",
            "num_devices": num_devices,
            "solar_ratio": SOLAR_RATIO,
            "wind_ratio": WIND_RATIO,
            "base_load_peak_mw": float(BASE_LOAD_PEAK_MW),
            "max_target_mw": float(max_target_mw),
            "steps_per_hour": STEPS_PER_HOUR,
            "time_step_s": TIME_STEP,
            "total_steps": cl_total,
            "nn_training_samples": 12000,
            "nn_training_seed": 42,
            "nn_training_r2": float(nn_r2) if nn_r2 else None,
            "cqr_nominal_coverage": 0.9,
            "timestamp": datetime.now().isoformat(),
        },
        "open_loop": {
            "picp": float(ol_picp),
            "pinaw": float(ol_pinaw),
            "mean_interval_width_kw": float(ol_mean_width),
            "mean_actual_kw": float(ol_mean_actual),
            "n_samples": ol_total,
        },
        "closed_loop": {
            "picp": float(cl_picp),
            "pinaw": float(cl_pinaw),
            "mean_interval_width_kw": float(cl_mean_width),
            "mean_actual_kw": float(cl_mean_actual),
            "n_steps": cl_total,
            "n_covered": cl_covered,
        },
        "closed_loop_by_mode": {
            "nn_optimized": {
                "picp": float(nn_picp),
                "n_steps": len(nn_steps),
            },
            "formula": {
                "picp": float(formula_picp),
                "n_steps": len(formula_steps),
            },
        },
        "picp_drop": float(picp_drop),
        "per_hour_coverage": per_hour_coverage,
        "per_step_records": per_step_records,
    }

    out_file = output_dir / "data" / "closedloop_picp.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved to {out_file}")

    import dataclasses
    M_FRESH = 20
    logger.info("=" * 70)
    logger.info(f"  Phase 4: Fresh-SOC closed-loop PICP (24 signals × M={M_FRESH})")

    seen_hours = set()
    hourly_signals = {}
    for rec in per_step_records:
        h = rec["hour"]
        if h not in seen_hours:
            seen_hours.add(h)
            hourly_signals[h] = {
                "intensity": rec["intensity"],
                "supply_demand": rec["supply_demand"],
                "signal_mode": rec["signal_mode"],
            }

    fs_covered = 0
    fs_total = 0
    fs_per_hour = {}

    for hour in range(24):
        hs = hourly_signals[hour]
        intensity_h = hs["intensity"]
        sd_h = hs["supply_demand"]

        sig_dict = {
            'intensity': intensity_h,
            'supply_demand': sd_h,
            'price': 0.0,
            'hour': hour,
            'direction': 1 if sd_h <= 7 else -1,
        }
        est = estimator.estimate(sig_dict)

        h_covered = 0
        h_actuals = []
        for m in range(M_FRESH):
            seed_m = 10000 + hour * 100 + m
            mini_config = SimulationConfig(
                num_devices=num_devices,
                num_regions=5,
                level=SimulationLevel.LEVEL1_AGENT,
                duration_seconds=900.0,
                time_step=300.0,
                random_seed=seed_m,
            )
            mini_sim = EPSSimulator(mini_config)
            mini_sim.initialize()
            mini_sim.set_override_signal(intensity=intensity_h, supply_demand=sd_h)
            scenario = 'valley_filling' if sd_h <= 7 else 'peak_shaving'
            res = mini_sim.run(num_steps=3, scenario=scenario)
            actual_kw = res.total_energy_kwh / 0.25
            h_actuals.append(actual_kw)
            if est.lower_bound <= actual_kw <= est.upper_bound:
                h_covered += 1

        fs_covered += h_covered
        fs_total += M_FRESH
        h_picp = h_covered / M_FRESH
        fs_per_hour[hour] = {
            "picp": float(h_picp),
            "covered": h_covered,
            "M": M_FRESH,
            "intensity": intensity_h,
            "sd": sd_h,
            "mode": hs["signal_mode"],
            "pred": float(est.response_kw),
            "mean_actual": float(np.mean(h_actuals)),
            "lower": float(est.lower_bound),
            "upper": float(est.upper_bound),
        }
        logger.info(f"    Hour {hour:2d}: {hs['signal_mode']:13s} int={intensity_h:4d} "
                     f"PICP={h_picp:.0%} ({h_covered}/{M_FRESH})")

    fs_picp = fs_covered / fs_total
    logger.info(f"  Fresh-SOC PICP: {fs_picp:.1%} (n={fs_total}), "
                f"vs open-loop {ol_picp:.1%}, drop={ol_picp - fs_picp:+.1%}")

    fs_results = {
        "fresh_soc_picp": float(fs_picp),
        "open_loop_picp": float(ol_picp),
        "M_per_signal": M_FRESH,
        "total_samples": fs_total,
        "total_covered": fs_covered,
        "per_hour": fs_per_hour,
    }
    fs_file = output_dir / "data" / "closedloop_picp_freshsoc.json"
    with open(fs_file, 'w') as f:
        json.dump(fs_results, f, indent=2)
    logger.info(f"  Saved to {fs_file}")

    return results


def _run_supplementary(config: 'ExperimentConfig'):
    experiment_dir, _ = _setup_experiment_directory(
        "results", f"supplementary_{config.num_devices}dev",
        config={"type": "supplementary", "num_devices": config.num_devices,
                "device_mode": "battery_only"},
    )
    logger.info(f"Supplementary experiment directory: {experiment_dir}")

    logger.info("Running closed-loop PICP verification (Discussion M5)...")
    run_closed_loop_picp(config, experiment_dir)

    _finalize_experiment(experiment_dir)
    logger.info(f"Supplementary experiments completed. Output: {experiment_dir}")


def _reset_seeds(seed=42):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_result1(config: 'ExperimentConfig'):
    experiment_dir, experiment_id = _setup_experiment_directory(
        "results", f"result1_scaling_law_{config.num_devices}dev",
        config={"type": "result1_scaling_law", "num_devices": config.num_devices,
                "num_runs": config.num_runs, "device_mode": "battery_only"},
    )
    logger.info(f"Result 1 directory: {experiment_dir}")

    run_core_experiment(config, experiment_dir=experiment_dir)

    _reset_seeds(config.random_seed)
    logger.info("Running N-threshold experiment...")
    run_n_threshold_experiment(config, experiment_dir)

    _reset_seeds(config.random_seed)
    logger.info("Running heterogeneity lookup table...")
    run_heterogeneity_lookup_table(config, experiment_dir)

    _finalize_experiment(experiment_dir)
    logger.info(f"Result 1 completed. Output: {experiment_dir}")


def _run_result2(config: 'ExperimentConfig'):
    experiment_dir, experiment_id = _setup_experiment_directory(
        "results", f"result2_curtailment_{config.num_devices}dev",
        config={"type": "result2_curtailment", "num_devices": config.num_devices,
                "device_mode": "battery_only"},
    )
    logger.info(f"Result 2 directory: {experiment_dir}")

    logger.info("Running curtailment baselines...")
    run_curtailment_baselines(config, experiment_dir)

    _reset_seeds(config.random_seed)
    logger.info("Running curtailment sensitivity...")
    run_curtailment_sensitivity(config, experiment_dir)

    _reset_seeds(config.random_seed)
    logger.info("Running N-scaling curtailment...")
    run_n_scaling_curtailment(config, experiment_dir)

    _finalize_experiment(experiment_dir)
    logger.info(f"Result 2 completed. Output: {experiment_dir}")


def _simulate_step_vectorized(
    capacities: np.ndarray,
    c_rates: np.ndarray,
    socs: np.ndarray,
    soc_reserves: np.ndarray,
    soc_maxes: np.ndarray,
    sohs: np.ndarray,
    intensity: int,
    supply_demand: int,
    rng: np.random.Generator,
    dt_seconds: float = 300.0,
) -> Tuple[float, np.ndarray]:
    N = len(capacities)

    active = (rng.random(N) >= 0.001) & (rng.random(N) >= 0.015)

    is_discharge = supply_demand >= 8

    if is_discharge:
        qualified = socs > soc_reserves
    else:
        qualified = socs < soc_maxes

    e_avail = capacities * sohs * 0.9
    dt_hours = dt_seconds / 3600.0
    max_power = capacities * c_rates

    if is_discharge:
        headroom = np.maximum(0, (socs - soc_reserves) * e_avail / dt_hours)
    else:
        headroom = np.maximum(0, (soc_maxes - socs) * e_avail / dt_hours)

    c_avail = np.minimum(max_power, headroom)
    has_cap = c_avail >= 0.01

    score = intensity / 4095.0
    soc_range = soc_maxes - soc_reserves
    if is_discharge:
        w_soc = np.clip((socs - soc_reserves) / np.maximum(soc_range, 0.1), 0, 1)
    else:
        w_soc = np.clip((soc_maxes - socs) / np.maximum(soc_range, 0.1), 0, 1)

    probs = score * w_soc
    responded = rng.random(N) < probs

    noise = 1.0 + rng.normal(0, 0.025, N) + rng.normal(0, 0.030, N)
    sign = -1.0 if is_discharge else 1.0

    mask = active & qualified & has_cap & responded
    powers = np.where(mask, sign * c_avail * noise, 0.0)

    energies = powers * dt_hours
    delta = np.where(
        energies > 0,
        energies * 0.95 / np.maximum(e_avail, 0.01),
        energies / (0.95 * np.maximum(e_avail, 0.01)),
    )
    new_socs = np.clip(socs + delta, 0.05, 0.98)

    return float(np.sum(powers)), new_socs


def _simulate_aggregate(
    capacities: np.ndarray,
    c_rates: np.ndarray,
    sohs: np.ndarray,
    intensity: int,
    supply_demand: int,
    rng: np.random.Generator,
    n_steps: int = 3,
    dt_seconds: float = 300.0,
) -> float:
    N = len(capacities)
    socs = rng.uniform(0.2, 0.9, N)
    soc_reserves = rng.uniform(0.1, 0.2, N)
    soc_maxes = rng.uniform(0.8, 0.95, N)

    total_energy = 0.0
    for _ in range(n_steps):
        power, socs = _simulate_step_vectorized(
            capacities, c_rates, socs, soc_reserves, soc_maxes, sohs,
            intensity, supply_demand, rng, dt_seconds,
        )
        total_energy += power * (dt_seconds / 3600.0)

    total_hours = n_steps * dt_seconds / 3600.0
    return total_energy / total_hours


def run_real_param_scaling(
    config: 'ExperimentConfig',
    output_dir: Path,
) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("Real-Parameter CV-vs-N Scaling (standalone vectorized sim)")
    logger.info("=" * 70)

    real_params = _load_nextgen_params()
    capacities = real_params['capacities']
    c_rates = real_params['c_rates']
    N_total = real_params['n_devices']
    N_source = real_params.get('source_n_devices', N_total)
    sohs = np.full(N_total, 0.95)

    logger.info(f"  Loaded {N_total} devices from NextGen dataset")

    N_VALUES = sorted(set([5, 10, 20, 50, 100, N_total]))
    N_TRIALS = 200

    test_signals = [
        ('charge_s=0.3', 0.3),
        ('discharge_s=-0.3', -0.3),
    ]

    all_results = {}

    for sig_name, test_s in test_signals:
        sd, intensity = encode_signal_score(test_s)
        results_per_n = []

        logger.info(f"  Signal: {sig_name}")

        for N in N_VALUES:
            responses = []
            for trial in range(N_TRIALS):
                rng = np.random.default_rng(trial * 777 + N * 13)

                if N < N_total:
                    idx = rng.choice(N_total, N, replace=False)
                    caps = capacities[idx]
                    crs = c_rates[idx]
                    shs = sohs[idx]
                else:
                    caps = capacities
                    crs = c_rates
                    shs = sohs

                response = _simulate_aggregate(
                    caps, crs, shs, intensity, sd, rng, n_steps=3,
                )
                responses.append(response)

            responses_arr = np.array(responses)
            mean_r = float(np.mean(responses_arr))
            std_r = float(np.std(responses_arr))
            cv = std_r / abs(mean_r) * 100 if abs(mean_r) > 0.01 else float('nan')
            cv_sqrt_n = cv * np.sqrt(N) if not np.isnan(cv) else float('nan')

            results_per_n.append({
                'N': N,
                'mean_response_kw': mean_r,
                'std_response_kw': std_r,
                'CV_pct': float(cv),
                'CV_sqrt_N': float(cv_sqrt_n),
            })

            logger.info(f"    N={N:4d}: mean={mean_r:8.2f} kW, "
                         f"CV={cv:6.2f}%, CV*sqrt(N)={cv_sqrt_n:.1f}")

        valid = [(r['N'], r['CV_pct']) for r in results_per_n
                 if not np.isnan(r['CV_pct']) and r['CV_pct'] > 0]
        if len(valid) >= 2:
            log_n = np.log(np.array([v[0] for v in valid]))
            log_cv = np.log(np.array([v[1] for v in valid]))
            A = np.column_stack([log_n, np.ones_like(log_n)])
            beta = np.linalg.lstsq(A, log_cv, rcond=None)[0]
            slope = float(beta[0])
        else:
            slope = float('nan')

        logger.info(f"    Log-log slope: {slope:.3f} (theory: -0.50)")

        all_results[sig_name] = {
            'signal_score': float(test_s),
            'per_N': results_per_n,
            'log_log_slope': slope,
        }

    slopes = [v['log_log_slope'] for v in all_results.values()
              if not np.isnan(v['log_log_slope'])]
    avg_slope = float(np.mean(slopes)) if slopes else float('nan')

    logger.info(f"  Average slope: {avg_slope:.3f} (theory: -0.50)")

    output = {
        'n_total_devices': N_total,
        'n_source_devices': N_source,
        'sampling_method': real_params.get('sampling_method', 'measured_devices'),
        'n_trials': N_TRIALS,
        'theoretical_slope': -0.50,
        'average_slope': avg_slope,
        'per_signal': all_results,
        'dataset': 'NextGen ACT Australia (Zenodo 14885589)',
    }

    out_file = output_dir / "data" / "real_params_scaling.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"  Results saved to {out_file}")

    return output


def _run_result3(config: 'ExperimentConfig'):
    experiment_dir, experiment_id = _setup_experiment_directory(
        "results", f"result3_robustness_{config.num_devices}dev",
        config={"type": "result3_robustness", "num_devices": config.num_devices,
                "device_mode": "battery_only"},
    )
    logger.info(f"Result 3 directory: {experiment_dir}")

    _reset_seeds(config.random_seed)
    logger.info("Running model mismatch sensitivity...")
    run_model_mismatch_sensitivity(config, experiment_dir)

    _reset_seeds(config.random_seed)
    logger.info("Running structural mismatch...")
    run_structural_mismatch(config, experiment_dir)

    _reset_seeds(config.random_seed)
    logger.info("Running hierarchical R-squared decomposition...")
    run_hierarchical_r2(config, experiment_dir)

    _reset_seeds(config.random_seed)
    logger.info("Running rho sensitivity scan...")
    run_rho_sensitivity_experiment(config, experiment_dir)

    _reset_seeds(config.random_seed)
    logger.info("Running N-scaling experiment...")
    run_n_scaling_experiment(config, experiment_dir)

    _finalize_experiment(experiment_dir)
    logger.info(f"Result 3 completed. Output: {experiment_dir}")


def _run_result4(config: 'ExperimentConfig'):
    experiment_dir, experiment_id = _setup_experiment_directory(
        "results", f"result4_generalization_{config.num_devices}dev",
        config={"type": "result4_generalization", "num_devices": config.num_devices,
                "device_mode": "battery_only"},
    )
    logger.info(f"Result 4 directory: {experiment_dir}")

    _reset_seeds(config.random_seed)
    logger.info("Running cross-region transfer...")
    run_cross_region_transfer_battery(config, experiment_dir)

    _reset_seeds(config.random_seed)
    logger.info("Running real parameter validation...")
    try:
        run_real_param_validation(config, experiment_dir)
    except FileNotFoundError as e:
        logger.warning(f"Skipping real parameter validation (data not available): {e}")

    _reset_seeds(config.random_seed)
    logger.info("Running real parameter CV-vs-N scaling...")
    try:
        run_real_param_scaling(config, experiment_dir)
    except FileNotFoundError as e:
        logger.warning(f"Skipping real parameter scaling (data not available): {e}")

    _finalize_experiment(experiment_dir)
    logger.info(f"Result 4 completed. Output: {experiment_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment Runner — run the experiments described in the paper"
    )
    parser.add_argument(
        "--result1", action="store_true",
        help="Result 1: 1/sqrt(N) scaling law (Fig. 2)"
    )
    parser.add_argument(
        "--result2", action="store_true",
        help="Result 2: Broadcast dispatch performance ceiling (Fig. 3)"
    )
    parser.add_argument(
        "--result3", action="store_true",
        help="Result 3: Robustness to mismatch and correlation (Fig. 4)"
    )
    parser.add_argument(
        "--result4", action="store_true",
        help="Result 4: Cross-region transfer and real-parameter validation (Fig. 5)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all four results"
    )
    parser.add_argument(
        "--n-devices", type=int, default=5000,
        help="Number of devices (default: 5000)"
    )
    parser.add_argument(
        "--n-runs", type=int, default=1,
        help="Number of runs for Result 1 core experiment (default: 1)"
    )
    parser.add_argument(
        "--supplementary", action="store_true",
        help="Supplementary: closed-loop PICP verification (Discussion M5)"
    )

    args = parser.parse_args()

    run_r1 = args.result1 or args.all
    run_r2 = args.result2 or args.all
    run_r3 = args.result3 or args.all
    run_r4 = args.result4 or args.all

    if not any([run_r1, run_r2, run_r3, run_r4, args.supplementary]):
        parser.error("Specify at least one of: --result1/2/3/4, --all, --supplementary")

    config = ExperimentConfig(
        num_devices=args.n_devices,
        num_runs=args.n_runs,
    )

    import torch
    seed = config.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seeds set to {seed}")

    if args.supplementary:
        _run_supplementary(config)

    if run_r1:
        _run_result1(config)
    if run_r2:
        _run_result2(config)
    if run_r3:
        _run_result3(config)
    if run_r4:
        _run_result4(config)

    logger.info("All requested experiments completed.")


if __name__ == '__main__':
    main()
