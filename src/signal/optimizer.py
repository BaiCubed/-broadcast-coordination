from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
from enum import Enum
import numpy as np

from ..estimation.estimator import EPSEstimator, EstimationResult


class OptimizationConfidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ScenarioRiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class OptimizationTarget:
    target_response_mw: float
    tolerance_fraction: float = 0.1

    max_intensity: int = 4095
    min_intensity: int = 0

    prefer_low_intensity: bool = True

    risk_level: ScenarioRiskLevel = ScenarioRiskLevel.MODERATE
    require_interval_coverage: bool = False


@dataclass
class RiskAssessment:
    target_in_interval: bool
    interval_coverage_ratio: float
    interval_width_mw: float
    interval_lower_mw: float = 0.0
    interval_upper_mw: float = 0.0

    pinaw: float = 0.0

    confidence: OptimizationConfidence = OptimizationConfidence.MEDIUM
    risk_adjusted: bool = False
    safety_margin_applied: float = 0.0


@dataclass
class OptimizedSignal:
    intensity: int
    price_value: float
    supply_demand: int
    priority: int

    predicted_response_mw: float
    prediction_confidence: float
    prediction_interval: Tuple[float, float]

    iterations: int
    converged: bool
    target_achievable: bool

    risk_assessment: Optional[RiskAssessment] = None

    def to_signal_dict(self) -> Dict[str, Any]:
        return {
            'intensity': self.intensity,
            'price': self.price_value,
            'supply_demand': self.supply_demand,
            'priority': self.priority,
        }

    @property
    def confidence_level(self) -> str:
        if self.risk_assessment:
            return self.risk_assessment.confidence.value
        return "unknown"


class SignalOptimizer:

    PINAW_THRESHOLD_CHARGE = 0.30
    PINAW_THRESHOLD_DISCHARGE = 0.35

    MIN_ABSOLUTE_TOL_KW = 30.0
    PREDICTOR_FLOOR_KW = 50.0
    MIN_GRID_RESOLUTION_KW = 50.0
    MAX_RELATIVE_TOL = 0.10

    UNCERTAINTY_FACTOR = 0.4

    def __init__(
        self,
        estimator: EPSEstimator,
        device_states: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.estimator = estimator
        self.device_states = device_states or {}

    def _infer_risk_level(
        self,
        priority: int,
        supply_demand: int,
        hour: Optional[int] = None,
    ) -> ScenarioRiskLevel:
        if priority >= 12:
            return ScenarioRiskLevel.CONSERVATIVE

        return ScenarioRiskLevel.MODERATE

    def optimize(
        self,
        target: OptimizationTarget,
        supply_demand: int = 8,
        priority: int = 8,
        signal_context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 20,
        auto_infer_risk_level: bool = True,
    ) -> OptimizedSignal:
        hour = signal_context.get('hour') if signal_context else None
        if auto_infer_risk_level:
            inferred_risk_level = self._infer_risk_level(priority, supply_demand, hour)
        else:
            inferred_risk_level = target.risk_level

        if target.target_response_mw > 0 and supply_demand >= 8:
            supply_demand = 4
        elif target.target_response_mw < 0 and supply_demand < 8:
            supply_demand = 12

        effective_target = OptimizationTarget(
            target_response_mw=target.target_response_mw,
            tolerance_fraction=target.tolerance_fraction,
            max_intensity=target.max_intensity,
            min_intensity=target.min_intensity,
            prefer_low_intensity=target.prefer_low_intensity,
            risk_level=inferred_risk_level,
            require_interval_coverage=target.require_interval_coverage,
        )

        target_kw = effective_target.target_response_mw * 1000

        best_intensity, best_error = self._intensity_grid_search(
            effective_target, supply_demand, priority, max_iterations, signal_context
        )

        final_signal = {
            'intensity': best_intensity,
            'price': 0,
            'supply_demand': supply_demand,
            'priority': priority,
        }
        if signal_context:
            final_signal.update(signal_context)
        final_result = self.estimator.estimate(
            final_signal, device_states=self.device_states
        )

        risk_assessment = self._assess_risk(target, final_result)

        target_kw = target.target_response_mw * 1000
        base_tol_kw = max(abs(target_kw), 1e-6) * target.tolerance_fraction
        dynamic_tol_kw = min(
            max(base_tol_kw, self.PREDICTOR_FLOOR_KW, self.MIN_GRID_RESOLUTION_KW),
            abs(target_kw) * self.MAX_RELATIVE_TOL
        )
        point_close = abs(final_result.response_kw - target_kw) <= dynamic_tol_kw
        converged = point_close

        return OptimizedSignal(
            intensity=best_intensity,
            price_value=0.0,
            supply_demand=supply_demand,
            priority=priority,
            predicted_response_mw=final_result.response_kw / 1000,
            prediction_confidence=final_result.confidence,
            prediction_interval=(
                final_result.lower_bound / 1000,
                final_result.upper_bound / 1000
            ),
            iterations=max_iterations,
            converged=converged,
            target_achievable=point_close,
            risk_assessment=risk_assessment,
        )

    def _assess_risk(
        self,
        target: OptimizationTarget,
        result: EstimationResult,
    ) -> RiskAssessment:
        target_kw = target.target_response_mw * 1000
        target_low_raw = target_kw * (1 - target.tolerance_fraction)
        target_high_raw = target_kw * (1 + target.tolerance_fraction)
        target_low = min(target_low_raw, target_high_raw)
        target_high = max(target_low_raw, target_high_raw)

        lower_kw = result.lower_bound
        upper_kw = result.upper_bound
        interval_width_kw = upper_kw - lower_kw

        target_in_interval = (lower_kw <= target_kw <= upper_kw)

        overlap_low = max(lower_kw, target_low)
        overlap_high = min(upper_kw, target_high)
        target_range = target_high - target_low

        if overlap_high > overlap_low and target_range > 0:
            interval_coverage_ratio = (overlap_high - overlap_low) / target_range
        else:
            interval_coverage_ratio = 0.0

        pinaw = interval_width_kw / max(abs(result.response_kw), 1.0)

        interval_ok = target_in_interval

        error_pct = abs(result.response_kw - target_kw) / max(abs(target_kw), 1.0)
        error_ok = error_pct <= target.tolerance_fraction

        pass_count = sum([interval_ok, error_ok])
        if pass_count >= 2:
            confidence = OptimizationConfidence.HIGH
        elif pass_count == 1:
            confidence = OptimizationConfidence.MEDIUM
        else:
            confidence = OptimizationConfidence.LOW

        return RiskAssessment(
            target_in_interval=target_in_interval,
            interval_coverage_ratio=interval_coverage_ratio,
            interval_width_mw=interval_width_kw / 1000,
            interval_lower_mw=lower_kw / 1000,
            interval_upper_mw=upper_kw / 1000,
            pinaw=pinaw,
            confidence=confidence,
        )

    def _intensity_grid_search(
        self,
        target: OptimizationTarget,
        supply_demand: int,
        priority: int,
        max_iterations: int,
        signal_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, float]:
        target_kw = target.target_response_mw * 1000
        base_tol_kw = max(abs(target_kw), 1e-6) * target.tolerance_fraction

        int_low, int_high = target.min_intensity, target.max_intensity

        def evaluate(intensity: int) -> Tuple[float, float, float]:
            signal = {
                'intensity': intensity,
                'price': 0,
                'supply_demand': supply_demand,
                'priority': priority,
            }
            if signal_context:
                signal.update(signal_context)
            result = self.estimator.estimate(signal, device_states=self.device_states)
            error = abs(result.response_kw - target_kw)
            return result.response_kw, error, result.interval_width

        def get_dynamic_tol(interval_width_kw: float) -> float:
            return min(
                max(base_tol_kw, self.PREDICTOR_FLOOR_KW, self.MIN_GRID_RESOLUTION_KW),
                abs(target_kw) * self.MAX_RELATIVE_TOL
            )

        coarse_budget = max(8, int(max_iterations * 0.6))
        coarse_step = max(1, (int_high - int_low) // coarse_budget)

        best_intensity = (int_low + int_high) // 2
        best_error = float('inf')

        for i in range(coarse_budget + 1):
            intensity = min(int_low + i * coarse_step, int_high)
            _, error, _ = evaluate(intensity)
            if error < best_error:
                best_error = error
                best_intensity = intensity

        fine_budget = max_iterations - coarse_budget
        if fine_budget > 0:
            fine_low = max(int_low, best_intensity - coarse_step)
            fine_high = min(int_high, best_intensity + coarse_step)
            fine_step = max(1, (fine_high - fine_low) // max(1, fine_budget))

            for i in range(fine_budget + 1):
                intensity = min(fine_low + i * fine_step, fine_high)
                _, error, interval_width = evaluate(intensity)
                if error < best_error:
                    best_error = error
                    best_intensity = intensity
                if error <= get_dynamic_tol(interval_width):
                    break

        return best_intensity, best_error

    def find_feasible_range(
        self,
        supply_demand: int = 8,
        priority: int = 8,
    ) -> Dict[str, Tuple[float, float]]:
        min_signal = {
            'intensity': 100,
            'price': 0,
            'supply_demand': supply_demand,
            'priority': priority,
        }
        min_result = self.estimator.estimate(min_signal, device_states=self.device_states)

        max_signal = {
            'intensity': 4095,
            'price': 0,
            'supply_demand': supply_demand,
            'priority': priority,
        }
        max_result = self.estimator.estimate(max_signal, device_states=self.device_states)

        return {
            'min_response': (min_result.response_kw, min_result.upper_bound),
            'max_response': (max_result.response_kw, max_result.upper_bound),
            'controllable_range': (min_result.response_kw, max_result.response_kw),
        }


