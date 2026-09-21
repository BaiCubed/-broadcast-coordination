from .estimator import EPSEstimator, EstimationResult, EstimatorConfig

from .conformal import (
    ConformalMethod,
    PredictionInterval,
    NonconformityScore,
    CalibrationSet,
    ConformalPredictor,
    CoverageTracker,
)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = [
    "EPSEstimator",
    "EstimationResult",
    "EstimatorConfig",
    "ConformalMethod",
    "PredictionInterval",
    "NonconformityScore",
    "CalibrationSet",
    "ConformalPredictor",
    "CoverageTracker",
    "TORCH_AVAILABLE",
]
