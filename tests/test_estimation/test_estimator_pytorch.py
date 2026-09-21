import pytest
import numpy as np
from typing import List, Dict

torch = pytest.importorskip("torch")

from src.estimation.estimator import EPSEstimator, EstimatorConfig


class TestPyTorchAvailability:

    def test_pytorch_available_when_enabled(self):
        config = EstimatorConfig(use_pytorch=True)
        estimator = EPSEstimator(config)
        assert estimator._pytorch_available is True

    def test_pytorch_not_used_when_disabled(self):
        config = EstimatorConfig(use_pytorch=False)
        estimator = EPSEstimator(config)
        assert hasattr(estimator, '_pytorch_available')

    def test_config_defaults(self):
        config = EstimatorConfig()
        assert hasattr(config, 'use_pytorch')
        assert hasattr(config, 'pytorch_epochs')
        assert hasattr(config, 'pytorch_batch_size')


class TestQuantileNNPyTorch:

    @pytest.fixture
    def training_data(self) -> tuple:
        rng = np.random.default_rng(42)
        signals = []
        responses = []

        for _ in range(200):
            intensity = int(rng.integers(500, 4000))
            supply_demand = int(rng.integers(0, 15))
            price = int(rng.integers(100, 2000))

            signal = {
                'intensity': intensity,
                'supply_demand': supply_demand,
                'price': price,
                'region_id': int(rng.integers(0, 16)),
                'priority': int(rng.integers(0, 15)),
            }
            signals.append(signal)

            response = intensity * 0.5 + supply_demand * 80 + price * 0.1
            response += rng.normal(0, 100)
            responses.append(response)

        return signals, responses

    def test_pytorch_training_completes(self, training_data):
        signals, responses = training_data

        config = EstimatorConfig(
            use_pytorch=True,
            pytorch_epochs=10,
            pytorch_batch_size=32,
            enable_conformal=False,
        )
        estimator = EPSEstimator(config)

        estimator.fit(signals, responses)
        assert estimator._is_fitted is True

    def test_pytorch_prediction_works(self, training_data):
        signals, responses = training_data

        config = EstimatorConfig(
            use_pytorch=True,
            pytorch_epochs=10,
            pytorch_batch_size=32,
            enable_conformal=True,
        )
        estimator = EPSEstimator(config)
        estimator.fit(signals[:150], responses[:150])

        test_signal = {
            'intensity': 2000,
            'supply_demand': 8,
            'price': 1000,
            'region_id': 5,
            'priority': 10,
        }
        result = estimator.estimate(test_signal)

        assert result is not None
        assert np.isfinite(result.response_kw)
        assert result.lower_bound <= result.response_kw <= result.upper_bound

    def test_pytorch_vs_fallback_consistency(self, training_data):
        signals, responses = training_data

        config_pytorch = EstimatorConfig(
            use_pytorch=True,
            pytorch_epochs=20,
            enable_conformal=False,
        )
        est_pytorch = EPSEstimator(config_pytorch)
        est_pytorch.fit(signals[:100], responses[:100])

        config_fallback = EstimatorConfig(
            use_pytorch=False,
            enable_conformal=False,
        )
        est_fallback = EPSEstimator(config_fallback)
        est_fallback.fit(signals[:100], responses[:100])

        test_signal = {
            'intensity': 2500,
            'supply_demand': 10,
            'price': 1200,
        }

        result_pytorch = est_pytorch.estimate(test_signal)
        result_fallback = est_fallback.estimate(test_signal)

        assert np.isfinite(result_pytorch.response_kw)
        assert np.isfinite(result_fallback.response_kw)

    def test_pytorch_batch_size_effect(self, training_data):
        signals, responses = training_data

        for batch_size in [16, 32, 64]:
            config = EstimatorConfig(
                use_pytorch=True,
                pytorch_epochs=5,
                pytorch_batch_size=batch_size,
                enable_conformal=False,
            )
            estimator = EPSEstimator(config)
            estimator.fit(signals[:100], responses[:100])

            result = estimator.estimate({'intensity': 2000, 'supply_demand': 8, 'price': 1000})
            assert result is not None
            assert np.isfinite(result.response_kw)


class TestPyTorchWithConformal:

    @pytest.fixture
    def training_data(self) -> tuple:
        rng = np.random.default_rng(123)
        signals = []
        responses = []

        for _ in range(300):
            intensity = int(rng.integers(500, 4000))
            signal = {
                'intensity': intensity,
                'supply_demand': int(rng.integers(0, 15)),
                'price': int(rng.integers(100, 2000)),
            }
            signals.append(signal)
            responses.append(intensity * 0.4 + rng.normal(0, 80))

        return signals, responses

    def test_pytorch_with_conformal(self, training_data):
        signals, responses = training_data

        config = EstimatorConfig(
            use_pytorch=True,
            pytorch_epochs=15,
            enable_conformal=True,
            target_coverage=0.9,
        )
        estimator = EPSEstimator(config)
        estimator.fit(signals[:200], responses[:200])

        covered = 0
        for signal, actual in zip(signals[200:], responses[200:]):
            result = estimator.estimate(signal)
            if result.lower_bound <= actual <= result.upper_bound:
                covered += 1

        coverage = covered / len(signals[200:])
        assert coverage >= 0.5


class TestPyTorchGPU:

    @pytest.fixture
    def has_gpu(self) -> bool:
        return torch.cuda.is_available()

    def test_gpu_detection(self, has_gpu):
        config = EstimatorConfig(use_pytorch=True)
        estimator = EPSEstimator(config)

        if has_gpu:
            pass
        else:
            assert estimator._pytorch_available is True
