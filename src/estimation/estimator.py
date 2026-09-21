from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import time
import math

from .conformal import (
    ConformalPredictor,
    CoverageTracker,
    PredictionInterval,
    CQRPredictor,
)
@dataclass
class EstimationResult:

    response_kw: float

    lower_bound: float
    upper_bound: float

    coverage: float = 0.9

    confidence: float = 0.5

    timestamp: float = field(default_factory=time.time)

    tail_risk: Optional[Any] = None

    layer_contributions: Optional[Dict[str, float]] = None

    @property
    def interval_width(self) -> float:
        return self.upper_bound - self.lower_bound

    @property
    def relative_uncertainty(self) -> float:
        if abs(self.response_kw) < 1e-6:
            return float('inf')
        return self.interval_width / abs(self.response_kw)

    def is_within_interval(self, actual: float) -> bool:
        return self.lower_bound <= actual <= self.upper_bound


@dataclass
class EstimatorConfig:

    target_coverage: float = 0.9

    min_confidence: float = 0.3

    history_window: float = 3600.0

    enable_conformal: bool = True

    use_cqr: bool = True

    conformal_adaptive: bool = True
    conformal_window_size: int = 100
    conformal_coverage_margin: float = 0.0

    online_validation_window: int = 50
    online_validation_patience: int = 3

    use_pytorch: bool = False
    pytorch_epochs: int = 100
    pytorch_batch_size: int = 32
    pytorch_learning_rate: float = 0.001

    use_single_nn: bool = False


class EPSEstimator:

    def __init__(self, config: Optional[EstimatorConfig] = None):
        self.config = config or EstimatorConfig()
        self._history: List[Dict[str, Any]] = []
        self._is_fitted = False

        self._validation_window: List[Dict[str, Any]] = []
        self._validation_errors: List[float] = []
        self._nn_update_frozen = False
        self._validation_deterioration_count = 0

        self._pytorch_available = False
        if self.config.use_pytorch:
            try:
                import torch
                self._pytorch_available = True
            except ImportError:
                self._pytorch_available = False

        self._conformal: Optional[ConformalPredictor] = None
        self._cqr: Optional[CQRPredictor] = None
        self._coverage_tracker: Optional[CoverageTracker] = None
        if self.config.enable_conformal:
            self._init_conformal()

        self._aggregate_nn: Optional[Any] = None
        self._is_quantile_model: bool = False
        self._learned_params: Optional[Dict[str, Any]] = None

    def _init_conformal(self) -> None:
        from .conformal import ConformalConfig, ConformalMethod

        conformal_config = ConformalConfig(
            target_coverage=self.config.target_coverage,
            coverage_margin=self.config.conformal_coverage_margin,
            adaptive_window=self.config.conformal_window_size,
            method=ConformalMethod.SPLIT,
            use_difficulty_adjustment=False,
        )

        self._conformal = ConformalPredictor(
            config=conformal_config,
            score_type="absolute",
        )

        if self.config.use_cqr:
            self._cqr = CQRPredictor(config=conformal_config)

        self._coverage_tracker = CoverageTracker(
            window_size=self.config.conformal_window_size,
        )

    def _extract_features(
        self,
        signal: Dict[str, Any],
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[float]:
        intensity = signal.get('intensity', 2048) / 4095.0
        supply_demand = signal.get('supply_demand', 8) / 15.0
        region_id = signal.get('region_id', 0) / 16.0
        priority = signal.get('priority', 8) / 15.0

        direction = 2.0 * (supply_demand - 0.5)

        hour = signal.get('hour', 12)
        if isinstance(hour, float):
            hour = int(hour) % 24
        else:
            hour = hour % 24

        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)

        is_peak = 1.0 if (7 <= hour <= 11) or (17 <= hour <= 21) else 0.0

        day_of_week = signal.get('day_of_week', None)
        if day_of_week is not None:
            is_weekend = 1.0 if day_of_week >= 5 else 0.0
        else:
            is_weekend = 0.0

        features = [
            intensity, supply_demand, region_id, priority,
            intensity * direction, direction,
            hour_sin, hour_cos, is_peak, is_weekend,
        ]

        return features

    def fit(
        self,
        historical_signals: List[Dict[str, Any]],
        historical_responses: List[float],
        context_features: Optional[List[Dict[str, Any]]] = None,
        warm_start_state: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> 'EPSEstimator':
        if len(historical_signals) == 0 or len(historical_responses) == 0:
            raise ValueError("Cannot fit with empty signals or responses")

        if len(historical_signals) != len(historical_responses):
            raise ValueError(
                f"Signal count ({len(historical_signals)}) must match "
                f"response count ({len(historical_responses)})"
            )

        n_samples = len(historical_signals)

        for i, (signal, response) in enumerate(zip(historical_signals, historical_responses)):
            entry = {
                'signal': signal,
                'response': response,
                'context': context_features[i] if context_features else None,
                'timestamp': time.time() - (n_samples - i) * 60,
            }
            self._history.append(entry)

        import numpy as np
        if not hasattr(self, '_fit_count'):
            self._fit_count = 0
        self._fit_count += 1
        rng_split = np.random.default_rng(42 + self._fit_count)
        pos_idx = [i for i, r in enumerate(historical_responses) if r > 0]
        neg_idx = [i for i, r in enumerate(historical_responses) if r <= 0]
        pos_idx_arr = np.array(pos_idx)
        neg_idx_arr = np.array(neg_idx)
        rng_split.shuffle(pos_idx_arr)
        rng_split.shuffle(neg_idx_arr)

        pos_split = int(len(pos_idx_arr) * 0.7)
        neg_split = int(len(neg_idx_arr) * 0.7)

        train_indices = sorted(pos_idx_arr[:pos_split].tolist() + neg_idx_arr[:neg_split].tolist())
        cal_indices = sorted(pos_idx_arr[pos_split:].tolist() + neg_idx_arr[neg_split:].tolist())

        train_signals = [historical_signals[i] for i in train_indices]
        train_responses = [historical_responses[i] for i in train_indices]
        cal_signals = [historical_signals[i] for i in cal_indices]
        cal_responses = [historical_responses[i] for i in cal_indices]

        self._fit_regression(train_signals, train_responses, warm_start_state)

        if self.config.enable_conformal:
            self._fit_conformal_from_split(cal_signals, cal_responses)

        self._is_fitted = True
        return self

    def _fit_regression(
        self,
        signals: List[Dict[str, Any]],
        responses: List[float],
        warm_start_state: Optional[Dict[str, Any]] = None
    ) -> None:
        if len(signals) < 10:
            self._learned_params = None
            return

        n = len(signals)

        X = []
        training_history = []
        for i, signal in enumerate(signals):
            features = self._extract_features(signal, history=training_history)
            X.append(features)
            training_history.append({
                'signal': signal,
                'response': responses[i],
            })

        y = list(responses)

        if self.config.use_pytorch:
            try:
                import torch
                import torch.nn as nn
                import torch.optim as optim

                class QuantileAggregateNN(nn.Module):
                    def __init__(self, input_dim):
                        super().__init__()
                        self.shared = nn.Sequential(
                            nn.Linear(input_dim, 128),
                            nn.ReLU(),
                            nn.Dropout(0.15),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Dropout(0.15),
                            nn.Linear(64, 32),
                            nn.ReLU(),
                        )
                        self.q10_head = nn.Linear(32, 1)
                        self.q50_head = nn.Linear(32, 1)
                        self.q90_head = nn.Linear(32, 1)

                    def forward(self, x):
                        h = self.shared(x)
                        q10 = self.q10_head(h)
                        q50 = self.q50_head(h)
                        q90 = self.q90_head(h)
                        return torch.cat([q10, q50, q90], dim=1)

                def huber_quantile_loss(pred, target, quantiles=[0.1, 0.5, 0.9], weights=None, delta=1.0):
                    losses = []
                    for i, q in enumerate(quantiles):
                        pred_q = pred[:, i:i+1]
                        error = target - pred_q
                        abs_error = torch.abs(error)
                        huber_mask = (abs_error <= delta).float()
                        quadratic = 0.5 * error ** 2
                        linear = delta * (abs_error - 0.5 * delta)
                        huber_error = huber_mask * quadratic + (1 - huber_mask) * linear
                        quantile_weight = torch.where(error >= 0, q, 1 - q)
                        loss_q = quantile_weight * huber_error
                        losses.append(loss_q)
                    total_loss = torch.cat(losses, dim=1).mean(dim=1)

                    if weights is not None:
                        total_loss = total_loss * weights

                    return total_loss.mean()

                def train_single_nn(X_data, y_data, input_dim, epochs, batch_size, warm_start=None,
                                    quantiles=[0.1, 0.5, 0.9]):
                    if len(X_data) < 10:
                        return None, 0.0, 1.0, 0.0, {'epochs': [], 'train_loss': [], 'val_loss': []}

                    n_samples = len(X_data)

                    val_size = max(int(n_samples * 0.1), 2)
                    train_size = n_samples - val_size

                    import random
                    indices = list(range(n_samples))
                    random.Random(42).shuffle(indices)
                    train_indices = indices[:train_size]
                    val_indices = indices[train_size:]

                    train_y = [y_data[i] for i in train_indices]
                    y_mean = sum(train_y) / len(train_y)
                    y_std = (sum((yi - y_mean) ** 2 for yi in train_y) / len(train_y)) ** 0.5
                    if y_std < 1e-6:
                        y_std = 1.0
                    y_normalized = [(yi - y_mean) / y_std for yi in y_data]

                    X_tensor = torch.tensor(X_data, dtype=torch.float32)
                    y_tensor = torch.tensor(y_normalized, dtype=torch.float32).unsqueeze(1)

                    X_train = X_tensor[train_indices]
                    y_train = y_tensor[train_indices]
                    X_val = X_tensor[val_indices]
                    y_val = y_tensor[val_indices]

                    y_abs = torch.tensor([abs(yi) for yi in y_data], dtype=torch.float32)
                    y_abs_median = torch.median(y_abs) if len(y_abs) > 0 else torch.tensor(1.0)
                    sample_weights = 1.0 + torch.log1p(y_abs / (y_abs_median + 1e-6))
                    sample_weights = sample_weights / sample_weights.mean()
                    train_weights = sample_weights[train_indices]

                    model = QuantileAggregateNN(input_dim)

                    use_warm_start = False
                    if warm_start is not None:
                        try:
                            model.load_state_dict(warm_start)
                            use_warm_start = True
                        except Exception:
                            pass

                    lr = 0.001 if use_warm_start else 0.003
                    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
                    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=20, T_mult=2, eta_min=1e-5
                    )

                    best_loss = float('inf')
                    best_model_state = None
                    patience = 40
                    patience_counter = 0

                    training_history = {
                        'epochs': [],
                        'train_loss': [],
                        'val_loss': [],
                    }

                    for epoch in range(epochs):
                        model.train()
                        train_idx_shuffled = torch.randperm(train_size)
                        total_train_loss = 0.0
                        n_batches = 0

                        for i in range(0, train_size, batch_size):
                            batch_idx = train_idx_shuffled[i:i+batch_size]
                            if len(batch_idx) < 2:
                                continue
                            batch_X = X_train[batch_idx]
                            batch_y = y_train[batch_idx]
                            batch_weights = train_weights[batch_idx]

                            optimizer.zero_grad()
                            pred = model(batch_X)
                            loss = huber_quantile_loss(pred, batch_y, quantiles=quantiles,
                                                       weights=batch_weights, delta=1.0)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()

                            total_train_loss += loss.item()
                            n_batches += 1

                        avg_train_loss = total_train_loss / max(n_batches, 1)

                        model.eval()
                        with torch.no_grad():
                            val_pred = model(X_val)
                            val_loss = huber_quantile_loss(val_pred, y_val, quantiles=quantiles,
                                                           delta=1.0).item()

                        scheduler.step()

                        training_history['epochs'].append(epoch + 1)
                        training_history['train_loss'].append(float(avg_train_loss))
                        training_history['val_loss'].append(float(val_loss))

                        if val_loss < best_loss - 1e-5:
                            best_loss = val_loss
                            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                break

                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)

                    model.eval()
                    with torch.no_grad():
                        pred_all = model(X_tensor)
                        pred_normalized = pred_all[:, 1].tolist()
                        predictions = [p * y_std + y_mean for p in pred_normalized]

                    mse = sum((p - a) ** 2 for p, a in zip(predictions, y_data)) / n_samples
                    var_y = sum((yi - y_mean) ** 2 for yi in y_data) / n_samples
                    r2 = 1 - mse / max(var_y, 1e-6)

                    return model, y_mean, y_std, r2, training_history

                input_dim = len(X[0])
                epochs = max(self.config.pytorch_epochs, 150)
                batch_size = min(self.config.pytorch_batch_size, n)

                if self.config.use_single_nn:
                    unified_warm_start = None
                    if warm_start_state is not None:
                        unified_warm_start = warm_start_state.get('charge_nn')

                    unified_nn, unified_y_mean, unified_y_std, unified_r2, unified_history = train_single_nn(
                        X, y, input_dim, epochs, batch_size, unified_warm_start,
                        quantiles=[0.1, 0.5, 0.9],
                    )

                    self._charge_nn = unified_nn
                    self._discharge_nn = unified_nn
                    self._is_dual_model = False
                    self._is_quantile_model = True
                    self._aggregate_nn = unified_nn

                    all_predictions = []
                    if unified_nn:
                        unified_nn.eval()
                        with torch.no_grad():
                            X_tensor_all = torch.tensor(X, dtype=torch.float32)
                            pred_all = unified_nn(X_tensor_all)
                            all_predictions = [
                                pred_all[i, 1].item() * unified_y_std + unified_y_mean
                                for i in range(n)
                            ]
                    else:
                        all_predictions = [0.0] * n

                    mse = sum((p - a) ** 2 for p, a in zip(all_predictions, y)) / n
                    y_mean_all = sum(y) / n
                    var_y = sum((yi - y_mean_all) ** 2 for yi in y) / n
                    r2_overall = 1 - mse / max(var_y, 1e-6)

                    residuals = [p - a for p, a in zip(all_predictions, y)]
                    residual_std = (sum(r ** 2 for r in residuals) / n) ** 0.5

                    self._learned_params = {
                        'model_type': 'dual_quantile_nn',
                        'charge_model': unified_nn,
                        'charge_y_mean': unified_y_mean,
                        'charge_y_std': unified_y_std,
                        'charge_r2': unified_r2,
                        'charge_n_samples': n,
                        'discharge_model': unified_nn,
                        'discharge_y_mean': unified_y_mean,
                        'discharge_y_std': unified_y_std,
                        'discharge_r2': unified_r2,
                        'discharge_n_samples': n,
                        'n_samples': n,
                        'r2': r2_overall,
                        'residual_std': residual_std,
                        'quantiles': [0.1, 0.5, 0.9],
                        'charge_training_history': unified_history,
                        'discharge_training_history': unified_history,
                    }
                    return

                else:
                    charge_indices = [i for i in range(n) if y[i] >= 0]
                    discharge_indices = [i for i in range(n) if y[i] < 0]

                    X_charge = [X[i] for i in charge_indices]
                    y_charge = [y[i] for i in charge_indices]

                    X_discharge = [X[i] for i in discharge_indices]
                    y_discharge = [y[i] for i in discharge_indices]

                    charge_warm_start = None
                    discharge_warm_start = None
                    if warm_start_state is not None:
                        charge_warm_start = warm_start_state.get('charge_nn')
                        discharge_warm_start = warm_start_state.get('discharge_nn')

                    charge_nn, charge_y_mean, charge_y_std, charge_r2, charge_history = train_single_nn(
                        X_charge, y_charge, input_dim, epochs, batch_size, charge_warm_start,
                        quantiles=[0.1, 0.5, 0.9],
                    )

                    discharge_nn, discharge_y_mean, discharge_y_std, discharge_r2, discharge_history = train_single_nn(
                        X_discharge, y_discharge, input_dim, epochs, batch_size, discharge_warm_start,
                        quantiles=[0.1, 0.5, 0.9],
                    )

                    self._charge_nn = charge_nn
                    self._discharge_nn = discharge_nn
                    self._is_dual_model = True
                    self._is_quantile_model = True

                    self._aggregate_nn = charge_nn if charge_nn else discharge_nn

                    all_predictions = []
                    for i in range(n):
                        if y[i] >= 0 and charge_nn:
                            model = charge_nn
                            y_mean_local = charge_y_mean
                            y_std_local = charge_y_std
                        elif y[i] < 0 and discharge_nn:
                            model = discharge_nn
                            y_mean_local = discharge_y_mean
                            y_std_local = discharge_y_std
                        else:
                            model = self._aggregate_nn
                            y_mean_local = charge_y_mean if charge_nn else discharge_y_mean
                            y_std_local = charge_y_std if charge_nn else discharge_y_std

                        if model:
                            model.eval()
                            with torch.no_grad():
                                x_tensor = torch.tensor([X[i]], dtype=torch.float32)
                                pred = model(x_tensor)
                                pred_val = pred[0, 1].item() * y_std_local + y_mean_local
                                all_predictions.append(pred_val)
                        else:
                            all_predictions.append(0.0)

                    mse = sum((p - a) ** 2 for p, a in zip(all_predictions, y)) / n
                    y_mean_all = sum(y) / n
                    var_y = sum((yi - y_mean_all) ** 2 for yi in y) / n
                    r2_overall = 1 - mse / max(var_y, 1e-6)

                    residuals = [p - a for p, a in zip(all_predictions, y)]
                    residual_std = (sum(r ** 2 for r in residuals) / n) ** 0.5

                    self._learned_params = {
                        'model_type': 'dual_quantile_nn',
                        'charge_model': charge_nn,
                        'charge_y_mean': charge_y_mean,
                        'charge_y_std': charge_y_std,
                        'charge_r2': charge_r2,
                        'charge_n_samples': len(charge_indices),
                        'discharge_model': discharge_nn,
                        'discharge_y_mean': discharge_y_mean,
                        'discharge_y_std': discharge_y_std,
                        'discharge_r2': discharge_r2,
                        'discharge_n_samples': len(discharge_indices),
                        'n_samples': n,
                        'r2': r2_overall,
                        'residual_std': residual_std,
                        'quantiles': [0.1, 0.5, 0.9],
                        'charge_training_history': charge_history,
                        'discharge_training_history': discharge_history,
                    }
                return

            except ImportError:
                pass
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"PyTorch quantile NN training failed: {type(e).__name__}: {e}. "
                    "Falling back to linear regression."
                )

        X_with_bias = [[1.0] + x for x in X]
        n_features = len(X_with_bias[0])

        XtX = [[0.0] * n_features for _ in range(n_features)]
        for i in range(n_features):
            for j in range(n_features):
                XtX[i][j] = sum(X_with_bias[k][i] * X_with_bias[k][j] for k in range(n))

        Xty = [sum(X_with_bias[k][i] * y[k] for k in range(n)) for i in range(n_features)]

        lambda_reg = 1e-3
        for i in range(n_features):
            XtX[i][i] += lambda_reg

        weights = self._solve_linear_system(XtX, Xty)

        predictions = []
        for x in X_with_bias:
            pred = sum(w * f for w, f in zip(weights, x))
            predictions.append(pred)

        mse = sum((p - a) ** 2 for p, a in zip(predictions, y)) / n
        r2 = 1 - mse / max(sum((r - sum(y)/n) ** 2 for r in y) / n, 1e-6)

        residuals = [p - a for p, a in zip(predictions, y)]
        residual_std = (sum(r ** 2 for r in residuals) / n) ** 0.5

        self._learned_params = {
            'model_type': 'linear',
            'weights': weights,
            'n_samples': n,
            'mse': mse,
            'r2': r2,
            'response_mean': sum(responses) / n,
            'response_std': (sum((r - sum(responses)/n) ** 2 for r in responses) / n) ** 0.5,
            'residual_std': residual_std,
        }

    def _solve_linear_system(self, A: List[List[float]], b: List[float]) -> List[float]:
        n = len(b)
        aug = [row[:] + [b[i]] for i, row in enumerate(A)]

        for i in range(n):
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            aug[i], aug[max_row] = aug[max_row], aug[i]

            if abs(aug[i][i]) < 1e-10:
                continue

            for k in range(i + 1, n):
                factor = aug[k][i] / aug[i][i]
                for j in range(i, n + 1):
                    aug[k][j] -= factor * aug[i][j]

        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            if abs(aug[i][i]) < 1e-10:
                x[i] = 0.0
            else:
                x[i] = (aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))) / aug[i][i]

        return x

    def _simple_predict(self, signal: Dict[str, Any]) -> float:
        if hasattr(self, '_learned_params') and self._learned_params is not None:
            return self._learned_predict(signal)

        intensity = signal.get('intensity', 2048)
        supply_demand = signal.get('supply_demand', 8)

        base = (intensity / 4095) * 100
        sd_factor = 1.0 + max(0, (supply_demand - 8) / 7) * 0.3

        return base * sd_factor

    def _learned_predict(self, signal: Dict[str, Any]) -> float:
        features = self._extract_features(signal)

        model_type = self._learned_params.get('model_type', 'linear')

        if model_type == 'dual_quantile_nn':
            try:
                import torch

                supply_demand = signal.get('supply_demand', 8)

                if supply_demand < 8:
                    model = self._learned_params.get('charge_model')
                    y_mean = self._learned_params.get('charge_y_mean', 0)
                    y_std = self._learned_params.get('charge_y_std', 1)
                else:
                    model = self._learned_params.get('discharge_model')
                    y_mean = self._learned_params.get('discharge_y_mean', 0)
                    y_std = self._learned_params.get('discharge_y_std', 1)

                if model is None:
                    model = self._learned_params.get('charge_model') or self._learned_params.get('discharge_model')
                    y_mean = self._learned_params.get('charge_y_mean', self._learned_params.get('discharge_y_mean', 0))
                    y_std = self._learned_params.get('charge_y_std', self._learned_params.get('discharge_y_std', 1))

                if model:
                    model.eval()
                    with torch.no_grad():
                        x = torch.tensor([features], dtype=torch.float32)
                        pred_output = model(x)
                        pred_normalized = pred_output[0, 1].item()
                        prediction = pred_normalized * y_std + y_mean
                    return prediction
            except Exception:
                pass

        if 'weights' in self._learned_params:
            features_with_bias = [1.0] + features
            prediction = sum(w * f for w, f in zip(self._learned_params['weights'], features_with_bias))
            return prediction

        return 0.0

    def _fit_conformal_from_split(
        self,
        cal_signals: List[Dict[str, Any]],
        cal_responses: List[float],
    ) -> None:
        n = len(cal_signals)
        if n < 20:
            return

        cal_predictions = []
        cal_q10_predictions = []
        cal_q90_predictions = []
        cal_actuals = list(cal_responses)

        has_quantile_nn = (
            (hasattr(self, '_aggregate_nn') and self._aggregate_nn is not None) or
            (hasattr(self, '_charge_nn') and self._charge_nn is not None) or
            (hasattr(self, '_discharge_nn') and self._discharge_nn is not None)
        )
        use_cqr = (
            self.config.use_cqr
            and self._cqr is not None
            and has_quantile_nn
            and getattr(self, '_is_quantile_model', False)
        )

        for signal in cal_signals:
            if use_cqr:
                q10, q50, q90 = self._get_quantile_predictions(signal)
                cal_predictions.append(q50)
                cal_q10_predictions.append(q10)
                cal_q90_predictions.append(q90)
            else:
                pred = self._simple_predict(signal)
                cal_predictions.append(pred)

        self._conformal.calibrate(
            y_true=cal_actuals,
            y_pred=cal_predictions,
        )

        if use_cqr and len(cal_q10_predictions) > 0:
            cqr_stats = self._cqr.calibrate(
                y_true=cal_actuals,
                q10_pred=cal_q10_predictions,
                q90_pred=cal_q90_predictions,
            )
            self._cqr_calibration_stats = cqr_stats

        residuals = [abs(a - p) for a, p in zip(cal_actuals, cal_predictions)]
        self._conformal_calibration_stats = {
            'n_calibration': len(cal_actuals),
            'mean_residual': sum(residuals) / len(residuals) if residuals else 0,
            'max_residual': max(residuals) if residuals else 0,
            'median_residual': sorted(residuals)[len(residuals) // 2] if residuals else 0,
            'q90_residual': sorted(residuals)[int(len(residuals) * 0.9)] if residuals else 0,
            'use_cqr': use_cqr,
        }

    def _get_quantile_predictions(self, signal: Dict[str, Any]) -> tuple:
        try:
            import torch

            features = self._extract_features(signal)
            model_type = self._learned_params.get('model_type', 'unknown')

            if model_type == 'dual_quantile_nn':
                supply_demand = signal.get('supply_demand', 8)

                if supply_demand < 8:
                    model = self._learned_params.get('charge_model')
                    y_mean = self._learned_params.get('charge_y_mean', 0)
                    y_std = self._learned_params.get('charge_y_std', 1)
                else:
                    model = self._learned_params.get('discharge_model')
                    y_mean = self._learned_params.get('discharge_y_mean', 0)
                    y_std = self._learned_params.get('discharge_y_std', 1)

                if model is None:
                    model = self._learned_params.get('charge_model') or self._learned_params.get('discharge_model')
                    y_mean = self._learned_params.get('charge_y_mean', self._learned_params.get('discharge_y_mean', 0))
                    y_std = self._learned_params.get('charge_y_std', self._learned_params.get('discharge_y_std', 1))

                if model:
                    model.eval()
                    with torch.no_grad():
                        x = torch.tensor([features], dtype=torch.float32)
                        pred_output = model(x)

                        if pred_output.shape[1] == 3:
                            q10_norm, q50_norm, q90_norm = pred_output[0].tolist()
                            q10 = q10_norm * y_std + y_mean
                            q50 = q50_norm * y_std + y_mean
                            q90 = q90_norm * y_std + y_mean
                            q10, q90 = min(q10, q90), max(q10, q90)
                            q50 = max(q10, min(q50, q90))
                            return (q10, q50, q90)
                        else:
                            pred = pred_output.item() * y_std + y_mean
                            return (pred, pred, pred)


        except Exception:
            pass

        pred = self._simple_predict(signal)
        return (pred, pred, pred)

    def get_training_history(self) -> Dict[str, Any]:
        if not self._is_fitted or not hasattr(self, '_learned_params') or not self._learned_params:
            return {
                'model_type': 'not_fitted',
                'epochs': [],
                'train_loss': [],
                'val_loss': [],
            }

        model_type = self._learned_params.get('model_type', 'unknown')

        if model_type == 'dual_quantile_nn':
            charge_history = self._learned_params.get('charge_training_history', {})
            discharge_history = self._learned_params.get('discharge_training_history', {})

            charge_epochs = charge_history.get('epochs', [])
            discharge_epochs = discharge_history.get('epochs', [])

            if len(charge_epochs) >= len(discharge_epochs):
                merged_epochs = charge_epochs
                merged_train_loss = charge_history.get('train_loss', [])
                merged_val_loss = charge_history.get('val_loss', [])
            else:
                merged_epochs = discharge_epochs
                merged_train_loss = discharge_history.get('train_loss', [])
                merged_val_loss = discharge_history.get('val_loss', [])

            return {
                'model_type': 'dual_quantile_nn',
                'epochs': merged_epochs,
                'train_loss': merged_train_loss,
                'val_loss': merged_val_loss,
                'charge': charge_history,
                'discharge': discharge_history,
            }
        else:
            return {
                'model_type': model_type,
                'epochs': [],
                'train_loss': [],
                'val_loss': [],
            }

    def estimate(
        self,
        signal: Dict[str, Any],
        device_distribution: Optional[Dict[str, int]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> EstimationResult:
        if not self._is_fitted:
            raise RuntimeError(
                "Estimator must be fitted before calling estimate(). "
                "Call fit() with training data first."
            )

        layer_contributions = {}

        base_prediction = 0.0
        prediction_source = 'none'

        is_quantile = getattr(self, '_is_quantile_model', False)
        is_dual = getattr(self, '_is_dual_model', False)

        if is_quantile and hasattr(self, '_learned_params') and self._learned_params:
            try:
                q10, q50, q90 = self._get_quantile_predictions(signal)

                base_prediction = q50
                layer_contributions['quantile_q10'] = q10
                layer_contributions['quantile_q50'] = q50
                layer_contributions['quantile_q90'] = q90

                if is_dual:
                    supply_demand = signal.get('supply_demand', 8)
                    if supply_demand < 8:
                        prediction_source = 'dual_quantile_nn_charge'
                        layer_contributions['model_used'] = 'charge_nn'
                    else:
                        prediction_source = 'dual_quantile_nn_discharge'
                        layer_contributions['model_used'] = 'discharge_nn'
                else:
                    prediction_source = 'quantile_nn'

                layer_contributions['aggregate_nn'] = base_prediction

            except Exception:
                pass

        if prediction_source == 'none':
            base_prediction = self._simple_predict(signal)
            prediction_source = 'learned_regression'
            layer_contributions['learned_regression'] = base_prediction

        point_estimate = base_prediction

        if device_distribution:
            total_devices = sum(device_distribution.values())
            point_estimate *= total_devices / 1000

        layer_contributions['final_source'] = prediction_source

        interval_source = 'empirical'

        is_quantile_source = prediction_source in (
            'quantile_nn',
            'dual_quantile_nn_charge',
            'dual_quantile_nn_discharge',
        )

        if (
            self.config.enable_conformal
            and self.config.use_cqr
            and self._cqr is not None
            and self._cqr._is_calibrated
            and is_quantile_source
        ):
            q10 = layer_contributions.get('quantile_q10', point_estimate)
            q50 = layer_contributions.get('quantile_q50', point_estimate)
            q90 = layer_contributions.get('quantile_q90', point_estimate)

            if device_distribution:
                total_devices = sum(device_distribution.values())
                scale_factor = total_devices / 1000
                q10 *= scale_factor
                q90 *= scale_factor

            interval = self._cqr.predict(q10=q10, q50=point_estimate, q90=q90)
            lower_bound = interval.lower_bound
            upper_bound = interval.upper_bound
            confidence = interval.estimated_coverage
            layer_contributions['cqr_adjustment'] = interval.quantile_threshold
            layer_contributions['interval_source'] = 'cqr'
            interval_source = 'cqr'

        elif self.config.enable_conformal and self._conformal and self._conformal._is_calibrated:
            interval = self._conformal.predict(point_estimate=point_estimate)
            lower_bound = interval.lower_bound
            upper_bound = interval.upper_bound
            confidence = interval.estimated_coverage
            layer_contributions['conformal'] = interval.quantile_threshold
            layer_contributions['interval_source'] = 'conformal'
            interval_source = 'conformal'

        else:
            if hasattr(self, '_learned_params') and self._learned_params is not None:
                residual_std = self._learned_params.get('residual_std', abs(point_estimate) * 0.3)
            else:
                residual_std = abs(point_estimate) * 0.3

            z_score = 1.645 if self.config.target_coverage <= 0.9 else 1.96
            margin = z_score * residual_std

            if hasattr(self, '_learned_params') and self._learned_params is not None:
                r2 = self._learned_params.get('r2', 0)
                if r2 < 0.5:
                    margin *= 1.5
            else:
                margin *= 2.0

            lower_bound = point_estimate - margin
            upper_bound = point_estimate + margin
            confidence = 0.5
            layer_contributions['interval_source'] = 'empirical'

        return EstimationResult(
            response_kw=point_estimate,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            coverage=self.config.target_coverage,
            confidence=confidence,
            layer_contributions=layer_contributions if layer_contributions else None,
        )

    def update(
        self,
        signal: Dict[str, Any],
        actual_response: float,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        entry = {
            'signal': signal,
            'response': actual_response,
            'context': context,
            'timestamp': time.time(),
        }
        self._history.append(entry)

        cutoff = time.time() - self.config.history_window * 24
        self._history = [h for h in self._history if h['timestamp'] > cutoff]

        if self.config.enable_conformal:
            estimated = self.estimate(signal, context=context)

            if self.config.use_cqr and self._cqr and hasattr(self._cqr, 'update'):
                q10, q50, q90 = self._get_quantile_predictions(signal)
                self._cqr.update(
                    y_true=actual_response,
                    q10_pred=q10,
                    q50_pred=q50,
                    q90_pred=q90,
                )
            elif self._conformal:
                self._conformal.update(estimated.response_kw, actual_response)

            if self._coverage_tracker:
                interval = PredictionInterval(
                    point_estimate=estimated.response_kw,
                    lower_bound=estimated.lower_bound,
                    upper_bound=estimated.upper_bound,
                    target_coverage=self.config.target_coverage,
                )
                self._coverage_tracker.record(interval, actual_response)

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'is_fitted': self._is_fitted,
            'history_size': len(self._history),
            'config': {
                'target_coverage': self.config.target_coverage,
                'enable_conformal': self.config.enable_conformal,
            },
        }

        if self._coverage_tracker:
            stats['coverage'] = {
                'empirical': self._coverage_tracker.get_empirical_coverage(),
                'target': self.config.target_coverage,
                'n_predictions': len(self._coverage_tracker._records),
            }

        if self.config.enable_conformal and self._conformal:
            stats['conformal'] = {
                'is_calibrated': self._conformal._is_calibrated,
                'target_coverage': self._conformal.config.target_coverage,
            }

        return stats

    def reset(self) -> None:
        self._history.clear()
        self._is_fitted = False

        if self._conformal:
            self._conformal.reset()

        if self._coverage_tracker:
            self._coverage_tracker._records.clear()

    def get_layer_diagnostics(self) -> Dict[str, Any]:
        diagnostics = {}

        if self.config.enable_conformal and self._conformal:
            diagnostics['conformal'] = {
                'is_calibrated': self._conformal._is_calibrated,
                'target_coverage': self._conformal.config.target_coverage,
                'method': self._conformal.config.method.value,
                'score_type': self._conformal.scorer.score_type,
            }

        return diagnostics

    def _online_update_single_nn(
        self,
        model: Any,
        buffer: List[Dict[str, Any]],
        y_mean: float,
        y_std: float,
        learning_rate: float,
        model_key: str = 'default',
        anchor_lambda: float = 100.0,
    ) -> float:
        import torch
        import torch.optim as optim

        if not hasattr(self, '_online_optimizers'):
            self._online_optimizers = {}
        if not hasattr(self, '_initial_weights'):
            self._initial_weights = {}

        if model_key not in self._initial_weights:
            self._initial_weights[model_key] = {
                name: param.clone().detach()
                for name, param in model.named_parameters()
            }

        if model_key not in self._online_optimizers:
            self._online_optimizers[model_key] = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.999),
            )
        optimizer = self._online_optimizers[model_key]

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        X_batch = []
        y_batch = []
        for item in buffer:
            features = self._extract_features(item['signal'])
            X_batch.append(features)
            y_normalized = (item['response'] - y_mean) / max(y_std, 1e-6)
            y_batch.append(y_normalized)

        X_tensor = torch.tensor(X_batch, dtype=torch.float32)
        y_tensor = torch.tensor(y_batch, dtype=torch.float32).unsqueeze(1)

        model.train()
        optimizer.zero_grad()
        pred = model(X_tensor)

        quantiles = [0.1, 0.5, 0.9]
        delta = 1.0
        losses = []
        for i, q in enumerate(quantiles):
            pred_q = pred[:, i:i+1]
            error = y_tensor - pred_q
            abs_error = torch.abs(error)

            huber = torch.where(
                abs_error <= delta,
                0.5 * error ** 2,
                delta * (abs_error - 0.5 * delta)
            )
            weight = torch.where(error >= 0, q, 1 - q)
            loss_q = weight * huber
            losses.append(loss_q)

        total_loss = torch.cat(losses, dim=1).mean()

        anchor_loss = 0.0
        for name, param in model.named_parameters():
            if name in self._initial_weights[model_key]:
                anchor_loss += torch.sum((param - self._initial_weights[model_key][name]) ** 2)
        total_loss = total_loss + anchor_lambda * anchor_loss

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        model.eval()
        return total_loss.item()

    def _compute_validation_error(self) -> float:
        if not self._validation_window:
            return 0.0

        errors = []
        for sample in self._validation_window:
            signal = sample['signal']
            actual = sample['response']

            try:
                prediction = self.estimate(signal)
                pred_value = prediction.response_kw
                relative_error = abs(pred_value - actual) / max(abs(actual), 1.0)
                errors.append(relative_error)
            except Exception:
                continue

        if not errors:
            return 0.0

        import numpy as np
        return float(np.mean(errors))

    def online_update(
        self,
        signal: Dict[str, Any],
        actual_response: float,
        learning_rate: float = 0.00005,
        error_threshold: float = 0.15,
        update_nn: bool = False,
        mini_batch_size: int = None,
        anchor_lambda: float = None,
        **kwargs,
    ) -> Dict[str, float]:
        prediction = self.estimate(signal)
        prediction_error = actual_response - prediction.response_kw

        relative_error = abs(prediction_error) / max(abs(actual_response), 1.0)

        update_stats = {
            'prediction': prediction.response_kw,
            'actual': actual_response,
            'error': prediction_error,
            'relative_error': relative_error,
            'updated': False,
            'nn_updated': False,
            'charge_nn_updated': False,
            'discharge_nn_updated': False,
            'skipped_low_error': False,
        }

        self._history.append({
            'signal': signal,
            'response': actual_response,
            'predicted': prediction.response_kw,
            'actual': actual_response,
            'error': prediction_error,
            'timestamp': time.time(),
        })

        self._validation_window.append({
            'signal': signal,
            'response': actual_response,
            'predicted': prediction.response_kw,
        })
        if len(self._validation_window) > self.config.online_validation_window:
            self._validation_window.pop(0)

        skip_conformal_update = relative_error < error_threshold
        if skip_conformal_update:
            update_stats['skipped_low_error'] = True

        if not hasattr(self, '_online_buffer'):
            self._online_buffer = []
            self._online_update_count = 0
        if not hasattr(self, '_online_buffer_charge'):
            self._online_buffer_charge = []
        if not hasattr(self, '_online_buffer_discharge'):
            self._online_buffer_discharge = []


        self._online_update_count += 1

        is_dual_model = getattr(self, '_is_dual_model', False)
        model_type = self._learned_params.get('model_type', '') if self._learned_params else ''

        if update_nn and not self._nn_update_frozen:
            if is_dual_model and model_type == 'dual_quantile_nn':
                supply_demand = signal.get('supply_demand', 8)

                if supply_demand < 8:
                    self._online_buffer_charge.append({
                        'signal': signal,
                        'response': actual_response,
                    })
                else:
                    self._online_buffer_discharge.append({
                        'signal': signal,
                        'response': actual_response,
                    })

                _mbs = mini_batch_size if mini_batch_size is not None else 100
                _al = anchor_lambda if anchor_lambda is not None else 100.0

                try:
                    import torch
                    import torch.optim as optim

                    if len(self._online_buffer_charge) >= _mbs:
                        charge_model = self._learned_params.get('charge_model')
                        if charge_model is not None:
                            charge_y_mean = self._learned_params.get('charge_y_mean', 0)
                            charge_y_std = self._learned_params.get('charge_y_std', 1)

                            loss = self._online_update_single_nn(
                                model=charge_model,
                                buffer=self._online_buffer_charge[-_mbs:],
                                y_mean=charge_y_mean,
                                y_std=charge_y_std,
                                learning_rate=learning_rate,
                                model_key='charge',
                                anchor_lambda=_al,
                            )

                            update_stats['charge_nn_updated'] = True
                            update_stats['charge_loss'] = loss
                            update_stats['nn_updated'] = True

                            self._online_buffer_charge = self._online_buffer_charge[-max(32, _mbs // 3):]

                    if len(self._online_buffer_discharge) >= _mbs:
                        discharge_model = self._learned_params.get('discharge_model')
                        if discharge_model is not None:
                            discharge_y_mean = self._learned_params.get('discharge_y_mean', 0)
                            discharge_y_std = self._learned_params.get('discharge_y_std', 1)

                            loss = self._online_update_single_nn(
                                model=discharge_model,
                                buffer=self._online_buffer_discharge[-_mbs:],
                                y_mean=discharge_y_mean,
                                y_std=discharge_y_std,
                                learning_rate=learning_rate,
                                model_key='discharge',
                                anchor_lambda=_al,
                            )

                            update_stats['discharge_nn_updated'] = True
                            update_stats['discharge_loss'] = loss
                            update_stats['nn_updated'] = True

                            self._online_buffer_discharge = self._online_buffer_discharge[-max(32, _mbs // 3):]

                except Exception as e:
                    update_stats['nn_error'] = str(e)

                if update_stats['nn_updated'] and len(self._validation_window) >= 20:
                    validation_error = self._compute_validation_error()
                    self._validation_errors.append(validation_error)
                    update_stats['validation_error'] = validation_error

                    if len(self._validation_errors) >= 2:
                        prev_error = self._validation_errors[-2]
                        if validation_error > prev_error * 1.05:
                            self._validation_deterioration_count += 1
                        else:
                            self._validation_deterioration_count = 0

                        if self._validation_deterioration_count >= self.config.online_validation_patience:
                            self._nn_update_frozen = True
                            update_stats['nn_frozen'] = True
                            update_stats['freeze_reason'] = f'validation_error increased {self._validation_deterioration_count} times consecutively'

            else:
                self._online_buffer.append({
                    'signal': signal,
                    'response': actual_response,
                })

                _mbs_legacy = mini_batch_size if mini_batch_size is not None else 10
                if (len(self._online_buffer) >= _mbs_legacy and
                    hasattr(self, '_aggregate_nn') and self._aggregate_nn is not None and
                    hasattr(self, '_learned_params') and self._learned_params is not None):

                    try:
                        import torch
                        import torch.optim as optim

                        model = self._aggregate_nn
                        y_mean = self._learned_params.get('y_mean', 0)
                        y_std = self._learned_params.get('y_std', 1)

                        _al_legacy = anchor_lambda if anchor_lambda is not None else 100.0
                        loss = self._online_update_single_nn(
                            model=model,
                            buffer=self._online_buffer[-_mbs_legacy:],
                            y_mean=y_mean,
                            y_std=y_std,
                            learning_rate=learning_rate,
                            model_key='legacy',
                            anchor_lambda=_al_legacy,
                        )

                        update_stats['nn_updated'] = True
                        update_stats['loss'] = loss

                        self._online_buffer = self._online_buffer[-5:]

                    except Exception as e:
                        update_stats['nn_error'] = str(e)

        if not skip_conformal_update:
            try:
                if self.config.use_cqr and self._cqr and hasattr(self._cqr, 'update'):
                    q10, q50, q90 = self._get_quantile_predictions(signal)
                    self._cqr.update(
                        y_true=actual_response,
                        q10_pred=q10,
                        q50_pred=q50,
                        q90_pred=q90,
                    )
                    update_stats['updated'] = True
                    update_stats['cqr_updated'] = True
                elif self._conformal and hasattr(self._conformal, 'update'):
                    self._conformal.update(prediction.response_kw, actual_response)
                    update_stats['updated'] = True
                elif self._conformal and hasattr(self._conformal, 'add_batch'):
                    import numpy as np
                    self._conformal.add_batch(
                        np.array([prediction.response_kw]),
                        np.array([actual_response]),
                    )
                    update_stats['updated'] = True
            except Exception as e:
                update_stats['conformal_update_error'] = str(e)

        if self._coverage_tracker and hasattr(self._coverage_tracker, 'record'):
            try:
                interval = PredictionInterval(
                    point_estimate=prediction.response_kw,
                    lower_bound=prediction.lower_bound,
                    upper_bound=prediction.upper_bound,
                    target_coverage=self.config.target_coverage,
                )
                self._coverage_tracker.record(interval, actual_response)
            except Exception:
                pass

        return update_stats

    def get_online_learning_stats(self) -> Dict[str, Any]:
        if not self._history:
            return {'n_samples': 0}

        import numpy as np

        errors = []
        for record in self._history:
            if 'error' in record:
                errors.append(record['error'])
            elif 'predicted' in record and 'actual' in record:
                errors.append(record['actual'] - record['predicted'])

        if not errors:
            return {'n_samples': len(self._history)}

        errors = np.array(errors)

        stats = {
            'n_samples': len(errors),
            'mae': float(np.mean(np.abs(errors))),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'bias': float(np.mean(errors)),
            'std': float(np.std(errors)),
        }

        if self._coverage_tracker:
            coverage_stats = self._coverage_tracker.get_summary()
            stats['coverage'] = coverage_stats

        return stats

    def save_online_state(self, path: str) -> None:
        import json
        from pathlib import Path

        state = {
            'history': self._history[-1000:],
            'is_fitted': self._is_fitted,
            'stats': self.get_online_learning_stats(),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load_online_state(self, path: str) -> None:
        import json

        with open(path, 'r') as f:
            state = json.load(f)

        self._history = state.get('history', [])
        self._is_fitted = state.get('is_fitted', False)
