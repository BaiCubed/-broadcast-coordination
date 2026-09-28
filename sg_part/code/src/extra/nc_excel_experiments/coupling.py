"""把设备间耦合从「合成注入」换成「数据自带」。

旧 E1 的相关性来自 protocol.py:487 的 zone_shocks —— 每次仿真按 zone 抽一个正态标量、
乘到该区所有设备上。它的分布只由 zone_correlation 参数决定，跟数据集无关，
所以 14 个数据集测出来的 rho 一致到 3.6e-8，iid 条件下 N_eff/N 恒等于 1.000。
15 个真实数据集在 E1 里唯一的作用退化成「能跑到多大的 N」。

这里换成：**设备的可用性由它自己那条真实功率序列的残差决定**。
共同的天气/作息驱动会让同一时刻多台设备一起变得不可用，耦合结构因此直接等于
该数据集真实的协动结构，不需要标定，也没有 rho 旋钮可调。

两个刻意的设计约束：

1. **没有自由幅度参数**。可用性取残差的经验分位数（秩变换），
   所以每台设备的可用性边缘分布都是 Uniform[0,1]，与数据集、与设备无关。
   数据集之间唯一的差别就是这些序列**互相之间对不对齐**，也就是我们要测的那个量。
2. **iid 对照臂用同一批序列做逐设备独立置换**。两臂的边缘分布逐设备完全相同，
   只有跨设备的时间对齐被打散。所以两臂之差不可能来自可用性水平或容量，
   只能来自耦合本身。

日内轮廓先回归掉再算残差：确定性的作息曲线是聚合控制器本来就能预测的部分
（`trace_features` 里就有 load_kw 这一列），把它算进耦合等于高估了不可预测的协动。
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# 日内轮廓基：常数 + 每小时一个指示变量。回归掉它剩下的才是控制器预测不了的协动。
DIURNAL_HOURS = 24


def _hour_of_day_design(step_indices: np.ndarray, steps_per_day: int) -> np.ndarray:
    hours = (step_indices * DIURNAL_HOURS // max(steps_per_day, 1)) % DIURNAL_HOURS
    columns = [np.ones(len(step_indices))]
    for hour in range(DIURNAL_HOURS):
        indicator = (hours == hour).astype(float)
        if indicator.sum() > 0:
            columns.append(indicator)
    return np.column_stack(columns)


def device_load_matrix(records: Sequence[Any]) -> np.ndarray:
    """(steps, devices) 的真实功率矩阵，按最短的那条序列截齐。"""
    steps = min(len(record.load_kw) for record in records)
    return np.column_stack([
        np.asarray(record.load_kw, dtype=float)[:steps] for record in records
    ])


def diurnal_residual(records: Sequence[Any], steps_per_day: int) -> np.ndarray:
    """(steps, devices)：真实功率减掉逐设备拟合的日内轮廓。

    共同的天气与作息扰动留在残差里 —— 那正是 N_eff < N 的来源。
    """
    power = device_load_matrix(records)
    design = _hour_of_day_design(np.arange(power.shape[0]), steps_per_day)
    coefficients, _, _, _ = np.linalg.lstsq(design, power, rcond=None)
    return power - design @ coefficients


def rank_availability(residual: np.ndarray) -> np.ndarray:
    """把残差按每台设备自己的经验分位数翻成可用性，取值 (0, 1)。

    残差越大（这一刻自己用得越凶）可用性越低。秩变换保证每台设备的边缘分布
    完全一样，数据集之间的差别只剩下跨设备的对齐程度。
    """
    steps = residual.shape[0]
    order = np.argsort(np.argsort(residual, axis=0), axis=0)
    percentile = (order + 0.5) / steps
    return 1.0 - percentile


def availability_at_steps(availability: np.ndarray, profile_steps: Sequence[int]) -> np.ndarray:
    """把整条序列上的可用性取到仿真实际用的那些时刻上。(conditions, devices)。"""
    indices = np.clip(np.asarray(profile_steps, dtype=int), 0, availability.shape[0] - 1)
    return availability[indices]


def decoupled_copy(availability: np.ndarray, seed: int) -> np.ndarray:
    """逐设备独立置换时间轴：边缘分布一模一样，跨设备对齐被打散。"""
    rng = np.random.default_rng(seed)
    shuffled = np.empty_like(availability)
    for device in range(availability.shape[1]):
        shuffled[:, device] = rng.permutation(availability[:, device])
    return shuffled


def mean_pairwise_rho(matrix: np.ndarray) -> float:
    """(samples, devices) 的平均成对相关，不构造 D x D 相关阵。"""
    deviation = np.std(matrix, axis=0)
    keep = deviation > 1e-12
    matrix = matrix[:, keep]
    device_count = matrix.shape[1]
    if device_count < 2:
        return 0.0
    standardized = (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)
    row_sums = np.sum(standardized, axis=1)
    numerator = float(np.mean(row_sums * row_sums) - device_count)
    return numerator / (device_count * (device_count - 1))


def broadcast_residual_rho(
    resource: np.ndarray, basis: np.ndarray, broadcast_columns: int
) -> float:
    """N_eff 的正确口径：沿 condition 轴、对广播基函数回归后的残差算平均成对相关。

    ``resource`` 是 (replication, condition, device)，``basis`` 是 (condition, feature)。
    只取 basis 的前 ``broadcast_columns`` 列 —— 那些是广播信号本身（强度/方向/样条）；
    再往后是 load、变压器与电压裕度等电网状态，那恰恰是共同环境驱动，
    一起回归掉就把要测的东西测没了。

    旧口径把残差取在 replication 轴上（同一天同一设备重跑一遍的差），
    真实数据在那个轴上是常数，会被均值整个减掉。
    """
    design = basis[:, :broadcast_columns]
    values = []
    for replication in range(resource.shape[0]):
        response = resource[replication]
        coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
        values.append(mean_pairwise_rho(response - design @ coefficients))
    return float(np.mean(values)) if values else 0.0
