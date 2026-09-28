from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


matplotlib.rcParams["font.family"] = "SimSun"
matplotlib.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
E23 = ROOT / "results/E23"
E24 = ROOT / "results/E24"

ALGORITHMS = (
    "no_coordination",
    "local_rules",
    "mpc_optimal",
    "mean_field_control",
    "virtual_battery",
    "packetized_energy_management",
    "transactive_control",
    "eps_ieee69_fused",
    "eps_global_mixed",
    "centralized_optimal",
)
LABELS = {
    "no_coordination": "No coordination",
    "local_rules": "Local SOC rules",
    "mpc_optimal": "MPC",
    "mean_field_control": "Mean-field control",
    "virtual_battery": "Virtual battery",
    "packetized_energy_management": "Packetized Energy Management",
    "transactive_control": "Transactive control",
    "eps_ieee69_fused": "EPS IEEE-69 fused",
    "eps_global_mixed": "EPS global mixed",
    "centralized_optimal": "Centralized greedy UB",
}
COLORS = {
    "no_coordination": "#9c9c9c",
    "local_rules": "#ed7d31",
    "mpc_optimal": "#7057ff",
    "mean_field_control": "#4e79a7",
    "virtual_battery": "#59a14f",
    "packetized_energy_management": "#af7aa1",
    "transactive_control": "#d55e00",
    "eps_ieee69_fused": "#1f5aa6",
    "eps_global_mixed": "#1f5aa6",
    "centralized_optimal": "#3a9d66",
}
SPATIAL_ORDER = ("H0", "H1", "H2_25", "H2_50", "H2_75", "H3", "H4", "H5")
SPATIAL_LABELS = {
    "H0": "H0",
    "H1": "H1",
    "H2_25": "H2-25",
    "H2_50": "H2-50",
    "H2_75": "H2-75",
    "H3": "H3",
    "H4": "H4",
    "H5": "H5",
}
SPATIAL_MARKERS = {
    "H0": "o",
    "H1": "s",
    "H2_25": "^",
    "H2_50": "D",
    "H2_75": "P",
    "H3": "X",
    "H4": "v",
    "H5": "<",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stable_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)


def mean_ci(values: list[float], label: str) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(array) == 1:
        return float(array[0]), float(array[0]), float(array[0])
    rng = np.random.default_rng(stable_seed(label))
    indices = rng.integers(0, len(array), size=(2000, len(array)))
    boot = np.mean(array[indices], axis=1)
    return float(np.mean(array)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def grouped(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def save(fig: plt.Figure, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_pressure(rows: list[dict[str, str]], output: Path) -> None:
    axes_info = (
        ("request_intensity", "请求强度 q", "q"),
        ("line_derating", "线路容量压力 λ_line", "λ_line"),
        ("spatial_concentration", "远端空间集中度 κ", "κ"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharey=True)
    for axis, (axis_name, xlabel, short) in zip(axes, axes_info):
        selected = [row for row in rows if row["axis"] == axis_name]
        groups = grouped(selected, ("algorithm", "pressure_value"))
        values = sorted({float(row["pressure_value"]) for row in selected})
        for algorithm in ALGORITHMS:
            if not any(row["algorithm"] == algorithm for row in selected):
                continue
            means = []
            lows = []
            highs = []
            for value in values:
                group = groups.get((algorithm, str(value)), [])
                if not group:
                    group = [row for row in selected if row["algorithm"] == algorithm and float(row["pressure_value"]) == value]
                mean, low, high = mean_ci([float(row["response_r2"]) for row in group], f"{axis_name}|{algorithm}|{value}")
                means.append(mean)
                lows.append(low)
                highs.append(high)
            axis.plot(values, means, marker="o", linewidth=1.5, color=COLORS[algorithm], label=LABELS[algorithm])
            axis.fill_between(values, lows, highs, color=COLORS[algorithm], alpha=0.08)
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.axhline(0.95, color="#b23b3b", linestyle="--", linewidth=0.8)
        axis.set_xlabel(xlabel)
        axis.set_title(short)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("响应跟踪 R²")
    axes[-1].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("E23：连续压力下的响应跟踪 R²")
    save(fig, output / "e23_r2_pressure_curves")


def plot_spatial(rows: list[dict[str, str]], output: Path) -> None:
    groups = grouped(rows, ("condition", "algorithm"))
    fig, axis = plt.subplots(figsize=(14, 6))
    x = np.arange(len(SPATIAL_ORDER))
    for algorithm in ALGORITHMS:
        means = []
        lows = []
        highs = []
        for condition in SPATIAL_ORDER:
            group = groups.get((condition, algorithm), [])
            mean, low, high = mean_ci([float(row["response_r2"]) for row in group], f"spatial|{condition}|{algorithm}")
            means.append(mean)
            lows.append(low)
            highs.append(high)
        if np.all(~np.isfinite(means)):
            continue
        axis.plot(x, means, marker="o", linewidth=1.6, color=COLORS[algorithm], label=LABELS[algorithm])
        axis.fill_between(x, lows, highs, color=COLORS[algorithm], alpha=0.07)
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.axhline(0.95, color="#b23b3b", linestyle="--", linewidth=0.8, label="R²=0.95")
    axis.set_xticks(x, [SPATIAL_LABELS[item] for item in SPATIAL_ORDER])
    axis.set_xlabel("IEEE-69 空间条件")
    axis.set_ylabel("响应跟踪 R²")
    axis.set_title("E23：H0-H5 空间条件下的响应跟踪 R²")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=8, ncol=3)
    save(fig, output / "e23_r2_spatial_conditions")


def plot_tradeoff(rows: list[dict[str, str]], output: Path) -> None:
    groups = grouped(rows, ("condition", "algorithm"))
    fig, axis = plt.subplots(figsize=(10.5, 7))
    for algorithm in ALGORITHMS:
        for condition in SPATIAL_ORDER:
            group = groups.get((condition, algorithm), [])
            if not group:
                continue
            reduction = float(np.mean([float(row["mean_reduction_pct"]) for row in group]))
            r2 = float(np.mean([float(row["response_r2"]) for row in group]))
            axis.scatter(reduction, r2, color=COLORS[algorithm], marker=SPATIAL_MARKERS[condition], s=75, alpha=0.85, edgecolors="white", linewidths=0.4)
    algorithm_handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[item], markersize=7, label=LABELS[item]) for item in ALGORITHMS]
    condition_handles = [plt.Line2D([0], [0], marker=SPATIAL_MARKERS[item], color="#555555", linestyle="None", markersize=7, label=SPATIAL_LABELS[item]) for item in SPATIAL_ORDER]
    first = axis.legend(handles=algorithm_handles, title="颜色：算法", fontsize=7, title_fontsize=8, loc="lower left", ncol=2)
    axis.add_artist(first)
    axis.legend(handles=condition_handles, title="点形：空间条件", fontsize=7, title_fontsize=8, loc="lower right", ncol=2)
    axis.axhline(0.95, color="#b23b3b", linestyle="--", linewidth=0.8)
    axis.set_xlabel("可交付弃电降低率 (%)")
    axis.set_ylabel("响应跟踪 R²")
    axis.set_title("E23：弃电消纳效果与响应跟踪能力")
    axis.grid(alpha=0.22)
    save(fig, output / "e23_reduction_vs_r2_spatial")


def plot_e24(rows: list[dict[str, str]], output: Path) -> None:
    groups = grouped(rows, ("scenario", "algorithm"))
    scenarios = []
    labels = {}
    for row in rows:
        if row["scenario"] not in scenarios:
            scenarios.append(row["scenario"])
            labels[row["scenario"]] = row["scenario_label"]
    fig, axis = plt.subplots(figsize=(13, 6))
    x = np.arange(len(scenarios))
    for algorithm in ALGORITHMS:
        means = []
        lows = []
        highs = []
        for scenario in scenarios:
            group = groups.get((scenario, algorithm), [])
            mean, low, high = mean_ci([float(row["response_r2"]) for row in group], f"e24|{scenario}|{algorithm}")
            means.append(mean)
            lows.append(low)
            highs.append(high)
        if np.all(~np.isfinite(means)):
            continue
        axis.plot(x, means, marker="o", linewidth=1.6, color=COLORS[algorithm], label=LABELS[algorithm])
        axis.fill_between(x, lows, highs, color=COLORS[algorithm], alpha=0.07)
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.axhline(0.95, color="#b23b3b", linestyle="--", linewidth=0.8)
    axis.set_xticks(x, [labels[item] for item in scenarios], rotation=12)
    axis.set_xlabel("IEEE-123 网络场景")
    axis.set_ylabel("响应跟踪 R²")
    axis.set_title("E24：IEEE-123 场景下的响应跟踪 R²")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(fontsize=8, ncol=3)
    save(fig, output / "e24_r2_scenario_comparison")


def main() -> None:
    output = E23 / "figures"
    output.mkdir(parents=True, exist_ok=True)
    e23 = read_csv(E23 / "data/e23_by_seed.csv")
    spatial = read_csv(E23 / "data/spatial_heterogeneity_by_seed.csv")
    e24 = read_csv(E24 / "data/e24_ieee123_by_seed.csv")
    plot_pressure(e23, output)
    plot_spatial(spatial, output)
    plot_tradeoff(spatial, output)
    plot_e24(e24, E24 / "figures")


if __name__ == "__main__":
    main()
