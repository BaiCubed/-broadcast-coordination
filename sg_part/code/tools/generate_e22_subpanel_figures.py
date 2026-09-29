#!/usr/bin/env python3
"""生成 E22 IEEE-69 子图的统一重绘版本。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
import matplotlib.ticker as ticker

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tools.ncstyle import (
    BAND,
    C3,
    DASH_SUMM,
    FAINT,
    FS_ANNOT,
    FS_ANNOT_HI,
    FS_LABEL,
    FS_LEGEND,
    FS_PANEL,
    FS_AUDIT_CELL,
    FS_AUDIT_TICK,
    FS_DISTRIBUTION_LEGEND,
    FS_DISTRIBUTION_TICK,
    FS_LINE_LEGEND,
    FS_LINE_TICK,
    FS_SUBPANEL_ANNOT,
    FS_SUBPANEL_CELL,
    FS_SUBPANEL_LEGEND,
    FS_SUBPANEL_TICK,
    FS_TICK,
    FS_TITLE,
    GRID,
    INK,
    LW_AXIS,
    LW_MAIN,
    LW_OTHER,
    LW_SUMM,
    MIX_FILL,
    MIX_INK,
    MIX_LINE,
    MS,
    MS_BIG,
    MM,
    RULE,
    SEQ,
    WIDTH_2COL,
    ax_mm,
    annot,
    cloud,
    configure,
    shade,
    tidy,
    title,
    )
    import tools.nckeys as K
except ModuleNotFoundError:
    from ncstyle import (
        BAND, C3, DASH_SUMM, FAINT, FS_ANNOT, FS_ANNOT_HI, FS_LABEL, FS_LEGEND,
        FS_PANEL, FS_AUDIT_CELL, FS_AUDIT_TICK, FS_DISTRIBUTION_LEGEND,
        FS_DISTRIBUTION_TICK, FS_LINE_LEGEND, FS_LINE_TICK, FS_SUBPANEL_ANNOT,
        FS_SUBPANEL_CELL, FS_SUBPANEL_LEGEND, FS_SUBPANEL_TICK, FS_TICK,
        FS_TITLE, GRID, INK, LW_AXIS, LW_MAIN, LW_OTHER, LW_SUMM, MIX_FILL,
        MIX_INK, MIX_LINE, MS, MS_BIG, MM, RULE, SEQ, WIDTH_2COL, ax_mm,
        annot, cloud, configure, shade, tidy, title,
    )
    import nckeys as K


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/figs/subpanel"
E22 = ROOT / "results/E22/data/e22_by_seed.csv"
E22_DIRECT = ROOT / "results/E22/trained_eps_ieee69_direct/data/trained_eps_by_seed.csv"
E22_IEEE33_DIRECT = ROOT / "results/E22/trained_eps_ieee33_direct/data/trained_eps_by_seed.csv"
E24_DIRECT = ROOT / "results/E24/trained_eps_ieee123_direct/data/trained_eps_by_seed.csv"
E23 = ROOT / "results/E23/data/e23_by_seed.csv"
SINGLE_CONSTRAINT = ROOT / "results/E22/single_constraint_effect/data/single_constraint_effect_summary.csv"
SINGLE_CONSTRAINT_MANIFEST = ROOT / "results/E22/single_constraint_effect/manifest.json"
TRANSFORMER_REPLAY = ROOT / "results/E22/standardized_constraint_replay/data/transformer_panel_replay_summary.csv"
E23_SPATIAL_DIRECT = ROOT / "results/E23/direct_training/data/spatial_heterogeneity_by_seed.csv"
E24_AUDIT = ROOT / "results/E24/data/e24_ieee123_by_seed.csv"
PAIRWISE_CURTAILMENT = ROOT / "results/E21/pairwise_curtailment/data/pairwise_curtailment_by_seed.csv"
PAIRWISE_SUMMARY = ROOT / "results/E21/pairwise_curtailment/data/pairwise_curtailment_summary.csv"
DIRECT_PAIRWISE = ROOT / "results/E21/pairwise_curtailment/direct_training/data/pairwise_eps_direct_by_seed.csv"
DIRECT_E23 = ROOT / "results/E23/data/e23_by_seed.csv"
DIRECT_SUPPLEMENT = ROOT / "results/E22/subpanel_direct_supplement/data/pairwise_eps_target_direct_by_seed.csv"
BASELINE_COMPLETION_PAIRWISE = ROOT / "results/E22/subpanel_direct_supplement/data/pairwise_baselines_target_by_seed.csv"
BASELINE_COMPLETION_ORIGINAL = ROOT / "results/E22/subpanel_direct_supplement/data/ieee123_original_baselines_m0_m4_m5_m6_by_seed.csv"
DIRECT_MODE = False

ALGORITHMS = [
    "no_coordination",
    "local_rules",
    "mpc_optimal",
    "mean_field_control",
    "virtual_battery",
    "packetized_energy_management",
    "transactive_control",
    "eps_ieee69_fused",
    "centralized_optimal",
]
COMPARISON_ALGORITHMS = ALGORITHMS[:-1]
ALGORITHM_CODES = {algorithm: chr(ord("A") + index) for index, algorithm in enumerate(ALGORITHMS)}
LABELS = {
    "no_coordination": "No coordination",
    "local_rules": "Local SOC rules",
    "mpc_optimal": "MPC",
    "mean_field_control": "Mean-field control",
    "virtual_battery": "Virtual battery",
    "packetized_energy_management": "Packetized Energy Management",
    "transactive_control": "Transactive control",
    "eps_ieee69_fused": "EPS",
    "centralized_optimal": "Centralized greedy UB",
}
COMPLEXITY = {
    "no_coordination": "O(0)",
    "local_rules": "O(1) per device",
    "mpc_optimal": "O(HN)",
    "mean_field_control": "O(N)",
    "virtual_battery": "O(N)",
    "packetized_energy_management": "O(N log N)",
    "transactive_control": "O(N log N)",
    "eps_ieee69_fused": "O(1)",
    "centralized_optimal": "O(N)",
}
COLORS = {
    "no_coordination": "#777777",
    "local_rules": "#E69F00",
    "mpc_optimal": "#0072B2",
    "mean_field_control": "#56B4E9",
    "virtual_battery": "#009E73",
    "packetized_energy_management": "#CC79A7",
    "transactive_control": "#D55E00",
    "eps_ieee69_fused": "#1B7837",
    "centralized_optimal": "#222222",
}
MARKERS = {
    "no_coordination": "o",
    "local_rules": "s",
    "mpc_optimal": "^",
    "mean_field_control": "D",
    "virtual_battery": "P",
    "packetized_energy_management": "X",
    "transactive_control": "v",
    "eps_ieee69_fused": "*",
    "centralized_optimal": "h",
}
TOPOLOGY_COLORS = {"ieee33": "#365A7C", "ieee69": "#55A6B5", "ieee123": "#E5633E"}
TOPOLOGY_LABELS = {"ieee33": "IEEE-33", "ieee69": "IEEE-69", "ieee123": "IEEE-123"}
DATASET_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#F0E442", "#882255", "#117733", "#332288",
    "#AA4499", "#44AA99", "#999933", "#88CCEE", "#DDCC77",
    "#CC6677", "#6699CC",
]
STRESS = ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]
STRESS_LABELS = {
    "M0": "Matched operating point",
    "M1": "Deep-feeder high load",
    "M2": "Dual bottleneck reverse flow",
    "M3": "Compound derating and forecast error",
    "M4": "Uniform device placement",
    "M5": "50% feeder concentration",
    "M6": "80% distal-node concentration",
}
STRESS_X_LABELS = {
    "M0": "Matched\npoint",
    "M1": "Deep-feeder\nload",
    "M2": "Dual\nbottleneck",
    "M3": "Derating\nerror",
    "M4": "Uniform\nplacement",
    "M5": "50%\nfeeder",
    "M6": "80% distal",
}
STRESS_COLORS = {
    "M0": "#4C78A8",
    "M1": "#F58518",
    "M2": "#E45756",
    "M3": "#72B7B2",
    "M4": "#54A24B",
    "M5": "#B279A2",
    "M6": "#FF9DA6",
}
HEATMAP_VMIN = 0.0
HEATMAP_VMAX = 100.0
CURTAILMENT_EFFECT_REFERENCE = 80.0
RESPONSE_FIDELITY_REFERENCE = 0.95
SPATIAL_CONDITION_LABELS = {
    "H0": "Regionally\nmixed",
    "H1": "Type-zoned",
    "H2_25": "25% density\nconcentration",
    "H2_50": "50% density\nconcentration",
    "H2_75": "75% density\nconcentration",
    "H3": "Joint type and\ndensity imbalance",
    "H4": "Adverse network\ncoupling",
    "H5": "Favorable network\ncoupling",
}
E24_SCENARIO_LABELS = {
    "S0_uniform": "Uniform deployment",
    "S1_feeder_50": "50% distal-feeder concentration",
    "S2_node_80": "80% distal-node concentration",
    "S3_feeder_50_derated": "50% distal-feeder concentration with 70% line capacity",
}
SCENARIO_COLORS = {
    "S0_uniform": "#365A7C",
    "S1_feeder_50": "#009E73",
    "S2_node_80": "#E69F00",
    "S3_feeder_50_derated": "#D55E00",
}

configure()


def save(fig: plt.Figure, name: str, *, tight: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    save_options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.035} if tight else {"facecolor": "white", "bbox_inches": None}
    fig.savefig(OUT / f"{name}.png", dpi=600, **save_options)
    fig.savefig(OUT / f"{name}.pdf", **save_options)
    plt.close(fig)


def format_percent_colorbar(colorbar: matplotlib.colorbar.Colorbar) -> None:
    ticks = np.linspace(HEATMAP_VMIN, HEATMAP_VMAX, 6)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value:.0f}%" for value in ticks])


def text_color(value: float, vmin: float, vmax: float) -> str:
    normalized = (value - vmin) / max(vmax - vmin, 1e-12)
    return "white" if normalized > 0.58 else "#111111"


def floor_to_10(value: float) -> float:
    return float(np.floor(value / 10.0) * 10.0)


def ceil_to_10(value: float) -> float:
    return float(np.ceil(value / 10.0) * 10.0)


def ensure_nonzero_floor(value: float) -> float:
    if abs(value) < 1e-12:
        return -10.0
    return value


def plot_boxline_groups(
    axis: plt.Axes,
    positions: np.ndarray,
    groups: list[np.ndarray],
    color: str,
    line_color: str,
    point_color: str | None = None,
    width: float = 0.16,
    marker: str = "o",
    linewidth: float = 2.0,
    markersize: float = 4.4,
    seed: int = 0,
    point_alpha: float = 0.52,
    point_size: float = 15.0,
    box_alpha: float = 0.22,
    point_jitter: float = 0.03,
) -> list[float]:
    """在同一横坐标上绘制箱型图、散点和连线。"""
    rng = np.random.default_rng(seed)
    medians: list[float] = []
    for position, values in zip(positions, groups):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            medians.append(np.nan)
            continue
        medians.append(float(np.median(values)))
        jitter = rng.uniform(-point_jitter, point_jitter, size=len(values))
        axis.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=point_size,
            color=point_color or color,
            alpha=point_alpha,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        axis.boxplot(
            [values],
            positions=[position],
            widths=width,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": shade(color, dl=0.30, ds=-0.20), "edgecolor": line_color, "linewidth": 1.1, "alpha": box_alpha},
            whiskerprops={"color": line_color, "linewidth": 0.95},
            capprops={"color": line_color, "linewidth": 0.95},
            medianprops={"color": line_color, "linewidth": 2.0},
            zorder=4,
        )
    axis.plot(
        positions,
        medians,
        color=line_color,
        marker=marker,
        markersize=markersize,
        linewidth=linewidth,
        zorder=5,
    )
    return medians


def annotate_heatmap(axis: plt.Axes, matrix: np.ndarray, fmt: str, vmin: float, vmax: float) -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if not np.isfinite(value):
                continue
            label = fmt.format(value)
            artist = axis.text(column, row, label, ha="center", va="center", color=text_color(value, vmin, vmax), fontsize=FS_SUBPANEL_CELL, fontweight="medium")
            artist.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground="black" if artist.get_color() == "white" else "white", alpha=0.25)])


def load_e22() -> pd.DataFrame:
    return pd.read_csv(E22, low_memory=False)


def load_raw_aggregate() -> pd.DataFrame:
    rows = []
    pattern = "results/*_ieee33_real_load/coverage_fix/network_constrained_new/data/curtailment_baselines_final_fixed5000_aggregate_data_driven.json"
    for path in sorted(ROOT.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = path.parent.parent.parent.parent.name.replace("_ieee33_real_load", "")
        for algorithm, result in payload["results"].items():
            for seed_row in result.get("seed_results", []):
                rows.append({"dataset": dataset, "algorithm": algorithm, **seed_row})
    return pd.DataFrame(rows)


def load_direct_original(topology: str) -> pd.DataFrame:
    """读取按数据集和网络条件直接训练得到的 EPS 原始数据。"""
    rows: list[dict[str, object]] = []
    if topology == "ieee69":
        data = pd.read_csv(E22_DIRECT, low_memory=False)
        data = data[
            (data["topology"] == topology)
            & (data["algorithm"] == "eps_ieee69_fused")
        ].copy()
        data["algorithm"] = "eps_ieee69_fused"
        return data
    if topology == "ieee123":
        data = pd.read_csv(E24_DIRECT, low_memory=False)
        data = data[
            (data["topology"] == topology)
            & (data["algorithm"] == "eps_ieee69_fused")
        ].copy()
        data["algorithm"] = "eps_ieee69_fused"
        return data
    if topology != "ieee33":
        raise ValueError(f"未知网络条件：{topology}")
    if E22_IEEE33_DIRECT.is_file():
        data = pd.read_csv(E22_IEEE33_DIRECT, low_memory=False)
        data = data[
            (data["topology"] == topology)
            & (data["algorithm"] == "eps_ieee69_fused")
        ].copy()
        if data["dataset"].nunique() == len(K.DATASET_LABELS):
            return data
    pattern = "results/*_ieee33_real_load/coverage_fix/network_constrained_new/data/curtailment_baselines_final_fixed5000_ieee33_data_driven.json"
    for path in sorted(ROOT.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = path.parent.parent.parent.parent.name.replace("_ieee33_real_load", "")
        result = payload.get("results", {}).get("eps_broadcast", {})
        for seed_row in result.get("seed_results", []):
            rows.append({
                "dataset": dataset,
                "dataset_label": K.DATASET_LABELS.get(dataset, dataset),
                "topology": "ieee33",
                "algorithm": "eps_ieee69_fused",
                **seed_row,
            })
    if not rows:
        raise FileNotFoundError("缺少 IEEE-33 各数据集直接训练 EPS 结果")
    return pd.DataFrame(rows)


def normalize_algorithm(value: str) -> str | None:
    aliases = {
        "eps_broadcast": "eps_ieee69_fused",
        "eps_e20_pooled": "eps_ieee69_fused",
        "eps_e21_mixed": "eps_ieee69_fused",
        "eps_ieee69_fused": "eps_ieee69_fused",
        "eps_global_mixed": "eps_ieee69_fused",
    }
    return aliases.get(str(value), str(value) if str(value) in ALGORITHMS else None)


def load_pairwise_mixed(network_mode: str) -> pd.DataFrame:
    """读取 E21 两两组合的全算法结果，不将 aggregate 借作 IEEE-69。"""
    data = pd.read_csv(ROOT / "results/E21/curtailment_baseline_supplement/data/baseline_by_seed.csv", low_memory=False)
    data = data[(data["composition_kind"] == "pairwise") & (data["network_mode"] == network_mode)].copy()
    data = data[~data["algorithm"].isin(["eps_e20_pooled", "eps_e21_mixed"])].copy()
    data["algorithm"] = data["algorithm"].map(normalize_algorithm)
    data = data[data["algorithm"].notna()].copy()
    if not DIRECT_MODE or not DIRECT_PAIRWISE.is_file():
        return data
    direct = pd.read_csv(DIRECT_PAIRWISE, low_memory=False)
    direct = direct[direct["network_mode"] == network_mode].copy()
    direct["composition_id"] = direct["pair_id"]
    direct["algorithm"] = "eps_ieee69_fused"
    direct["mean_reduction_pct"] = direct["curtailment_reduction_pct"]
    direct["source"] = "pairwise_direct_training"
    direct = direct[["composition_id", "algorithm", "mean_reduction_pct"]].copy()
    return pd.concat([data, direct], ignore_index=True)


def load_direct_pairwise(topology: str) -> pd.DataFrame:
    """读取指定目标网络上的 EPS 两两组合直接训练结果。"""
    frames: list[pd.DataFrame] = []
    if DIRECT_SUPPLEMENT.is_file():
        data = pd.read_csv(DIRECT_SUPPLEMENT, low_memory=False)
        data = data[data["topology"] == topology].copy()
        if not data.empty:
            data["composition_id"] = data["pair_id"]
            data["algorithm"] = "eps_ieee69_fused"
            frames.append(data[["composition_id", "algorithm", "mean_reduction_pct"]])
    if BASELINE_COMPLETION_PAIRWISE.is_file():
        data = pd.read_csv(BASELINE_COMPLETION_PAIRWISE, low_memory=False)
        data = data[data["topology"] == topology].copy()
        if not data.empty:
            frames.append(data[["composition_id", "algorithm", "mean_reduction_pct"]])
    if not frames:
        return pd.DataFrame(columns=["composition_id", "algorithm", "mean_reduction_pct"])
    return pd.concat(frames, ignore_index=True)


def load_direct_original_baselines(topology: str, stress_modes: tuple[str, ...] | None = None) -> pd.DataFrame:
    """读取 IEEE-123 原始数据集的非EPS目标网络结果。"""
    if topology != "ieee123" or not BASELINE_COMPLETION_ORIGINAL.is_file():
        return pd.DataFrame(columns=["dataset", "topology", "algorithm", "stress_mode", "mean_reduction_pct"])
    data = pd.read_csv(BASELINE_COMPLETION_ORIGINAL, low_memory=False)
    if stress_modes is not None:
        data = data[data["stress_mode"].isin(stress_modes)].copy()
    return data


def load_uniform_topology_data(topology: str) -> pd.DataFrame:
    """读取指定拓扑在正常均匀部署下的原始数据。"""
    if topology in {"ieee33", "ieee69"}:
        data = pd.read_csv(E22, low_memory=False)
        data = data[(data["topology"] == topology) & (data["stress_mode"] == "M0")].copy()
        data = data[~data["algorithm"].isin(["eps_e20_pooled", "eps_e21_mixed"])].copy()
        data["algorithm"] = data["algorithm"].map(normalize_algorithm)
        data = data[data["algorithm"].notna()].copy()
        eps = load_direct_original(topology)
        eps = eps[eps["stress_mode"] == "M0"].copy()
        data = pd.concat([data, eps], ignore_index=True, sort=False)
        return data
    if topology != "ieee123":
        raise ValueError(f"未知拓扑：{topology}")
    data = load_direct_original_baselines("ieee123", ("M0",)).copy()
    data["algorithm"] = data["algorithm"].map(normalize_algorithm)
    data = data[data["algorithm"].notna()].copy()
    eps = load_direct_original("ieee123").copy()
    eps = eps[eps["stress_mode"] == "M0"].copy()
    return pd.concat([data, eps], ignore_index=True, sort=False)


def load_uniform_pairwise_data(topology: str) -> pd.DataFrame:
    """读取指定拓扑在正常均匀部署下的两两 mixed 数据。"""
    data = pd.read_csv(BASELINE_COMPLETION_PAIRWISE, low_memory=False)
    data = data[data["topology"] == topology].copy()
    if "composition_id" not in data.columns:
        data["composition_id"] = data["pair_id"]
    data = data[~data["algorithm"].isin(["eps_e20_pooled", "eps_e21_mixed"])].copy()
    data["algorithm"] = data["algorithm"].map(normalize_algorithm)
    data = data[data["algorithm"].notna()].copy()
    if topology in {"ieee33", "ieee69", "ieee123"}:
        eps = pd.read_csv(DIRECT_SUPPLEMENT, low_memory=False)
        eps = eps[eps["topology"] == topology].copy()
        eps["composition_id"] = eps["pair_id"]
        eps["algorithm"] = "eps_ieee69_fused"
        data = pd.concat([data, eps], ignore_index=True, sort=False)
    return data


def load_uniform_r2_data(topology: str) -> pd.DataFrame:
    """读取指定拓扑正常均匀部署下的响应 R²。"""
    data = load_uniform_topology_data(topology)
    data["response_r2"] = pd.to_numeric(data["response_r2"], errors="coerce")
    return data[data["response_r2"].notna()].copy()


def merge_direct_eps(raw: pd.DataFrame, topology: str) -> pd.DataFrame:
    """用目标网络直接训练 EPS 替换对应拓扑的旧 pooled EPS。"""
    base = raw[raw["algorithm"] != "eps_ieee69_fused"].copy()
    return pd.concat([base, load_direct_original(topology)], ignore_index=True)


def raw_point_values(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby(["dataset", "algorithm"], as_index=False, observed=False)["mean_reduction_pct"].mean()


def mixed_point_values(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby(["composition_id", "algorithm"], as_index=False, observed=False)["mean_reduction_pct"].mean()


def draw_algorithm_cloud_combined(
    axis: plt.Axes,
    raw: pd.DataFrame,
    mixed: pd.DataFrame,
    y0: float,
    algorithm: str,
    dataset_colours: dict[str, str],
    seed: int,
) -> None:
    raw_values = raw.loc[raw["algorithm"] == algorithm, "mean_reduction_pct"].dropna().to_numpy()
    mixed_values = mixed.loc[mixed["algorithm"] == algorithm, "mean_reduction_pct"].dropna().to_numpy()
    raw_points = raw_point_values(raw[raw["algorithm"] == algorithm])
    mixed_points = mixed_point_values(mixed[mixed["algorithm"] == algorithm])
    combined_values = np.concatenate([raw_values, mixed_values])
    combined_points = np.concatenate([
        raw_points["mean_reduction_pct"].to_numpy(),
        mixed_points["mean_reduction_pct"].to_numpy(),
    ])
    combined_colours = [
        dataset_colours.get(str(dataset), INK) for dataset in raw_points["dataset"]
    ] + [MIX_INK] * len(mixed_points)
    cloud(
        axis,
        combined_values,
        y0 - 0.08,
        COLORS[algorithm],
        fill=shade(COLORS[algorithm], dl=0.18, ds=-0.12),
        height=0.42,
        points=True,
        point_values=combined_points,
        pt_colours=combined_colours,
        pt_alpha=0.95,
        ms=2.7,
        seed=seed,
        bw_floor=0.10,
    )


def draw_metric_cloud_combined(
    axis: plt.Axes,
    raw: pd.DataFrame,
    mixed: pd.DataFrame,
    y0: float,
    algorithm: str,
    metric: str,
    dataset_colours: dict[str, str],
    seed: int,
    bw_floor: float,
) -> None:
    raw_values = raw.loc[raw["algorithm"] == algorithm, metric].dropna().to_numpy()
    mixed_values = mixed.loc[mixed["algorithm"] == algorithm, metric].dropna().to_numpy()
    raw_points = (
        raw[raw["algorithm"] == algorithm]
        .groupby("dataset", as_index=False, observed=False)[metric]
        .mean()
    )
    mixed_points = (
        mixed[mixed["algorithm"] == algorithm]
        .groupby("composition_id", as_index=False, observed=False)[metric]
        .mean()
    )
    combined_values = np.concatenate([raw_values, mixed_values])
    point_values = np.concatenate(
        [raw_points[metric].to_numpy(), mixed_points[metric].to_numpy()]
    )
    point_colours = [
        dataset_colours.get(str(dataset), INK) for dataset in raw_points["dataset"]
    ] + [MIX_INK] * len(mixed_points)
    cloud(
        axis,
        combined_values,
        y0 - 0.08,
        COLORS[algorithm],
        fill=shade(COLORS[algorithm], dl=0.18, ds=-0.12),
        height=0.42,
        points=True,
        point_values=point_values,
        pt_colours=point_colours,
        pt_alpha=0.95,
        ms=2.7,
        seed=seed,
        bw_floor=bw_floor,
    )


def write_source_data(name: str, rows: pd.DataFrame) -> None:
    source_dir = OUT / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(source_dir / f"{name}.csv", index=False)


def canonical_e22(frame: pd.DataFrame, topology: str | None = None, paired_eps: bool = False) -> pd.DataFrame:
    data = frame.copy()
    if topology is not None:
        data = data[data["topology"] == topology].copy()
    data = data[~data["algorithm"].isin(["eps_e20_pooled", "eps_e21_mixed"])].copy()
    if DIRECT_MODE:
        direct_topologies = [topology] if topology is not None else sorted(frame["topology"].dropna().unique())
        eps = pd.concat([load_direct_original(value) for value in direct_topologies], ignore_index=True)
    elif paired_eps:
        eps = frame[frame["algorithm"] == "eps_e20_pooled"].copy()
    else:
        eps = pd.read_csv(E22_DIRECT, low_memory=False)
        eps = eps[eps["algorithm"] == "eps_ieee69_fused"].copy()
    if topology is not None:
        eps = eps[eps["topology"] == topology].copy()
    eps["algorithm"] = "eps_ieee69_fused"
    return pd.concat([data, eps], ignore_index=True)


def ordered_algorithm_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame[frame["algorithm"].isin(ALGORITHMS)].copy()
    data["algorithm"] = pd.Categorical(data["algorithm"], categories=ALGORITHMS, ordered=True)
    return data


def dataset_color_map(frame: pd.DataFrame) -> tuple[list[str], dict[str, str], dict[str, str]]:
    metadata = frame[["dataset", "dataset_label"]].drop_duplicates("dataset")
    datasets = metadata["dataset"].tolist()
    colors = {dataset: K.DATASET_COLOURS.get(dataset, DATASET_COLORS[index % len(DATASET_COLORS)]) for index, dataset in enumerate(datasets)}
    labels = {dataset: K.DATASET_LABELS.get(dataset, str(label)) for dataset, label in zip(metadata["dataset"], metadata["dataset_label"])}
    return datasets, colors, labels


def dataset_handles(datasets: list[str], colors: dict[str, str], labels: dict[str, str]) -> list[plt.Line2D]:
    return [plt.Line2D([0], [0], color=colors[dataset], marker="o", linestyle="none", markersize=MS, label=labels[dataset]) for dataset in datasets]


def make_acceptance() -> None:
    data = ordered_algorithm_data(canonical_e22(load_e22(), "ieee69"))
    matrix = data.groupby(["algorithm", "stress_mode"], observed=False)["network_acceptance_ratio"].mean().unstack().reindex(index=ALGORITHMS, columns=STRESS) * 100.0
    fig, axis = plt.subplots(figsize=(11.5, 6.5))
    image = axis.imshow(matrix.to_numpy(), cmap=SEQ, vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, aspect="auto")
    axis.set_xticks(range(7), [STRESS_X_LABELS[value] for value in STRESS], rotation=0)
    axis.set_yticks(range(len(ALGORITHMS)), [ALGORITHM_CODES[a] for a in ALGORITHMS])
    axis.tick_params(axis="both", labelsize=FS_SUBPANEL_TICK)
    axis.tick_params(axis="x", pad=10)
    axis.set_xlabel("")
    annotate_heatmap(axis, matrix.to_numpy(), "{:.1f}%", HEATMAP_VMIN, HEATMAP_VMAX)
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    format_percent_colorbar(colorbar)
    fig.axes[-1].tick_params(labelsize=FS_SUBPANEL_TICK)
    fig.tight_layout()
    save(fig, "00_ieee69_network_acceptance")


def make_constraint_audit() -> None:
    if not SINGLE_CONSTRAINT.is_file():
        raise FileNotFoundError(f"缺少单约束实验结果：{SINGLE_CONSTRAINT}")
    data = pd.read_csv(SINGLE_CONSTRAINT, low_memory=False)
    # The single-constraint summary contains stale one-seed compatibility
    # rows.  Keep only the completed 30-seed Line/voltage sweep.
    data = data[data["seed_count"] == 30].copy()
    if not TRANSFORMER_REPLAY.is_file():
        raise FileNotFoundError(f"缺少 fixed-desired Transformer replay：{TRANSFORMER_REPLAY}")
    transformer_replay = pd.read_csv(TRANSFORMER_REPLAY, low_memory=False)
    transformer_replay = transformer_replay[
        (transformer_replay["constraint"] == "Transformer")
        & transformer_replay["fixed_desired"].astype(bool)
    ].copy()
    transformer_replay["mean_reduction_pct_median"] = transformer_replay["curtailment_value_pct"]
    transformer_replay["mean_reduction_pct_mean"] = transformer_replay["curtailment_value_pct"]
    transformer_replay["response_r2_median"] = transformer_replay["response_r2_pct"] / 100.0
    transformer_replay["response_r2_mean"] = transformer_replay["response_r2_pct"] / 100.0
    transformer_replay["network_acceptance_ratio_median"] = transformer_replay["network_acceptance_pct"] / 100.0
    transformer_replay["network_acceptance_ratio_mean"] = transformer_replay["network_acceptance_pct"] / 100.0
    transformer_replay["seed_count"] = 1
    data = pd.concat(
        [data[data["constraint"] != "Transformer"], transformer_replay],
        ignore_index=True,
        sort=False,
    )
    constraints = ("Line", "Transformer", "Minimum voltage", "Maximum voltage")
    fig, axes = plt.subplots(1, 4, figsize=(11.41, 3.60), sharex=False)
    offsets = {"ieee33": -0.22, "ieee69": 0.0, "ieee123": 0.22}
    source_rows: list[pd.DataFrame] = []
    bar_pair_width = 0.09
    figure_legend_handles: list[object] = []
    for column_index, constraint in enumerate(constraints):
        axis = axes[column_index]
        right_axis = axis.twinx()
        subset_constraint = data[data["constraint"] == constraint].copy()
        raw_values = subset_constraint["constraint_value"].dropna().unique()
        if constraint in {"Line", "Transformer"}:
            values = sorted(raw_values, reverse=True)
            tick_labels = [f"{100.0 * value:.0f}%" for value in values]
        else:
            scan_values = sorted(raw_values, reverse=constraint == "Maximum voltage")
            initial_value = 0.95 if constraint == "Minimum voltage" else 1.05
            initial_rows = subset_constraint[
                np.isclose(subset_constraint["constraint_value"], initial_value)
            ]
            initial_effect = float(initial_rows["mean_reduction_pct_median"].mean())
            values = []
            for value in scan_values:
                rows_at_value = subset_constraint[
                    np.isclose(subset_constraint["constraint_value"], value)
                ]
                effect = float(rows_at_value["mean_reduction_pct_median"].mean())
                if initial_effect - effect > 5.0 and values:
                    break
                values.append(value)
            tick_labels = [f"{100.0 * value:.1f}%" for value in values]
        tick_positions = np.arange(len(values), dtype=float)
        if len(values) > 6:
            tick_indices = np.linspace(0, len(values) - 1, 5, dtype=int)
            tick_positions = tick_indices.astype(float)
            tick_labels = [tick_labels[index] for index in tick_indices]
        all_left_values: list[np.ndarray] = []
        all_right_values: list[np.ndarray] = []
        topology_handles: list[plt.Line2D] = []
        for topology in ("ieee33", "ieee69", "ieee123"):
            position = np.arange(len(values), dtype=float) + offsets[topology]
            left_groups: list[np.ndarray] = []
            r2_groups: list[np.ndarray] = []
            accept_groups: list[np.ndarray] = []
            left_medians: list[float] = []
            r2_medians: list[float] = []
            accept_medians: list[float] = []
            for value in values:
                selected = subset_constraint[
                    (subset_constraint["topology"] == topology)
                    & np.isclose(subset_constraint["constraint_value"], value)
                ].copy()
                if selected.empty:
                    left_groups.append(np.array([]))
                    r2_groups.append(np.array([]))
                    accept_groups.append(np.array([]))
                    left_medians.append(np.nan)
                    r2_medians.append(np.nan)
                    accept_medians.append(np.nan)
                    continue
                left_values = selected["mean_reduction_pct_median"].to_numpy(dtype=float)
                r2_values = selected["response_r2_median"].to_numpy(dtype=float) * 100.0
                accept_values = selected["network_acceptance_ratio_median"].to_numpy(dtype=float) * 100.0
                left_groups.append(left_values)
                r2_groups.append(r2_values)
                accept_groups.append(accept_values)
                left_medians.append(float(np.median(left_values)))
                r2_medians.append(float(np.median(r2_values)))
                accept_medians.append(float(np.median(accept_values)))
                source_rows.append(
                    selected.assign(
                        curtailment_value_pct=left_values,
                        response_r2_pct=r2_values,
                        network_acceptance_pct=accept_values,
                    )
                )
            all_left_values.extend(left_groups)
            all_right_values.extend(r2_groups)
            all_right_values.extend(accept_groups)
            plot_boxline_groups(
                axis,
                position,
                left_groups,
                color=TOPOLOGY_COLORS[topology],
                line_color=TOPOLOGY_COLORS[topology],
                point_color=TOPOLOGY_COLORS[topology],
                width=0.12,
                marker="o",
                linewidth=2.0,
                markersize=4.8,
                seed=20261200 + column_index * 10 + (0 if topology == "ieee33" else 1 if topology == "ieee69" else 2),
                point_alpha=0.56,
                point_size=14.0,
                box_alpha=0.26,
            )
            topology_handles.append(
                plt.Line2D(
                    [0], [0], color=TOPOLOGY_COLORS[topology], marker="o",
                    linewidth=1.8, markersize=4.5, label=TOPOLOGY_LABELS[topology],
                )
            )
            right_axis.bar(
                position - bar_pair_width / 2,
                r2_medians,
                width=bar_pair_width,
                color=TOPOLOGY_COLORS[topology],
                alpha=0.70,
                edgecolor=TOPOLOGY_COLORS[topology],
                linewidth=0.55,
                bottom=0.0,
                zorder=2,
            )
            right_axis.bar(
                position + bar_pair_width / 2,
                accept_medians,
                width=bar_pair_width,
                color=shade(TOPOLOGY_COLORS[topology], dl=0.14, ds=-0.08),
                alpha=0.62,
                hatch="//",
                edgecolor=TOPOLOGY_COLORS[topology],
                linewidth=0.55,
                bottom=0.0,
                zorder=2,
            )
        left_values_all = np.concatenate([values for values in all_left_values if len(values)]) if all_left_values else np.array([0.0])
        right_values_all = np.concatenate([values for values in all_right_values if len(values)]) if all_right_values else np.array([0.0])
        left_min = float(np.floor(np.nanmin(left_values_all) / 5.0) * 5.0)
        left_max = float(np.ceil(np.nanmax(left_values_all) / 5.0) * 5.0)
        right_min = 20.0
        right_max = 180.0
        if left_min == left_max:
            left_min, left_max = left_min - 10.0, left_max + 10.0
        axis.axhline(0.0, color=RULE, linestyle=(0, (3.0, 2.0)), linewidth=1.0)
        axis.set_ylim(left_min, left_max)
        right_axis.set_ylim(right_min, right_max)
        axis.set_xticks(tick_positions, tick_labels)
        axis.yaxis.set_major_locator(ticker.MultipleLocator(10.0))
        right_axis.yaxis.set_major_locator(ticker.MultipleLocator(20.0))
        axis.tick_params(axis="both", labelsize=13)
        axis.tick_params(axis="x", pad=5, rotation=30)
        right_axis.tick_params(axis="y", labelsize=13)
        axis.grid(axis="y", color=GRID, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        right_axis.spines[["top", "left"]].set_visible(False)
        if column_index == 0:
            axis.set_ylabel(
                "Network acceptance (%)" if constraint == "Transformer" else "Curtailment reduction (%)",
                fontsize=14,
            )
            right_axis.tick_params(axis="y", labelright=False, right=False)
        else:
            axis.tick_params(axis="y", labelleft=False, left=False)
            axis.spines["left"].set_visible(False)
        if column_index < len(constraints) - 1:
            right_axis.tick_params(axis="y", labelright=False, right=False)
            right_axis.spines["right"].set_visible(False)
        else:
            right_axis.spines["right"].set_position(("outward", 6))
        title_text = {
            "Line": "Line loading limit",
            "Transformer": "Transformer capacity",
            "Minimum voltage": "Minimum voltage",
            "Maximum voltage": "Maximum voltage",
        }[constraint]
        if constraint == "Transformer":
            title_text += "\n(fixed-desired replay)"
        elif constraint == "Maximum voltage" and len(values) == len(raw_values):
            title_text += "\n(no 5-pp loss observed)"
        axis.set_title(title_text, fontsize=13, pad=6)
        if column_index == len(constraints) - 1:
            right_axis.set_ylabel("R² and network acceptance (%)", fontsize=13)
        if column_index == 0:
            metric_handles = [
                plt.Rectangle((0, 0), 1, 1, color=C3, alpha=0.70, label="R²"),
                plt.Rectangle((0, 0), 1, 1, color=C3, alpha=0.62, hatch="//", label="Network acceptance"),
            ]
            figure_legend_handles = topology_handles + metric_handles
    write_source_data("01_ieee69_constraint_audit", pd.concat(source_rows, ignore_index=True))
    fig.legend(
        figure_legend_handles,
        [handle.get_label() for handle in figure_legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        fontsize=11,
        frameon=False,
        ncol=5,
        columnspacing=1.2,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.065, right=0.90, bottom=0.26, top=0.78, wspace=0.32)
    save(fig, "01_ieee69_constraint_audit", tight=False)


def make_effect_safety() -> None:
    data = pd.read_csv(E23_SPATIAL_DIRECT, low_memory=False)
    data["algorithm"] = data["algorithm"].map(normalize_algorithm)
    data = data[data["algorithm"].notna()].copy()
    conditions = tuple(SPATIAL_CONDITION_LABELS)
    x = np.arange(len(conditions), dtype=float)
    fig, axis = plt.subplots(figsize=(11.475, 5.7))
    offsets = np.linspace(-0.34, 0.34, len(COMPARISON_ALGORITHMS))
    for algorithm_index, algorithm in enumerate(COMPARISON_ALGORITHMS):
        subset = data[data["algorithm"] == algorithm]
        grouped = subset.groupby("condition", observed=False)["mean_reduction_pct"]
        groups = [grouped.get_group(condition).to_numpy() if condition in grouped.groups else np.array([]) for condition in conditions]
        algorithm_x = x + offsets[algorithm_index]
        plot_boxline_groups(
            axis,
            algorithm_x,
            groups,
            color=COLORS[algorithm],
            line_color=COLORS[algorithm],
            point_color=COLORS[algorithm],
            width=0.095,
            marker=MARKERS[algorithm],
            linewidth=3.4 if algorithm == "eps_ieee69_fused" else 2.2,
            markersize=8.1 if algorithm == "eps_ieee69_fused" else 6.5,
            seed=20260960 + ALGORITHMS.index(algorithm),
            point_alpha=0.52,
            point_size=14.0,
            box_alpha=0.23,
            point_jitter=0.018,
        )
    upper_bound = (
        data[data["algorithm"] == "centralized_optimal"]
        .groupby("condition", observed=False)["mean_reduction_pct"]
        .median()
        .reindex(conditions)
        .to_numpy()
    )
    axis.plot(
        x,
        upper_bound,
        color=COLORS["centralized_optimal"],
        linestyle="--",
        linewidth=2.0,
        label="Centralized greedy upper bound  O(N)",
    )
    axis.set_xticks(x, [SPATIAL_CONDITION_LABELS[value] for value in conditions])
    axis.set_ylabel("")
    axis.set_xlim(-0.48, len(conditions) - 0.52)
    axis.set_ylim(0, 105)
    axis.tick_params(axis="both", labelsize=24)
    axis.tick_params(axis="x", pad=7, labelsize=17, labelrotation=40)
    for label in axis.get_xticklabels():
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    handles = [plt.Line2D([0], [0], color=COLORS[alg], marker=MARKERS[alg], linewidth=1.6) for alg in ALGORITHMS]
    handles[-1].set_linestyle("--")
    labels = [f"{LABELS[alg]}  {COMPLEXITY[alg]}" for alg in ALGORITHMS]
    labels[5] = "Packetized Energy Management\nO(N log N)"
    labels[-1] = "Centralized greedy upper bound\nO(N)"
    axis.legend(handles, labels, loc="lower left", ncol=3, fontsize=11.5,
                frameon=False, columnspacing=0.8, handlelength=1.5,
                handletextpad=0.45, borderaxespad=0.4)
    write_source_data("02_algorithm_effect_raw_vs_pairwise_mixed", data)
    fig.subplots_adjust(left=0.08, right=0.975, bottom=0.34, top=0.97)
    save(fig, "02_algorithm_effect_raw_vs_pairwise_mixed", tight=False)


def make_spatial_retention() -> None:
    frames = []
    base = load_e22()
    for topology in ("ieee33", "ieee69"):
        selected = base[
            (base["topology"] == topology)
            & (base["stress_mode"].isin(["M4", "M5", "M6"]))
            & (~base["algorithm"].isin(["eps_e20_pooled", "eps_e21_mixed"]))
        ].copy()
        selected["algorithm"] = selected["algorithm"].map(normalize_algorithm)
        direct = load_direct_original(topology)
        direct = direct[direct["stress_mode"].isin(["M4", "M5", "M6"])].copy()
        frames.extend([selected, direct])
    ieee123 = load_direct_original_baselines("ieee123", ("M4", "M5", "M6")).copy()
    ieee123["algorithm"] = ieee123["algorithm"].map(normalize_algorithm)
    frames.extend([ieee123, load_direct_original("ieee123").query("stress_mode in ['M4', 'M5', 'M6']")])
    data = pd.concat(frames, ignore_index=True, sort=False)
    grouped = (
        data.groupby(["dataset", "topology", "stress_mode", "algorithm"], observed=False)["mean_reduction_pct"]
        .mean()
        .reset_index()
    )
    rows = []
    for (dataset, topology, algorithm), subset in grouped.groupby(["dataset", "topology", "algorithm"], observed=False):
        reference = subset.loc[subset["stress_mode"] == "M4", "mean_reduction_pct"]
        if reference.empty:
            continue
        reference_value = float(reference.iloc[0])
        for stress_mode in ("M5", "M6"):
            value = subset.loc[subset["stress_mode"] == stress_mode, "mean_reduction_pct"]
            if value.empty:
                continue
            retention_value = (
                float(value.iloc[0]) / reference_value
                if reference_value > 1e-12
                else 0.0 if algorithm == "no_coordination" else np.nan
            )
            rows.append({
                "dataset": dataset,
                "topology": topology,
                "algorithm": algorithm,
                "stress_mode": stress_mode,
                "retention": retention_value,
            })
    retention = pd.DataFrame(rows)
    columns = pd.MultiIndex.from_product(
        [["ieee33", "ieee69", "ieee123"], ["M5", "M6"]]
    )
    write_source_data("03_ieee69_spatial_retention", retention)
    # Use six independent x axes so each topology/condition keeps its own
    # useful range instead of being forced into a single crowded scale.
    fig, axes = plt.subplots(
        1, 6, figsize=(11.5, 6.5), sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    topology_colours = {"ieee33": "#DCEAF7", "ieee69": "#DDF2E8", "ieee123": "#FBE7D3"}
    condition_names = {"M5": "50% feeder", "M6": "80% distal node"}
    datasets = sorted(retention["dataset"].dropna().unique())
    dataset_colors = {
        dataset: K.DATASET_COLOURS.get(dataset, DATASET_COLORS[index % len(DATASET_COLORS)])
        for index, dataset in enumerate(datasets)
    }
    mixed_by_topology = {
        topology: load_uniform_pairwise_data(topology)
        for topology in ("ieee33", "ieee69", "ieee123")
    }
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        y = len(ALGORITHMS) - 1 - algorithm_index
        colour = COLORS[algorithm]
        for topology_index, topology in enumerate(("ieee33", "ieee69", "ieee123")):
            for condition_index, stress_mode in enumerate(("M5", "M6")):
                column_index = topology_index * 2 + condition_index
                axis = axes[column_index]
                axis.set_facecolor("white")
                values = retention.loc[
                    (retention["topology"] == topology)
                    & (retention["stress_mode"] == stress_mode)
                    & (retention["algorithm"] == algorithm),
                    "retention",
                ].dropna().to_numpy(dtype=float) * 100.0
                if values.size == 0:
                    continue
                source_rows = retention.loc[
                    (retention["topology"] == topology)
                    & (retention["stress_mode"] == stress_mode)
                    & (retention["algorithm"] == algorithm),
                    ["dataset", "retention"],
                ].dropna(subset=["retention"])
                y_jitter = np.random.default_rng(20260901 + algorithm_index * 100 + column_index).uniform(-0.07, 0.07, len(source_rows))
                axis.scatter(
                    source_rows["retention"].to_numpy(dtype=float) * 100.0,
                    y + y_jitter,
                    s=28,
                    c=colour,
                    alpha=1.0,
                    edgecolors="none",
                    zorder=1,
                )
                # Original-data point cloud and arithmetic mean only.
                mean_value = float(np.mean(values))
                q25, q75 = np.quantile(values, [0.25, 0.75])
                axis.plot([float(np.min(values)), float(np.max(values))], [y, y], color=colour, linewidth=2.2, zorder=2)
                axis.plot([q25, q75], [y, y], color=colour, linewidth=6.5, solid_capstyle="butt", zorder=3)
                axis.scatter([mean_value], [y], s=195, color="white", edgecolor=colour, linewidth=3.4, zorder=4)
    for column_index, axis in enumerate(axes):
        topology = ("ieee33", "ieee69", "ieee123")[column_index // 2]
        stress_mode = ("M5", "M6")[column_index % 2]
        subset = retention.loc[
            (retention["topology"] == topology) & (retention["stress_mode"] == stress_mode),
            "retention",
        ].dropna().to_numpy(dtype=float) * 100.0
        upper = max(100.0, float(np.nanmax(subset)) if subset.size else 100.0)
        upper = float(np.ceil((upper + 5.0) / 10.0) * 10.0)
        axis.set_xlim(-5.0, upper + 5.0)
        axis.set_ylim(-0.65, len(ALGORITHMS) - 0.35)
        tick_values = np.array([0.0, upper])
        axis.set_xticks(tick_values)
        tick_labels = [f"{value:.0f}" for value in tick_values]
        if column_index != 0:
            tick_labels[0] = ""
        axis.set_xticklabels(tick_labels)
        axis.tick_params(axis="x", labelsize=24, pad=3, length=3)
        axis.tick_params(axis="y", labelsize=22)
        axis.grid(axis="x", color=GRID, linewidth=0.45, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("")
        if column_index == 0:
            axis.set_yticks(range(len(ALGORITHMS)), [ALGORITHM_CODES[a] for a in ALGORITHMS[::-1]])
        else:
            axis.tick_params(axis="y", labelleft=False)
    # Lower the x-axis baseline to match panel a while retaining a tall plot.
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.08, top=0.98)
    save(fig, "03_ieee69_spatial_retention", tight=False)


def make_topology_boxpoint() -> None:
    """Draw topology comparison as a compact forest plot.

    Each algorithm has a coloured original-data interval and a grey pairwise-
    mixed interval.  The raw points remain visible, with their dataset colour,
    so the plot changes the presentation without changing the source records.
    """
    topologies = ("ieee33", "ieee69", "ieee123")
    source_by_topology = {topology: load_uniform_topology_data(topology) for topology in topologies}
    mixed_by_topology = {topology: load_uniform_pairwise_data(topology) for topology in topologies}
    dataset_colors = {
        dataset: K.DATASET_COLOURS.get(dataset, DATASET_COLORS[index % len(DATASET_COLORS)])
        for index, dataset in enumerate(
            sorted(pd.concat(source_by_topology.values(), ignore_index=True)["dataset"].dropna().unique())
        )
    }
    figure_height = 90.9
    fig = plt.figure(figsize=(WIDTH_2COL * MM, figure_height * MM), facecolor="white")
    axes = {
        topology: fig.add_axes([0.095 + panel_index * 0.30, 0.11, 0.27, 0.86])
        for panel_index, topology in enumerate(topologies)
    }
    row_positions = np.arange(len(COMPARISON_ALGORITHMS), dtype=float)
    for panel_index, topology in enumerate(topologies):
        axis = axes[topology]
        raw = source_by_topology[topology]
        mixed = mixed_by_topology[topology]
        for row, algorithm in zip(row_positions, COMPARISON_ALGORITHMS):
            raw_values = raw.loc[raw["algorithm"] == algorithm, "mean_reduction_pct"].dropna().to_numpy(dtype=float)
            mixed_values = mixed.loc[mixed["algorithm"] == algorithm, "mean_reduction_pct"].dropna().to_numpy(dtype=float)
            raw_values = raw_values[np.isfinite(raw_values)]
            mixed_values = mixed_values[np.isfinite(mixed_values)]
            if len(raw_values):
                q10, q25, median, q75, q90 = np.quantile(raw_values, [0.10, 0.25, 0.50, 0.75, 0.90])
                axis.plot(np.array([q10, q90]) / 100.0, [row + 0.13, row + 0.13], color=TOPOLOGY_COLORS[topology], linewidth=1.2, zorder=2)
                axis.plot(np.array([q25, q75]) / 100.0, [row + 0.13, row + 0.13], color=TOPOLOGY_COLORS[topology], linewidth=5.0, solid_capstyle="butt", zorder=3)
                raw_points = raw.loc[raw["algorithm"] == algorithm, ["dataset", "mean_reduction_pct"]].dropna(subset=["mean_reduction_pct"])
                jitter = np.random.default_rng(20260900 + panel_index * 100 + int(row)).uniform(-0.055, 0.055, len(raw_points))
                axis.scatter(
                    raw_points["mean_reduction_pct"] / 100.0,
                    row + 0.13 + jitter,
                    s=8.0,
                    c=[dataset_colors.get(str(dataset), INK) for dataset in raw_points["dataset"]],
                    alpha=0.22,
                    edgecolors="none",
                    zorder=1,
                )
                axis.scatter([median / 100.0], [row + 0.13], s=34, marker="o", facecolor="white", edgecolor=TOPOLOGY_COLORS[topology], linewidth=1.25, zorder=5)
            if len(mixed_values):
                q10, q25, median, q75, q90 = np.quantile(mixed_values, [0.10, 0.25, 0.50, 0.75, 0.90])
                axis.plot(np.array([q10, q90]) / 100.0, [row - 0.13, row - 0.13], color=MIX_INK, linewidth=1.1, zorder=2)
                axis.plot(np.array([q25, q75]) / 100.0, [row - 0.13, row - 0.13], color=MIX_INK, linewidth=5.0, solid_capstyle="butt", zorder=3)
                mixed_points = mixed.loc[mixed["algorithm"] == algorithm, "mean_reduction_pct"].dropna().to_numpy(dtype=float)
                jitter = np.random.default_rng(20260930 + panel_index * 100 + int(row)).uniform(-0.055, 0.055, len(mixed_points))
                axis.scatter(mixed_points / 100.0, row - 0.13 + jitter, s=7.0, color=MIX_INK, alpha=0.15, edgecolors="none", zorder=1)
                axis.scatter([median / 100.0], [row - 0.13], s=34, marker="D", facecolor="white", edgecolor=MIX_INK, linewidth=1.1, zorder=5)
        upper_bound_values = pd.concat([
            raw.loc[raw["algorithm"] == "centralized_optimal", "mean_reduction_pct"],
            mixed.loc[mixed["algorithm"] == "centralized_optimal", "mean_reduction_pct"],
        ]).dropna()
        if not upper_bound_values.empty:
            axis.axvline(float(upper_bound_values.median()) / 100.0, color=COLORS["centralized_optimal"], linestyle=(0, (3.0, 2.0)), linewidth=1.4, zorder=0)
        axis.set_yticks(row_positions, [ALGORITHM_CODES[algorithm] for algorithm in COMPARISON_ALGORITHMS])
        axis.set_ylim(-0.48, len(COMPARISON_ALGORITHMS) - 0.52)
        axis.set_xlim(0, 1.05)
        axis.set_xticks([0, 1], ["0", "1"])
        # Panel e: reduce only the coordinate tick labels by one size step.
        axis.tick_params(axis="both", labelsize=FS_SUBPANEL_TICK - 2)
        axis.tick_params(axis="x", labelsize=FS_SUBPANEL_TICK - 2)
        axis.tick_params(axis="x", pad=4)
        if panel_index > 0:
            axis.tick_params(axis="y", labelleft=False)
        tidy(axis, grid="x")
    axes["ieee33"].set_ylabel("")
    axes["ieee69"].set_xlabel("")
    mixed_source = []
    for topology, frame in mixed_by_topology.items():
        if frame.empty:
            continue
        source_frame = frame.copy()
        source_frame["topology"] = topology
        mixed_source.append(source_frame.assign(source="pairwise_mixed"))
    write_source_data("04_topology_effect_raw_vs_pairwise_mixed", pd.concat([
        pd.concat([frame for frame in source_by_topology.values() if not frame.empty], ignore_index=True).assign(source="original"),
        pd.concat(mixed_source, ignore_index=True),
    ], ignore_index=True))
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.25, top=0.90, wspace=0.16)
    save(fig, "04_topology_effect_raw_vs_pairwise_mixed", tight=False)


def make_dataset_retention() -> None:
    topologies = ("ieee33", "ieee69", "ieee123")
    original_by_topology = {topology: load_uniform_r2_data(topology) for topology in topologies}
    mixed_by_topology = {topology: load_uniform_pairwise_data(topology) for topology in topologies}
    for frame in mixed_by_topology.values():
        frame["response_r2"] = pd.to_numeric(frame["response_r2"], errors="coerce")
    data = pd.concat(original_by_topology.values(), ignore_index=True, sort=False)
    datasets, dataset_colors, _ = dataset_color_map(data.assign(dataset_label=data["dataset"].astype(str)))
    figure_height = 94.0
    fig = plt.figure(figsize=(WIDTH_2COL * MM, figure_height * MM), facecolor="white")
    axes = {
        topology: ax_mm(fig, 24.0 + panel_index * 52.0, 8.0, 48.0, 78.0, figure_height)
        for panel_index, topology in enumerate(topologies)
    }
    y_positions = np.arange(len(COMPARISON_ALGORITHMS), dtype=float)
    source_rows = []
    for panel_index, topology in enumerate(topologies):
        axis = axes[topology]
        original = original_by_topology[topology]
        mixed = mixed_by_topology[topology]
        for index, algorithm in enumerate(COMPARISON_ALGORITHMS):
            if algorithm == "no_coordination":
                continue
            draw_metric_cloud_combined(
                axis, original, mixed, float(index), algorithm, "response_r2",
                dataset_colors, 20261100 + panel_index * 100 + index, 0.01,
            )
        upper_bound_values = pd.concat([
            original.loc[original["algorithm"] == "centralized_optimal", "response_r2"],
            mixed.loc[mixed["algorithm"] == "centralized_optimal", "response_r2"],
        ]).dropna()
        if not upper_bound_values.empty:
            axis.axvline(float(upper_bound_values.median()), color=COLORS["centralized_optimal"], linestyle="--", linewidth=1.5)
        original_source = original[["dataset", "algorithm", "response_r2"]].copy()
        original_source["topology"] = topology
        original_source["source"] = "original"
        mixed_source = mixed[["composition_id", "algorithm", "response_r2"]].copy()
        mixed_source["topology"] = topology
        mixed_source["source"] = "pairwise_mixed"
        source_rows.extend([original_source, mixed_source])
        axis.set_yticks(y_positions, [ALGORITHM_CODES[algorithm] for algorithm in COMPARISON_ALGORITHMS])
        axis.set_ylim(-0.62, len(COMPARISON_ALGORITHMS) - 0.18)
        axis.set_xlim(-0.55, 1.05)
        # Panel d: reduce only the coordinate tick labels by one size step.
        axis.tick_params(axis="both", labelsize=FS_DISTRIBUTION_TICK - 2)
        if topology != "ieee33":
            axis.tick_params(axis="y", labelleft=False)
        tidy(axis, grid="x")
        annot(
            axis,
            0.03,
            0.04,
            "0.0%",
            color=RULE,
            fontsize=FS_DISTRIBUTION_LEGEND,
            ha="left",
            va="center",
        )
    write_source_data("05_response_fidelity_r2", pd.concat(source_rows, ignore_index=True, sort=False))
    save(fig, "05_ieee69_dataset_algorithm_retention")


def make_stress_lines() -> None:
    if not E24_AUDIT.is_file():
        raise FileNotFoundError(f"缺少 IEEE-123 直接训练审计结果：{E24_AUDIT}")
    data = pd.read_csv(E24_AUDIT, low_memory=False)
    data["algorithm"] = data["algorithm"].map(normalize_algorithm)
    data = data[data["algorithm"].notna()].copy()
    scenarios = tuple(E24_SCENARIO_LABELS)
    fig, axis = plt.subplots(figsize=(11.475, 5.7))
    row_positions = np.arange(len(ALGORITHMS), dtype=float)
    scenario_offsets = np.linspace(-0.27, 0.27, len(scenarios))
    median_records: dict[str, list[float]] = {algorithm: [] for algorithm in ALGORITHMS}

    def draw_distribution(
        values: np.ndarray,
        y: float,
        colour: str,
        *,
        upper_bound: bool = False,
        eps_row: bool = False,
        seed: int = 0,
    ) -> float | None:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return None
        quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
        violin = axis.violinplot([values], positions=[y], vert=False, widths=0.15, showextrema=False, showmedians=False)
        for body in violin["bodies"]:
            body.set_facecolor(colour)
            body.set_edgecolor(INK if upper_bound else colour)
            body.set_linewidth(1.1 if eps_row or upper_bound else 0.7)
            body.set_alpha(0.16 if upper_bound else 0.30 if eps_row else 0.22)
        axis.plot(quantiles[[0, 4]], [y, y], color=colour, linewidth=1.5 if eps_row else 1.0, zorder=3)
        axis.plot(quantiles[[1, 3]], [y, y], color=colour, linewidth=5.4 if eps_row else 4.2, solid_capstyle="butt", zorder=4)
        rng = np.random.default_rng(seed)
        jitter = rng.uniform(-0.045, 0.045, len(values))
        axis.scatter(values, y + jitter, s=7.0, color=colour, alpha=0.12 if upper_bound else 0.22, edgecolors="none", zorder=2)
        axis.scatter(
            [quantiles[2]], [y], s=42 if upper_bound else 42 if eps_row else 32,
            marker="D" if upper_bound else "o",
            facecolor="white", edgecolor=INK if upper_bound else colour,
            linewidth=1.7 if upper_bound else 1.5 if eps_row else 1.2, zorder=6,
        )
        return float(quantiles[2])

    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        row_medians: list[float] = []
        for scenario_index, scenario in enumerate(scenarios):
            values = data.loc[
                (data["algorithm"] == algorithm) & (data["scenario"] == scenario),
                "mean_reduction_pct",
            ].to_numpy(dtype=float)
            median = draw_distribution(
                values,
                row_positions[algorithm_index] + scenario_offsets[scenario_index],
                SCENARIO_COLORS[scenario] if algorithm != "centralized_optimal" else COLORS["centralized_optimal"],
                upper_bound=algorithm == "centralized_optimal",
                eps_row=algorithm == "eps_ieee69_fused",
                seed=20260901 + algorithm_index * 100 + scenario_index,
            )
            row_medians.append(np.nan if median is None else median)
        median_records[algorithm] = row_medians
        if algorithm != "centralized_optimal":
            medians = np.asarray(row_medians, dtype=float)
            valid = np.isfinite(medians)
            if valid.sum() > 1:
                axis.plot(
                    scenario_offsets[valid] + row_positions[algorithm_index],
                    medians[valid],
                    color=FAINT,
                    linewidth=0.8,
                    zorder=1,
                )

    axis.set_yticks(row_positions, [ALGORITHM_CODES[algorithm] for algorithm in ALGORITHMS])
    axis.set_ylabel("")
    axis.set_xlabel("")
    axis.set_xlim(0, 100)
    axis.set_ylim(-0.58, len(ALGORITHMS) - 0.42)
    axis.set_xticks([0, 20, 40, 60, 80, 100])
    axis.tick_params(axis="both", labelsize=24)
    axis.tick_params(axis="x", pad=7)
    axis.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.axhline(row_positions[-1], color=INK, linestyle=(0, (3.0, 2.0)), linewidth=0.9, alpha=0.7, zorder=0)
    # Scenario legend is editable text above the image in the presentation.
    write_source_data("06_ieee69_all_algorithms_m0_m6", data)
    fig.subplots_adjust(left=0.08, right=0.975, bottom=0.34, top=0.97)
    save(fig, "06_ieee69_all_algorithms_m0_m6", tight=False)


def make_physical_boundary() -> None:
    data = pd.read_csv(E23, low_memory=False)
    data = data[data["axis"] == "request_intensity"].copy()
    data["m_line"] = 1.0 - data["maximum_branch_loading"]
    data["m_transformer"] = 1.0 - data["maximum_transformer_loading"]
    data["m_voltage"] = np.minimum((data["minimum_voltage_pu"] - 0.95) / 0.05, (1.05 - data["maximum_voltage_pu"]) / 0.05)
    data["m_phys"] = data[["m_line", "m_transformer", "m_voltage"]].min(axis=1)
    data["algorithm"] = pd.Categorical(data["algorithm"], categories=ALGORITHMS, ordered=True)
    grouped = data.groupby(["algorithm", "pressure_value"], observed=False)[["m_line", "m_transformer", "m_voltage", "m_phys", "network_acceptance_ratio", "mean_reduction_pct"]].mean().reset_index()
    pressures = sorted(grouped.pressure_value.unique())
    fig, (axis, boundary_axis) = plt.subplots(2, 1, figsize=(14.5, 8.5), sharex=True, gridspec_kw={"height_ratios": [4.8, 1.8], "hspace": 0.08})
    overall = grouped.groupby("pressure_value", observed=False)[["m_line", "m_transformer", "m_voltage", "m_phys"]].median().reindex(pressures)
    axis.plot(pressures, overall["m_line"], color="#D55E00", linewidth=2, label="Line margin")
    axis.plot(pressures, overall["m_transformer"], color="#0072B2", linewidth=2, label="Transformer margin")
    axis.plot(pressures, overall["m_voltage"], color="#009E73", linewidth=2, label="Voltage margin")
    axis.plot(pressures, overall["m_phys"], color="#222222", linewidth=2.8, label="Minimum physical margin")
    axis.axhline(0, color="#222222", linestyle="--", linewidth=1, label="Feasibility boundary")
    physical = overall["m_phys"].to_numpy()
    physical_quantiles = data.groupby("pressure_value", observed=False)["m_phys"].quantile([0.10, 0.90]).unstack().reindex(pressures)
    axis.fill_between(pressures, physical_quantiles[0.10].to_numpy(), physical_quantiles[0.90].to_numpy(), color="#555555", alpha=0.12, linewidth=0, label="Physical-margin 10–90% range")
    axis.fill_between(pressures, 0, physical, where=physical >= 0, color="#B7E4C7", alpha=0.35, label="Feasible physical region")
    axis.fill_between(pressures, 0, physical, where=physical < 0, color="#F4A6A6", alpha=0.35, label="Infeasible physical region")
    safe = []
    for algorithm in ALGORITHMS:
        sub = grouped[grouped.algorithm == algorithm].sort_values("pressure_value")
        feasible = sub.loc[sub.m_phys >= 0, "pressure_value"]
        safe.append(float(feasible.max()) if len(feasible) else np.nan)
    acceptance = grouped.groupby("pressure_value", observed=False)["network_acceptance_ratio"].median().reindex(pressures) * 100.0
    acceptance_quantiles = data.groupby("pressure_value", observed=False)["network_acceptance_ratio"].quantile([0.10, 0.90]).unstack().reindex(pressures) * 100.0
    right_axis = axis.twinx()
    right_axis.plot(pressures, acceptance.to_numpy(), color="#6A3D9A", linewidth=2.0, linestyle="-.", label="Median network acceptance")
    right_axis.fill_between(pressures, acceptance_quantiles[0.10].to_numpy(), acceptance_quantiles[0.90].to_numpy(), color="#6A3D9A", alpha=0.035, linewidth=0, label="Acceptance 10–90% range")
    right_axis.set_ylim(0, 105)
    right_axis.tick_params(axis="y", colors="#6A3D9A")
    right_axis.tick_params(axis="y", labelsize=FS_SUBPANEL_TICK)
    boundary_positions = np.arange(len(ALGORITHMS))
    for position, (algorithm, boundary) in enumerate(zip(ALGORITHMS, safe)):
        if np.isfinite(boundary):
            boundary_axis.scatter(boundary, position, c=COLORS[algorithm], marker=MARKERS[algorithm], s=58, edgecolor="white", linewidth=0.6, zorder=3)
            boundary_axis.annotate(f"{boundary:.2f}", (boundary, position), xytext=(5, 0), textcoords="offset points", va="center", fontsize=FS_SUBPANEL_ANNOT)
    boundary_axis.set_yticks(boundary_positions, [ALGORITHM_CODES[algorithm] for algorithm in ALGORITHMS], fontsize=FS_SUBPANEL_TICK)
    boundary_axis.set_xlim(min(pressures) - 0.05, max(pressures) + 0.05)
    boundary_axis.set_ylim(-0.5, len(ALGORITHMS) - 0.5)
    boundary_axis.grid(axis="x", alpha=0.2)
    axis.grid(alpha=0.2)
    handles, labels = axis.get_legend_handles_labels()
    right_handles, right_labels = right_axis.get_legend_handles_labels()
    axis.legend(handles + right_handles, labels + right_labels, loc="upper right", fontsize=FS_SUBPANEL_LEGEND, ncol=2, frameon=True)
    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.10, top=0.94, hspace=0.10)
    save(fig, "07_physical_to_computational_boundary")


def write_readme() -> None:
    text = """# 配电网约束子图

图中使用目标拓扑直接训练的 EPS。Centralized greedy upper bound 只表示设备层理论上界，不参与算法排名。

|文件|内容|
|-|-|
|00|IEEE-69 七种运行条件下的网络接受率|
|01|Line、Transformer、Minimum voltage、Maximum voltage 四类约束分别单独启用时，弃电率降低、响应保真度和网络接受率随实际约束数值的变化|
|02|IEEE-69 区域完全混合、类型分区、设备密度集中、类型与密度联合不均衡、不利网络耦合和有利网络耦合下的弃电率降低|
|03|IEEE-33、IEEE-69、IEEE-123 在 50% 馈线集中和 80% 末端节点集中时，相对均匀部署的效果保持率|
|04|三种拓扑均匀部署下，17 个原始数据集和 105 个两两混合数据集的弃电率降低分布|
|05|三种拓扑均匀部署下，17 个原始数据集和 105 个两两混合数据集的响应 R² 分布|
|06|IEEE-123 四种完整部署条件下的算法弃电率降低曲线|
|07|请求强度变化对应的物理裕度、网络接受率和可行上限|

01 中 Line 和 Transformer 的刻度直接显示剩余容量百分比；Minimum voltage 和 Maximum voltage 直接显示 p.u. 数值。弃电率降低的 80% 线和响应 R² 的 0.95 线是预先声明的工程有效性参考，不是数学定理。网络接受率不设参考线。

04 和 05 中彩色点表示原始数据集，灰色点表示两两混合数据集。钢琴分布将二者合并，数据点仍保留来源区分。

生成脚本：`tools/generate_e22_subpanel_figures.py`

组合脚本：`tools/build_e22_subpanel_presentation.py`
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


PLOTS = {
    "00_ieee69_network_acceptance": make_acceptance,
    "01_ieee69_constraint_audit": make_constraint_audit,
    "02_algorithm_effect_raw_vs_pairwise_mixed": make_effect_safety,
    "03_ieee69_spatial_retention": make_spatial_retention,
    "04_topology_effect_raw_vs_pairwise_mixed": make_topology_boxpoint,
    "05_ieee69_dataset_algorithm_retention": make_dataset_retention,
    "06_ieee69_all_algorithms_m0_m6": make_stress_lines,
    "07_physical_to_computational_boundary": make_physical_boundary,
}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", choices=sorted(PLOTS))
    args = parser.parse_args()
    if OUT.exists():
        for path in OUT.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.plot:
        PLOTS[args.plot]()
    else:
        for plot in PLOTS.values():
            plot()
        write_readme()
    print(f"已生成：{OUT}")


if __name__ == "__main__":
    main()
