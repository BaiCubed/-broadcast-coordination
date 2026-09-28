#!/usr/bin/env python3
"""Build a numbered, reviewer-oriented appendix from E20-E24 figures."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, minimum_filter
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/results-E20-E24/source_figures"
APP = ROOT / "outputs/appendix"
FIG = APP / "figures"

BASE_ENTRIES = [
    ("E20 transfer heatmap", "E20/figures/e20_transfer_heatmap.png", "results/E20/fig3a_transfer/fig3a_transfer_summary.csv", "src/extra/ieee33_device_day_simulation/figures/run_e20_fig3a_transfer.py", "比较五个目标数据集在不同迁移方法下的效果，检查迁移性能是否随目标域而变化。", "横坐标为 in-domain、zero-shot 和 target-calibrated；纵坐标为目标数据集；颜色和格内数字表示弃电降低率(%)。", "先看同一数据集的横向变化，再看不同目标域之间的颜色差异。颜色越深表示可交付效果越高；Irish/OPSD 等偏移较大的域应与其他域分开解释。"),
    ("E20 multi-split transfer heatmap", "E20/rotations/figures/e20_multisplit_transfer_heatmap.png", "results/E20/rotations/data/e20_multisplit_transfer_summary.csv", "src/extra/ieee33_device_day_simulation/figures/run_e20_rotations.py", "检验迁移结论是否依赖某一个数据集划分或旋转折叠。", "横坐标为迁移方法或折叠；纵坐标为目标数据集；颜色表示 R² 或相对性能。", "若相邻折叠保持相近颜色，说明迁移结论具有划分稳健性；明显变色提示结果对训练/测试切分敏感。"),
    ("E21 Gamma-R² and physical scale", "E21/gamma_universal_boundary/figures/e21_gamma_scatter.png", "results/E21/gamma_universal_boundary/data/gamma_points.csv; results/E21/gamma_universal_boundary/data/dataset_calibration.csv", "src/extra/ieee33_device_day_simulation/figures/run_e21_gamma_universal_boundary.py", "同时观察归一化 Gamma 和真实物理设备规模，避免把尺度坍缩误读为设备数本身的规律。", "左横轴为 Gamma，右横轴为物理设备数 N，纵轴为独立测试 R²；点大小编码 N，虚线为 R²=0.95。", "左图看 Gamma 是否与 R² 单调相关，右图核对这种关系是否只是 N 增长造成的。OPSD 的 profile reuse 不能当作独立设备扩展。"),
    ("E21 critical Gamma distribution", "E21/gamma_universal_boundary/figures/e21_gamma_critical_distribution.png", "results/E21/gamma_universal_boundary/data/gamma_critical.csv; results/E21/gamma_universal_boundary/data/gamma_points.csv", "src/extra/ieee33_device_day_simulation/figures/run_e21_gamma_universal_boundary.py", "给出每个数据集达到 R²=0.95 时的实测临界 Gamma，检查统一边界是否掩盖数据集差异。", "横坐标为按 Gamma_c 排序的数据集，纵坐标为临界 Gamma_c；点和误差线表示估计值及 bootstrap 区间。", "看 Gamma_c 是否集中在 1 附近，以及哪些数据集明显偏离。右删失数据集表示扫描范围内未达到阈值，不是零值。"),
    ("E21 mixed composition heatmap", "E21/gamma_mixed_boundary/figures/mixed_composition_heatmap.png", "results/E21/gamma_mixed_boundary/data/fleet_audit.json", "src/extra/ieee33_device_day_simulation/figures/run_e21_mixed_gamma_boundary.py", "审计混合 Gamma 实验是否真的使用了预定义的物理混合 fleet，而不是事后平均单数据集结果。", "横坐标为源数据集，纵坐标为 S1-A 至 S6-C 混合场景，颜色表示名义设备比例。", "先检查每个场景是否包含声明的数据集，再比较不同场景的组成结构；空白代表该数据集不在该场景中。"),
    ("E21 mixed Gamma scaling", "E21/gamma_mixed_boundary/figures/mixed_gamma_scaling.png", "results/E21/gamma_mixed_boundary/data/mixed_gamma_points.csv; results/E21/gamma_mixed_boundary/data/mixed_gamma_all_arms.csv", "src/extra/ieee33_device_day_simulation/figures/run_e21_mixed_gamma_boundary.py", "比较混合场景在物理 N 与 Gamma 两种坐标下的 R² 扩展趋势，并同时保留 data-coupled 与 decoupled 对照。", "横轴分别为 N 和 Gamma，纵轴为独立测试 R²；颜色区分场景，线型/点形区分两条实验臂。", "看两种横轴对跨场景预测误差的解释力，不应仅凭曲线更平滑就宣称 Gamma 优于 N。"),
    ("E21 curtailment opportunity versus reduction", "E21/pairwise_curtailment/figures/curtailment_opportunity_vs_reduction.png", "results/E21/pairwise_curtailment/data/pairwise_curtailment_by_seed.csv", "src/extra/ieee33_device_day_simulation/figures/run_e21_pairwise_curtailment.py", "区分可利用的弃电机会量与算法实际消纳比例，检查高降低率是否只是由高机会量造成。", "横轴为基线弃电机会量，纵轴为弃电降低率；颜色或点形区分数据集组合。", "看是否存在明显的机会量-效果耦合，以及离群组合是否集中在特定数据集。"),
    ("E21 pairwise R² and N95", "E21/pairwise_r2/figures/pairwise_r2_n95_overall.png", "results/E21/pairwise_r2/data/pairwise_r2_points.csv; results/E21/pairwise_r2/data/pairwise_r2_n95.csv", "src/extra/ieee33_device_day_simulation/figures/run_e21_pairwise_r2.py", "汇总 105 个两两组合的响应拟合质量和达到 R²=0.95 所需规模。", "横轴为设备规模或组合，纵轴为 R²/N95；参考线为 R²=0.95。", "先看组合间离散程度，再看 N95 是否集中；组合差异大时应报告分布而不是单一平均值。"),
    ("E22 branch-time heatmap", "E22/figures/ieee69_branch_time_heatmap.png", "results/E22/data/ieee69_m2_eps_trace.json; results/E22/data/e22_summary.csv", "src/extra/ieee33_device_day_simulation/figures/run_e22_ieee69_complexity.py", "展示 EPS 在 IEEE-69 不同支路和时间步上的负载压力，定位网络瓶颈而不是只报告一个总均值。", "横轴为时间步，纵轴为支路；颜色表示支路负载率，1.0 对应热限。", "沿时间方向寻找连续热点，再沿支路方向判断热点是否局部集中；超过 1.0 的区域表示仍存在过载风险。"),
    ("E22 stress-profile comparison", "E22/figures/ieee69_stress_profiles_comparison.png", "results/E22/data/e22_by_seed.csv; results/E22/data/e22_summary.csv", "src/extra/ieee33_device_day_simulation/figures/run_e22_ieee69_complexity.py", "比较 M0-M6 网络压力和空间位置条件下的算法性能变化。", "横轴为 M0-M6 场景，纵轴为可交付弃电降低率或响应 R²；不同颜色代表算法。", "M0-M3 是网络压力变化，M4-M6 是空间位置变化。若同一算法在 M5/M6 明显下降，说明设备数量不能替代空间约束审计。"),
    ("E22 trained-versus-transfer dataset heatmap", "E22/trained_eps_ieee69_direct/figures/trained_vs_transfer_dataset_heatmap_ieee69.png", "results/E22/trained_eps_ieee69_direct/data/combined_e22_overall.csv; results/E22/trained_eps_ieee69_direct/data/combined_e22_summary.csv", "src/extra/ieee33_device_day_simulation/figures/run_e22_trained_eps_comparison.py", "比较目标域直接训练与迁移模型在不同数据集上的性能差异。", "横坐标为训练方式/算法，纵坐标为数据集，颜色和数字表示 R² 或效果保持率。", "看直接训练是否只改善少数数据集，还是在整个数据集族上稳定改善；不要把颜色差异解释为网络安全差异。"),
    ("E23 physical violation curves", "E23/figures/e23_physical_violation_curves.png", "results/E23/data/e23_by_seed.csv; results/E23/data/e23_boundaries.csv", "src/extra/ieee33_device_day_simulation/figures/run_e23_ieee69_relative_boundary.py", "检查安全裁剪前的原始广播请求是否会产生线路、电压或主变风险。", "横轴分别为 q、线路容量压力和空间集中比例，纵轴为新增物理违规请求比例；5% 为审计参考线。", "超过 5% 且在相邻压力点持续出现时，表示算法需要安全层频繁修正，不能称为天然安全。"),
    ("E23 spatial-condition R²", "E23/figures/e23_r2_spatial_conditions.png", "results/E23/data/e23_by_seed.csv; results/E23/data/spatial_interaction_effects.csv", "src/extra/ieee33_device_day_simulation/figures/run_e23_spatial_heterogeneity.py", "观察不同空间部署条件下响应跟踪 R² 是否发生系统性下降。", "横轴为空间条件/压力水平，纵轴为响应 R²；颜色区分算法或数据集。", "与弃电降低率图配合阅读：R² 下降说明预测响应失真，效果下降说明可交付执行受限，二者原因不同。"),
    ("E24 IEEE-123 R² scenario comparison", "E24/figures/e24_r2_scenario_comparison.png", "results/E24/data/e24_ieee123_by_seed.csv; results/E24/data/e24_ieee123_summary.csv", "src/extra/ieee33_device_day_simulation/figures/run_e24_ieee123_audit.py", "比较 IEEE-123 四种部署场景下的响应跟踪质量，确认空间集中和容量降额对预测层的影响。", "横轴为算法，纵轴为响应 R²，四个面板分别为 S0-S3。", "比较同一算法在四个面板的高度变化；R² 下降只说明响应跟踪受损，不能直接等同于网络违规。"),
    ("E24 IEEE-123 safety audit", "E24/figures/e24_ieee123_safety_audit.png", "results/E24/data/e24_ieee123_by_seed.csv; results/E24/data/e24_ieee123_paired.csv", "src/extra/ieee33_device_day_simulation/figures/run_e24_ieee123_audit.py", "同时报告安全裁剪前的新增违规请求比例和裁剪后的网络接受率。", "横轴为算法，纵轴为比例(%)，四个面板为 S0-S3；红色为新增风险，蓝色为网络接受率，5% 为审计参考线。", "红柱回答请求是否天然安全，蓝柱回答安全层之后实际执行了多少；两者必须分开解释。"),
]

ENTRIES = [entry for entry in BASE_ENTRIES if not entry[0].startswith("E24 IEEE-123")]
ENTRIES = [
    entry[:5]
    + ("横轴分别为 N 和 Gamma，纵轴为独立测试 R²，显示范围从 0.8 开始；颜色区分场景，线型/点形区分两条实验臂。", "在相同 0.8 起点下比较两种尺度的跨场景离散性，不把更平滑的视觉效果直接解释为更强证据。")
    if entry[0] == "E21 mixed Gamma scaling" else entry
    for entry in ENTRIES
]

ADDED_ENTRIES = [
    ("E20 R² versus device scale", "outputs/01-e20-迁移N95与效果/e20_r2_vs_N_boxplot.png", "results/E20/fig3a_transfer/fig3a_transfer_summary.csv", "tools/generate_boxplot_outputs.py", "比较迁移方法在不同物理设备规模下的 R² 分布，补充单条均值曲线对离散性展示不足的问题。", "横坐标为物理设备数量 N，纵坐标为 Independent test R²；半小提琴表示分布，点表示观测值，箱线显示四分位区间和中位数。", "先看同一 N 下不同方法的分布宽度，再看是否达到 R²=0.95 附近的性能区间。"),
    ("E21 original and mixed algorithm effect", "outputs/02-e21-原始与混合算法效果/e21_raw_vs_mixed_algorithm_boxplot.png", "results/E21/curtailment_baseline_supplement/data/baseline_by_seed.csv", "tools/generate_boxplot_outputs.py", "比较原始数据集与混合组合在各算法下的弃电降低率分布。", "横坐标为算法，纵坐标为弃电率降低率(%)；半小提琴和箱线表示原始分布，点集补充混合组合结果。", "重点看混合组合是否改变算法排序，以及分布尾部是否暴露稳定性问题。"),
    ("E21 pairwise R² by device scale", "outputs/04-e21-两两组合R2规模/e21_pairwise_r2_vs_N_boxplot.png", "results/E21/pairwise_r2/data/pairwise_r2_points.csv", "tools/generate_boxplot_outputs.py", "展示 105 个两两组合在不同设备规模下的 R² 分布。", "横坐标为设备数量 N，纵坐标为 Independent test R²；每个 N 使用半小提琴、点集和箱线。", "先看组合间离散程度，再判断规模增加是否稳定地把整体分布推向 R²=0.95。"),
    ("E21 Gamma-R² distribution", "outputs/05-e21-Gamma边界/e21_gamma_r2_boxplot.png", "results/E21/gamma_universal_boundary/data/gamma_points.csv; results/E21/gamma_mixed_boundary/data/mixed_gamma_points.csv", "tools/generate_boxplot_outputs.py", "比较原始数据集和 mixed 场景在 Gamma 分箱下的 R² 分布。", "横坐标为 log10(Gamma) 分箱，纵坐标为响应跟踪 R²；两面板分别表示原始数据集和 mixed 场景。", "看同一 Gamma 分箱内的分布宽度与中位数，避免只根据一条拟合线作结论。"),
    ("E22 IEEE-69 algorithm and R² distributions", "outputs/06-e22-IEEE69算法与R2/e22_ieee69_algorithm_boxplots.png", "results/E22/data/e22_by_seed.csv", "tools/generate_boxplot_outputs.py", "同时比较 IEEE-69 各算法的弃电降低率与响应跟踪 R² 分布。", "左面板纵轴为弃电率降低率(%)，右面板纵轴为响应跟踪 R²；横轴为算法。", "算法效果和响应拟合需要联合阅读，不能用单一效果指标代替响应质量或网络约束审计。"),
    ("E23 line-derating pressure distribution", "outputs/07-e23-IEEE69压力曲线/e23_line_derating_boxplot.png", "results/E23/data/e23_by_seed.csv", "tools/generate_boxplot_outputs.py", "展示线路容量降额压力下各算法的弃电降低率分布。", "横轴为线路降额压力参数，纵轴为弃电率降低率(%)；不同小面板代表算法。", "观察容量降额增加时中位数、四分位区间和点集是否同步下降。"),
    ("E23 request-intensity pressure distribution", "outputs/07-e23-IEEE69压力曲线/e23_request_intensity_boxplot.png", "results/E23/data/e23_by_seed.csv", "tools/generate_boxplot_outputs.py", "展示请求强度增加时各算法的可交付效果分布。", "横轴为 request intensity，纵轴为弃电率降低率(%)；不同小面板代表算法。", "高请求强度下的分布尾部反映安全裁剪和设备能力限制，不应只看平均值。"),
    ("E23 spatial-concentration pressure distribution", "outputs/07-e23-IEEE69压力曲线/e23_spatial_concentration_boxplot.png", "results/E23/data/e23_by_seed.csv", "tools/generate_boxplot_outputs.py", "展示设备空间集中比例增加时各算法的可交付效果分布。", "横轴为空间集中参数，纵轴为弃电率降低率(%)；不同小面板代表算法。", "如果集中比例增加导致整体分布下移，说明局部网络瓶颈不能由增加设备数量消除。"),
    ("E24 IEEE-123 delivered curtailment reduction", "generated/e24_delivered_reduction.png", "results/E24/data/e24_ieee123_summary.csv", "tools/build_appendix.py", "Compare delivered curtailment reduction across the four IEEE-123 deployment scenarios.", "The x-axis lists algorithms and the y-axis reports delivered curtailment reduction (%).", "Compare each algorithm across scenarios; results are after the network safety layer has constrained the request."),
    ("E24 IEEE-123 network safety audit", "generated/e24_network_safety.png", "results/E24/data/e24_ieee123_summary.csv", "tools/build_appendix.py", "Separate the risk in requested control from the share of control accepted by the IEEE-123 network.", "The x-axis lists algorithms; red bars show requested added-violation risk and blue bars show network acceptance (%).", "A high acceptance rate and a low requested-risk rate are distinct properties and should be evaluated together."),
    ("E24 IEEE-123 direct-training validation R2", "generated/e24_direct_training_r2.png", "results/E24/trained_eps_ieee123_direct/data/training_metadata.json", "tools/build_appendix.py", "Audit held-out validation quality of the IEEE-123 target-domain EPS models.", "The x-axis lists source datasets and the y-axis shows held-out R2 for each direct-training model.", "This is a training-quality audit; it does not replace the delivered-effect and network-safety results."),
    ("E22 IEEE-69 constraint-component audit heatmap", "outputs/figs/e22_ieee69_appendix/ieee69_constraint_component_audit_appendix.png", "results/E22/data/e22_summary.csv", "tools/build_e22_constraint_component_heatmap.py", "Audit four executed network-constraint components across the seven IEEE-69 stress modes.", "The four panels show maximum branch loading, minimum voltage, maximum transformer loading, and safety-layer gain in violation-free steps; rows are algorithms and columns are M0-M6. A dagger marks displayed ties at the current stress-mode maximum; the shared value is caused by that mode's maximum.", "Read each panel with its colorbar and annotations. The matrix averages the 17 dataset summaries, each computed from 30 paired seeds. Dagger-marked cells are tied displayed maxima within their stress-mode column, so their repeated number reflects the active mode maximum rather than duplicated raw records.")
]
ENTRIES.extend(ADDED_ENTRIES)

HEATMAP_TITLES = {"E20 transfer heatmap", "E20 multi-split transfer heatmap", "E21 mixed composition heatmap", "E22 branch-time heatmap", "E22 trained-versus-transfer dataset heatmap"}
REF_CMAP = LinearSegmentedColormap.from_list("appendix_reference", ["#E5633E", "#E49A61", "#E8ECF0", "#9CC6CE", "#5B83AD", "#365A7C"])

ALGORITHM_ORDER = (
    "no_coordination",
    "local_rules",
    "mpc_optimal",
    "mean_field_control",
    "virtual_battery",
    "packetized_energy_management",
    "transactive_control",
    "eps_ieee69_fused",
    "centralized_optimal",
)
ALGORITHM_LABELS = {
    "no_coordination": "No coordination",
    "local_rules": "Local SOC rules",
    "mpc_optimal": "MPC",
    "mean_field_control": "Mean-field control",
    "virtual_battery": "Virtual battery",
    "packetized_energy_management": "Packetized EM",
    "transactive_control": "Transactive control",
    "eps_ieee69_fused": "EPS IEEE-123 direct",
    "centralized_optimal": "Centralized greedy UB",
}
ALGORITHM_COLORS = {
    "no_coordination": "#9C9C9C",
    "local_rules": "#E5633E",
    "mpc_optimal": "#7057FF",
    "mean_field_control": "#4E79A7",
    "virtual_battery": "#59A14F",
    "packetized_energy_management": "#AF7AA1",
    "transactive_control": "#D55E00",
    "eps_ieee69_fused": "#365A7C",
    "centralized_optimal": "#3A9D66",
}
SCENARIO_LABELS = {
    "S0_uniform": "S0\nUniform",
    "S1_feeder_50": "S1\n50% distal feeder",
    "S2_node_80": "S2\n80% distal node",
    "S3_feeder_50_derated": "S3\n50% feeder, 70% capacity",
}
GENERATED_TITLES = {
    "E23 physical violation curves",
    "E24 IEEE-123 delivered curtailment reduction",
    "E24 IEEE-123 network safety audit",
    "E24 IEEE-123 R² scenario comparison",
    "E24 IEEE-123 response tracking by scenario",
    "E24 IEEE-123 direct-training validation R2",
    "E23 spatial-condition R²",
}


def _configure_english_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save_generated(figure: plt.Figure, target: Path) -> None:
    figure.tight_layout()
    figure.savefig(target, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _bootstrap_interval(values: np.ndarray, label: str) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(sum(label.encode("utf-8")))
    samples = rng.integers(0, len(values), size=(2000, len(values)))
    means = values[samples].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _algorithm_tick_labels() -> list[str]:
    return [ALGORITHM_LABELS[algorithm] for algorithm in ALGORITHM_ORDER]


def _plot_e23_physical_risk(target: Path) -> None:
    data = pd.read_csv(ROOT / "results/E23/data/e23_by_seed.csv")
    axes_info = (
        ("request_intensity", "Request intensity q", "q"),
        ("line_derating", "Line derating", "Fractional derating"),
        ("spatial_concentration", "Spatial concentration", "Distal-node share"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharey=True)
    for axis, (axis_name, title, xlabel) in zip(axes, axes_info):
        subset = data[data["axis"] == axis_name]
        for algorithm in ALGORITHM_ORDER:
            rows = subset[subset["algorithm"] == algorithm]
            if rows.empty:
                continue
            grouped = rows.groupby("pressure_value", sort=True)["requested_added_violation_steps"].mean()
            values = 100.0 * grouped.to_numpy(dtype=float) / 288.0
            axis.plot(grouped.index, values, marker="o", markersize=3, linewidth=1.25,
                      color=ALGORITHM_COLORS[algorithm], label=ALGORITHM_LABELS[algorithm])
        axis.axhline(5.0, color="#D1495B", linestyle="--", linewidth=1.0, label="5% reference")
        axis.set_title(title, fontsize=11)
        axis.set_xlabel(xlabel)
        axis.grid(axis="y", color="#E8ECF0", linewidth=0.7)
    axes[0].set_ylabel("Requested added-violation risk (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        fontsize=7,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.24, wspace=0.08)
    figure.savefig(target, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _plot_e23_spatial_r2(target: Path) -> None:
    data = pd.read_csv(ROOT / "results/E23/data/spatial_heterogeneity_by_seed.csv")
    algorithm_order = (
        "no_coordination",
        "local_rules",
        "mpc_optimal",
        "mean_field_control",
        "virtual_battery",
        "packetized_energy_management",
        "transactive_control",
        "eps_global_mixed",
        "centralized_optimal",
    )
    labels = {
        "no_coordination": "No coordination",
        "local_rules": "Local SOC rules",
        "mpc_optimal": "MPC",
        "mean_field_control": "Mean-field control",
        "virtual_battery": "Virtual battery",
        "packetized_energy_management": "Packetized Energy Management",
        "transactive_control": "Transactive control",
        "eps_global_mixed": "EPS global mixed",
        "centralized_optimal": "Centralized greedy UB",
    }
    colours = {
        "no_coordination": "#9C9C9C",
        "local_rules": "#E5633E",
        "mpc_optimal": "#7057FF",
        "mean_field_control": "#4E79A7",
        "virtual_battery": "#59A14F",
        "packetized_energy_management": "#AF7AA1",
        "transactive_control": "#D55E00",
        "eps_global_mixed": "#365A7C",
        "centralized_optimal": "#3A9D66",
    }
    conditions = ("H0", "H1", "H2_25", "H2_50", "H2_75", "H3", "H4", "H5")
    condition_labels = ("H0", "H1", "H2-25", "H2-50", "H2-75", "H3", "H4", "H5")
    figure, axis = plt.subplots(figsize=(13.5, 5.8))
    x = np.arange(len(conditions))
    for algorithm in algorithm_order:
        means, lows, highs = [], [], []
        for condition in conditions:
            values = data.loc[
                (data["algorithm"] == algorithm) & (data["condition"] == condition),
                "response_r2",
            ].to_numpy(dtype=float)
            mean, low, high = _bootstrap_interval(values, f"spatial|{condition}|{algorithm}")
            means.append(mean)
            lows.append(low)
            highs.append(high)
        axis.plot(x, means, marker="o", linewidth=1.5, markersize=4,
                  color=colours[algorithm], label=labels[algorithm])
        axis.fill_between(x, lows, highs, color=colours[algorithm], alpha=0.08)
    axis.axhline(0.95, color="#D1495B", linestyle="--", linewidth=1.0, label="R2 = 0.95")
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_xticks(x, condition_labels)
    axis.set_xlabel("IEEE-69 spatial condition")
    axis.set_ylabel("Response-tracking R2")
    axis.grid(axis="y", color="#E8ECF0", linewidth=0.7)
    axis.legend(fontsize=7.5, ncol=3, loc="lower right", frameon=True)
    _save_generated(figure, target)


def _plot_e24_effect(target: Path) -> None:
    data = pd.read_csv(ROOT / "results/E24/data/e24_ieee123_summary.csv")
    scenarios = list(SCENARIO_LABELS)
    figure, axes = plt.subplots(1, len(scenarios), figsize=(18, 5.6), sharey=True)
    for axis, scenario in zip(axes, scenarios):
        selected = data[data["scenario"] == scenario].set_index("algorithm")
        values = [float(selected.loc[algorithm, "effect_mean_pct"]) for algorithm in ALGORITHM_ORDER]
        bars = axis.bar(np.arange(len(ALGORITHM_ORDER)), values,
                        color=[ALGORITHM_COLORS[algorithm] for algorithm in ALGORITHM_ORDER])
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=6.5, rotation=90)
        axis.set_title(SCENARIO_LABELS[scenario].replace("\n", ": "), fontsize=10)
        axis.set_xticks(np.arange(len(ALGORITHM_ORDER)), _algorithm_tick_labels(), rotation=55, ha="right", fontsize=6.5)
        axis.grid(axis="y", color="#E8ECF0", linewidth=0.7)
        axis.set_ylim(0, 110)
    axes[0].set_ylabel("Delivered curtailment reduction (%)")
    _save_generated(figure, target)


def _plot_e24_safety(target: Path) -> None:
    data = pd.read_csv(ROOT / "results/E24/data/e24_ieee123_summary.csv")
    scenarios = list(SCENARIO_LABELS)
    figure, axes = plt.subplots(1, len(scenarios), figsize=(18, 5.6), sharey=True)
    for axis, scenario in zip(axes, scenarios):
        selected = data[data["scenario"] == scenario].set_index("algorithm")
        risk = [float(selected.loc[algorithm, "requested_added_violation_mean_pct"]) for algorithm in ALGORITHM_ORDER]
        acceptance = [np.nan if str(selected.loc[algorithm, "network_acceptance_mean_pct"]) == "NA"
                      else float(selected.loc[algorithm, "network_acceptance_mean_pct"])
                      for algorithm in ALGORITHM_ORDER]
        x = np.arange(len(ALGORITHM_ORDER))
        axis.bar(x - 0.19, risk, 0.38, color="#D1495B", label="Requested risk")
        axis.bar(x + 0.19, acceptance, 0.38, color="#4E79A7", label="Network acceptance")
        axis.axhline(5.0, color="#D1495B", linestyle="--", linewidth=1.2, label="5% reference")
        axis.set_title(SCENARIO_LABELS[scenario].replace("\n", ": "), fontsize=12, fontweight="bold")
        axis.set_xticks(x, _algorithm_tick_labels(), rotation=55, ha="right", fontsize=8, fontweight="bold")
        axis.set_ylim(0, 110)
        axis.grid(axis="y", color="#E8ECF0", linewidth=0.7)
    axes[0].set_ylabel("Share of requested control (%)", fontsize=11, fontweight="bold")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(target, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _plot_e24_response_r2(target: Path) -> None:
    data = pd.read_csv(ROOT / "results/E24/data/e24_ieee123_by_seed.csv")
    scenarios = list(SCENARIO_LABELS)
    figure, axis = plt.subplots(figsize=(12.2, 5.5))
    x = np.arange(len(scenarios))
    for algorithm in ALGORITHM_ORDER:
        means, lows, highs = [], [], []
        for scenario in scenarios:
            rows = data[(data["algorithm"] == algorithm) & (data["scenario"] == scenario)]
            mean, low, high = _bootstrap_interval(rows["response_r2"].to_numpy(), f"{algorithm}|{scenario}")
            means.append(mean); lows.append(low); highs.append(high)
        axis.plot(x, means, marker="o", markersize=4, linewidth=1.5,
                  color=ALGORITHM_COLORS[algorithm], label=ALGORITHM_LABELS[algorithm])
        axis.fill_between(x, lows, highs, color=ALGORITHM_COLORS[algorithm], alpha=0.09)
    axis.axhline(0.95, color="#D1495B", linestyle="--", linewidth=1.0, label="R2 = 0.95")
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_xticks(x, [SCENARIO_LABELS[scenario].replace("\n", ": ") for scenario in scenarios])
    axis.set_ylabel("Response-tracking R2")
    axis.grid(axis="y", color="#E8ECF0", linewidth=0.7)
    axis.legend(fontsize=7, ncol=3, frameon=False, loc="lower left")
    _save_generated(figure, target)


def _plot_e24_training_r2(target: Path) -> None:
    metadata = json.loads((ROOT / "results/E24/trained_eps_ieee123_direct/data/training_metadata.json").read_text(encoding="utf-8"))
    metadata.sort(key=lambda row: row["dataset"])
    labels = [row["dataset"].replace("_", " ") for row in metadata]
    values = [float(row["held_out_r2"]) for row in metadata]
    figure, axis = plt.subplots(figsize=(14, 5.4))
    axis.bar(np.arange(len(labels)), values, color="#365A7C")
    axis.axhline(0.95, color="#D1495B", linestyle="--", linewidth=1.0, label="R2 = 0.95")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    axis.set_ylim(min(-0.1, min(values) - 0.05), 1.05)
    axis.set_ylabel("Held-out validation R2")
    axis.grid(axis="y", color="#E8ECF0", linewidth=0.7)
    axis.legend(frameon=False)
    _save_generated(figure, target)


def _generate_english_figure(title: str, target: Path) -> None:
    _configure_english_style()
    if title == "E23 physical violation curves":
        _plot_e23_physical_risk(target)
    elif title == "E23 spatial-condition R²":
        _plot_e23_spatial_r2(target)
    elif title == "E24 IEEE-123 delivered curtailment reduction":
        _plot_e24_effect(target)
    elif title == "E24 IEEE-123 network safety audit":
        _plot_e24_safety(target)
    elif title == "E24 IEEE-123 R² scenario comparison":
        _plot_e24_response_r2(target)
    elif title == "E24 IEEE-123 response tracking by scenario":
        _plot_e24_response_r2(target)
    elif title == "E24 IEEE-123 direct-training validation R2":
        _plot_e24_training_r2(target)
    else:
        raise ValueError(f"No English renderer registered for {title}")


def _rerender_heatmap(source: Path, target: Path) -> None:
    image = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
    luminance = image.astype(float).mean(axis=2) / 255.0
    local_max = maximum_filter(image, size=(7, 7, 1))
    local_min = minimum_filter(image, size=(7, 7, 1))
    uniform = (local_max.astype(int) - local_min.astype(int)).max(axis=2) <= 14
    data_mask = (np.max(image, axis=2) < 238) & uniform
    # In the reference palette, low values are orange/red and high values are
    # dark blue. The source heatmaps use the inverse brightness ordering.
    mapped = (REF_CMAP(1.0 - luminance)[..., :3] * 255).astype(np.uint8)
    output = np.where(data_mask[..., None], mapped, image)
    Image.fromarray(output).save(target)


def _rerender_mixed_gamma(target: Path) -> None:
    coupled = pd.read_csv(ROOT / "results/E21/gamma_mixed_boundary/data/mixed_gamma_points.csv")
    all_arms = pd.read_csv(ROOT / "results/E21/gamma_mixed_boundary/data/mixed_gamma_all_arms.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    colours = {s: REF_CMAP(i / max(1, coupled.scenario.nunique() - 1)) for i, s in enumerate(sorted(coupled.scenario.dropna().unique()))}
    for ax, x in zip(axes, ["N", "Gamma"]):
        for scenario in sorted(coupled.scenario.dropna().unique()):
            for arm, ls in [("data_coupled", "-"), ("decoupled", "--")]:
                d = all_arms[(all_arms.scenario == scenario) & (all_arms.arm == arm)].sort_values(x)
                if len(d): ax.plot(d[x], d.R2, marker="o", ms=2.5, lw=1.0, ls=ls, color=colours[scenario], alpha=.8)
        ax.axhline(.95, color="#4D4D4D", ls=(0, (1, 1.5)), lw=.7); ax.set_xlabel("Physical N" if x == "N" else "Gamma"); ax.set_ylim(.8, 1.01); ax.grid(axis="y", color="#E8ECF0", lw=.4)
    axes[0].set_ylabel("Independent test $R^2$"); fig.tight_layout(); fig.savefig(target, dpi=220, facecolor="white"); plt.close(fig)


def _english_narrative(title: str, data: str, script: str) -> tuple[str, str, str]:
    return (
        f"This supplementary figure documents {title} using the experiment output listed below.",
        "Axis labels, legends, reference lines, and annotations define the reported metric and comparison groups.",
        f"Read this figure together with `{data}` and reproduce it with `{script}` when numerical detail is required.",
    )


def build():
    if APP.exists(): shutil.rmtree(APP)
    FIG.mkdir(parents=True)
    md = ["# Appendix Figures", "", "This appendix selects reviewer-oriented figures from E20-E24. Figures are renumbered in reading order; charts that previously used Chinese labels are regenerated in English from the underlying experiment results.", ""]
    doc = Document()
    doc.styles["Normal"].font.name = "Arial"
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading("Appendix Figures", level=0)
    doc.add_paragraph("This appendix collects figures that support the E20-E24 results. Figures are renumbered in reading order, and all chart labels are in English.")
    for idx, (title, rel, data, script, answer, axes, reading) in enumerate(ENTRIES, 1):
        num = f"{idx:02d}"; png_name = f"Appendix_Fig_{num}.png"; pdf_name = f"Appendix_Fig_{num}.pdf"; src = SOURCE / rel; out = FIG / png_name
        src = ROOT / rel if rel.startswith("outputs/") else SOURCE / rel
        if title in GENERATED_TITLES:
            _generate_english_figure(title, out)
        elif title == "E21 mixed Gamma scaling":
            _rerender_mixed_gamma(out)
        elif title in HEATMAP_TITLES:
            shutil.copy2(src, out)
        else:
            shutil.copy2(src, out)
        Image.open(out).convert("RGB").save(FIG / pdf_name, "PDF", resolution=600.0)
        english_answer, english_axes, english_reading = _english_narrative(title, data, script)
        if title in GENERATED_TITLES and title != "E23 physical violation curves":
            english_answer, english_axes, english_reading = answer, axes, reading
        md += [f"## {idx}. {title}", "", f"![{title}](figures/{png_name})", "", "Files:", "", f"- PNG: `figures/{png_name}`", f"- PDF: `figures/{pdf_name}`", f"- Data: `{data}`", f"- Generator: `{script}`", "", "### Question", "", english_answer, "", "### Axes and Figure Elements", "", english_axes, "", "### Reading Guidance", "", english_reading, ""]
        doc.add_heading(f"{idx}. {title}", level=1)
        paragraph = doc.add_paragraph(); paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER; paragraph.add_run().add_picture(str(out), width=Inches(6.5))
        caption = doc.add_paragraph(f"Figure {idx}. {title}"); caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph(f"Files:\nPNG: figures/{png_name}\nPDF: figures/{pdf_name}\nData: {data}\nGenerator: {script}")
        doc.add_heading("Question", level=2); doc.add_paragraph(english_answer)
        doc.add_heading("Axes and Figure Elements", level=2); doc.add_paragraph(english_axes)
        doc.add_heading("Reading Guidance", level=2); doc.add_paragraph(english_reading)
    (APP / "appendix_figures.md").write_text("\n".join(md), encoding="utf-8")
    doc.save(APP / "appendix_figures.docx")
    (APP / "README.md").write_text(f"# Appendix\n\nReviewer-oriented appendix with {len(ENTRIES)} renumbered figures. Chinese-labelled charts are regenerated in English from their source experiment data.\n\n- Markdown: `appendix_figures.md`\n- Word: `appendix_figures.docx`\n- Figures: `figures/`\n", encoding="utf-8")
    print(f"appendix_figures={len(ENTRIES)} destination={APP}")


if __name__ == "__main__": build()
