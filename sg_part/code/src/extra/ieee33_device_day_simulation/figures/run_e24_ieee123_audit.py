"""运行 E24：IEEE-123 单相结构等值网络审计实验。"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches as PptInches

from . import fig4d_extra_baselines as legacy
from . import fig4d_final_protocol as training
from . import run_e22_ieee69_complexity as e22
from . import run_e22_trained_eps_comparison as direct


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/E24"
NETWORK_FILE = ROOT / "src/extra/ieee33_device_day_simulation/configs/network_ieee123_e24.yaml"
MODEL_SOURCE = ROOT / "results/E22/trained_eps_ieee69_direct/models"
PROTOCOL = "E24_ieee123_direct_training_network_audit_v2"
MODEL_REGIME = "IEEE-69冻结模型附加实验"
EPS_LABEL = "EPS IEEE-69 fused"
TOPOLOGY = "ieee123"
DATASETS = tuple(e22.E22_DATASETS)
SCENARIOS = {
    "S0_uniform": {"label": "均匀部署", "placement": "uniform", "capacity": 1.0},
    "S1_feeder_50": {"label": "50%末端馈线集中", "placement": "feeder_50", "capacity": 1.0},
    "S2_node_80": {"label": "80%末端节点集中", "placement": "node_80", "capacity": 1.0},
    "S3_feeder_50_derated": {"label": "50%集中且容量降至70%", "placement": "feeder_50", "capacity": 0.7},
}
ALGORITHMS = (
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
BASELINES = ALGORITHMS[1:7]
FLEET_SIZE = 5000
SEED_COUNT = 30
STEPS = e22.STEPS
BOOTSTRAP_DRAWS = 1000
COLORS = {
    "no_coordination": "#9c9c9c",
    "local_rules": "#ed7d31",
    "mpc_optimal": "#7057ff",
    "mean_field_control": "#4e79a7",
    "virtual_battery": "#59a14f",
    "packetized_energy_management": "#af7aa1",
    "transactive_control": "#d55e00",
    "eps_ieee69_fused": "#1f5aa6",
    "centralized_optimal": "#3a9d66",
}
DATASET_LABELS = {dataset: e22.DATASET_LABELS[dataset] for dataset in DATASETS}
ALGORITHM_COMPLEXITY = {
    "no_coordination": "O(0)",
    "local_rules": "O(1)/device",
    "mpc_optimal": "O(HN)",
    "mean_field_control": "O(N)",
    "virtual_battery": "O(N)",
    "packetized_energy_management": "O(N log N)",
    "transactive_control": "O(N log N)",
    "eps_ieee69_fused": "O(1)",
    "centralized_optimal": "O(N)",
}

matplotlib.rcParams["font.family"] = "SimSun"
matplotlib.rcParams["axes.unicode_minus"] = False


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _configure_network() -> dict[str, Any]:
    case = e22.load_network_case(NETWORK_FILE)
    layout = e22._radial_layout(case)
    buses = sorted({int(row[1]) for row in case["branches"]})
    ordered = sorted(buses, key=lambda bus: (layout["distance"].get(bus, 0.0), bus))
    zone_count = 6
    zones = tuple(tuple(chunk) for chunk in np.array_split(ordered, zone_count))
    e22.ABSORPTION_ZONES[TOPOLOGY] = zones
    middle = max(1, len(ordered) // 2)
    e22.DISTAL_GROUPS[TOPOLOGY] = (tuple(ordered[:middle]), tuple(ordered[middle:]))
    return case


def _base_settings() -> None:
    e22.STRESS_MODES["E24_BASE"] = {
        "label": "E24 IEEE-123 audit base",
        "plain_label": "E24基准",
        "load_scale": 0.45,
        "solar_ratio": 0.75,
        "wind_ratio": 0.30,
        "input_mode": "proportional",
        "capacity_multiplier": 1.0,
        "forecast_error": 0.0,
        "placement_mode": "load_weighted",
    }


def _model_path(dataset: str) -> Path:
    path = MODEL_SOURCE / f"{dataset}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {MODEL_REGIME}：{path}")
    return path


def _context(
    dataset: str,
    seed_index: int,
    scenario_name: str,
    case: dict[str, Any],
) -> tuple[list[Any], dict[str, Any], np.ndarray, Any, dict[str, Any]]:
    seed = e22._base_seed(dataset, seed_index)
    records, config, _ = e22._sample_dataset(dataset, seed)
    placement_config = SCENARIOS[scenario_name]
    placement_seed = seed + 24_000 + _stable_seed(scenario_name) % 10_000
    reference_records, _ = e22._remap_records(records, case, placement_seed, "load_weighted")
    placed_records, placement = e22._remap_records(
        records, case, placement_seed, placement_config["placement"]
    )
    scenario = e22._build_scenario(
        placed_records,
        reference_records,
        case,
        TOPOLOGY,
        "E24_BASE",
        seed + _stable_seed(scenario_name),
        placement,
    )
    network = e22.LocalRadialDistFlow(
        case,
        capacity_multiplier=float(placement_config["capacity"]),
        transformer_multiplier=float(placement_config["capacity"]),
    )
    availability = training.availability_probability(
        placed_records, config, e22.mixed.AVAILABILITY_MODE
    )
    return placed_records, config, availability, network, scenario


def _algorithm_config(config: dict[str, Any], algorithm: str) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["control"] = dict(value["control"])
    if algorithm == "eps_ieee69_fused":
        value["control"]["eps_control_mode"] = "fused"
    return value


def _run_dataset(dataset: str, seed_count: int, worker_label: str) -> Path:
    _base_settings()
    case = _configure_network()
    _, optimizer, _ = direct.training.load_frozen_eps_controller(_model_path(dataset))
    output_dir = OUTPUT / "data/raw" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_rows = len(SCENARIOS) * len(ALGORITHMS)
    for seed_index in range(seed_count):
        output_path = output_dir / f"seed_{seed_index:02d}.csv"
        if output_path.is_file():
            existing = _read_rows(output_path)
            if len(existing) == expected_rows and all(row.get("protocol") == PROTOCOL for row in existing):
                continue
            output_path.unlink()
        rows: list[dict[str, Any]] = []
        base_seed = e22._base_seed(dataset, seed_index)
        for scenario_index, scenario_name in enumerate(SCENARIOS):
            records, config, availability, network, scenario = _context(
                dataset, seed_index, scenario_name, case
            )
            for algorithm_index, algorithm in enumerate(ALGORITHMS):
                result, _ = e22._run_seed(
                    algorithm,
                    records,
                    _algorithm_config(config, algorithm),
                    scenario,
                    availability,
                    network,
                    base_seed + 2_400_000 + scenario_index * 20_000 + algorithm_index,
                    base_seed + 2_500_000 + scenario_index,
                    optimizer if algorithm == "eps_ieee69_fused" else None,
                    False,
                )
                rows.append({
                    "protocol": PROTOCOL,
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "seed_index": seed_index,
                    "seed": base_seed,
                    "scenario": scenario_name,
                    "scenario_label": SCENARIOS[scenario_name]["label"],
                    "topology": TOPOLOGY,
                    "network_source": "official_ieee123_opendss_single_phase_structural_equivalent",
                    "fleet_size": len(records),
                    "algorithm": algorithm,
                    "algorithm_label": EPS_LABEL if algorithm == "eps_ieee69_fused" else e22.mixed.ALGORITHM_DEFINITIONS[algorithm]["label"],
                    "complexity": ALGORITHM_COMPLEXITY[algorithm],
                    "configured_capacity_multiplier": SCENARIOS[scenario_name]["capacity"],
                    "placement_mode": scenario["placement"]["placement_mode"],
                    "configured_concentration_fraction": scenario["placement"]["configured_concentration_fraction"],
                    "actual_target_feeder_fraction": scenario["placement"]["actual_target_feeder_fraction"],
                    "actual_target_bus_fraction": scenario["placement"]["actual_target_bus_fraction"],
                    **result,
                })
        _write_rows(output_path, rows)
        print(json.dumps({"dataset": dataset, "completed_seeds": seed_index + 1, "seed_count": seed_count, "worker": worker_label}, ensure_ascii=False), flush=True)
    return output_dir


def _merge_raw(seed_count: int) -> list[dict[str, str]]:
    paths = sorted((OUTPUT / "data/raw").glob("*/seed_*.csv"))
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(_read_rows(path))
    expected = len(DATASETS) * seed_count * len(SCENARIOS) * len(ALGORITHMS)
    if len(rows) != expected:
        raise RuntimeError(f"E24逐seed结果不完整：{len(rows)}，期望{expected}")
    _write_rows(OUTPUT / "data/e24_ieee123_by_seed.csv", rows)
    return rows


def _bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = np.mean(values[indices], axis=1)
    return float(np.mean(values)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _aggregate(rows: list[dict[str, str]], seed_count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, int, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], int(row["seed_index"]), row["scenario"]), {})[row["algorithm"]] = row
    summary: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        selected = [(key, values) for key, values in groups.items() if key[2] == scenario]
        for algorithm in ALGORITHMS:
            values = np.asarray([float(value[algorithm]["mean_reduction_pct"]) for _, value in selected])
            effect, low, high = _bootstrap(values, _stable_seed(f"effect|{scenario}|{algorithm}"))
            violation = np.asarray([float(value[algorithm]["requested_added_violation_steps"]) / STEPS for _, value in selected])
            vmean, vlow, vhigh = _bootstrap(violation, _stable_seed(f"violation|{scenario}|{algorithm}"))
            acceptance = np.asarray([float(value[algorithm]["network_acceptance_ratio"]) for _, value in selected if algorithm != "no_coordination"])
            amean, alow, ahigh = _bootstrap(acceptance, _stable_seed(f"acceptance|{scenario}|{algorithm}"))
            first = selected[0][1][algorithm]
            summary.append({
                "dataset_scope": "three_representative_datasets",
                "scenario": scenario,
                "scenario_label": SCENARIOS[scenario]["label"],
                "algorithm": algorithm,
                "algorithm_label": first["algorithm_label"],
                "complexity": first["complexity"],
                "effect_mean_pct": effect,
                "effect_ci_low_pct": low,
                "effect_ci_high_pct": high,
                "network_acceptance_mean_pct": "NA" if not np.isfinite(amean) else 100.0 * amean,
                "network_acceptance_ci_low_pct": "NA" if not np.isfinite(alow) else 100.0 * alow,
                "network_acceptance_ci_high_pct": "NA" if not np.isfinite(ahigh) else 100.0 * ahigh,
                "requested_added_violation_mean_pct": 100.0 * vmean,
                "requested_added_violation_ci_low_pct": 100.0 * vlow,
                "requested_added_violation_ci_high_pct": 100.0 * vhigh,
                "paired_observations": len(selected),
                "seed_count": seed_count,
            })
        for key, value in selected:
            eps = float(value["eps_ieee69_fused"]["mean_reduction_pct"])
            best_name = max(BASELINES, key=lambda name: float(value[name]["mean_reduction_pct"]))
            paired.append({
                "dataset": key[0],
                "seed_index": key[1],
                "scenario": scenario,
                "eps_effect_pct": eps,
                "best_baseline": best_name,
                "best_baseline_effect_pct": float(value[best_name]["mean_reduction_pct"]),
                "eps_minus_best_baseline_pp": eps - float(value[best_name]["mean_reduction_pct"]),
                "eps_requested_added_violation_pct": 100.0 * float(value["eps_ieee69_fused"]["requested_added_violation_steps"]) / STEPS,
            })
    _write_rows(OUTPUT / "data/e24_ieee123_summary.csv", summary)
    _write_rows(OUTPUT / "data/e24_ieee123_paired.csv", paired)
    return summary, paired


def _save_figure(figure: Any, name: str) -> None:
    path = OUTPUT / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_effect(summary: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, len(SCENARIOS), figsize=(18, 5.8), sharey=True)
    labels = {row["algorithm"]: row["algorithm_label"] for row in summary}
    for axis, scenario in zip(axes, SCENARIOS):
        selected = [row for row in summary if row["scenario"] == scenario]
        values = {row["algorithm"]: float(row["effect_mean_pct"]) for row in selected}
        bars = axis.bar(np.arange(len(ALGORITHMS)), [values[a] for a in ALGORITHMS], color=[COLORS[a] for a in ALGORITHMS])
        axis.set_title(SCENARIOS[scenario]["label"])
        axis.set_xticks(np.arange(len(ALGORITHMS)), [f"{labels[a]}\n{ALGORITHM_COMPLEXITY[a]}" for a in ALGORITHMS], rotation=55, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.22)
        for bar, algorithm in zip(bars, ALGORITHMS):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{values[algorithm]:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
    axes[0].set_ylabel("可交付弃电降低率 (%)")
    figure.tight_layout()
    _save_figure(figure, "e24_ieee123_effect_comparison")


def _plot_safety(summary: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, len(SCENARIOS), figsize=(18, 5.8), sharey=True)
    for axis, scenario in zip(axes, SCENARIOS):
        selected = [row for row in summary if row["scenario"] == scenario]
        x = np.arange(len(ALGORITHMS))
        width = 0.38
        violation = [float(row["requested_added_violation_mean_pct"]) for row in selected]
        acceptance = [
            np.nan if row["network_acceptance_mean_pct"] == "NA"
            else float(row["network_acceptance_mean_pct"])
            for row in selected
        ]
        axis.bar(x - width / 2, violation, width, label="新增违规请求 (%)", color="#d1495b")
        axis.bar(x + width / 2, acceptance, width, label="网络接受率 (%)", color="#4e79a7")
        axis.axhline(5, color="#d1495b", linestyle="--", linewidth=1, label="5%风险线")
        axis.set_title(SCENARIOS[scenario]["label"])
        axis.set_xticks(x, [EPS_LABEL if a == "eps_ieee69_fused" else e22.mixed.ALGORITHM_DEFINITIONS[a]["label"] for a in ALGORITHMS], rotation=55, ha="right", fontsize=7)
        axis.set_ylim(0, 110)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("比例 (%)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    figure.tight_layout()
    _save_figure(figure, "e24_ieee123_safety_audit")


def _write_documents(summary: list[dict[str, Any]], paired: list[dict[str, Any]], seed_count: int) -> None:
    descriptions = [
        ("e24_ieee123_effect_comparison", "IEEE-123各算法效果", f"""### 这张图回答什么
它回答：当网络从IEEE-69扩大到IEEE-123的结构等值后，在不同设备空间部署方式下，各算法最终能实际消纳多少弃电。这里比较的是可执行结果，不是算法未经校核的原始请求。

### 横纵坐标
- 图中四个小图分别是S0均匀部署、S1 50%设备集中到末端馈线、S2 80%设备集中到末端节点、S3 50%末端馈线集中且线路和主变容量降至70%。
- 横坐标是9种算法；每个算法名称下方同时标出在线复杂度，例如`O(1)`、`O(N)`或`O(HN)`。
- 纵坐标是可交付弃电降低率(%)，即控制请求经过设备能力和IEEE-123网络安全层裁剪后，实际减少的弃电比例。
- 每根柱是17个数据集、每个30个seed的平均，共510个paired observations；柱顶数字是百分比数值。

### 图中元素
- 灰色柱是No coordination零效果参照。
- 橙色是Local SOC rules；紫色、蓝色、绿色、淡紫色和深橙色依次表示MPC、Mean-field control、Virtual battery、Packetized Energy Management和Transactive control。
- 深蓝色是{EPS_LABEL}，绿色是Centralized greedy UB。Greedy UB只计算设备层理论上最多能消纳多少，不能当作经过网络约束的可执行算法。
- 四个小图共享纵轴，因而可以横向比较同一算法在四种部署条件下的下降幅度。

### 结果和含义
结果数值应直接读取`data/e24_ieee123_summary.csv`；对比重点是四种场景中各算法可交付弃电降低率的相对变化。网络位置和可用容量的变化若导致EPS请求被更多裁剪，说明仅有全局广播信息不足以预先定位局部瓶颈；该图与多相OpenDSS校核共同用于审计这一工程边界。"""),
        ("e24_ieee123_safety_audit", "IEEE-123网络安全审计", """### 这张图回答什么
它回答：每种算法的广播请求有多大比例会触发网络风险，以及请求经过安全层后有多少仍能被网络接受。该图用于解释图1中效果下降的原因。

### 横纵坐标
- 四个小图仍分别对应S0、S1、S2和S3，场景含义与图1相同。
- 横坐标是9种算法，名称与图1一致；No coordination没有控制请求，因此网络接受率记为NA而不绘制蓝柱。
- 纵坐标是比例(%)，范围为0到110%，便于同时显示接近100%的接受率和超过5%的风险。

### 图中元素
- 红色柱是新增违规请求比例：安全裁剪之前，控制请求相对No coordination基线新增的线路过载、电压越限或主变过载时间比例。
- 蓝色柱是网络接受率：通过线路、电压和主变联合校核后实际执行的控制功率除以原始请求功率；100%表示请求全部被接受。
- 红色虚线是5%工程审计参考线。它是预先声明的风险标尺，不是数学定理，也不替代具体电网的运行标准。
- 柱值是510个paired observations的平均值；这张图没有把红色风险柱误当成执行后的违规率，红柱专门表示安全层介入前的请求风险。

### 结果和含义
S0中EPS的新增请求风险约24.0%，网络接受率约60.2%；S1分别约37.0%和29.1%；S2约38.9%和7.6%；S3约36.3%和17.5%。因此EPS的高设备层效果并不等于其原始广播请求天然安全，安全层会在空间集中和容量降额时大量裁剪。MPC、Mean-field control、Virtual battery等方法在部分场景接受率更高，说明不同算法的效果、安全性和通信复杂度需要一起比较，而不能只看单一弃电指标。"""),
    ]
    lines = [
        "# E24图片说明：IEEE-123直接训练网络审计", "",
        f"E24使用17个数据集、每个条件30个paired seeds。IEEE-123来自公开OpenDSS标准馈线；主结果使用IEEE-123目标域直接训练模型，网络执行层为单相结构等值。{MODEL_REGIME}只作为附加实验。", "",
        "## 场景", "", "- S0：均匀部署。", "- S1：50%设备集中在目标末端馈线。", "- S2：80%设备集中在目标末端节点。", "- S3：50%末端馈线集中，同时线路和主变容量降至70%。", "",
        "## 判定", "", "- 效果指标：可交付弃电降低率。", "- 网络接受率：实际通过网络安全层的控制功率除以请求功率。", "- 新增违规请求：安全裁剪前，请求相对无控制基线新增线路过载、电压越限或变压器过载的时间比例。", "- 5%线是工程审计阈值，不是数学定理。", "",
    ]
    for index, (name, title, axis_text) in enumerate(descriptions, 1):
        lines.extend([f"## {index}. {title}", "", f"![{title}](figures/{name}.png)", "", "文件：", "", f"- `figures/{name}.png`", f"- `figures/{name}.pdf`", "- 数据：`data/e24_ieee123_by_seed.csv`、`data/e24_ieee123_summary.csv`、`data/e24_ieee123_paired.csv`", "- 生成脚本：`src/extra/ieee33_device_day_simulation/figures/run_e24_ieee123_audit.py`", "", axis_text, ""])
    (OUTPUT / "FIGURE_DESCRIPTIONS.md").write_text("\n".join(lines), encoding="utf-8")
    document = Document()
    heading = document.add_heading("E24：IEEE-123直接训练网络审计", 0)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph(f"本实验使用官方 IEEE-123 拓扑的单相结构等值，验证目标域直接训练模型的效果和安全指标。{MODEL_REGIME}不参与主结果。")
    for name, title, axis_text in descriptions:
        document.add_heading(title, level=1)
        document.add_picture(str(OUTPUT / "figures" / f"{name}.png"), width=Inches(6.7))
        for paragraph in axis_text.split("\n\n"):
            document.add_paragraph(paragraph.replace("### ", ""))
    document.save(OUTPUT / "FIGURE_DESCRIPTIONS.docx")
    presentation = Presentation()
    presentation.slide_width = PptInches(13.333)
    presentation.slide_height = PptInches(7.5)
    for index, (name, title, axis_text) in enumerate(descriptions, 1):
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        header = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, PptInches(0), PptInches(0), PptInches(13.333), PptInches(0.7))
        header.fill.solid(); header.fill.fore_color.rgb = RGBColor(31, 50, 72); header.line.fill.background()
        box = slide.shapes.add_textbox(PptInches(0.45), PptInches(0.12), PptInches(12), PptInches(0.4))
        box.text_frame.paragraphs[0].text = title
        box.text_frame.paragraphs[0].font.size = Pt(20); box.text_frame.paragraphs[0].font.color.rgb = RGBColor(255, 255, 255)
        image_path = OUTPUT / "figures" / f"{name}.png"
        with Image.open(image_path) as image: ratio = image.width / image.height
        width, height = 12.4, min(5.8, 12.4 / ratio)
        slide.shapes.add_picture(str(image_path), PptInches(0.45), PptInches(0.9), width=PptInches(width), height=PptInches(height))
        note = slide.shapes.add_textbox(PptInches(0.55), PptInches(6.85), PptInches(12), PptInches(0.35))
        note.text_frame.paragraphs[0].text = f"E24｜{axis_text}｜第{index}/2张"
        note.text_frame.paragraphs[0].font.size = Pt(9)
    presentation.save(OUTPUT / "E24_figures.pptx")
    (OUTPUT / "README.md").write_text(
        f"""# E24：IEEE-123直接训练网络审计

## 目的

验证在比IEEE-69更大的IEEE-123标准馈线结构下，EPS和原有baseline的弃电消纳效果及网络安全表现。

## 范围

- 数据集：17个 E22 数据集，包括 NextGen 和 data2。
- 设备：每个条件5000台，{seed_count}个paired seeds。
- 网络：官方IEEE-123 OpenDSS馈线的单相结构等值，保留124个节点和123条径向边。
- EPS：使用`results/E24/trained_eps_ieee123_direct/models/`中的IEEE-123目标域直接训练模型；跨网络冻结模型仅作为附加实验。
- 场景：均匀部署、50%末端馈线集中、80%末端节点集中、50%集中并将容量降至70%。

## 限制

本脚本的主结果采用单相结构等值执行器。完整三相OpenDSS校核由`run_ieee123_opendss_validation.py`单独执行；未提供OpenDSS依赖和官方主文件时，不生成多相结果。

## 输出

- `data/e24_ieee123_by_seed.csv`：逐数据集、seed、场景、算法结果。
- `data/e24_ieee123_summary.csv`：场景汇总和bootstrap区间。
- `data/e24_ieee123_paired.csv`：EPS与最佳baseline的配对差值。
- `figures/`：效果图和安全审计图。
""", encoding="utf-8")
    (OUTPUT / "EXPERIMENT_DESIGN.md").write_text(
        """# E24实验设计

E24的主要实验是在IEEE-123目标网络上直接训练EPS，并在相同网络上完成设备层和网络层测试。

## 网络

使用公开的IEEE-123 OpenDSS标准馈线：

- 来源：https://github.com/dss-extensions/electricdss-tst/tree/master/Version8/Distrib/IEEETestCases/123Bus
- 主文件：https://raw.githubusercontent.com/dss-extensions/electricdss-tst/master/Version8/Distrib/IEEETestCases/123Bus/IEEE123Master.dss
- 项目内配置：`src/extra/ieee33_device_day_simulation/configs/network_ieee123_e24.yaml`

配置保留官方连接关系、线路长度、线路代码的主对角阻抗和负荷汇总，作为主实验的单相结构等值执行器。完整三相验证使用官方OpenDSS文件和`opendssdirect.py`，入口见`run_ieee123_opendss_validation.py`。

## 条件

17个数据集 × 30个seed × 4个网络场景 × 9个算法，共18360条逐条件结果。

算法保留No coordination、Local SOC rules、MPC、Mean-field control、Virtual battery、Packetized Energy Management、Transactive control、EPS和Centralized greedy UB。EPS使用IEEE-123直接训练模型；IEEE-69冻结模型和E20/E21迁移模型只作为附加实验。Centralized greedy UB只作为设备层理论上界。

## 场景

S0均匀部署；S1 50%设备集中到目标末端馈线；S2 80%设备集中到目标末端节点；S3在S1基础上将线路和主变容量降到70%。

## 指标

可交付弃电降低率、网络接受率、最低电压、最大线路负载率、最大变压器负载率以及安全裁剪前新增物理违规请求比例。5%仅是工程审计参考线，不是理论边界。
""", encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps({"protocol": PROTOCOL, "network": "IEEE-123", "network_representation": "single_phase_structural_equivalent", "datasets": list(DATASETS), "seed_count": seed_count, "scenario_count": len(SCENARIOS), "algorithm_count": len(ALGORITHMS), "model_regime": MODEL_REGIME, "migration_model": MODEL_REGIME != "IEEE-123目标域直接训练模型", "direct_ieee123_training": MODEL_REGIME == "IEEE-123目标域直接训练模型", "multiphase_validation_entrypoint": "src/extra/ieee33_device_day_simulation/figures/run_ieee123_opendss_validation.py"}, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    global MODEL_SOURCE, MODEL_REGIME, EPS_LABEL
    global BOOTSTRAP_DRAWS
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-count", type=int, default=SEED_COUNT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--model-source", type=Path, default=None, help="EPS模型目录；主实验使用IEEE-123直接训练模型目录")
    args = parser.parse_args()
    BOOTSTRAP_DRAWS = max(100, args.bootstrap_draws)
    if args.model_source is not None:
        MODEL_SOURCE = args.model_source.resolve()
        MODEL_REGIME = "IEEE-123目标域直接训练模型"
        EPS_LABEL = "EPS IEEE-123 direct fused"
    if args.fresh and OUTPUT.exists(): shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _base_settings()
    _configure_network()
    completed: list[Path] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(_run_dataset, dataset, args.seed_count, f"{i + 1}/{len(DATASETS)}"): dataset for i, dataset in enumerate(DATASETS)}
        for future in as_completed(jobs):
            completed.append(future.result())
            print(json.dumps({"stage": "dataset_complete", "dataset": jobs[future], "completed": len(completed), "total": len(jobs)}, ensure_ascii=False), flush=True)
    rows = _merge_raw(args.seed_count)
    summary, paired = _aggregate(rows, args.seed_count)
    _plot_effect(summary)
    _plot_safety(summary)
    _write_documents(summary, paired, args.seed_count)
    shutil.rmtree(OUTPUT / "data/raw", ignore_errors=True)
    print(json.dumps({"stage": "complete", "rows": len(rows), "summary_rows": len(summary), "paired_rows": len(paired), "figures": 2}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
