"""为 E22 已完成数据集生成独立测试 R² 随物理设备规模变化的阶段性图。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import run_e22_ieee69_complexity as e22


DEFAULT_E22_ROOT = Path("results/E22")
DEFAULT_OUTPUT = DEFAULT_E22_ROOT / "interim/r2_scaling"
R2_THRESHOLD = 0.95
EXPECTED_ROWS_PER_DATASET = (
    e22.SEED_COUNT
    * len(e22.TOPOLOGIES)
    * len(e22.STRESS_MODES)
    * len(e22.ALGORITHM_ORDER)
)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _completed_datasets(e22_root: Path) -> list[dict[str, str]]:
    checkpoint = json.loads((e22_root / "checkpoint.json").read_text(encoding="utf-8"))
    output = []
    for dataset in checkpoint["dataset_ids"]:
        raw_path = e22_root / "raw" / f"{dataset}.json"
        if not raw_path.is_file():
            continue
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        rows = payload.get("seed_results", [])
        seed_count = len({int(row["seed_index"]) for row in rows})
        if (
            payload.get("protocol") == e22.PROTOCOL
            and seed_count == int(checkpoint["seed_count"])
            and len(rows) == EXPECTED_ROWS_PER_DATASET
        ):
            output.append({
                "dataset": dataset,
                "dataset_label": str(payload["dataset_label"]),
                "e22_raw": str(raw_path),
            })
    return output


def _threshold_path(dataset: str) -> Path:
    return (
        Path("results")
        / f"{dataset}_ieee33_real_load"
        / "coverage_fix/network_constrained_new/data/n_threshold.json"
    )


def _load_points(dataset: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    threshold_path = _threshold_path(dataset["dataset"])
    if not threshold_path.is_file():
        raise FileNotFoundError(
            f"{dataset['dataset']} 缺少 Figure 3A 规模结果：{threshold_path}"
        )
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol_details", {})
    if payload.get("protocol") != "e1_data_coupled_fixed_condition_multi_replication":
        raise ValueError(f"{threshold_path} 不是当前独立测试 R² 协议")
    points = []
    for n in sorted(int(value) for value in payload["N_values"]):
        row = payload["per_N"][str(n)]
        points.append({
            "dataset": dataset["dataset"],
            "dataset_label": dataset["dataset_label"],
            "N": n,
            "R2": float(row["r2"]),
            "R2_ci_lower": float(row["r2_ci_lower"]),
            "R2_ci_upper": float(row["r2_ci_upper"]),
            "N95_first_sampled": payload.get("threshold_N_95"),
            "N95_log_interpolated": payload.get("threshold_N_95_interpolated"),
            "train_conditions": protocol.get("train_conditions"),
            "test_conditions": protocol.get("eval_conditions"),
            "train_replications": protocol.get("train_replications"),
            "test_replications": protocol.get("eval_replications"),
            "bootstrap_draws": protocol.get("bootstrap_draws"),
            "source_file": str(threshold_path),
            "e22_completion_evidence": dataset["e22_raw"],
        })
    return points, payload


def _plot_curve(axis: Any, rows: list[dict[str, Any]], title: str) -> None:
    n_values = np.asarray([row["N"] for row in rows], dtype=float)
    r2 = np.asarray([row["R2"] for row in rows], dtype=float)
    lower = np.asarray([row["R2_ci_lower"] for row in rows], dtype=float)
    upper = np.asarray([row["R2_ci_upper"] for row in rows], dtype=float)
    n95 = rows[0]["N95_log_interpolated"]
    axis.fill_between(n_values, lower, upper, color="#2878b5", alpha=0.18, linewidth=0)
    axis.plot(n_values, r2, "o-", color="#2878b5", linewidth=1.8, markersize=4.2)
    axis.axhline(
        R2_THRESHOLD,
        color="#b2182b",
        linestyle=":",
        linewidth=1.2,
        label=r"$R^2=0.95$",
    )
    if n95 is not None:
        axis.axvline(float(n95), color="#1b7837", linestyle="--", linewidth=1.0)
        boundary = rf"log-interpolated $N_{{95}}={float(n95):.1f}$"
    else:
        boundary = rf"$R^2<0.95$ through $N={int(np.max(n_values))}$"
    axis.set_xscale("log")
    axis.set_xlim(0.8, float(np.max(n_values)) * 1.25)
    axis.set_ylim(max(-0.05, float(np.min(lower)) - 0.05), 1.01)
    axis.set_title(f"{title}\n{boundary}", fontsize=10)
    axis.set_xlabel(r"Physical fleet size $N$")
    axis.set_ylabel(r"Independent test $R^2$")
    axis.grid(alpha=0.22)


def _save(fig: Any, path: Path) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    pdf = path.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [str(path), str(pdf)]


def _write_readme(
    output: Path,
    datasets: list[dict[str, str]],
    rows: list[dict[str, Any]],
    figures: list[str],
) -> None:
    summary_lines = [
        "|数据集|采样 N 范围|首个采样 N95|对数插值 N95|最大 N 的 R²|",
        "|-|-|-|-|-|",
    ]
    for dataset in datasets:
        block = [row for row in rows if row["dataset"] == dataset["dataset"]]
        last = max(block, key=lambda row: int(row["N"]))
        sampled = block[0]["N95_first_sampled"]
        interpolated = block[0]["N95_log_interpolated"]
        summary_lines.append(
            f"|{dataset['dataset_label']}|{min(row['N'] for row in block)}-"
            f"{max(row['N'] for row in block)}|{sampled if sampled is not None else '未达到'}|"
            f"{float(interpolated):.2f}|{last['R2']:.6f}|"
        )
    text = f"""# E22 阶段性 R²-N 规模图

该目录只纳入 E22 主实验中已经完整完成 30 个 seed、2 种拓扑、7 个 M 场景和 10 种算法的数据集。当前共 {len(datasets)} 个数据集。重新运行绘图脚本时，新增完成的数据集会自动加入组图。

## 重要口径

这不是 E22 主实验 `response_r2` 的重绘。E22 主实验固定 `N=5000`，其中 `response_r2` 衡量 288 个时间步上实际吸收功率对目标富余功率的跟踪程度，无法构成 `R²-N` 曲线。

本目录复用各数据集已经完成的 Figure 3A 独立规模实验：在每个 N 下，用训练重复估计各广播条件的期望响应，再用独立测试重复计算 `R² = 1 - SSE/SST`。阴影为按条件 bootstrap 的 95% 置信区间，绿色虚线为相邻采样点间沿 `log10(N)` 插值得到的 N95。

## 文件

- `data/e22_completed_r2_points.csv`：当前已完成数据集的全部逐 N 点和来源路径。
- `figures/e22_completed_r2_group.png/pdf`：当前阶段组图。
- `figures/by_dataset/`：每个已完成数据集的单独曲线。
- 生成脚本：`src/extra/ieee33_device_day_simulation/figures/plot_e22_interim_r2_scaling.py`。

## 当前结果

{chr(10).join(summary_lines)}

## 与 E22 最终总图的关系

阶段性文件全部位于 `results/E22/interim/r2_scaling/`，不会修改 `results/E22/raw/`、检查点或最终图目录。它用于提前审阅已完成数据集的统计规模规律；最终 E22 网络复杂度图仍由固定 `N=5000` 的 M0-M6 主实验单独生成。当前生成图文件数为 {len(figures)}。
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e22-root", type=Path, default=DEFAULT_E22_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    datasets = _completed_datasets(args.e22_root)
    if not datasets:
        raise SystemExit("E22 尚无完整数据集")
    all_rows: list[dict[str, Any]] = []
    available_datasets = []
    for dataset in datasets:
        try:
            points, _ = _load_points(dataset)
        except FileNotFoundError as error:
            print(str(error), flush=True)
            continue
        all_rows.extend(points)
        available_datasets.append(dataset)
    if not all_rows:
        raise SystemExit("E22 已完成数据集均缺少 Figure 3A 规模结果")

    figures = []
    for dataset in available_datasets:
        rows = [row for row in all_rows if row["dataset"] == dataset["dataset"]]
        fig, axis = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
        _plot_curve(axis, rows, dataset["dataset_label"])
        figures.extend(_save(
            fig,
            args.output / "figures/by_dataset" / f"{dataset['dataset']}_r2_vs_n.png",
        ))

    columns = 3
    rows_count = math.ceil(len(available_datasets) / columns)
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(15.5, 4.8 * rows_count),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, dataset in zip(axes.flat, available_datasets):
        rows = [row for row in all_rows if row["dataset"] == dataset["dataset"]]
        _plot_curve(axis, rows, dataset["dataset_label"])
    for axis in axes.flat[len(available_datasets):]:
        axis.axis("off")
    fig.suptitle(
        "Independent test R² versus physical fleet size\n"
        "Figure 3A protocol; datasets completed in E22",
        fontsize=14,
    )
    figures.extend(_save(fig, args.output / "figures/e22_completed_r2_group.png"))
    _write_rows(args.output / "data/e22_completed_r2_points.csv", all_rows)
    _write_readme(args.output, available_datasets, all_rows, figures)
    manifest = {
        "status": "completed",
        "dataset_count": len(available_datasets),
        "datasets": [dataset["dataset"] for dataset in available_datasets],
        "point_count": len(all_rows),
        "figures": figures,
        "note": "Figure 3A independent-test R2; not E22 fixed-N response_r2",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
