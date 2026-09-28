"""运行 E24：IEEE-123 目标域直接训练 EPS。"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import run_e22_ieee69_complexity as e22
from . import run_e22_trained_eps_comparison as direct
from . import run_e24_ieee123_audit as audit


ROOT = Path(__file__).resolve().parents[4]
TOPOLOGY = "ieee123"
OUTPUT = ROOT / "results/E24/trained_eps_ieee123_direct"
PROTOCOL = "E24_ieee123_direct_training_v1"


def _configure_runtime() -> dict[str, Any]:
    """为复用的训练器注入 IEEE-123 网络，不改变 E22 默认全局配置。"""
    case = e22.load_network_case(audit.NETWORK_FILE)
    layout = e22._radial_layout(case)
    buses = sorted(int(bus) for bus in layout["buses"])
    ordered = sorted(buses, key=lambda bus: (layout["distance"].get(bus, 0.0), bus))
    zones = tuple(tuple(int(bus) for bus in chunk) for chunk in np.array_split(ordered, 6))
    midpoint = max(1, len(ordered) // 2)
    e22.TOPOLOGIES = (TOPOLOGY,)
    e22.NETWORK_FILES = {TOPOLOGY: audit.NETWORK_FILE}
    e22.DISTAL_GROUPS[TOPOLOGY] = (tuple(ordered[:midpoint]), tuple(ordered[midpoint:]))
    e22.ABSORPTION_ZONES[TOPOLOGY] = zones
    branch_keys = [f"{int(row[0])}-{int(row[1])}" for row in case["branches"]]
    e22.DERATINGS[TOPOLOGY] = {
        key: value for key, value in zip(branch_keys, (0.65, 0.55, 0.60))
    }

    def network_cases() -> dict[str, dict[str, Any]]:
        return {TOPOLOGY: case}

    e22._network_cases = network_cases
    direct.TRAINING_TOPOLOGY = TOPOLOGY
    direct.PROTOCOL = PROTOCOL
    direct.OUTPUT_ROOT = OUTPUT
    for definition in direct.NATIVE_DEFINITIONS.values():
        definition["label"] = definition["label"].replace("IEEE-69", "IEEE-123")
        definition["description"] = definition["description"].replace("IEEE-69", "IEEE-123")
    direct.NATIVE_DEFINITION["label"] = "EPS IEEE-123 fused"
    direct.NATIVE_DEFINITION["description"] = "每个数据集在 IEEE-123 M0-M6 上直接训练，测试阶段保持 O(1) 广播控制。"
    for algorithm, definition in direct.NATIVE_DEFINITIONS.items():
        e22.mixed.ALGORITHM_DEFINITIONS[algorithm] = definition
    return case


def _run_dataset_job(
    dataset: str,
    seed_count: int,
    stress_modes: tuple[str, ...],
    force: bool,
    retrain_all: bool,
    training_samples: int | None,
    validation_samples: int | None,
) -> dict[str, Any]:
    _configure_runtime()
    return direct._run_dataset_job(
        dataset,
        str(OUTPUT),
        seed_count,
        (TOPOLOGY,),
        stress_modes,
        force,
        retrain_all,
        training_samples,
        validation_samples,
    )


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    e22._write_rows(path, rows)


def _plot_training_r2(metadata: list[dict[str, Any]]) -> None:
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = [e22.DATASET_LABELS[row["dataset"]] for row in metadata]
    values = [float(row.get("held_out_r2", float("nan"))) for row in metadata]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(np.arange(len(labels)), values, color="#1f5aa6")
    axis.axhline(0.95, color="#d1495b", linestyle="--", linewidth=1.2, label="R²=0.95")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    axis.set_ylim(min(-0.1, float(np.nanmin(values)) - 0.05), 1.05)
    axis.set_ylabel("Held-out R²")
    axis.grid(axis="y", alpha=0.22)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "ieee123_direct_training_r2.png", dpi=220)
    figure.savefig(figure_dir / "ieee123_direct_training_r2.pdf")
    plt.close(figure)


def _write_documents(metadata: list[dict[str, Any]], seed_count: int, stress_modes: tuple[str, ...]) -> None:
    training_rows = "\n".join(
        f"|{e22.DATASET_LABELS[row['dataset']]}|{int(row.get('training_samples', 0))}|"
        f"{int(row.get('validation_samples', 0))}|{float(row.get('held_out_r2', float('nan'))):.4f}|"
        for row in metadata
    )
    (OUTPUT / "EXPERIMENT_DESIGN.md").write_text(
        f"""# E24：IEEE-123直接训练与网络验证

## 主要实验

E24 的主要实验在 IEEE-123 目标网络上为每个数据集独立训练 EPS。训练标签由 IEEE-123 的 M0-M6 网络约束仿真生成，测试阶段使用相同网络、设备、场景和 paired seed。训练与测试使用不同的 profile partition；IEEE-123 测试集不参与训练。

IEEE-123 的训练模型目录为 `models/`，训练拓扑是 `ieee123`，在线控制仍只发送一个广播强度，复杂度为 O(1)。

## 固定设置

- 17 个数据集，5000 台逻辑设备，{seed_count} 个 paired seeds。
- M0-M6：常规运行、深馈线高负载、双瓶颈反向潮流、线路降额与预测误差、均匀部署、50%末端馈线集中、80%末端节点集中。
- 主结果算法：No coordination、Local SOC rules、MPC、Mean-field control、Virtual battery、Packetized Energy Management、Transactive control、IEEE-123直接训练 EPS 和 Centralized greedy UB。
- Centralized greedy UB 是设备层理论上界，按 O(N) 计算设备可用 headroom；它不包含线路、电压和变压器安全校核。

## 训练审计

|数据集|训练样本|验证样本|Held-out R²|
|-|-:|-:|-:|
{training_rows}

## 附加实验

跨网络迁移结果单独作为附加实验保存。附加实验使用 E20/E21 的冻结模型，目的是测量跨域泛化，不参与 IEEE-123 直接训练主结论的排名。

## 多相验证

IEEE-123 的官方 OpenDSS 多相验证入口为：

```bash
python -m src.extra.ieee33_device_day_simulation.figures.run_ieee123_opendss_validation --master path/to/IEEE123Master.dss
```

该入口使用 OpenDSS 的三相节点电压、线路电流和变压器负载结果进行校核。没有安装 `opendssdirect.py` 或没有提供官方算例文件时，脚本会直接报出安装或文件错误，不会把单相结构等值结果标记为多相结果。

## 输出

- `data/trained_eps_by_seed.csv`：IEEE-123直接训练 EPS 的逐 seed、场景和消融结果。
- `data/trained_eps_summary.csv`：直接训练模型的效果汇总。
- `data/training_metadata.json`：训练样本、验证 R²、seed 和模型哈希。
- `figures/ieee123_direct_training_r2.png`：各数据集目标域模型的验证 R²。
""", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        """# E24：IEEE-123直接训练主实验

本目录保存 IEEE-123 目标域直接训练的 EPS 模型和训练审计结果。主要结论使用 `models/` 中在 IEEE-123 M0-M6 上训练的模型；跨网络迁移模型只在附加实验中使用。

运行：

```bash
python -m src.extra.ieee33_device_day_simulation.figures.run_e24_ieee123_direct_training --seed-count 30 --workers 2
```

生成模型后运行 IEEE-123 网络审计：

```bash
python -m src.extra.ieee33_device_day_simulation.figures.run_e24_ieee123_audit --model-source results/E24/trained_eps_ieee123_direct/models --seed-count 30
```

官方 IEEE-123 OpenDSS 三相校核：

```bash
python -m src.extra.ieee33_device_day_simulation.figures.run_ieee123_opendss_validation --master path/to/IEEE123Master.dss
```
""", encoding="utf-8"
    )
    (OUTPUT / "FIGURE_DESCRIPTIONS.md").write_text(
        """# E24图片说明

## 1. IEEE-123直接训练模型的验证 R²

![IEEE-123直接训练模型的验证 R²](figures/ieee123_direct_training_r2.png)

横坐标是数据集名称，纵坐标是目标域验证集上的 R²。每根柱对应一个只使用该数据集训练分区的 IEEE-123 EPS 模型；虚线是 R²=0.95 的参考线。R² 衡量模型对未参与训练的 IEEE-123 网络响应标签的解释程度，越接近 1 表示拟合越好。该图只审计训练质量，不替代最终的弃电降低率和网络安全指标。
""", encoding="utf-8"
    )
    manifest = {
        "protocol": PROTOCOL,
        "status": "completed",
        "network": "IEEE-123",
        "network_representation": "single_phase_structural_equivalent_for_training",
        "direct_ieee123_training": True,
        "migration_model": False,
        "training_stress_modes": list(stress_modes),
        "seed_count": seed_count,
        "datasets": [row["dataset"] for row in metadata],
        "model_directory": "models",
        "multiphase_validation_entrypoint": "src/extra/ieee33_device_day_simulation/figures/run_ieee123_opendss_validation.py",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-datasets", type=int, default=len(e22.E22_DATASETS))
    parser.add_argument("--max-stress-modes", type=int, default=len(e22.STRESS_MODES))
    parser.add_argument("--training-samples", type=int)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retrain-all", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seed_count <= 30:
        raise ValueError("seed-count 必须在 1 至 30 之间")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers 必须在 1 至 8 之间")
    if args.training_samples is not None and args.training_samples < 256:
        raise ValueError("training-samples 不能少于256，以保证充电和放电分支都有训练样本")
    if args.validation_samples is not None and args.validation_samples < 64:
        raise ValueError("validation-samples 不能少于64")
    _configure_runtime()
    datasets = e22.E22_DATASETS[: max(1, min(args.max_datasets, len(e22.E22_DATASETS)))]
    stress_modes = tuple(e22.STRESS_MODES)[: max(1, min(args.max_stress_modes, len(e22.STRESS_MODES)))]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata_by_dataset: dict[str, dict[str, Any]] = {}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(datasets)), mp_context=context) as executor:
        jobs = {
            executor.submit(
                _run_dataset_job,
                dataset,
                args.seed_count,
                stress_modes,
                args.force,
                args.retrain_all,
                args.training_samples,
                args.validation_samples,
            ): dataset
            for dataset in datasets
        }
        for future in as_completed(jobs):
            dataset = jobs[future]
            result = future.result()
            metadata_by_dataset[dataset] = result["metadata"]
            e22._write_json(OUTPUT / "data/training_metadata.json", [metadata_by_dataset[name] for name in datasets if name in metadata_by_dataset])
            print(json.dumps({"stage": "dataset_complete", "dataset": dataset, "completed": len(metadata_by_dataset), "total": len(datasets)}, ensure_ascii=False), flush=True)

    metadata = [metadata_by_dataset[dataset] for dataset in datasets]
    payloads = [json.loads((OUTPUT / "raw" / f"{dataset}.json").read_text(encoding="utf-8")) for dataset in datasets]
    rows = [row for payload in payloads for row in payload["seed_results"]]
    summaries = e22._summaries(rows, 2000)
    _write_rows(OUTPUT / "data/trained_eps_by_seed.csv", rows)
    _write_rows(OUTPUT / "data/trained_eps_summary.csv", summaries)
    _plot_training_r2(metadata)
    _write_documents(metadata, args.seed_count, stress_modes)
    e22._write_json(OUTPUT / "checkpoint.json", {"protocol": PROTOCOL, "status": "completed", "dataset_count": len(datasets), "seed_count": args.seed_count, "row_count": len(rows)})
    print(json.dumps({"status": "complete", "datasets": len(datasets), "rows": len(rows), "summary_rows": len(summaries)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
