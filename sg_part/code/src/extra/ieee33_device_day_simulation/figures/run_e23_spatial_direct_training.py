"""运行 E23 的逐空间条件直接训练 EPS 对照实验。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from src.estimation import EPSEstimator, EstimatorConfig
from src.signal import SignalOptimizer

from ..original_adapter.eps_adapter import OriginalEPSAdapter
from . import fig4d_extra_baselines as legacy
from . import fig4d_final_protocol as protocol
from . import run_e22_ieee69_complexity as e22
from . import run_e23_spatial_heterogeneity as spatial
from . import run_e22_trained_eps_comparison as direct
from . import run_e20_transfer as transfer


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "results/E23/direct_training"
PUBLISH_OUTPUT = ROOT / "outputs/figs/direct_train"
SOURCE = ROOT / "results/E23"
PROTOCOL_NAME = "E23_ieee69_spatial_condition_direct_training_v1"
TRAINING_TOPOLOGY = "ieee69"
FLEET_SIZE = spatial.FLEET_SIZE
TRAIN_FLEET_SEEDS = tuple(range(0, 24))
VALIDATION_FLEET_SEEDS = tuple(range(24, 30))
TRAIN_SAMPLES_PER_FLEET = 500
VALIDATION_SAMPLES_PER_FLEET = 200
RESPONSE_SCALE_KW = 5000.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"没有可写入的数据：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _network_context() -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    case = spatial._network_cases()[TRAINING_TOPOLOGY]
    base_weights = spatial._region_weights(case)
    headroom = spatial._headroom_by_region(
        case, e22._network(case, TRAINING_TOPOLOGY, "M0")
    )
    global_counts = spatial._allocate_counts(
        np.ones(len(spatial.DATASETS)), FLEET_SIZE
    )
    return case, base_weights, headroom, {
        "global_counts": global_counts,
    }


def _mapped_fleet(
    fleet_index: int,
    condition: str,
    case: dict[str, Any],
    base_weights: np.ndarray,
    headroom: np.ndarray,
    global_counts: np.ndarray,
) -> tuple[list[Any], dict[str, Any], np.ndarray, np.ndarray]:
    records, config, _ = spatial._sample_global_fleet(900000 + fleet_index)
    matrix, region_counts = spatial._condition_matrix(
        condition, global_counts, base_weights, headroom
    )
    mapped = spatial._assign_records(
        records,
        matrix,
        region_counts,
        case,
        900000 + fleet_index + spatial._stable_seed(condition),
    )
    return mapped, config, matrix, region_counts


def _response_samples(
    records: list[Any],
    config: dict[str, Any],
    sample_count: int,
    seed: int,
    case: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[float]]:
    scenario = e22._build_scenario(
        records,
        records,
        case,
        TRAINING_TOPOLOGY,
        "M0",
        seed,
        {"placement_mode": "spatial_condition_direct_training"},
    )
    network = e22._network(case, TRAINING_TOPOLOGY, "M0")
    availability = protocol.availability_probability(
        records, config, spatial.mixed.AVAILABILITY_MODE
    )
    adapter = OriginalEPSAdapter(records, config, seed + 1)
    rng = np.random.default_rng(seed + 2)
    dt = float(config["simulation"]["time_step_seconds"]) / 3600.0
    zone_count = len(config["zones"]["zones"])
    signals: list[dict[str, Any]] = []
    responses: list[float] = []
    for index in range(sample_count):
        supply_demand, intensity, hour = protocol._signal_parameters(
            index, sample_count, rng
        )
        profile_step = int(hour * 12 + rng.integers(0, 12))
        adapter.reset_to_initial_state()
        energy_kwh = 0.0
        for offset in range(3):
            step = (profile_step + offset) % legacy.STEPS
            generated = [
                adapter.simulator._signal_generator.generate_signal(
                    zone, supply_demand, intensity, priority=10
                )
                for zone in range(zone_count)
            ]
            batch = adapter.step(generated, step, availability[step])
            load_map, input_map = e22._maps(scenario, step)
            dispatch = network.dispatch(
                load_map,
                input_map,
                np.asarray(batch.desired_kw, dtype=float),
                scenario["device_buses"],
            )
            actual = adapter.apply_dispatch(dispatch.accepted_kw)
            energy_kwh += float(np.sum(actual)) * dt
        signals.append(
            {
                "supply_demand": supply_demand,
                "intensity": intensity,
                "price": 0.0,
                "hour": hour,
                "day_of_week": 0,
                "direction": 1 if supply_demand <= 7 else -1,
            }
        )
        responses.append(
            energy_kwh / max(3.0 * dt, 1e-12) / RESPONSE_SCALE_KW
        )
    return signals, responses


def _train_condition(condition: str, output: str, force: bool) -> dict[str, Any]:
    output_root = Path(output)
    model_path = output_root / "models" / f"{condition}.pt"
    metadata_path = output_root / "models" / f"{condition}.json"
    if model_path.is_file() and metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("protocol") == PROTOCOL_NAME:
            return metadata

    case, base_weights, headroom, context = _network_context()
    train_signals: list[dict[str, Any]] = []
    train_responses: list[float] = []
    validation_signals: list[dict[str, Any]] = []
    validation_responses: list[float] = []
    for fleet_index in TRAIN_FLEET_SEEDS:
        records, config, _, _ = _mapped_fleet(
            fleet_index,
            condition,
            case,
            base_weights,
            headroom,
            context["global_counts"],
        )
        signals, responses = _response_samples(
            records,
            config,
            TRAIN_SAMPLES_PER_FLEET,
            700000 + fleet_index + e22._stable_seed(condition),
            case,
        )
        train_signals.extend(signals)
        train_responses.extend(responses)
        print(
            json.dumps(
                {
                    "stage": "direct_training_samples",
                    "condition": condition,
                    "fleet": fleet_index + 1,
                    "fleet_count": len(TRAIN_FLEET_SEEDS),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    for fleet_index in VALIDATION_FLEET_SEEDS:
        records, config, _, _ = _mapped_fleet(
            fleet_index,
            condition,
            case,
            base_weights,
            headroom,
            context["global_counts"],
        )
        signals, responses = _response_samples(
            records,
            config,
            VALIDATION_SAMPLES_PER_FLEET,
            800000 + fleet_index + e22._stable_seed(condition),
            case,
        )
        validation_signals.extend(signals)
        validation_responses.extend(responses)

    try:
        import torch

        torch.manual_seed(3100000 + e22._stable_seed(condition) % 100000)
        torch.set_num_threads(1)
    except ImportError:
        pass
    estimator = EPSEstimator(
        EstimatorConfig(
            target_coverage=0.9,
            enable_conformal=True,
            use_cqr=True,
            use_pytorch=True,
            pytorch_epochs=legacy.EPS_TRAINING_EPOCHS,
            pytorch_batch_size=64,
            pytorch_learning_rate=0.001,
        )
    )
    estimator.fit(train_signals, train_responses)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    legacy._save_eps_model(estimator, model_path)
    direct._ensure_bidirectional_artifact(model_path)
    predicted = np.asarray(
        [estimator.estimate(signal).response_kw for signal in validation_signals],
        dtype=float,
    )
    actual = np.asarray(validation_responses, dtype=float)
    residual = actual - predicted
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    parameters = estimator._learned_params or {}
    metadata = {
        "protocol": PROTOCOL_NAME,
        "condition": condition,
        "condition_label": spatial.CONDITIONS[condition]["label"],
        "training_topology": TRAINING_TOPOLOGY,
        "training_scope": "IEEE-69 spatial condition direct training",
        "training_fleet_seeds": list(TRAIN_FLEET_SEEDS),
        "validation_fleet_seeds": list(VALIDATION_FLEET_SEEDS),
        "training_samples": len(train_signals),
        "validation_samples": len(validation_signals),
        "epochs": legacy.EPS_TRAINING_EPOCHS,
        "model_type": str(parameters.get("model_type", "unknown")),
        "internal_training_r2": float(parameters.get("r2", float("nan"))),
        "held_out_r2": float(
            1.0 - np.sum(residual**2) / max(denominator, 1e-12)
        ),
        "held_out_rmse_normalized": float(np.sqrt(np.mean(residual**2))),
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "model_reused": False,
        "input_protocol": "one broadcast signal and aggregate scalar feedback",
        "communication_complexity": "O(1) with respect to N",
    }
    _write_json(metadata_path, metadata)
    return metadata


def _train_job(condition: str, output: str, force: bool) -> dict[str, Any]:
    spatial.CACHE_ROOT = SOURCE / "data/.fleet_cache"
    return _train_condition(condition, output, force)


def _evaluate_direct(
    condition: str,
    seed_index: int,
    model_path: str,
) -> dict[str, Any]:
    spatial.CACHE_ROOT = SOURCE / "data/.fleet_cache"
    case, base_weights, headroom, context = _network_context()
    records, config, matrix, region_counts = _mapped_fleet(
        seed_index,
        condition,
        case,
        base_weights,
        headroom,
        context["global_counts"],
    )
    spatial.RUN_ALGORITHMS = ("eps_global_mixed",)
    rows, metric, regions = spatial._run_condition(
        records,
        config,
        condition,
        matrix,
        region_counts,
        case,
        base_weights,
        headroom,
        SignalOptimizer(
            transfer.ScaledEstimator(
                protocol.load_frozen_eps_controller(Path(model_path))[0],
                RESPONSE_SCALE_KW,
                0.0,
            )
        ),
        900000 + seed_index,
    )
    return {"rows": rows, "metric": metric, "regions": regions}


def _write_docs(output: Path, metadata: list[dict[str, Any]], figures: list[str]) -> None:
    lines = [
        "# E23 IEEE-69 全直接训练版本",
        "",
        "本目录是 E23 空间异质性实验的逐条件直接训练版本。每个 H 条件分别使用 IEEE-69 中该条件构造的设备、区域映射和网络约束生成训练标签，再训练一个 EPS 模型；测试时使用同一条件的留出设备 fleet 和固定 seed。",
        "",
        "## 训练设置",
        "",
        "- 总设备数：5000；数据集：17 个；网络：IEEE-69。",
        "- 条件：H0、H1、H2_25、H2_50、H2_75、H3、H4、H5。",
        "- 每个条件使用 24 个 fleet seed 生成 12000 个训练样本，另用 6 个 fleet seed 生成 1200 个验证样本。",
        "- 训练标签是经过 IEEE-69 网络 dispatch 后的可交付聚合响应，保持原有设备层仿真、SOC、可用率和网络约束。",
        "- EPS 在线阶段仍只发送一个广播信号并接收一个聚合反馈，通信复杂度为 O(1)。",
        "- 原 E23 的 baseline 数据直接复用同一 seed、设备和网络构造；本版本只重新计算 EPS，保证前后对比只反映训练来源变化。",
        "",
        "## H 条件",
        "",
        "|条件|含义|",
        "|-|-|",
        "|H0|每个区域按相同数据类型比例混合，设备数量按区域基础负荷分配。|",
        "|H1|不同区域由不同数据类型主导，但区域设备数量仍按基础负荷分配。|",
        "|H2_25/H2_50/H2_75|数据类型比例保持全局一致，设备密度分别以 25%、50%、75% 强度集中。|",
        "|H3|类型分区与 50% 密度集中同时存在。|",
        "|H4|H3 的设备构成放到网络裕度较低的区域，形成不利耦合。|",
        "|H5|H3 的设备构成放到网络裕度较高的区域，形成有利耦合。|",
        "",
        "## 文件",
        "",
        "- `models/H*.pt`：每个 H 条件独立训练的 EPS 模型。",
        "- `models/H*.json`：训练样本、验证 R²、模型哈希和通信协议。",
        "- `data/spatial_heterogeneity_by_seed.csv`：baseline 与直接训练 EPS 的逐 seed 结果。",
        "- `data/spatial_heterogeneity_summary.csv`：各条件和算法的汇总结果。",
        "- `data/imbalance_metrics.csv`、`data/region_assignments.csv`：空间构造审计数据。",
        "- `figures/`：与原 E23 相同的效果、不平衡和安全图。",
        "",
        "## 如何判断是否更好",
        "",
        "重点比较同一 H 条件、同一 seed 下 EPS 的可交付弃电降低率、网络接受率、线路负载、电压和变压器负载。若直接训练版本在多数条件下提高弃电降低率且不增加网络越限，说明目标空间条件的训练分布与测试分布更匹配；若只提高效果但增加约束违规，则不能视为更优。",
        "",
        "## 图片",
        "",
    ]
    lines.extend(f"- `{Path(path).relative_to(output)}`" for path in figures)
    lines.extend(["", "## 模型验证", ""])
    for item in metadata:
        lines.append(
            f"- `{item['condition']}`：验证 R²={float(item['held_out_r2']):.4f}，训练样本={item['training_samples']}，验证样本={item['validation_samples']}。"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "EXPERIMENT_DESIGN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _publish_figures(figures: list[str]) -> list[str]:
    PUBLISH_OUTPUT.mkdir(parents=True, exist_ok=True)
    published: list[str] = []
    for figure in figures:
        source = Path(figure)
        for candidate in (source, source.with_suffix(".pdf")):
            if not candidate.is_file():
                continue
            target = PUBLISH_OUTPUT / candidate.name
            shutil.copy2(candidate, target)
            published.append(str(target))
    return published


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.seed_count != 30:
        raise ValueError("该版本固定使用 30 个 seed")
    if args.workers < 1:
        raise ValueError("workers 必须为正数")
    output = OUTPUT
    (output / "models").mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    spatial.CACHE_ROOT = SOURCE / "data/.fleet_cache"
    conditions = tuple(spatial.CONDITIONS)
    metadata_by_condition: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(conditions))) as executor:
        futures = {
            executor.submit(_train_job, condition, str(output), args.force): condition
            for condition in conditions
        }
        for future in as_completed(futures):
            condition = futures[future]
            metadata_by_condition[condition] = future.result()
            _write_json(
                output / "data/training_metadata.json",
                [metadata_by_condition[key] for key in conditions if key in metadata_by_condition],
            )
            print(
                json.dumps(
                    {
                        "stage": "model_complete",
                        "condition": condition,
                        "completed": len(metadata_by_condition),
                        "total": len(conditions),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    original_rows = _read_rows(SOURCE / "data/spatial_heterogeneity_by_seed.csv")
    original_metrics = _read_rows(SOURCE / "data/imbalance_metrics.csv")
    original_regions = _read_rows(SOURCE / "data/region_assignments.csv")
    baseline_rows = [row for row in original_rows if row["algorithm"] != "eps_global_mixed"]
    direct_rows: list[dict[str, Any]] = []
    spatial.RUN_ALGORITHMS = ("eps_global_mixed",)
    for seed_index in range(args.seed_count):
        for condition in conditions:
            result = _evaluate_direct(
                condition,
                seed_index,
                str(output / "models" / f"{condition}.pt"),
            )
            direct_rows.extend(result["rows"])
            print(
                json.dumps(
                    {
                        "stage": "evaluation",
                        "seed_index": seed_index,
                        "condition": condition,
                        "rows": len(direct_rows),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    rows = baseline_rows + direct_rows
    rows.sort(key=lambda row: (int(row["seed_index"]), conditions.index(row["condition"]), row["algorithm"]))
    _write_rows(output / "data/spatial_heterogeneity_by_seed.csv", rows)
    _write_rows(output / "data/imbalance_metrics.csv", original_metrics)
    _write_rows(output / "data/region_assignments.csv", original_regions)
    spatial.OUTPUT = output
    spatial.RUN_ALGORITHMS = spatial.ALGORITHMS
    figures = spatial._plot_outputs(rows, original_metrics, original_regions)
    published_figures = _publish_figures(figures)
    summary_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for algorithm in spatial.ALGORITHMS:
            selected = [
                row for row in rows
                if row["condition"] == condition and row["algorithm"] == algorithm
            ]
            if not selected:
                continue
            values = np.asarray([float(row["mean_reduction_pct"]) for row in selected])
            summary_rows.append(
                {
                    "condition": condition,
                    "condition_label": spatial.CONDITIONS[condition]["label"],
                    "algorithm": algorithm,
                    "algorithm_label": spatial.ALGORITHM_LABELS[algorithm],
                    "complexity": spatial.ALGORITHM_COMPLEXITY[algorithm],
                    "seed_count": len(selected),
                    "mean_reduction_pct": float(np.mean(values)),
                    "std_reduction_pct": float(np.std(values, ddof=1)),
                    "network_acceptance_mean_pct": float(np.mean([float(row["network_acceptance_ratio"]) for row in selected]) * 100.0),
                }
            )
    _write_rows(output / "data/spatial_heterogeneity_summary.csv", summary_rows)
    _write_json(
        output / "manifest.json",
        {
            "protocol": PROTOCOL_NAME,
            "status": "completed",
            "topology": TRAINING_TOPOLOGY,
            "dataset_count": len(spatial.DATASETS),
            "fleet_size": FLEET_SIZE,
            "seed_count": args.seed_count,
            "conditions": list(conditions),
            "direct_training_models": [str(output / "models" / f"{condition}.pt") for condition in conditions],
            "baseline_source": str(SOURCE / "data/spatial_heterogeneity_by_seed.csv"),
            "figures": figures,
            "published_figures": published_figures,
        },
    )
    _write_docs(output, [metadata_by_condition[key] for key in conditions], figures)
    print(json.dumps({"status": "complete", "output": str(output), "figures": figures}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
