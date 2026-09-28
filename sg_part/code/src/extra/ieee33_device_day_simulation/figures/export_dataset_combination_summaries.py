"""将既有两两组合和预定义混合场景导出为逐组合 CSV 数据集。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PAIRWISE_SOURCE = Path("results/E21/pairwise_curtailment")
MIXED_SOURCE = Path("results/E21/data/e21_results.json")
DEFAULT_OUTPUT = Path("data/dataset_combination_summaries")

DATASET_LABELS = {
    "bdg1_building_data_genome": "BDG1",
    "bdg2_building_data_genome": "BDG2",
    "low_carbon_london": "LCL",
    "camsl_japan_smart_meters": "CAMSL",
    "irish_domestic_smart_meters": "Irish",
    "goiener_smart_meters": "GoiEner",
    "smart_grid_smart_city": "SGSC",
    "danish_smart_heat_meters": "Danish",
    "heapo_heat_pumps": "HEAPO",
    "european_lv_rural_2731": "EU-Rural",
    "european_lv_urban_35297": "EU-35297",
    "european_lv_urban_8087": "EU-8087",
    "norway_ami_energy_distribution": "Norway",
    "complete_energy_community": "CEC",
    "opsd_household_data": "OPSD",
}

SCENARIO_PURPOSES = {
    "S1-A": "相似住宅半小时电表融合",
    "S1-B": "建筑、热泵和带受控负荷住宅的高差异融合",
    "S2-A": "相似的大规模配网与 AMI 场景",
    "S2-B": "建筑、住宅、热需求、热泵和综合社区融合",
    "S3-A": "十数据集电力场景，加入少量热泵扰动",
    "S3-B": "十数据集多行业复杂场景",
    "S4-A": "数据集等权的极端异质性压力测试",
    "S4-B": "类型等权的主全量混合场景",
    "S5-A": "住宅主导长尾场景",
    "S5-B": "建筑和热负荷主导长尾场景",
    "S5-C": "网络和 DER 主导长尾场景",
    "S6-A": "相似网络负荷按上游、中游和末端空间聚集",
    "S6-B": "建筑、住宅和热负荷分层形成拥塞",
    "S6-C": "综合 DER 和双向交换子群集中在不同支路",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    remaining = sorted({key for row in rows for key in row} - set(preferred))
    fieldnames = [
        key for key in preferred if any(key in row for row in rows)
    ] + remaining
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _json_text(value) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _export_pairwise(source: Path, output: Path) -> list[dict[str, Any]]:
    seed_rows = _read_csv(source / "data/pairwise_curtailment_by_seed.csv")
    summary_rows = {
        row["pair_id"]: row
        for row in _read_csv(source / "data/pairwise_curtailment_summary.csv")
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in seed_rows:
        grouped.setdefault(row["pair_id"], []).append(row)

    mappings: list[dict[str, Any]] = []
    for pair_id in sorted(grouped):
        summary = summary_rows[pair_id]
        dataset_a = summary["dataset_a"]
        dataset_b = summary["dataset_b"]
        exported_rows = []
        for row in sorted(grouped[pair_id], key=lambda item: int(item["seed_index"])):
            exported_rows.append(
                {
                    "combination_type": "pairwise_equal_mix",
                    "pair_id": pair_id,
                    "pair_label": summary["pair_label"],
                    "dataset_a": dataset_a,
                    "dataset_a_label": DATASET_LABELS[dataset_a],
                    "weight_a": 0.5,
                    "dataset_b": dataset_b,
                    "dataset_b_label": DATASET_LABELS[dataset_b],
                    "weight_b": 0.5,
                    "network_mode": "aggregate",
                    "availability_mode": "data_driven",
                    "protocol": "dataset_combinations_pairwise_summary_v1",
                    **row,
                }
            )
        file_name = f"{pair_id}.csv"
        _write_csv(
            output / file_name,
            exported_rows,
            [
                "combination_type",
                "pair_id",
                "pair_label",
                "dataset_a",
                "dataset_a_label",
                "weight_a",
                "devices_a",
                "dataset_b",
                "dataset_b_label",
                "weight_b",
                "devices_b",
                "fleet_size",
                "seed",
                "seed_index",
                "network_mode",
                "availability_mode",
                "baseline_curtailment_mwh",
                "remaining_curtailment_mwh",
                "accepted_absorption_mwh",
                "curtailment_reduction_pct",
                "availability_fraction",
                "mean_network_scale",
                "network_violation_steps",
                "max_soc_violation",
                "profile_bootstrap_a",
                "mean_profile_reuse_a",
                "profile_bootstrap_b",
                "mean_profile_reuse_b",
                "protocol",
            ],
        )
        mappings.append(
            {
                "file": f"pairwise/{file_name}",
                "pair_id": pair_id,
                "dataset_a": dataset_a,
                "dataset_b": dataset_b,
                "label_a": DATASET_LABELS[dataset_a],
                "label_b": DATASET_LABELS[dataset_b],
                "row_count": len(exported_rows),
            }
        )
    return mappings


def _export_mixed(source: Path, output: Path) -> list[dict[str, Any]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    definitions = payload["scenario_definitions"]
    algorithm_definitions = payload["algorithm_definitions"]
    grouped: dict[str, list[dict[str, Any]]] = {
        scenario: [] for scenario in definitions
    }

    for condition in payload["conditions"]:
        scenario = condition["scenario"]
        audits = condition["fleet_audit"]
        for algorithm, result in condition["results"].items():
            algorithm_definition = algorithm_definitions[algorithm]
            for seed_index, seed_result in enumerate(result["seed_results"]):
                audit = audits[seed_index]
                if seed_result["composition_regime"] != audit["composition_regime"]:
                    raise ValueError(
                        f"{scenario}/{condition['network_mode']}/{algorithm} "
                        f"的第 {seed_index} 个组成状态无法与 fleet 审计对齐"
                    )
                seed_metrics = {
                    key: value
                    for key, value in seed_result.items()
                    if key not in {"seed", "composition_regime"}
                }
                grouped[scenario].append(
                    {
                        "combination_type": "predefined_mixed_scenario",
                        "scenario": scenario,
                        "scenario_label": condition["scenario_label"],
                        "scenario_purpose": SCENARIO_PURPOSES[scenario],
                        "network_mode": condition["network_mode"],
                        "algorithm": algorithm,
                        "algorithm_label": algorithm_definition["label"],
                        "complexity": algorithm_definition["complexity"],
                        "seed_index": seed_index,
                        "fleet_seed": int(audit["seed"]),
                        "algorithm_seed": int(seed_result["seed"]),
                        "composition_regime": seed_result["composition_regime"],
                        "fleet_size": audit["fleet_size"],
                        "nominal_weights_json": condition["nominal_weights"],
                        "actual_weights_json": audit["weights"],
                        "device_counts_json": audit["counts"],
                        "dataset_audit_json": audit["datasets"],
                        "spatial_clustering": audit["spatial_clustering"],
                        "partition": audit["partition"],
                        "protocol": "dataset_combinations_mixed_summary_v1",
                        **seed_metrics,
                    }
                )

    mappings: list[dict[str, Any]] = []
    for scenario in definitions:
        rows = sorted(
            grouped[scenario],
            key=lambda item: (
                item["network_mode"],
                item["algorithm"],
                int(item["seed_index"]),
            ),
        )
        file_name = f"{scenario}.csv"
        _write_csv(
            output / file_name,
            rows,
            [
                "combination_type",
                "scenario",
                "scenario_label",
                "scenario_purpose",
                "network_mode",
                "algorithm",
                "algorithm_label",
                "complexity",
                "seed_index",
                "fleet_seed",
                "algorithm_seed",
                "composition_regime",
                "fleet_size",
                "nominal_weights_json",
                "actual_weights_json",
                "device_counts_json",
                "spatial_clustering",
                "partition",
                "baseline_curtailment_mwh",
                "remaining_curtailment_mwh",
                "accepted_absorption_mwh",
                "mean_reduction_pct",
                "availability_fraction",
                "mean_network_scale",
                "network_violation_steps",
                "max_soc_violation",
                "total_charging_mwh",
                "excess_grid_charging_mwh",
                "total_discharge_mwh",
                "dataset_audit_json",
                "protocol",
            ],
        )
        mappings.append(
            {
                "file": f"mixed_scenarios/{file_name}",
                "scenario": scenario,
                "label": definitions[scenario]["label"],
                "purpose": SCENARIO_PURPOSES[scenario],
                "weights": definitions[scenario]["weights"],
                "row_count": len(rows),
            }
        )
    return mappings


def _format_weights(weights: dict[str, float]) -> str:
    return "；".join(
        f"{DATASET_LABELS[dataset]} {weight * 100:.2f}%"
        for dataset, weight in weights.items()
    )


def _build_readme(
    pairwise_mappings: list[dict[str, Any]],
    mixed_mappings: list[dict[str, Any]],
) -> str:
    pairwise_rows = "\n".join(
        f"|`{row['file']}`|{row['label_a']}|{row['label_b']}|50% / 50%|{row['row_count']}|"
        for row in pairwise_mappings
    )
    mixed_rows = "\n".join(
        f"|`{row['file']}`|{_format_weights(row['weights'])}|{row['purpose']}|{row['row_count']}|"
        for row in mixed_mappings
    )
    dataset_rows = "\n".join(
        f"|{label}|`{dataset}`|`data/{dataset}/`|"
        for dataset, label in DATASET_LABELS.items()
    )
    return f"""# 数据集组合摘要

本目录将两类组合实验整理成逐组合 CSV。每一种组合对应一个文件，便于单独读取、复核和重新绘图。这里保存的是组合定义、随机种子、fleet 审计信息和实验结果，不包含每台设备每个时刻的完整输入矩阵。

## 目录结构

```text
data/dataset_combination_summaries/
├── pairwise/          # 105 个两两 50/50 组合，每个组合一个 CSV
├── mixed_scenarios/   # S1-A 至 S6-C，每个场景一个 CSV
├── manifest.json      # 来源、数量和 SHA256 校验信息
└── README.md
```

## 构造方法

### 两两组合

- 从 15 个数据集中任取两个，得到 `C(15, 2) = 105` 个不同组合。
- 每个组合固定为 50%/50%，5000 台逻辑设备分别由两个数据集提供 2500 台。
- 每个文件保存 30 个配对随机种子的结果，因此每个 CSV 有 30 行。
- 网络模式为 `aggregate`，可用率为 `data_driven`。
- 原始结果来自既有两两组合实验目录。

### 预定义混合场景

- 使用 S1-A 至 S6-C 共 14 个预定义场景，各场景按名义权重构造一个 5000 台设备的混合 fleet。
- 每个场景同时保存 `aggregate` 和 `ieee33` 两种网络模式、10 种算法和 30 个随机种子，共 600 行。
- 每组前 10 个 seed 使用名义比例；中间 10 个将最大子群增加 15 个百分点；最后 10 个将最大子群减少 15 个百分点，其余子群按比例调整。
- `fleet_seed` 用于构造共享的设备 fleet；`algorithm_seed` 是各算法运行时使用的独立随机种子；`seed_index` 用于将二者正确配对。
- `nominal_weights_json` 是场景名义比例，`actual_weights_json` 和 `device_counts_json` 是当前 seed 的实际构成。
- 原始结果来自既有混合场景实验结果。

## 主要字段

|字段|含义|
|-|-|
|`baseline_curtailment_mwh`|不进行协调时的弃电量|
|`remaining_curtailment_mwh`|算法执行后的剩余弃电量|
|`accepted_absorption_mwh`|被设备实际吸收的弃电量|
|`curtailment_reduction_pct` / `mean_reduction_pct`|弃电降低率，单位为百分比|
|`availability_fraction`|当前 seed 的设备平均可用率|
|`network_violation_steps`|网络约束违规时间步数|
|`max_soc_violation`|最大 SOC 约束违规量|
|`mean_profile_reuse_a/b`|两两组合中原始 profile 的平均重复使用次数|
|`composition_regime`|`nominal`、最大子群增加或最大子群减少|
|`dataset_audit_json`|各组成数据集的逻辑设备数、可用源数和 profile 重复采样审计|

弃电降低率按以下公式计算：

```text
100 * (baseline_curtailment_mwh - remaining_curtailment_mwh)
    / baseline_curtailment_mwh
```

## 数据集名称关系

|简称|内部数据集 ID|项目数据目录|
|-|-|-|
{dataset_rows}

## 混合场景文件关系

|文件|组成|设计目的|行数|
|-|-|-|-|
{mixed_rows}

## 两两组合文件关系

|文件|数据集 A|数据集 B|比例|行数|
|-|-|-|-|-|
{pairwise_rows}

## 重新导出

```bash
python -m src.extra.dataset_combinations.export_experiment_summaries
```

脚本只读取既有实验结果，不会重新运行实验，也不会修改原始结果文件。每次导出都会重新生成 CSV、README 和 `manifest.json`，并校验组合数量、seed 数量、权重和 fleet 大小。
"""


def _validate(
    pairwise_mappings: list[dict[str, Any]],
    mixed_mappings: list[dict[str, Any]],
) -> None:
    if len(pairwise_mappings) != 105:
        raise ValueError(f"两两组合数量错误：{len(pairwise_mappings)}，预期 105")
    if any(row["row_count"] != 30 for row in pairwise_mappings):
        raise ValueError("两两组合必须每个文件包含 30 个 seed")
    if len(mixed_mappings) != 14:
        raise ValueError(f"混合场景数量错误：{len(mixed_mappings)}，预期 14")
    if any(row["row_count"] != 600 for row in mixed_mappings):
        raise ValueError("混合场景必须每个文件包含 600 行结果")
    for row in mixed_mappings:
        if abs(sum(row["weights"].values()) - 1.0) > 1e-9:
            raise ValueError(f"{row['scenario']} 的名义权重之和不等于 1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-source", type=Path, default=PAIRWISE_SOURCE)
    parser.add_argument("--mixed-source", type=Path, default=MIXED_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pairwise_output = args.output / "pairwise"
    mixed_output = args.output / "mixed_scenarios"
    pairwise_output.mkdir(parents=True, exist_ok=True)
    mixed_output.mkdir(parents=True, exist_ok=True)

    pairwise_mappings = _export_pairwise(args.pairwise_source, pairwise_output)
    mixed_mappings = _export_mixed(args.mixed_source, mixed_output)
    _validate(pairwise_mappings, mixed_mappings)

    readme = _build_readme(pairwise_mappings, mixed_mappings)
    _write_text(args.output / "README.md", readme)

    generated_files = sorted(args.output.rglob("*.csv")) + [args.output / "README.md"]
    manifest = {
        "protocol": "dataset_combinations_summary_export_v1",
        "pairwise_combination_count": len(pairwise_mappings),
        "mixed_scenario_count": len(mixed_mappings),
        "pairwise_seed_row_count": sum(row["row_count"] for row in pairwise_mappings),
        "mixed_result_row_count": sum(row["row_count"] for row in mixed_mappings),
        "sources": [
            {
                "path": str(
                    args.pairwise_source / "data/pairwise_curtailment_by_seed.csv"
                ),
                "sha256": _sha256(
                    args.pairwise_source / "data/pairwise_curtailment_by_seed.csv"
                ),
            },
            {
                "path": str(args.mixed_source),
                "sha256": _sha256(args.mixed_source),
            },
        ],
        "generated_files": [
            {
                "path": str(path.relative_to(args.output)),
                "sha256": _sha256(path),
            }
            for path in generated_files
        ],
    }
    _write_json(args.output / "manifest.json", manifest)

    print(f"已导出 {len(pairwise_mappings)} 个两两组合 CSV")
    print(f"已导出 {len(mixed_mappings)} 个混合场景 CSV")
    print(f"输出目录：{args.output}")


if __name__ == "__main__":
    main()
