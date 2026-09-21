from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

from . import run_e21_mixed_gamma_boundary as mixed_gamma
from . import run_e21_pairwise_curtailment as pairwise
from . import run_e21_pairwise_r2 as pairwise_r2


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "results/E21/unique"
ORIGINAL_E21 = ROOT / "results/E21"
UNIQUE_PROTOCOL = "E21_source_unique_derived_experiments_v1"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _enrich_fleet_audit(audit: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(audit)
    datasets: dict[str, Any] = {}
    for dataset, row in audit["datasets"].items():
        selected = int(row["selected_unique_profile_sources"])
        logical = int(row["logical_devices"])
        if row["profile_bootstrap"] or selected != logical:
            raise ValueError(f"{dataset} is not a source-unique cell")
        datasets[dataset] = {
            **row,
            "physical_devices": logical,
            "sampling_without_replacement": True,
            "duplicate_source_count": 0,
            "max_source_reuse": 1,
        }
    enriched.update(
        {
            "datasets": datasets,
            "sampling_without_replacement": True,
            "duplicate_source_count": 0,
            "max_source_reuse": 1,
            "fleet_mode": "source_unique",
            "reuse_provenance": (
                "the fixed5000 sampler also draws without replacement when the real sources suffice; "
                "this directory keeps only the cells whose audit shows profile_bootstrap=false and as many selected real sources as devices"
            ),
        }
    )
    return enriched


def _materialize_mixed_scenarios(output_root: Path) -> dict[str, Any]:
    output = output_root / "mixed_scenarios"
    figure_dir = output / "figures"
    source_dir = output_root / "figures"
    mapping = {
        "e21_unique_algorithm_heatmap_aggregate.png": "e21_algorithm_heatmap_aggregate.png",
        "e21_unique_algorithm_heatmap_ieee33.png": "e21_algorithm_heatmap_ieee33.png",
        "e21_unique_algorithm_overall.png": "e21_algorithm_overall.png",
        "e21_unique_eps_mix_vs_pool.png": "e21_eps_mix_vs_pool.png",
        "e21_unique_fleet_sizes.png": "e21_source_unique_fleet_sizes.png",
    }
    generated = []
    for source_name, target_name in mapping.items():
        target = figure_dir / target_name
        _copy_file(source_dir / source_name, target)
        generated.append(str(target.relative_to(output_root)))
    return {"directory": str(output), "generated": generated}


def _materialize_mixed_gamma(output_root: Path) -> dict[str, Any]:
    source = ORIGINAL_E21 / "gamma_mixed_boundary"
    output = output_root / "gamma_mixed_boundary"
    raw_dir = output / "raw" / "responses"
    data_dir = output / "data"
    figure_dir = output / "figures"
    for directory in (raw_dir, data_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint = json.loads(
        (source / "data/checkpoint.json").read_text(encoding="utf-8")
    )
    selected: list[dict[str, Any]] = []
    for cell in checkpoint:
        audit = cell["audit"]
        if any(row["profile_bootstrap"] for row in audit["datasets"].values()):
            continue
        copied = dict(cell)
        copied["audit"] = _enrich_fleet_audit(audit)
        old_archive = ROOT / cell["response_archive"]
        new_archive = raw_dir / cell["scenario"] / old_archive.name
        _copy_file(old_archive, new_archive)
        copied["response_archive"] = str(new_archive.relative_to(ROOT))
        selected.append(copied)

    selected.sort(
        key=lambda row: (
            list(mixed_gamma.mixed.SCENARIOS).index(row["scenario"]),
            int(row["N"]),
        )
    )
    arm_rows = [row for cell in selected for row in cell["rows"]]
    main_rows, summaries = mixed_gamma._annotate_metrics(arm_rows)
    leave_one_out, leave_one_out_rows = mixed_gamma._leave_one_scenario_out(main_rows)
    collapse = mixed_gamma._collapse_metrics(main_rows, leave_one_out)
    mixed_gamma._write_json(data_dir / "checkpoint.json", selected)
    mixed_gamma._write_rows(data_dir / "mixed_gamma_all_arms.csv", arm_rows)
    mixed_gamma._write_rows(data_dir / "mixed_gamma_points.csv", main_rows)
    mixed_gamma._write_rows(data_dir / "mixed_gamma_scenario_summary.csv", summaries)
    mixed_gamma._write_rows(data_dir / "leave_one_scenario_out.csv", leave_one_out_rows)
    mixed_gamma._write_json(data_dir / "collapse_quality.json", collapse)
    mixed_gamma._write_json(
        data_dir / "fleet_audit.json",
        [
            {"scenario": row["scenario"], "N": row["N"], "audit": row["audit"]}
            for row in selected
        ],
    )
    figures = mixed_gamma._save_figures(main_rows, arm_rows, summaries, figure_dir)
    args = SimpleNamespace(
        n_values=sorted({int(row["N"]) for row in selected}),
        train_replications=10,
        test_replications=30,
        conditions=48,
        bootstrap_draws=400,
        seed=20260807,
    )
    manifest = {
        "protocol": UNIQUE_PROTOCOL,
        "experiment": "E21_source_unique_mixed_gamma_boundary",
        "status": "completed",
        "scenario_count": len(summaries),
        "cell_count": len(selected),
        "source_cell_count": len(checkpoint),
        "excluded_bootstrapped_cells": len(checkpoint) - len(selected),
        "figures": figures,
        "generated_files": [],
    }
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["generated_files"].append(
                {"path": str(path.relative_to(output)), "sha256": _sha256(path)}
            )
    _write_json(output / "manifest.json", manifest)
    return manifest


def _materialize_pairwise_r2(output_root: Path) -> dict[str, Any]:
    source = ORIGINAL_E21 / "pairwise_r2"
    output = output_root / "pairwise_r2"
    raw_dir = output / "raw" / "responses"
    data_dir = output / "data"
    figure_dir = output / "figures"
    pair_data_dir = data_dir / "pairs"
    for directory in (raw_dir, data_dir, figure_dir, pair_data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint = json.loads(
        (source / "data/checkpoint.json").read_text(encoding="utf-8")
    )
    selected: list[dict[str, Any]] = []
    for cell in checkpoint["cells"]:
        if cell["profile_bootstrap_a"] or cell["profile_bootstrap_b"]:
            continue
        copied = {
            **cell,
            "protocol": "E21_pairwise_equal_mix_r2_source_unique_v1",
            "sampling_without_replacement": True,
            "duplicate_source_count": 0,
            "max_source_reuse": 1,
        }
        old_archive = ROOT / cell["response_archive"]
        new_archive = raw_dir / cell["pair_id"] / old_archive.name
        _copy_file(old_archive, new_archive)
        copied["response_archive"] = str(new_archive.relative_to(ROOT))
        selected.append(copied)

    pairs = pairwise._pair_specs()
    selected.sort(key=lambda row: (int(row["pair_index"]), int(row["N"])))
    summaries = pairwise_r2._summaries(pairs, selected)
    pairwise_r2._write_rows(data_dir / "pairwise_r2_points.csv", selected)
    pairwise_r2._write_rows(data_dir / "pairwise_r2_n95.csv", summaries)
    for pair in pairs:
        pairwise_r2._write_rows(
            pair_data_dir / f"{pair['pair_id']}.csv",
            pairwise_r2._pair_rows(selected, pair["pair_id"]),
        )
    figures = pairwise_r2._save_curves(pairs, selected, summaries, figure_dir)
    figures.extend(pairwise_r2._save_n95_overall(summaries, figure_dir))
    settings = checkpoint["settings"]
    args = SimpleNamespace(**settings)
    _write_json(
        data_dir / "checkpoint.json",
        {
            "protocol": "E21_pairwise_equal_mix_r2_source_unique_v1",
            "settings": settings,
            "cells": selected,
        },
    )
    reached = [row for row in summaries if row["N95_interpolated"] is not None]
    manifest = {
        "protocol": "E21_pairwise_equal_mix_r2_source_unique_v1",
        "status": "completed",
        "pair_count": len(pairs),
        "reached_count": len(reached),
        "right_censored_count": len(pairs) - len(reached),
        "cell_count": len(selected),
        "source_cell_count": len(checkpoint["cells"]),
        "excluded_bootstrapped_cells": len(checkpoint["cells"]) - len(selected),
        "figures": figures,
        "generated_files": [],
    }
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["generated_files"].append(
                {"path": str(path.relative_to(output)), "sha256": _sha256(path)}
            )
    _write_json(output / "manifest.json", manifest)
    return manifest


def _materialize_universal_gamma(output_root: Path) -> dict[str, Any]:
    source = ORIGINAL_E21 / "gamma_universal_boundary"
    output = output_root / "gamma_universal_boundary"
    shutil.copytree(source, output, dirs_exist_ok=True)
    manifest = {
        "protocol": UNIQUE_PROTOCOL,
        "experiment": "E21_source_unique_gamma_universal_boundary",
        "status": "completed",
        "source": str(source.relative_to(ROOT)),
        "sampling_source": "results/e1_full/E1_scale_boundary_new",
        "sampling_without_replacement": True,
        "generated_files": [],
    }
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "source_unique_manifest.json":
            manifest["generated_files"].append(
                {"path": str(path.relative_to(output)), "sha256": _sha256(path)}
            )
    _write_json(output / "source_unique_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Command line entry point for run e21 unique derived experiments.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = {
        "mixed_scenarios": _materialize_mixed_scenarios(output),
        "mixed_gamma": _materialize_mixed_gamma(output),
        "pairwise_r2": _materialize_pairwise_r2(output),
        "universal_gamma": _materialize_universal_gamma(output),
    }
    pairwise_manifest = output / "pairwise_curtailment" / "manifest.json"
    if pairwise_manifest.is_file():
        reports["pairwise_curtailment"] = json.loads(
            pairwise_manifest.read_text(encoding="utf-8")
        )
    _write_json(
        output / "derived_experiments_manifest.json",
        {
            "protocol": UNIQUE_PROTOCOL,
            "status": "completed",
            "reports": reports,
        },
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "mixed_gamma_cells": reports["mixed_gamma"]["cell_count"],
                "pairwise_r2_cells": reports["pairwise_r2"]["cell_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
