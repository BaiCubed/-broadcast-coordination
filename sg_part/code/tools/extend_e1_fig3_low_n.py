"""增量计算 Figure 3 的低设备数采样点并同步旧结果目录。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extra.nc_excel_experiments.run import run_e1
from tools.sync_e1_fig3_results import synchronize


DEFAULT_PROTOCOL = ROOT / "src/extra/nc_excel_experiments/configs/protocol.yaml"
DEFAULT_SOURCE = ROOT / "results/e1_full/E1_scale_boundary_new"
DEFAULT_EXTENSION_ROOT = ROOT / "results/e1_fig3_low_n_extension"
LOW_N_VALUES = [1, 5, 15, 30]


def _copy_response_archives(extension: Path, source: Path) -> list[str]:
    copied: list[str] = []
    response_root = extension / "E1_scale_boundary_new/raw/responses"
    for archive in sorted(response_root.glob("*/responses_N*.npz")):
        destination = source / "raw/responses" / archive.parent.name / archive.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive, destination)
        copied.append(str(destination.relative_to(ROOT)))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--extension-root", type=Path, default=DEFAULT_EXTENSION_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sync-only", action="store_true")
    args = parser.parse_args()

    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    protocol["e1"]["n_candidates"] = LOW_N_VALUES
    protocol["e1"]["include_maximum_unique_fleet"] = False
    protocol["execution"]["dataset_workers"] = max(1, args.workers)

    extension_root = args.extension_root.resolve()
    extension_root.mkdir(parents=True, exist_ok=True)
    if args.sync_only:
        result = {"status": "sync_only_existing_archives"}
        copied = []
    else:
        result = run_e1(protocol, extension_root)
        copied = _copy_response_archives(extension_root, args.source.resolve())
    source = args.source.resolve()
    synchronized = synchronize(source, render=True)
    archive_audit = {}
    for dataset_dir in sorted((source / "raw/responses").iterdir()):
        if dataset_dir.is_dir():
            archive_audit[dataset_dir.name] = [
                fleet_size
                for fleet_size in LOW_N_VALUES
                if (dataset_dir / f"responses_N{fleet_size}_data_coupled.npz").is_file()
            ]

    manifest = {
        "protocol": "e1_fig3_low_n_incremental_extension",
        "requested_N_values": LOW_N_VALUES,
        "fleet_interpretation": "N is the logical fleet size; source-profile reuse must be disclosed when N exceeds measured source count",
        "extension_result": result,
        "copied_archive_count": len(copied),
        "copied_archives": copied,
        "existing_data_coupled_low_N": archive_audit,
        "synchronized_datasets": synchronized,
    }
    manifest_path = extension_root / "fig3_low_n_extension_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
