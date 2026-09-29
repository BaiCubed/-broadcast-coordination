from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "derived_data"


def main() -> None:
    required = [
        ROOT / "requirements.txt",
        ROOT / "configs/reproducibility.json",
        ROOT / "tools/data/preprocess.py",
        ROOT / "src/extra/dataset_combinations/cli.py",
        ROOT / "src/extra/ieee33_device_day_simulation/configs/default.yaml",
        DATA / "configuration/generation_config.json",
        DATA / "configuration/scenario_config.json",
        DATA / "index/scenario_seed_index.csv",
        DATA / "index/pairwise_index.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing release files:\n" + "\n".join(missing))

    with (ROOT / "configs/reproducibility.json").open(encoding="utf-8") as handle:
        protocol = json.load(handle)
    if protocol["randomness"]["global_simulation_seed"] != 20260714:
        raise SystemExit("Unexpected global simulation seed")
    if protocol["neural_network"]["hidden_layers"] != [128, 64, 32]:
        raise SystemExit("Unexpected neural-network architecture")

    scenario_files = sorted((DATA / "generated_data/mixed_scenarios").glob("*.csv"))
    pair_files = sorted((DATA / "generated_data/pairwise").glob("*.csv"))
    if len(scenario_files) != 14 or len(pair_files) != 105:
        raise SystemExit(f"Expected 14 scenario and 105 pairwise CSVs, got {len(scenario_files)} and {len(pair_files)}")

    for path in scenario_files + pair_files:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
            if not header:
                raise SystemExit(f"Empty CSV: {path}")
            if not any(name in header for name in ("mean_reduction_pct", "curtailment_reduction_pct")):
                raise SystemExit(f"No reduction metric in {path}")
            if next(reader, None) is None:
                raise SystemExit(f"CSV has no data rows: {path}")

    print(f"derived-data check passed: {len(scenario_files)} scenarios, {len(pair_files)} pairwise files")


if __name__ == "__main__":
    main()
