from __future__ import annotations

import copy
from pathlib import Path

import yaml

CONFIGS = Path("src/extra/nc_excel_experiments/configs")
BASE = CONFIGS / "protocol_e6_full.yaml"

PARTICIPATION = [1.0, 0.9, 0.8, 0.6, 0.4]

STRUCTURES = [
    "iid",
    "diurnal_class",
    "diurnal_antiphase",
    "diurnal_shift",
    "group_shared:1",
    "group_shared:5",
    "group_shared:20",
    "group_shared:100",
    "weather_block",
]

E15 = {
    "resources": 3000,
    "participation": PARTICIPATION,
    "structures": STRUCTURES,
    "markov_persistence": 0.9,
    "community_shared_fraction": 0.5,
    "diurnal_noise_weight": 0.5,
    "diurnal_classes": [
        {"name": "ev", "share": 0.4, "peak_hour": 12.0, "depth": 0.9},
        {"name": "home_battery", "share": 0.4, "peak_hour": 10.0, "depth": 0.5},
        {"name": "flat", "share": 0.2, "peak_hour": 0.0, "depth": 0.0},
    ],
    "broadcast_basis_columns": 9,
}


def main() -> None:
    base = yaml.safe_load(BASE.read_text())

    full = copy.deepcopy(base)
    full["results_root"] = "results/e15_full"
    full["e15"] = dict(E15, datasets=list(base["e6"]["datasets"]))
    full["execution"] = dict(base["execution"], experiments=["E15"], dataset_workers=15)
    (CONFIGS / "protocol_e15_full.yaml").write_text(
        yaml.safe_dump(full, sort_keys=False, allow_unicode=True))

    smoke = copy.deepcopy(base)
    smoke["results_root"] = "results/e15_smoke"
    smoke["common"] = dict(base["common"], test_replications=12, train_replications=4)
    smoke["e15"] = dict(
        E15,
        datasets=["low_carbon_london", "opsd_household_data"],
        participation=[1.0, 0.8, 0.4],
        resources=400,
    )
    smoke["execution"] = dict(base["execution"], experiments=["E15"], dataset_workers=2)
    (CONFIGS / "protocol_e15_smoke.yaml").write_text(
        yaml.safe_dump(smoke, sort_keys=False, allow_unicode=True))

    print("wrote", CONFIGS / "protocol_e15_full.yaml")
    print("wrote", CONFIGS / "protocol_e15_smoke.yaml")
    print("cells full  =", len(full["e15"]["datasets"]) * len(PARTICIPATION) * len(STRUCTURES))
    print("cells smoke =", 2 * 3 * len(STRUCTURES))


if __name__ == "__main__":
    main()
