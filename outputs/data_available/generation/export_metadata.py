"""Export the machine-readable public-release configuration and indexes.

The values are imported from the release generator itself so that the public
metadata cannot silently diverge from the implementation used by the paper.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "generation"
sys.path.insert(0, str(PACKAGE))

from dataset_combinations.constants import (  # noqa: E402
    ALL_DATASETS,
    DATASET_SOURCES,
    FLEET_SIZE,
    PAIR_ORDER,
    PAIR_SEED_BASE,
    PAIR_SEED_STRIDE,
    PARTITION_SEED,
    SCENARIOS,
    SCENARIO_SEEDS,
    TEST_SEED_COUNT,
    ZONE_BUSES,
)
from dataset_combinations.generator import pair_specs  # noqa: E402


SOURCE_URLS = {
    "bdg1_building_data_genome": "https://github.com/buds-lab/the-building-data-genome-project",
    "bdg2_building_data_genome": "https://github.com/buds-lab/building-data-genome-project-2",
    "complete_energy_community": "https://doi.org/10.5281/zenodo.7602546",
    "danish_smart_heat_meters": "https://doi.org/10.5281/zenodo.6563114",
    "european_lv_rural_2731": "https://doi.org/10.17632/gspyzvvrhm.2",
    "european_lv_urban_35297": "https://doi.org/10.17632/gspyzvvrhm.2",
    "goiener_smart_meters": "https://doi.org/10.5281/zenodo.7362094",
    "heapo_heat_pumps": "https://doi.org/10.5281/zenodo.15056919",
    "low_carbon_london": "https://data.london.gov.uk/download/vqm0d/3527bf39-d93e-4071-8451-df2ade1ea4f2/LCL-FullData.zip",
    "norway_ami_energy_distribution": "https://doi.org/10.17632/jv3rz8k35r.1",
    "camsl_japan_smart_meters": "https://doi.org/10.17632/cmpsyncmmk.1",
    "european_lv_urban_8087": "https://doi.org/10.17632/685vgp64sm.1",
    "irish_domestic_smart_meters": "https://doi.org/10.6084/m9.figshare.31851922.v2",
    "opsd_household_data": "https://data.open-power-system-data.org/household_data/opsd-household-data-2020-04-15.zip",
    "smart_grid_smart_city": "https://data.gov.au/data/dataset/smart-grid-smart-city-customer-trial-data",
}


# Relative paths and adapter parameters mirror the final DATASETS registry
# used by the project's canonical preprocessing stage.
PREPROCESSING_SPECS = {
    "bdg1_building_data_genome": {"adapter": "bdg1_building_data_genome", "input": "data/bdg1_building_data_genome/raw/temp_open_utc_complete.csv", "load_scale": 0.001, "max_sources": 507, "max_days_per_source": 20, "input_mapping": "load_shape_counterfactual"},
    "low_carbon_london": {"adapter": "low_carbon_london", "input": "data/low_carbon_london/raw/LCL-FullData.zip", "cache": "data/low_carbon_london/processed/canonical_device_days_bidirectional_v2.npz", "load_scale": 1.0, "max_sources": 5000, "max_days_per_source": 2, "input_mapping": "load_shape_counterfactual"},
    "bdg2_building_data_genome": {"adapter": "bdg2_building_data_genome", "input": "data/bdg2_building_data_genome/raw/electricity_cleaned.csv", "solar_input": "data/bdg2_building_data_genome/raw/solar_cleaned.csv", "cache": "data/bdg2_building_data_genome/processed/canonical_device_days.npz", "load_scale": 0.001, "max_sources": 1578, "max_days_per_source": 8, "input_mapping": "measured_or_load_shape_counterfactual"},
    "danish_smart_heat_meters": {"adapter": "danish_smart_heat_meters", "input": "data/danish_smart_heat_meters/raw/3_years_3021_smart_heat_meters_residential_denmark.zip", "cache": "data/danish_smart_heat_meters/processed/canonical_device_days.npz", "load_scale": 1.0, "max_sources": 2400, "max_days_per_source": 4, "input_mapping": "load_shape_counterfactual", "source_energy_type": "heat_proxy"},
    "smart_grid_smart_city": {"adapter": "smart_grid_smart_city", "input": "data/smart_grid_smart_city/raw/cdintervalreadingallnoquotes.csv.7z", "cache": "data/smart_grid_smart_city/processed/canonical_device_days_bidirectional_v2.npz", "load_scale": 1.0, "max_sources": 5000, "max_days_per_source": 2, "input_mapping": "measured_or_load_shape_counterfactual"},
    "heapo_heat_pumps": {"adapter": "heapo_heat_pumps", "input": "data/heapo_heat_pumps/raw/heapo_data.zip", "cache": "data/heapo_heat_pumps/processed/canonical_device_days.npz", "load_scale": 1.0, "max_sources": 1408, "max_days_per_source": 8, "input_mapping": "load_shape_counterfactual"},
    "goiener_smart_meters": {"adapter": "goiener_smart_meters", "input": "data/goiener_smart_meters/raw/imp-post.tzst", "cache": "data/goiener_smart_meters/processed/canonical_device_days_bidirectional_v2.npz", "load_scale": 1.0, "max_sources": 5000, "max_days_per_source": 2, "input_mapping": "load_shape_counterfactual"},
    "european_lv_urban_8087": {"adapter": "european_lv_urban_8087", "input": "data/european_lv_urban_8087/raw/Sim_files_190128_OK_V0.zip", "cache": "data/european_lv_urban_8087/processed/canonical_device_days.npz", "load_scale": 1.0, "max_sources": 5000, "max_days_per_source": 2, "input_mapping": "load_shape_counterfactual"},
    "european_lv_rural_2731": {"adapter": "european_lv_rural_2731", "input_dir": "data/european_lv_rural_2731/processed/PQ_csv", "cache": "data/european_lv_rural_2731/processed/canonical_device_days.npz", "load_scale": 1.0, "max_sources": 2731, "max_days_per_source": 4, "input_mapping": "load_shape_counterfactual"},
    "european_lv_urban_35297": {"adapter": "european_lv_urban_35297", "input_dir": "data/european_lv_urban_35297/processed/PQ_csv", "cache": "data/european_lv_urban_35297/processed/canonical_device_days.npz", "load_scale": 1.0, "max_sources": 12000, "max_days_per_source": 1, "input_mapping": "load_shape_counterfactual"},
    "norway_ami_energy_distribution": {"adapter": "norway_ami_energy_distribution", "input_dir": "data/norway_ami_energy_distribution/raw/Energy distribution models with AMI smart meter sensor dataset/data/ami", "cache": "data/norway_ami_energy_distribution/processed/canonical_device_days_measured_only.npz", "load_scale": 1.0, "max_sources": 3000, "max_days_per_source": 2, "input_mapping": "measured_only", "forbid_counterfactual_input": True},
    "camsl_japan_smart_meters": {"adapter": "camsl_japan_smart_meters", "input_dir": "data/camsl_japan_smart_meters/extracted/public/consumption_data/consumption_data", "cache": "data/camsl_japan_smart_meters/processed/canonical_device_days_measured_only.npz", "load_scale": 0.002, "max_sources": 1423, "max_days_per_source": 2, "input_mapping": "measured_only", "forbid_counterfactual_input": True},
    "irish_domestic_smart_meters": {"adapter": "irish_domestic_smart_meters", "input_dir": "data/irish_domestic_smart_meters/raw/SM Data 2.0", "cache": "data/irish_domestic_smart_meters/processed/canonical_device_days_measured_only_kw_label.npz", "load_scale": 1.0, "max_sources": 2989, "max_days_per_source": 2, "input_mapping": "measured_only", "forbid_counterfactual_input": True},
    "opsd_household_data": {"adapter": "opsd_household_data", "input": "data/opsd_household_data/raw/opsd-household_data-2020-04-15/household_data_15min_singleindex.csv", "cache": "data/opsd_household_data/processed/canonical_device_days_measured_only.npz", "load_scale": 4.0, "max_sources": 11, "max_days_per_source": 30, "input_mapping": "measured_only", "forbid_counterfactual_input": True},
    "complete_energy_community": {"adapter": "complete_energy_community", "input": "data/complete_energy_community/EC_EV_dataset.xlsx", "cache": "data/complete_energy_community/processed/canonical_device_days_measured_only.npz", "load_scale": 1.0, "max_sources": 250, "max_days_per_source": 1, "input_mapping": "measured_only", "forbid_counterfactual_input": True},
}


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    scenario_payload = {
        "schema": "broadcast-coordination-scenario-config-v1",
        "canonical_ids": list(SCENARIOS),
        "notation": "The paper shorthand S1A-S6C corresponds to canonical IDs S1-A-S6-C.",
        "sampling_modes": ["unique", "non_unique"],
        "fleet_size_non_unique": FLEET_SIZE,
        "test_seed_count": TEST_SEED_COUNT,
        "partition_seed": PARTITION_SEED,
        "zone_buses": {key: list(value) for key, value in ZONE_BUSES.items()},
        "dataset_sources": DATASET_SOURCES,
        "scenarios": {
            scenario_id: {
                "label": spec["label"],
                "weights": spec["weights"],
                **({"zones": spec["zones"]} if "zones" in spec else {}),
                "seeds": {
                    mode: {
                        "train": SCENARIO_SEEDS[mode][scenario_id][0],
                        "validation": SCENARIO_SEEDS[mode][scenario_id][1],
                        "test_start": SCENARIO_SEEDS[mode][scenario_id][2],
                        "test_formula": "test_start + seed_index",
                    }
                    for mode in ("unique", "non_unique")
                },
                "test_weight_regimes": {
                    "seed_index_0_to_9": "nominal",
                    "seed_index_10_to_19": "dominant_plus_15pp",
                    "seed_index_20_to_29": "dominant_minus_15pp",
                },
            }
            for scenario_id, spec in SCENARIOS.items()
        },
        "allocation": {
            "algorithm": "allocate_counts",
            "minimum_one_device_per_source": True,
            "remainder_rule": "largest fractional residual after reserving one device per source",
            "unique_mode": "largest feasible fleet under source capacities, sampled without replacement",
            "non_unique_mode": "fixed fleet_size_non_unique, bootstrap with replacement if source capacity is insufficient",
        },
        "placement": {
            "default": "retain source profile zone_id and bus_id",
            "spatial_scenarios": "assign each selected source cyclically to its configured zones and buses",
            "bus_rule": "ZONE_BUSES[zone][(local_index // number_of_zones) % len(ZONE_BUSES[zone])]",
        },
    }
    dump(ROOT / "configuration/scenario_config.json", scenario_payload)

    pairs = pair_specs()
    with (ROOT / "index/pairwise_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("pair_id", "pair_index", "dataset_a", "dataset_b", "weight_a", "weight_b", "seed_formula"))
        writer.writeheader()
        for pair in pairs:
            writer.writerow({
                "pair_id": pair["pair_id"],
                "pair_index": pair["pair_index"],
                "dataset_a": pair["dataset_a"],
                "dataset_b": pair["dataset_b"],
                "weight_a": "0.5",
                "weight_b": "0.5",
                "seed_formula": "20260808 + pair_index * 100000 + seed_index",
            })
    dump(ROOT / "configuration/pairwise_seed_config.json", {
        "schema": "broadcast-coordination-pairwise-seed-config-v1",
        "pair_count": len(pairs),
        "pair_order": list(PAIR_ORDER),
        "pair_id_order": [pair["pair_id"] for pair in pairs],
        "pair_seed_base": PAIR_SEED_BASE,
        "pair_seed_stride": PAIR_SEED_STRIDE,
        "seed_index_range": [0, TEST_SEED_COUNT - 1],
        "seed_formula": "pair_seed_base + pair_index * pair_seed_stride + seed_index",
        "weights": [0.5, 0.5],
    })

    dump(ROOT / "configuration/source_metadata.json", {
        "schema": "broadcast-coordination-source-metadata-v1",
        "distribution": "Source data are not included in this release.",
        "sources": {
            dataset: {
                **DATASET_SOURCES[dataset],
                "official_source": SOURCE_URLS[dataset],
            }
            for dataset in ALL_DATASETS
        },
    })
    dump(ROOT / "configuration/preprocessing_config.json", {
        "schema": "broadcast-coordination-canonical-preprocessing-v1",
        "description": "Raw public data to canonical device-day cache parameters used by the final composition pipeline.",
        "canonical_cache_format": "NPZ with source_id, day_id, device_id, zone_id, bus_id and 288-point fields",
        "zone_buses": {key: list(value) for key, value in ZONE_BUSES.items()},
        "fallback_device_parameters": {"capacity_kwh": 10.0, "peak_power_kw": 1.0, "soc_fraction": 0.50},
        "datasets": PREPROCESSING_SPECS,
    })
    dump(ROOT / "configuration/generation_config.json", {
        "schema": "broadcast-coordination-generation-config-v1",
        "preprocessing_entrypoint": "generation/preprocess.py",
        "generator": "generation/dataset_combinations",
        "python": ">=3.10",
        "dependencies": ["numpy", "pandas"],
        "input_root": "data",
        "output_layout": "<output-root>/<sampling-mode>/ieee33/{scenario|pairwise}/...",
        "scenario_partitions": ["train", "validation", "test/seed_00..seed_29"],
        "sampling_modes": ["unique", "non_unique"],
        "partition_seed": PARTITION_SEED,
        "fleet_size_non_unique": FLEET_SIZE,
        "test_seed_count": TEST_SEED_COUNT,
        "pair_seed_base": PAIR_SEED_BASE,
        "pair_seed_stride": PAIR_SEED_STRIDE,
        "profile_bootstrap": "non_unique only when source capacity is insufficient",
        "time_steps": 288,
        "network_mode": "ieee33",
    })

    with (ROOT / "index/scenario_seed_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("scenario_id", "sampling_mode", "partition", "seed_index", "seed", "composition_regime"))
        writer.writeheader()
        for scenario_id in SCENARIOS:
            for mode in ("unique", "non_unique"):
                train_seed, validation_seed, test_start = SCENARIO_SEEDS[mode][scenario_id]
                writer.writerow({"scenario_id": scenario_id, "sampling_mode": mode, "partition": "train", "seed_index": "", "seed": train_seed, "composition_regime": "fixed"})
                writer.writerow({"scenario_id": scenario_id, "sampling_mode": mode, "partition": "validation", "seed_index": "", "seed": validation_seed, "composition_regime": "fixed"})
                for seed_index in range(TEST_SEED_COUNT):
                    regime = "nominal" if seed_index < 10 else ("dominant_plus_15pp" if seed_index < 20 else "dominant_minus_15pp")
                    writer.writerow({"scenario_id": scenario_id, "sampling_mode": mode, "partition": "test", "seed_index": seed_index, "seed": test_start + seed_index, "composition_regime": regime})


if __name__ == "__main__":
    main()
