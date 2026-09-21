"""独立组合器使用的数据集、场景、网络位置和 seed 常量。"""

from __future__ import annotations


BDG1 = "bdg1_building_data_genome"
BDG2 = "bdg2_building_data_genome"
CEC = "complete_energy_community"
DANISH = "danish_smart_heat_meters"
EU_RURAL = "european_lv_rural_2731"
EU_35297 = "european_lv_urban_35297"
GOIENER = "goiener_smart_meters"
HEAPO = "heapo_heat_pumps"
LCL = "low_carbon_london"
NORWAY = "norway_ami_energy_distribution"
CAMSL = "camsl_japan_smart_meters"
EU_8087 = "european_lv_urban_8087"
IRISH = "irish_domestic_smart_meters"
OPSD = "opsd_household_data"
SGSC = "smart_grid_smart_city"

ALL_DATASETS = (
    BDG1, BDG2, CEC, DANISH, EU_RURAL, EU_35297, GOIENER, HEAPO,
    LCL, NORWAY, CAMSL, EU_8087, IRISH, OPSD, SGSC,
)

PAIR_ORDER = (
    BDG1, BDG2, LCL, CAMSL, IRISH, GOIENER, SGSC, DANISH, HEAPO,
    EU_RURAL, EU_35297, EU_8087, NORWAY, CEC, OPSD,
)

SHORT_NAMES = {
    BDG1: "BDG1", BDG2: "BDG2", LCL: "LCL", CAMSL: "CAMSL",
    IRISH: "Irish", GOIENER: "GoiEner", SGSC: "SGSC", DANISH: "Danish",
    HEAPO: "HEAPO", EU_RURAL: "EU-Rural", EU_35297: "EU-35k",
    EU_8087: "EU-8k", NORWAY: "Norway", CEC: "CEC", OPSD: "OPSD",
}

DATASET_SOURCES = {
    BDG1: {"kind": "bdg1_csv", "path": "bdg1_building_data_genome/raw/temp_open_utc_complete.csv"},
    BDG2: {"kind": "canonical_npz", "path": "bdg2_building_data_genome/processed/canonical_device_days.npz"},
    CEC: {"kind": "canonical_npz", "path": "complete_energy_community/processed/canonical_device_days_measured_only.npz"},
    DANISH: {"kind": "canonical_npz", "path": "danish_smart_heat_meters/processed/canonical_device_days.npz"},
    EU_RURAL: {"kind": "canonical_npz", "path": "european_lv_rural_2731/processed/canonical_device_days.npz"},
    EU_35297: {"kind": "canonical_npz", "path": "european_lv_urban_35297/processed/canonical_device_days.npz"},
    GOIENER: {"kind": "canonical_npz", "path": "goiener_smart_meters/processed/canonical_device_days_bidirectional_v2.npz"},
    HEAPO: {"kind": "canonical_npz", "path": "heapo_heat_pumps/processed/canonical_device_days.npz"},
    LCL: {"kind": "canonical_npz", "path": "low_carbon_london/processed/canonical_device_days_bidirectional_v2.npz"},
    NORWAY: {"kind": "canonical_npz", "path": "norway_ami_energy_distribution/processed/canonical_device_days_measured_only.npz"},
    CAMSL: {"kind": "canonical_npz", "path": "camsl_japan_smart_meters/processed/canonical_device_days_measured_only.npz"},
    EU_8087: {"kind": "canonical_npz", "path": "european_lv_urban_8087/processed/canonical_device_days.npz"},
    IRISH: {"kind": "canonical_npz", "path": "irish_domestic_smart_meters/processed/canonical_device_days_measured_only_kw_label.npz"},
    OPSD: {"kind": "canonical_npz", "path": "opsd_household_data/processed/canonical_device_days_measured_only.npz"},
    SGSC: {"kind": "canonical_npz", "path": "smart_grid_smart_city/processed/canonical_device_days_bidirectional_v2.npz"},
}

ZONE_BUSES = {
    "zone_1": (2, 3, 4, 5),
    "zone_2": (6, 7, 8, 9),
    "zone_3": (10, 11, 12, 13),
    "zone_4": (14, 15, 16, 17, 18),
    "zone_5": (19, 20, 21, 22, 23, 24),
    "zone_6": (25, 26, 27, 28, 29, 30, 31, 32, 33),
}
ZONE_ORDER = tuple(ZONE_BUSES)


def _equal_weights(datasets: tuple[str, ...]) -> dict[str, float]:
    return {dataset: 1.0 / len(datasets) for dataset in datasets}


SCENARIOS = {
    "S1-A": {"label": "Similar residential", "weights": _equal_weights((LCL, CAMSL, IRISH))},
    "S1-B": {"label": "Building-heat pump-residential", "weights": _equal_weights((BDG2, HEAPO, SGSC))},
    "S2-A": {"label": "LV network and AMI", "weights": {EU_35297: 0.40, EU_8087: 0.25, EU_RURAL: 0.15, NORWAY: 0.10, GOIENER: 0.10}},
    "S2-B": {"label": "Building-residential-thermal-DER", "weights": {BDG2: 0.35, LCL: 0.25, DANISH: 0.15, HEAPO: 0.15, CEC: 0.10}},
    "S3-A": {"label": "Ten-source electricity mix", "weights": {LCL: 0.18, SGSC: 0.14, CAMSL: 0.12, IRISH: 0.10, GOIENER: 0.08, NORWAY: 0.08, EU_RURAL: 0.07, EU_35297: 0.10, EU_8087: 0.08, HEAPO: 0.05}},
    "S3-B": {"label": "Ten-source multi-sector mix", "weights": {BDG1: 0.10, BDG2: 0.15, CEC: 0.10, DANISH: 0.10, HEAPO: 0.10, LCL: 0.10, SGSC: 0.10, EU_RURAL: 0.08, NORWAY: 0.12, OPSD: 0.05}},
    "S4-A": {"label": "All datasets equally weighted", "weights": _equal_weights(ALL_DATASETS)},
    "S4-B": {"label": "All dataset types equally weighted", "weights": {BDG1: 0.10, BDG2: 0.10, LCL: 0.04, CAMSL: 0.04, IRISH: 0.04, GOIENER: 0.04, SGSC: 0.04, DANISH: 0.10, HEAPO: 0.10, EU_RURAL: 0.05, EU_35297: 0.05, EU_8087: 0.05, NORWAY: 0.05, CEC: 0.15, OPSD: 0.05}},
    "S5-A": {"label": "Residential-dominant long tail", "weights": {LCL: 0.13, CAMSL: 0.13, IRISH: 0.13, GOIENER: 0.13, SGSC: 0.13, EU_RURAL: 0.05, EU_35297: 0.05, EU_8087: 0.05, NORWAY: 0.05, BDG1: 0.025, BDG2: 0.025, DANISH: 0.025, HEAPO: 0.025, CEC: 0.03, OPSD: 0.02}},
    "S5-B": {"label": "Building-thermal-dominant long tail", "weights": {BDG1: 0.20, BDG2: 0.20, DANISH: 0.15, HEAPO: 0.15, LCL: 0.03, CAMSL: 0.03, IRISH: 0.03, GOIENER: 0.03, SGSC: 0.03, EU_RURAL: 0.025, EU_35297: 0.025, EU_8087: 0.025, NORWAY: 0.025, CEC: 0.03, OPSD: 0.02}},
    "S5-C": {"label": "Network-DER-dominant long tail", "weights": {EU_RURAL: 0.1125, EU_35297: 0.1125, EU_8087: 0.1125, NORWAY: 0.1125, CEC: 0.20, OPSD: 0.05, LCL: 0.04, CAMSL: 0.04, IRISH: 0.04, GOIENER: 0.04, SGSC: 0.04, BDG1: 0.025, BDG2: 0.025, DANISH: 0.025, HEAPO: 0.025}},
    "S6-A": {"label": "Spatially clustered LV networks", "weights": {EU_35297: 0.35, EU_8087: 0.25, EU_RURAL: 0.20, NORWAY: 0.20}, "zones": {EU_35297: ("zone_1", "zone_2"), EU_8087: ("zone_3", "zone_4"), NORWAY: ("zone_5",), EU_RURAL: ("zone_6",)}},
    "S6-B": {"label": "Spatial cross-sector congestion", "weights": {BDG2: 0.35, HEAPO: 0.20, DANISH: 0.15, LCL: 0.20, SGSC: 0.10}, "zones": {BDG2: ("zone_1", "zone_2"), LCL: ("zone_3",), SGSC: ("zone_4",), HEAPO: ("zone_5",), DANISH: ("zone_6",)}},
    "S6-C": {"label": "Spatial DER and bidirectional mix", "weights": {CEC: 0.30, SGSC: 0.30, OPSD: 0.05, IRISH: 0.20, NORWAY: 0.15}, "zones": {NORWAY: ("zone_1",), IRISH: ("zone_2", "zone_3"), OPSD: ("zone_4",), SGSC: ("zone_5",), CEC: ("zone_6",)}},
}

SCENARIO_SEEDS = {
    "non_unique": {
        "S1-A": (3600000, 3602000, 4657375), "S1-B": (3610000, 3612000, 4692405),
        "S2-A": (3620000, 3622000, 4699504), "S2-B": (3630000, 3632000, 4678152),
        "S3-A": (3640000, 3642000, 4639045), "S3-B": (3650000, 3652000, 4692942),
        "S4-A": (3660000, 3662000, 4655942), "S4-B": (3670000, 3672000, 4676323),
        "S5-A": (3680000, 3682000, 4621945), "S5-B": (3690000, 3692000, 4689632),
        "S5-C": (3700000, 3702000, 4697256), "S6-A": (3710000, 3712000, 4675093),
        "S6-B": (3720000, 3722000, 4645607), "S6-C": (3730000, 3732000, 4641334),
    },
    "unique": {
        "S1-A": (5600000, 5602000, 6657375), "S1-B": (5610000, 5612000, 6692405),
        "S2-A": (5620000, 5622000, 6699504), "S2-B": (5630000, 5632000, 6678152),
        "S3-A": (5640000, 5642000, 6639045), "S3-B": (5650000, 5652000, 6692942),
        "S4-A": (5660000, 5662000, 6655942), "S4-B": (5670000, 5672000, 6676323),
        "S5-A": (5680000, 5682000, 6621945), "S5-B": (5690000, 5692000, 6689632),
        "S5-C": (5700000, 5702000, 6697256), "S6-A": (5710000, 5712000, 6675093),
        "S6-B": (5720000, 5722000, 6645607), "S6-C": (5730000, 5732000, 6641334),
    },
}

PARTITION_SEED = 20260714
PAIR_SEED_BASE = 20260808
PAIR_SEED_STRIDE = 100000
FLEET_SIZE = 5000
TEST_SEED_COUNT = 30
