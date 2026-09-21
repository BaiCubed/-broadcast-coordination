import json
from pathlib import Path

import numpy as np
import yaml

from src.extra.dataset_experiment.canonical_adapter import load_canonical_device_day_pool
from src.extra.ieee33_device_day_simulation.config_loader import load_config

CONFIG = Path("results/bdg1_building_data_genome_ieee33_real_load/config/default_weak_correlation.yaml")
CACHE = Path("data/bdg1_building_data_genome/processed/canonical_device_days.npz")

config = load_config(CONFIG)
spec = config["population"]["canonical_adapter"]
print("configured cache:", repr(spec.get("cache")))
spec["cache"] = str(CACHE.resolve())

pool = load_canonical_device_day_pool(config)
print("BDG1 pool: records=%d  unique_sources=%d" % (len(pool.records), pool.source_count))
print("cache written:", CACHE, CACHE.stat().st_size if CACHE.exists() else "MISSING")

index = {}
for i, r in enumerate(pool.records):
    day = np.datetime64(int(r.timestamp[0]), "s").astype("datetime64[D]")
    index[(r.source_device_id, str(day))] = i

from src.extra.dataset_combinations.catalog import ProfileCatalog
from src.extra.dataset_combinations.constants import BDG1, PARTITION_SEED

catalog = ProfileCatalog(Path("data"), PARTITION_SEED)
parts = catalog.partitions(BDG1)
total = hit = 0
missing_examples = []
for part in (parts.train, parts.validation, parts.test):
    for rec in part.records:
        total += 1
        column, day = rec.source_locator.split("|", 1)
        if (column, day) in index:
            hit += 1
        elif len(missing_examples) < 5:
            missing_examples.append(rec.source_locator)
print("BDG1 profiles in the mixture manifests: %d; matched in the pool by (source, day): %d (%.2f%%)" % (
    total, hit, 100.0 * hit / max(total, 1)))
if missing_examples:
    print("unmatched examples:", missing_examples)

Path("tools/_bdg1_join_report.json").write_text(json.dumps({
    "pool_records": len(pool.records), "pool_sources": int(pool.source_count),
    "combo_profiles": total, "joined": hit,
    "join_rate": hit / max(total, 1), "missing_examples": missing_examples,
}, indent=2), encoding="utf-8")
