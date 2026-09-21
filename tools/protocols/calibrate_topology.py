from __future__ import annotations
import json, sys, time
from pathlib import Path

from src.extra.dataset_experiment.run import _capacity_multiplier
from src.extra.ieee33_device_day_simulation.config_loader import load_config
from src.extra.ieee33_device_day_simulation.population.device_day_loader import load_device_day_pool
from src.extra.nc_excel_experiments.run import _dataset_config
from src.extra.nc_excel_experiments import topology as T

CACHE = Path("results") / "_topology_calibration.json"


def main() -> None:
    dataset = sys.argv[1]
    topologies = sys.argv[2:] or ["ieee33_published_config", "ieee33", "ieee69", "ieee123"]
    config_path, _ = _dataset_config(dataset)
    out = {}
    for label in topologies:
        t = time.time()
        config = load_config(config_path)
        if label != "ieee33_published_config":
            config["network"] = T.derive_network_config(label)
            config["zones"] = T.derive_zones_config(config["zones"], label)
        pool = load_device_day_pool(config)
        multiplier = _capacity_multiplier(config, pool, target_loading=0.90)
        out[T.calibration_key(dataset, label)] = {
            "dataset": dataset,
            "topology": label,
            "network_capacity_multiplier": float(multiplier),
            "network_name": config["network"]["name"],
            "bus_count": len({int(r[1]) for r in config["network"]["branches"]}) + 1,
            "target_loading": 0.90,
            "seconds": round(time.time() - t, 1),
        }
        print(json.dumps(out[T.calibration_key(dataset, label)], ensure_ascii=False), flush=True)
    part = CACHE.with_suffix(f".{dataset}.json")
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
