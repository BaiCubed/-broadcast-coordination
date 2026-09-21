from __future__ import annotations
import json, shutil, sys, time
from pathlib import Path
import yaml

from src.extra.dataset_experiment.run import (
    DATASETS, _capacity_multiplier, _fit_experiment_to_pool, _mode_config,
    _prepare_config, _write_yaml, dataset_results_root,
)
from src.extra.ieee33_device_day_simulation.config_loader import load_config
from src.extra.ieee33_device_day_simulation.population.device_day_loader import load_device_day_pool


def build(dataset: str) -> dict:
    started = time.time()
    root = dataset_results_root(dataset) / "coverage_fix"
    config_path = _prepare_config(dataset, root)
    config = load_config(config_path)
    pool = load_device_day_pool(config)
    fleet = _fit_experiment_to_pool(config_path, dataset, pool)
    config = load_config(config_path)
    multiplier = _capacity_multiplier(config, pool, target_loading=0.90)
    stress_default = _mode_config(config_path, "network_stress", multiplier)

    out_cfg = root / "network_constrained_new" / "config"
    if out_cfg.exists():
        shutil.rmtree(out_cfg)
    shutil.copytree(root / "config", out_cfg)
    payload = yaml.safe_load(stress_default.read_text(encoding="utf-8"))
    control = yaml.safe_load((out_cfg / str(payload["control"])).read_text(encoding="utf-8"))
    control["network_feedback"] = True
    _write_yaml(out_cfg / "control_network_constrained_new.yaml", control)
    experiment = yaml.safe_load((out_cfg / str(payload["experiment"])).read_text(encoding="utf-8"))
    experiment["force_network_feedback"] = True
    experiment["experiment_variant"] = "paired_network_constrained_new"
    experiment["constraint_calibration"] = {
        "calibration_strategy": "SKIPPED_config_only_rebuild",
        "note": "network_capacity_multiplier is unused here; the value follows the 0.90 target-loading calibration",
        "selected_multiplier": float(multiplier),
    }
    _write_yaml(out_cfg / str(payload["experiment"]), experiment)
    payload["control"] = "control_network_constrained_new.yaml"
    _write_yaml(out_cfg / "default_network_constrained_new.yaml", payload)
    return {
        "dataset": dataset, "seconds": round(time.time() - started, 1),
        "selected_main_resources": fleet["selected_main_resources"],
        "balanced_unique_limit": fleet["balanced_unique_limit"],
        "network_capacity_multiplier": round(float(multiplier), 6),
    }


if __name__ == "__main__":
    out = build(sys.argv[1])
    print(json.dumps(out, ensure_ascii=False), flush=True)
