from __future__ import annotations
import json, sys, time
from pathlib import Path
from src.extra.ieee33_device_day_simulation.figures import run_e22_ieee69_complexity as e22
from src.extra.ieee33_device_day_simulation.figures import run_e22_trained_eps_comparison as direct

if __name__ == "__main__":
    dataset = sys.argv[1]
    output = Path(direct.OUTPUT_ROOT)
    output.mkdir(parents=True, exist_ok=True)
    cases = e22._network_cases()
    stress_modes = tuple(e22.STRESS_MODES)
    t = time.time()
    path, meta = direct._ensure_native_model(dataset, output, cases, stress_modes, retrain_all=False)
    print(json.dumps({
        "dataset": dataset, "model": str(path), "seconds": round(time.time()-t, 1),
        "reused": bool(meta.get("model_reused", False)),
        "internal_training_r2": meta.get("internal_training_r2"),
        "training_samples": meta.get("training_samples"),
    }, ensure_ascii=False), flush=True)
