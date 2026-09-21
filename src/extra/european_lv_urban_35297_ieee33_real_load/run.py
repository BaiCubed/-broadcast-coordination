from __future__ import annotations
import json
import sys
from . import DATASET
from src.extra.dataset_experiment.run import run_dataset

if __name__ == "__main__":
    modes = ("network_stress",) if "--network-stress-only" in sys.argv else ("network_stress", "weak_correlation")
    print(json.dumps(run_dataset(DATASET, modes=modes, preflight_only="--preflight-only" in sys.argv), ensure_ascii=False))
