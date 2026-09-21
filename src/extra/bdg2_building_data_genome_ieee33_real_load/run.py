if __name__ == "__main__":
    from . import DATASET
    import sys
    from src.extra.dataset_experiment.run import run_dataset
    import json
    preflight = "--preflight-only" in sys.argv
    stress = "--network-stress-only" in sys.argv
    print(json.dumps(run_dataset(DATASET, modes=("network_stress",) if stress else ("network_stress", "weak_correlation"), preflight_only=preflight), ensure_ascii=False))
