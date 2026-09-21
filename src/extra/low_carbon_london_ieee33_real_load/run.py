from __future__ import annotations

import argparse
import json

from src.extra.dataset_experiment.run import run_dataset

from . import DATASET


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Low Carbon London through the unchanged IEEE33 real-load experiment.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--network-stress-only", action="store_true")
    args = parser.parse_args()
    modes = ("network_stress",) if args.network_stress_only else ("network_stress", "weak_correlation")
    print(json.dumps(run_dataset(DATASET, modes=modes, preflight_only=args.preflight_only), ensure_ascii=False))


if __name__ == "__main__":
    main()
