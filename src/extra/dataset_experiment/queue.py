from __future__ import annotations

import argparse
import json
import time

from .run import DATASETS, dataset_results_root, run_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the verified external-dataset experiments sequentially in the background.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS))
    args = parser.parse_args()
    datasets = args.datasets or sorted(DATASETS)
    started = time.time()
    print(json.dumps({"event": "queue_started", "datasets": datasets}, ensure_ascii=False), flush=True)
    for dataset in datasets:
        print(json.dumps({"event": "dataset_started", "dataset": dataset}, ensure_ascii=False), flush=True)
        try:
            result = run_dataset(dataset)
        except Exception as exc:
            result = {"dataset": dataset, "status": "failed", "error": repr(exc)}
            root = dataset_results_root(dataset)
            root.mkdir(parents=True, exist_ok=True)
            (root / "run_status.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"event": "dataset_finished", **result}, ensure_ascii=False), flush=True)
    print(json.dumps({"event": "queue_finished", "elapsed_seconds": time.time() - started}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
