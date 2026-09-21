from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("NC_ROOT", Path(__file__).resolve().parents[2]))
CFG = ROOT / "src/extra/nc_excel_experiments/configs"
BASE = {
    "E1": CFG / "e1_full_v2.yaml",
    "E2": CFG / "e2_full_v2.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write an E1 or E2 protocol for the mixture fleets of one seed.")
    parser.add_argument("--experiment", choices=("E1", "E2"), required=True)
    parser.add_argument("--pools-manifest",
                        default=str(ROOT / "data/dataset_combination_pools/pools_manifest.json"))
    parser.add_argument("--names", default="", help="comma separated; when given, only these datasets are used, otherwise the whole manifest")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    protocol = yaml.safe_load(BASE[args.experiment].read_text(encoding="utf-8"))
    if args.names:
        names = [value.strip() for value in args.names.split(",") if value.strip()]
    else:
        manifest = json.loads(Path(args.pools_manifest).read_text(encoding="utf-8"))
        names = [entry["name"] for entry in manifest["entries"]]

    key = args.experiment.lower()
    protocol[key]["datasets"] = names
    protocol["execution"]["dataset_workers"] = int(args.workers)
    protocol["execution"]["experiments"] = [args.experiment]
    protocol["results_root"] = args.results_root
    protocol["primary_dataset"] = names[0]
    protocol["primary_config"] = "results/%s_ieee33_real_load/config/default_weak_correlation.yaml" % names[0]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(protocol, sort_keys=True, allow_unicode=True), encoding="utf-8")
    print(json.dumps({"experiment": args.experiment, "datasets": len(names),
                      "workers": args.workers, "results_root": args.results_root,
                      "protocol": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
