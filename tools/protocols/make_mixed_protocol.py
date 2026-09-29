from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("NC_ROOT", Path(__file__).resolve().parents[2]))
CFG = ROOT / "src/extra/population_experiments/configs"

BASE: dict[str, Path] = {
    "E1": CFG / "E1_population_scale.yaml",
    "E2": CFG / "E2_controller_synchronization.yaml",
    "phase_coherence": CFG / "phase_coherence.yaml",
    "E4": CFG / "E4_controller_drift.yaml",
    "structured_availability": CFG / "structured_availability.yaml",
    "availability_second_family": CFG / "availability_second_family.yaml",
    "long_horizon_state": CFG / "long_horizon_state.yaml",
    "response_mechanisms": CFG / "response_mechanisms_and_capacity_concentration.yaml",
    "capacity_concentration": CFG / "response_mechanisms_and_capacity_concentration.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a protocol for one experiment over the constructed mixed populations.")
    parser.add_argument("--experiment", choices=sorted(BASE), required=True)
    parser.add_argument("--pools-manifest",
                        default=str(ROOT / "data/dataset_combination_pools/pools_manifest.json"))
    parser.add_argument("--names", default="", help="comma separated; when given, only these datasets are used")
    parser.add_argument("--seed-index", type=int, default=None,
                        help="without --names, take the 119 mixed populations of this seed from the manifest")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = BASE[args.experiment]
    protocol = yaml.safe_load(base.read_text(encoding="utf-8"))

    if args.names:
        names = [v.strip() for v in args.names.split(",") if v.strip()]
    else:
        manifest = json.loads(Path(args.pools_manifest).read_text(encoding="utf-8"))
        entries = manifest["entries"]
        if args.seed_index is not None:
            entries = [e for e in entries if e["seed_index"] == args.seed_index]
        names = [e["name"] for e in entries]
    if not names:
        raise SystemExit("empty dataset list; check --seed-index / --names")

    key = args.experiment.lower()
    if key not in protocol:
        raise SystemExit(f"{base} has no section '{key}'")
    protocol[key]["datasets"] = names
    protocol["execution"]["dataset_workers"] = int(args.workers)
    protocol["execution"]["experiments"] = [args.experiment]
    protocol["results_root"] = args.results_root
    protocol["primary_dataset"] = names[0]
    protocol["primary_config"] = (
        "results/%s_ieee33_real_load/config/default_weak_correlation.yaml" % names[0])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(protocol, sort_keys=True, allow_unicode=True),
                   encoding="utf-8")
    print(json.dumps({"experiment": args.experiment,
                      "base": str(base), "datasets": len(names),
                      "workers": args.workers, "results_root": args.results_root,
                      "protocol": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
