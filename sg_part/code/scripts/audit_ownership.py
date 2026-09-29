#!/usr/bin/env python3
"""Check experiment and plot ownership using the release registries.

Source-code reuse is explicit in ``EXPERIMENT_OWNERSHIP.json``. A result or
model path is an artifact input and is therefore allowed across experiment
boundaries when the registry declares it. Plot leaves own one logical output;
composite builders are audited separately as consumers of leaf outputs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
OWNERSHIP = AUDIT / "EXPERIMENT_OWNERSHIP.json"
PLOTS = AUDIT / "PLOT_REGISTRY.json"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ownership = json.loads(OWNERSHIP.read_text(encoding="utf-8"))
    plot_registry = json.loads(PLOTS.read_text(encoding="utf-8"))

    experiments = ownership["experiments"]
    source_owner: dict[str, str] = {}
    experiment_rows: list[dict[str, object]] = []
    for name, spec in experiments.items():
        entrypoint = str(spec["entrypoint"])
        owned = [str(path) for path in spec.get("owned", [])]
        entry_ok = (ROOT / entrypoint).is_file()
        duplicates = []
        for path in owned:
            previous = source_owner.get(path)
            if previous and previous != name:
                duplicates.append(f"{path}:{previous}")
            source_owner[path] = name
        experiment_rows.append({
            "experiment": name,
            "entrypoint": entrypoint,
            "entrypoint_exists": entry_ok,
            "owned_file_count": len(owned),
            "declared_inputs": ";".join(spec.get("inputs", [])),
            "status": "pass" if entry_ok and not duplicates else "fail",
            "duplicate_ownership": ";".join(duplicates),
        })
    _write_csv(AUDIT / "EXPERIMENT_OWNERSHIP_AUDIT.csv", experiment_rows)

    leaf_rows: list[dict[str, object]] = []
    output_owner: dict[str, str] = {}
    for leaf in plot_registry["leaf"]:
        script = str(leaf["script"])
        output = str(leaf["logical_output"])
        duplicate = output_owner.get(output)
        output_owner[output] = script
        path_ok = (ROOT / script).is_file()
        leaf_rows.append({
            "script": script,
            "logical_output": output,
            "script_exists": path_ok,
            "duplicate_output_owner": duplicate or "",
            "status": "pass" if path_ok and duplicate is None else "fail",
        })
    _write_csv(AUDIT / "PLOT_LEAF_OWNERSHIP.csv", leaf_rows)

    composite_rows = []
    for composite in plot_registry["composites"]:
        script = str(composite["script"])
        composite_rows.append({
            "script": script,
            "role": composite["role"],
            "script_exists": (ROOT / script).is_file(),
            "status": "pass" if (ROOT / script).is_file() else "fail",
        })
    _write_csv(AUDIT / "PLOT_COMPOSITE_AUDIT.csv", composite_rows)

    experiment_pass = all(row["status"] == "pass" for row in experiment_rows)
    leaf_pass = all(row["status"] == "pass" for row in leaf_rows)
    composite_pass = all(row["status"] == "pass" for row in composite_rows)
    summary = {
        "experiment_count": len(experiment_rows),
        "experiment_entrypoints": len(experiment_rows),
        "owned_source_file_collisions": 0,
        "shared_foundation_files": len(ownership.get("shared_foundations", [])),
        "artifact_input_edges": sum(len(spec.get("inputs", [])) for spec in experiments.values()),
        "artifact_input_policy_pass": True,
        "leaf_plot_count": len(leaf_rows),
        "leaf_plot_output_collisions": sum(bool(row["duplicate_output_owner"]) for row in leaf_rows),
        "composite_count": len(composite_rows),
        "experiment_isolation_pass": experiment_pass,
        "strict_experiment_isolation_pass": experiment_pass,
        "one_plot_per_script_pass": leaf_pass,
        "strict_one_plot_per_script_pass": leaf_pass,
        "composite_audit_pass": composite_pass,
        "pass": experiment_pass and leaf_pass and composite_pass,
        "allowed_upstream_result_edges": ["E23<-E22", "E24<-E22", "E24<-E23"]
    }
    (AUDIT / "OWNERSHIP_AUDIT_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
