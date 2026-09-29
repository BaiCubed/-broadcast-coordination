from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/mixed_populations"
OUTPUT = ROOT / "reproduced_tables"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], names: tuple[str, ...]) -> float | None:
    for name in names:
        if row.get(name, "") != "":
            return float(row[name])
    return None


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    scenarios = [_read(path) for path in sorted((DATA / "mixed_scenarios").glob("*.csv"))]
    pairwise = [_read(path) for path in sorted((DATA / "pairwise").glob("*.csv"))]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rows in scenarios:
        for row in rows:
            value = _number(row, ("mean_reduction_pct", "curtailment_reduction_pct"))
            if value is not None:
                grouped[(row.get("scenario", "unknown"), row.get("algorithm_label", row.get("algorithm", "unknown")))].append(value)
    scenario_rows = [
        {"scenario": scenario, "algorithm": algorithm, "n": len(values), "mean_reduction_pct": sum(values) / len(values)}
        for (scenario, algorithm), values in sorted(grouped.items())
    ]
    _write(OUTPUT / "scenario_algorithm_summary.csv", scenario_rows)

    pair_values: dict[str, list[float]] = defaultdict(list)
    for rows in pairwise:
        for row in rows:
            value = _number(row, ("curtailment_reduction_pct", "mean_reduction_pct"))
            if value is not None:
                pair_values[row.get("pair_id", "unknown")].append(value)
    pair_rows = [
        {"pair_id": pair_id, "n": len(values), "mean_reduction_pct": sum(values) / len(values)}
        for pair_id, values in sorted(pair_values.items())
    ]
    _write(OUTPUT / "pairwise_summary.csv", pair_rows)

    manifest = {
        "source": "datasets/mixed_populations",
        "scenario_files": len(scenarios),
        "pairwise_files": len(pairwise),
        "outputs": ["scenario_algorithm_summary.csv", "pairwise_summary.csv"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(OUTPUT / "scenario_algorithm_summary.csv")
    print(OUTPUT / "pairwise_summary.csv")


if __name__ == "__main__":
    main()
