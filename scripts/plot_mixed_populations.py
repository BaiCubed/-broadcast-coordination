from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/mixed_populations"
OUTPUT = ROOT / "reproduced_figures"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric(row: dict[str, str]) -> float | None:
    raw = row.get("mean_reduction_pct", row.get("curtailment_reduction_pct", ""))
    return float(raw) if raw else None


def _save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = [row for path in sorted((DATA / "mixed_scenarios").glob("*.csv")) for row in _read(path)]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = _metric(row)
        if value is not None:
            grouped[(row.get("scenario", "unknown"), row.get("algorithm_label", row.get("algorithm", "unknown")))].append(value)
    scenarios = sorted({key[0] for key in grouped})
    algorithms = sorted({key[1] for key in grouped})
    if not scenarios or not algorithms:
        raise SystemExit("No derived scenario CSV data found")

    matrix = np.full((len(algorithms), len(scenarios)), np.nan)
    for i, algorithm in enumerate(algorithms):
        for j, scenario in enumerate(scenarios):
            values = grouped.get((scenario, algorithm), [])
            if values:
                matrix[i, j] = np.mean(values)
    fig, axis = plt.subplots(figsize=(max(10, len(scenarios) * 0.75), 6.5))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis")
    axis.set_xticks(np.arange(len(scenarios)), scenarios, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(algorithms)), algorithms)
    axis.set_xlabel("Scenario")
    axis.set_ylabel("Algorithm")
    axis.set_title("Derived-data mean curtailment reduction (%)")
    fig.colorbar(image, ax=axis, label="Reduction (%)")
    _save(fig, "scenario_algorithm_heatmap")

    pair_rows = [row for path in sorted((DATA / "pairwise").glob("*.csv")) for row in _read(path)]
    pair_grouped: dict[str, list[float]] = defaultdict(list)
    for row in pair_rows:
        value = _metric(row)
        if value is not None:
            pair_grouped[row.get("pair_id", "unknown")].append(value)
    pair_ids = sorted(pair_grouped)
    pair_values = [float(np.mean(pair_grouped[pair_id])) for pair_id in pair_ids]
    fig, axis = plt.subplots(figsize=(16, 5.5))
    axis.bar(np.arange(len(pair_ids)), pair_values, color="#1f6f8b")
    axis.set_xticks(np.arange(len(pair_ids)), pair_ids, rotation=90)
    axis.set_ylabel("Mean reduction (%)")
    axis.set_title("Pairwise mixed-population summary")
    axis.grid(axis="y", alpha=0.25)
    _save(fig, "pairwise_reduction_summary")
    print(OUTPUT)


if __name__ == "__main__":
    main()
