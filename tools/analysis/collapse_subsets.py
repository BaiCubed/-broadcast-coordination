import csv
import json
from pathlib import Path

from src.extra.nc_excel_experiments.run import _e1_collapse_quality

ROOT = Path("results/e1_mix_s00/E1_scale_boundary_new")
BASELINE = Path("results/e1_full_v2/E1_scale_boundary_new")


def load(path):
    with (path / "summary.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("margin", "N", "p_ctrl", "N_eff"):
            if key in row:
                try:
                    row[key] = float(row[key])
                except (TypeError, ValueError):
                    row[key] = None
    return rows


def show(tag, rows):
    quality = _e1_collapse_quality(rows)
    ratio = quality["variance_ratio_margin_over_N"]
    n_datasets = len({r["dataset"] for r in rows if r.get("arm") == "data_coupled"})
    print("%-42s mixtures=%3d cells=%4d  margin_var=%.5f  N_axis_var=%.5f  ratio=%s  matched=%s" % (
        tag, n_datasets, quality["cells"],
        quality["margin_axis"]["mean_between_dataset_variance"] or float("nan"),
        quality["physical_N_axis"]["mean_between_dataset_variance"] or float("nan"),
        ("%.3f" % ratio) if ratio else "NA", quality["comparison_is_matched"]))
    return quality


mix = load(ROOT)
print("=== mixtures (seed_00) ===")
show("all 119 mixtures", mix)
show("the 14 predefined scenarios only", [r for r in mix if r["dataset"].startswith("mix_S")])
show("the 105 pairwise mixtures only", [r for r in mix if r["dataset"].startswith("mix_P")])

big = [r for r in mix if r["N"] and r["N"] >= 50]
show("cells with N>=50 only", big)

if (BASELINE / "summary.csv").is_file():
    print()
    print("=== baseline, 15 datasets (e1_full_v2) ===")
    base = load(BASELINE)
    show("all 15 datasets", base)
    show("cells with N>=50 only", [r for r in base if r["N"] and r["N"] >= 50])
else:
    print()
    print("baseline summary.csv is missing: %s" % (BASELINE / "summary.csv"))

print()
print("=== spread of p_ctrl across mixtures, on each axis ===")
by_dataset = {}
for row in mix:
    if row.get("arm") != "data_coupled" or row["N"] is None:
        continue
    by_dataset.setdefault(row["dataset"], []).append(row)
spans = []
for name, rows in by_dataset.items():
    ns = [r["N"] for r in rows]
    spans.append((max(ns), name))
spans.sort()
print("   upper N of the 119 mixtures: min %d, median %d, max %d" % (
    spans[0][0], spans[len(spans) // 2][0], spans[-1][0]))
print("   mixtures whose upper N is 3000: %d" % sum(1 for s in spans if s[0] >= 3000))
