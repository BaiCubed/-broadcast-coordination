import csv
import json
from pathlib import Path

RESULTS = Path("results")


def read_csv(path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def median(values):
    values = sorted(values)
    return values[len(values) // 2] if values else float("nan")


rows = []
for directory in sorted(RESULTS.glob("e1_mix_s*")):
    output = directory / "E1_scale_boundary_new"
    status_path = directory / "run_status.json"
    if not (output / "collapse_quality.json").is_file() or not status_path.is_file():
        continue
    status = json.loads(status_path.read_text())["experiments"]["E1"]
    if status.get("status") != "completed":
        continue
    quality = json.loads((output / "collapse_quality.json").read_text())
    datasets = read_csv(output / "dataset_summary.csv")
    neff = [r for r in read_csv(output / "neff_boundaries.csv") if r["arm"] == "data_coupled"]
    reached = [float(r["N_star_eff"]) for r in neff if r["status"] == "reached" and r["N_star_eff"]]
    summary = read_csv(output / "summary.csv")

    ratios = {"data_coupled": [], "decoupled": []}
    for row in summary:
        arm = row.get("arm")
        if arm not in ratios:
            continue
        try:
            n, n_eff = float(row["N"]), float(row["N_eff"])
        except (KeyError, ValueError, TypeError):
            continue
        if n > 0:
            ratios[arm].append(n_eff / n)

    rows.append({
        "seed": directory.name.replace("e1_mix_s", ""),
        "datasets": status.get("completed_datasets"),
        "ratio": quality["variance_ratio_margin_over_N"],
        "margin_var": quality["margin_axis"]["mean_between_dataset_variance"],
        "n_var": quality["physical_N_axis"]["mean_between_dataset_variance"],
        "reached": len(reached),
        "n_star_med": median(reached),
        "beta_coupled": median([float(r["beta_data_coupled"]) for r in datasets if r.get("beta_data_coupled")]),
        "beta_decoupled": median([float(r["beta_decoupled"]) for r in datasets if r.get("beta_decoupled")]),
        "neff_ratio_coupled": median(ratios["data_coupled"]),
        "neff_ratio_decoupled": median(ratios["decoupled"]),
    })

print("seed  mixtures  boundary_reached  median_N*_eff  beta_data  beta_shuffled  N_eff/N_data  shuffled  margin_var  N_axis_var  ratio")
for r in rows:
    print("s%-4s %4s   %3d/%s     %6.0f   %7.4f %7.4f    %5.3f  %5.3f   %8.5f %8.5f  %5.2f" % (
        r["seed"], r["datasets"], r["reached"], r["datasets"], r["n_star_med"],
        r["beta_coupled"], r["beta_decoupled"],
        r["neff_ratio_coupled"], r["neff_ratio_decoupled"],
        r["margin_var"], r["n_var"], r["ratio"]))

print()
print("baseline, 15 datasets: 11/15      494   -0.5005 -0.5005    0.691  0.993    0.00264  0.01261   0.21")
