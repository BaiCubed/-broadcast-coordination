from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
SNAPSHOT_ROOT = ROOT / "results" / "ieee33_real_snapshot_simulation" / "snapshot"
PURE_THRESHOLD = ROOT / "results" / "experiments" / "20260712_180802_result1_scaling_law_5000dev" / "data" / "n_threshold.json"
PURE_SCALING = ROOT / "results" / "experiments" / "20260713_072302_result3_robustness_5000dev" / "data" / "n_scaling.json"

GROUPS = {
    "independent_real_snapshot_no_network": {
        "label": "Independent snapshot\nno network",
        "color": "#087bbd",
    },
    "time_aligned_real_snapshot_no_network": {
        "label": "Time-aligned snapshot\nno network",
        "color": "#f47c2c",
    },
    "time_aligned_real_snapshot_network": {
        "label": "Time-aligned snapshot\nIEEE33 active",
        "color": "#d93419",
    },
}


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _style(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="left")
    ax.grid(True, color="#e8e8e8", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bdbdbd")
    ax.spines["bottom"].set_color("#bdbdbd")
    ax.tick_params(labelsize=8, colors="#333333")


def _snapshot_dirs(root: Path) -> list[tuple[str, Path]]:
    return [(name, root / name) for name in GROUPS if (root / name).is_dir()]


def _plot_scaling(ax: plt.Axes, pure: dict, snapshots: list[tuple[str, Path]]) -> None:
    pure_payload = pure["iid"]
    pure_rows = pure_payload["scaling_data"]
    ax.plot([row["N"] for row in pure_rows], [row["cv"] for row in pure_rows],
            "o--", color="#555555", linewidth=1.8, markersize=5,
            label="Pure simulation IID")
    for name, path in snapshots:
        payload = _read(path / "data" / "n_scaling.json")["real_snapshot"]
        rows = payload["scaling_data"]
        ax.plot([row["N"] for row in rows], [row["cv"] for row in rows],
                "o-", color=GROUPS[name]["color"], linewidth=2.0, markersize=5,
                label=GROUPS[name]["label"].replace("\n", " "))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of resources (N)")
    ax.set_ylabel("Aggregate CV")
    ax.legend(frameon=False, fontsize=7, loc="lower left")


def _plot_r2(ax: plt.Axes, pure: dict, snapshots: list[tuple[str, Path]]) -> None:
    pure_rows = [value for _, value in sorted(pure["per_N"].items(), key=lambda item: int(item[0]))]
    n = np.asarray([row["N"] for row in pure_rows])
    r2 = np.asarray([row["r2"] for row in pure_rows])
    ax.plot(n, r2, "o--", color="#555555", linewidth=1.8, markersize=4,
            label="Pure simulation")
    for name, path in snapshots:
        data = _read(path / "data" / "n_threshold.json")
        rows = [value for _, value in sorted(data["per_N"].items(), key=lambda item: int(item[0]))]
        n = np.asarray([row["N"] for row in rows])
        r2 = np.asarray([row["r2"] for row in rows])
        lo = np.asarray([row["r2_ci_lower"] for row in rows])
        hi = np.asarray([row["r2_ci_upper"] for row in rows])
        color = GROUPS[name]["color"]
        ax.plot(n, r2, "o-", color=color, linewidth=2.0, markersize=5,
                label=GROUPS[name]["label"].replace("\n", " "))
        ax.fill_between(n, lo, hi, color=color, alpha=0.10, linewidth=0)
    ax.axhline(0.95, color="#999999", linestyle=":", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("Number of resources (N)")
    ax.set_ylabel("$R^2$")
    ax.set_ylim(0.45, 1.02)
    ax.legend(frameon=False, fontsize=7, loc="lower right")


def _plot_rho(ax: plt.Axes, snapshots: list[tuple[str, Path]]) -> None:
    labels = [GROUPS[name]["label"] for name, _ in snapshots]
    raw = []
    residual = []
    for name, path in snapshots:
        data = _read(path / "data" / "resource_response_diagnostics.json")
        raw.append(data["mean_pairwise_rho"])
        residual.append(data["conditional_residual_mean_pairwise_rho"])
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, raw, width, color="#9a9a9a", label="Raw response rho")
    ax.bar(x + width / 2, residual, width, color="#069c8f", label="Condition residual rho")
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Mean pairwise correlation")
    ax.legend(frameon=False, fontsize=7, loc="upper right")


def _plot_residual_zone(ax: plt.Axes, snapshots: list[tuple[str, Path]]) -> None:
    labels = [GROUPS[name]["label"] for name, _ in snapshots]
    x = np.arange(len(labels))
    width = 0.34
    within_means, within_lo, within_hi = [], [], []
    between_means, between_lo, between_hi = [], [], []
    for name, path in snapshots:
        data = _read(path / "data" / "resource_response_diagnostics.json")
        for key, means, lows, highs in (
            ("conditional_residual_within_zone_correlation", within_means, within_lo, within_hi),
            ("conditional_residual_between_zone_correlation", between_means, between_lo, between_hi),
        ):
            item = data[key]
            means.append(item["mean"])
            lows.append(item["q05"])
            highs.append(item["q95"])
    ax.errorbar(x - width / 2, within_means,
                yerr=[np.asarray(within_means) - np.asarray(within_lo),
                      np.asarray(within_hi) - np.asarray(within_means)],
                fmt="o", color="#087bbd", capsize=3, label="Within zone")
    ax.errorbar(x + width / 2, between_means,
                yerr=[np.asarray(between_means) - np.asarray(between_lo),
                      np.asarray(between_hi) - np.asarray(between_means)],
                fmt="s", color="#f47c2c", capsize=3, label="Between zones")
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Conditional residual rho\n(mean; error bars: q05-q95)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")


def _plot_neff(ax: plt.Axes, snapshots: list[tuple[str, Path]]) -> None:
    labels = [GROUPS[name]["label"] for name, _ in snapshots]
    raw, residual = [], []
    for name, path in snapshots:
        data = _read(path / "data" / "resource_response_diagnostics.json")
        raw.append(data["pairwise_N_eff"])
        residual.append(data["conditional_residual_pairwise_N_eff"])
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, raw, width, color="#9a9a9a", label="Raw response")
    ax.bar(x + width / 2, residual, width, color="#069c8f", label="Condition residual")
    ax.axhline(3000, color="#777777", linestyle=":", linewidth=1.0, label="N=3000 reference")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Effective sample size $N_{eff}$")
    ax.legend(frameon=False, fontsize=7, loc="upper left")


def _plot_network_scale(ax: plt.Axes, snapshots: list[tuple[str, Path]]) -> None:
    labels = [GROUPS[name]["label"] for name, _ in snapshots]
    values = []
    for name, path in snapshots:
        values.append(_read(path / "data" / "network_timeseries.json")["network_scale"])
    ax.boxplot(values, tick_labels=labels, patch_artist=True,
               boxprops={"facecolor": "#d6e9f2", "edgecolor": "#777777"},
               medianprops={"color": "#d93419", "linewidth": 1.5},
               whiskerprops={"color": "#777777"}, capprops={"color": "#777777"})
    ax.axhline(1.0, color="#555555", linestyle=":", linewidth=1.0)
    ax.set_ylabel("Network dispatch scale")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="x", labelsize=7)


def make_figure6(snapshot_root: Path = SNAPSHOT_ROOT, output_dir: Path | None = None) -> Path:
    output_dir = output_dir or (snapshot_root / "Figs")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = _snapshot_dirs(snapshot_root)
    if len(snapshots) != 3:
        raise RuntimeError(f"Expected three completed snapshot groups, found {len(snapshots)}")
    pure_threshold = _read(PURE_THRESHOLD)
    pure_scaling = _read(PURE_SCALING)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.7), constrained_layout=True)
    _plot_scaling(axes[0, 0], pure_scaling, snapshots)
    _style(axes[0, 0], "A")
    _plot_r2(axes[0, 1], pure_threshold, snapshots)
    _style(axes[0, 1], "B")
    _plot_rho(axes[0, 2], snapshots)
    _style(axes[0, 2], "C")
    _plot_residual_zone(axes[1, 0], snapshots)
    _style(axes[1, 0], "D")
    _plot_neff(axes[1, 1], snapshots)
    _style(axes[1, 1], "E")
    _plot_network_scale(axes[1, 2], snapshots)
    _style(axes[1, 2], "F")
    fig.suptitle("Real-snapshot ablation: scaling, residual correlation and IEEE33 feedback",
                 fontsize=15, y=1.02)
    path = output_dir / "fig6_real_snapshot_ablation.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the real-snapshot ablation as a new Figure 6.")
    parser.add_argument("--snapshot-root", default=str(SNAPSHOT_ROOT))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    print(make_figure6(Path(args.snapshot_root), Path(args.output_dir) if args.output_dir else None))


if __name__ == "__main__":
    main()
