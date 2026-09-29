from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np


def _json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_all(result_root: Path, *, compact: bool = False) -> list[Path]:
    from .legacy_compat import generate_legacy_figures

    generated = generate_legacy_figures(result_root, result_root / "Figs")
    if compact:
        shutil.rmtree(result_root / "legacy_compat", ignore_errors=True)
        shutil.rmtree(result_root / "paper_figures", ignore_errors=True)
        return generated
    paper = result_root / "paper_figures"
    paper.mkdir(parents=True, exist_ok=True)
    for item in generated:
        (paper / item.name).write_bytes(item.read_bytes())
    return generated


def plot_figure4(result_root: Path, *, compact: bool = False) -> list[Path]:
    from .legacy_compat import _prepare
    from src.extra.paper_figures import make_figure4, make_figure4_self_consumption

    output_dir = result_root / "Figs"
    r1 = _prepare(result_root)
    generated = [
        make_figure4(r1, output_dir),
        make_figure4_self_consumption(r1, output_dir),
    ]
    if compact:
        shutil.rmtree(result_root / "legacy_compat", ignore_errors=True)
        shutil.rmtree(result_root / "paper_figures", ignore_errors=True)
    return generated

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    data = result_root / "data"
    figures = result_root / "Figs"
    paper = result_root / "paper_figures"
    figures.mkdir(parents=True, exist_ok=True)
    paper.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    estimation = _json(result_root / "estimation" / "estimation_validation_results.json")
    actual = np.asarray(estimation["test_data"]["actuals"], dtype=float)
    predicted = np.asarray(estimation["test_data"]["predictions"], dtype=float)
    nscale = _json(data / "n_scaling.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1))
    axes[0].scatter(actual, predicted, s=13, alpha=0.45, c="#087bbd", edgecolors="none")
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    axes[0].plot([low, high], [low, high], "--", c="#888888", lw=1)
    axes[0].set(xlabel="Observed EPS response (kW)", ylabel="Predicted response (kW)", title="Real device-day validation")
    axes[0].text(0.04, 0.94, f"R2 = {estimation['point_metrics']['r2']['value']:.3f}", transform=axes[0].transAxes, va="top")
    for label, color in [("iid", "#069c8f"), ("weak", "#087bbd"), ("moderate", "#f47c2c")]:
        rows = nscale[label]["scaling_data"]
        axes[1].plot([row["N"] for row in rows], [row["cv"] for row in rows], "o-", label=label, color=color)
    axes[1].set(xscale="log", yscale="log", xlabel="Number of device-day resources N", ylabel="Aggregate CV", title="Scaling of response variability")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    path = figures / "fig2_scaling_predictability.png"
    _save(fig, path)
    generated.append(path)

    scenarios = _json(data / "complete_results.json")["scenarios"]
    labels = list(scenarios)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for label in labels:
        trace = scenarios[label]
        axes[0, 0].plot(trace["time_index"], trace["accepted_control_kw"], label=label.replace("_", " "))
        axes[0, 1].plot(trace["time_index"], trace["transformer_loading"], label=label.replace("_", " "))
        axes[1, 0].plot(trace["time_index"], trace["minimum_voltage_pu"], label=label.replace("_", " "))
        axes[1, 1].plot(trace["time_index"], trace["overloaded_branches"], label=label.replace("_", " "))
    axes[0, 0].set(ylabel="Accepted EPS power (kW)", title="Network-constrained dispatch")
    axes[0, 1].set(ylabel="Transformer loading", title="Transformer constraint")
    axes[1, 0].set(xlabel="5-minute step", ylabel="Minimum voltage (p.u.)", title="Voltage constraint")
    axes[1, 1].set(xlabel="5-minute step", ylabel="Overloaded branches", title="Feeder capacity constraint")
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    path = figures / "fig3_network_constrained_dispatch.png"
    _save(fig, path)
    generated.append(path)

    trace = _json(data / "network_timeseries.json")
    t = np.asarray(trace["time_index"], dtype=float) / 12.0
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    axes[0].plot(t, trace["load_kw"], label="Real load", c="#333333")
    axes[0].plot(t, trace.get("energy_input_kw", trace["pv_kw"]), label="External energy input", c="#f47c2c")
    axes[0].plot(t, np.asarray(trace["net_grid_kw"]), label="Grid net power after EPS", c="#087bbd")
    axes[0].set(ylabel="Power (kW)", title="Real load and external energy-input profiles")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].plot(t, trace["desired_control_kw"], label="Desired EPS", c="#9a9a9a")
    axes[1].plot(t, trace["accepted_control_kw"], label="Accepted after network limits", c="#069c8f")
    axes[1].set(xlabel="Time of day (h)", ylabel="Control power (kW)", title="External IEEE 33 constraint response")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    path = figures / "fig4_real_pv_load.png"
    _save(fig, path)
    generated.append(path)

    fig, ax = plt.subplots(figsize=(10, 4.4))
    baseline = np.cumsum(trace["self_consumption_baseline_kwh"])
    eps = np.cumsum(trace["self_consumption_eps_kwh"])
    ax.plot(t, baseline, label="Input self-consumption without EPS", c="#888888")
    ax.plot(t, eps, label="Input self-consumption with EPS charging", c="#069c8f")
    ax.fill_between(t, baseline, eps, color="#65c2b7", alpha=0.25)
    ax.set(xlabel="Time of day (h)", ylabel="Cumulative self-consumed input (kWh)", title="External-input self-consumption accounting")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    path = figures / "fig4_self_consumption.png"
    _save(fig, path)
    generated.append(path)

    rho = _json(data / "rho_sensitivity.json")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for label, color in [("iid", "#069c8f"), ("weak", "#087bbd"), ("moderate", "#f47c2c")]:
        rows = nscale[label]["scaling_data"]
        axes[0].plot([row["N"] for row in rows], [row["cv"] for row in rows], "o-", label=label, color=color)
    axes[0].set(xscale="log", yscale="log", xlabel="N", ylabel="CV", title="Fleet scaling")
    rho_scan = rho.get("rho_scan", {key: rho[key] for key in ("iid", "weak", "moderate")})
    labels = list(rho_scan)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(labels)))
    axes[1].bar(labels, [rho_scan[label]["N_eff"] for label in labels], color=colors)
    axes[1].set(ylabel="Effective N", title="Spatial correlation")
    axes[2].plot(labels, [rho_scan[label]["cv_sqrt_neff"] for label in labels], "o-", c="#d93419")
    axes[2].set(ylabel="CV x sqrt(N_eff)", title="Correlation-adjusted convergence")
    for ax in axes:
        ax.grid(alpha=0.25, axis="y")
    for ax in axes[1:]:
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].legend(frameon=False)
    path = figures / "fig5_correlation_and_scale.png"
    _save(fig, path)
    generated.append(path)

    for item in generated:
        mirror = paper / item.name
        mirror.write_bytes(item.read_bytes())
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot IEEE 33 device-day figures")
    parser.add_argument("--results-root", default="results/ieee33_device_day_simulation")
    args = parser.parse_args()
    for path in plot_all(Path(args.results_root)):
        print(path)


if __name__ == "__main__":
    main()
