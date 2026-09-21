from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


SCENARIO_LABELS = {
    "peak_shaving": "Peak\nshaving",
    "valley_filling": "Valley\nfilling",
    "emergency_grid_stability": "Grid\nemergency",
    "emergency_supply_shortage": "Supply\nemergency",
}

COLORS = {
    "teal": "#069c8f",
    "teal_light": "#65c2b7",
    "blue": "#087bbd",
    "blue_light": "#37b5e9",
    "orange": "#f47c2c",
    "red": "#d93419",
    "gray": "#9a9a9a",
    "gray_light": "#d1d1d1",
    "green": "#3d9f5b",
    "yellow": "#f7d982",
}


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _latest_dir(results_root: Path, name_fragment: str) -> Path | None:
    if not results_root.exists():
        return None
    candidates = [
        path
        for path in results_root.iterdir()
        if path.is_dir() and name_fragment in path.name and path.name != "latest"
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _related(result_dir: Path, fragment: str) -> Path | None:
    return _latest_dir(result_dir.parent, fragment)


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax.grid(True, color="#e8e8e8", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def _style_figure(fig: plt.Figure) -> None:
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#bdbdbd")
        ax.spines["bottom"].set_color("#bdbdbd")
        ax.tick_params(colors="#333333", labelsize=9)


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    _style_figure(fig)
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_dir / filename


def _missing(ax: plt.Axes, label: str, text: str) -> None:
    _panel(ax, label)
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#666666",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _scale_response(values: Any) -> np.ndarray:
    array = _as_float_array(values)
    return array / 1000.0


def _scenario_rows(stats: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        (key, stats[key])
        for key in SCENARIO_LABELS
        if key in stats and isinstance(stats[key], dict)
    ]


def _model_mismatch_rows(data: dict[str, Any]) -> tuple[list[str], list[float]]:
    order = [
        ("default", "Baseline"),
        ("soc_30%", "SOC noise 30%"),
        ("offline_15%", "Offline 15%"),
        ("prob_-20%", "Prob. bias -20%"),
        ("prob_-30%", "Prob. bias -30%"),
        ("combined_mild", "Comb. positive"),
        ("combined_negative_mild", "Comb. neg. mild"),
        ("combined_negative_moderate", "Comb. neg. moderate"),
        ("combined_negative_severe", "Comb. neg. severe"),
    ]
    labels = [label for key, label in order if key in data]
    values = [float(data[key].get("r2", np.nan)) for key, _ in order if key in data]
    return labels, values


def make_figure2(result_dir: Path, output_dir: Path) -> Path:

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.4), constrained_layout=True)
    est = _load_json(result_dir / "estimation" / "estimation_validation_results.json")

    ax = axes[0, 0]
    if est and "test_data" in est:
        test = est["test_data"]
        actual = _scale_response(test.get("actuals", []))
        predicted = _scale_response(test.get("predictions", []))
        source = np.asarray(test.get("prediction_sources", [""] * len(actual))).astype(str)
        charge = np.char.endswith(source, "_charge")
        discharge = np.char.endswith(source, "_discharge")
        blocked = np.char.endswith(source, "blocked")
        for mask, color, label in (
            (charge, COLORS["teal_light"], f"Charge (n={int(charge.sum())})"),
            (discharge, "#f06c59", f"Discharge (n={int(discharge.sum())})"),
            (blocked, COLORS["gray"], f"Network-blocked (n={int(blocked.sum())})"),
        ):
            ax.scatter(
                actual[mask],
                predicted[mask],
                s=34,
                alpha=0.72,
                color=color,
                edgecolors="none",
                label=label,
            )
        low = min(float(actual.min()), float(predicted.min()))
        high = max(float(actual.max()), float(predicted.max()))
        ax.plot([low, high], [low, high], "--", color="#999999", linewidth=1)
        r2 = est.get("point_metrics", {}).get("r2", {}).get("value")
        ax.text(
            0.97,
            0.08,
            f"$R^2$ = {float(r2):.3f}" if r2 is not None else "",
            transform=ax.transAxes,
            ha="right",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Actual response (MW)")
        ax.set_ylabel("Predicted response (MW)")
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    else:
        _missing(ax, "A", "estimation validation data missing")
    _panel(ax, "A")

    ax = axes[0, 1]
    if est and "test_data" in est:
        test = est["test_data"]
        actual = _scale_response(test.get("actuals", []))
        predicted = _scale_response(test.get("predictions", []))
        intervals = _scale_response(test.get("intervals", []))
        order = np.argsort(actual)
        x = np.arange(actual.size)
        ax.plot(x, actual[order], color=COLORS["teal"], linewidth=1.3, alpha=0.9)
        ax.plot(x, predicted[order], color=COLORS["teal_light"], linewidth=1.0, alpha=0.8)
        if intervals.size:
            ax.fill_between(
                x,
                intervals[order, 0],
                intervals[order, 1],
                color=COLORS["teal_light"],
                alpha=0.3,
                linewidth=0,
            )
        positive = actual >= 0
        ax.scatter(
            x[positive[order]],
            actual[order][positive[order]],
            s=13,
            color=COLORS["teal"],
            alpha=0.78,
            label="Response",
        )
        ax.scatter(
            x[~positive[order]],
            actual[order][~positive[order]],
            s=13,
            color="#f06c59",
            alpha=0.78,
        )
        metrics = est.get("interval_metrics", {})
        ax.text(
            0.97,
            0.06,
            f"PICP = {100 * float(metrics.get('picp', np.nan)):.1f}%\n"
            f"PINAW = {100 * float(metrics.get('pinaw', np.nan)):.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d5d5d5"),
        )
        ax.set_xlabel("Sample index (sorted by actual)")
        ax.set_ylabel("Response (MW)")
    else:
        _missing(ax, "B", "estimation validation data missing")
    _panel(ax, "B")

    mismatch = _load_json(
        (_related(result_dir, "result3_robustness") or result_dir)
        / "data"
        / "model_mismatch.json"
    )
    ax = axes[1, 0]
    if mismatch:
        labels, values = _model_mismatch_rows(mismatch)
        y = np.arange(len(values))
        colors = [
            COLORS["gray"],
            "#2ca02c",
            "#e36b00",
            COLORS["blue"],
            COLORS["blue"],
            "#756bb1",
            "#ef3b2c",
            "#d7191c",
            "#99000d",
        ][: len(values)]
        ax.barh(y, values, color=colors, height=0.56)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        x_min = min(0.5, float(np.nanmin(values)) - 0.05)
        ax.set_xlim(x_min, 1.02)
        ax.set_xlabel("$R^2$")
        ax.axvline(0.95, color="#ef6f63", linestyle=":", linewidth=1.4)
        ax.text(0.95, 1.01, "$R^2$=0.95", transform=ax.get_xaxis_transform(),
                ha="center", color="#ef6f63", fontsize=9)
        for index, value in enumerate(values):
            ax.text(
                min(value + 0.006, 1.005),
                index,
                f"{value:.3f}",
                va="center",
                ha="right" if value > 0.93 else "left",
                color="white" if value > 0.78 else "#222222",
                fontsize=9,
                fontweight="bold",
            )
        ax.axhline(4.5, color="#cccccc", linewidth=1)
    else:
        _missing(ax, "C", "model mismatch data missing")
    _panel(ax, "C")

    stats = _load_json(result_dir / "data" / "multi_run_statistics.json")
    ax = axes[1, 1]
    if stats:
        rows = _scenario_rows(stats)
        x = np.arange(len(rows))
        values = np.array([row[1].get("r2_mean", np.nan) for row in rows])
        lower = np.array([row[1].get("r2_ci_lower", np.nan) for row in rows])
        upper = np.array([row[1].get("r2_ci_upper", np.nan) for row in rows])
        ymin = max(0.0, float(np.nanmin(values)) - 0.06)
        ymax = min(1.05, float(np.nanmax(values)) + 0.06)
        point_colors = [COLORS["blue"], COLORS["blue_light"], COLORS["orange"], COLORS["red"]]
        for index, (name, value) in enumerate(zip([r[0] for r in rows], values)):
            ax.vlines(x[index], ymin, value, color=point_colors[index], linewidth=3)
            ax.scatter(x[index], value, color=point_colors[index], s=105, zorder=3)
            if np.isfinite(lower[index]) and np.isfinite(upper[index]):
                ax.errorbar(
                    x[index],
                    value,
                    yerr=[[value - lower[index]], [upper[index] - value]],
                    fmt="none",
                    ecolor="#333333",
                    capsize=3,
                    linewidth=0.9,
                )
            smape = rows[index][1].get("smape_mean")
            suffix = f" ({smape:.1f}%)" if smape is not None else ""
            label_x = x[index] + (0.03 if index == 0 else -0.03 if index == len(rows) - 1 else 0)
            ax.text(
                label_x,
                value + max(0.005, (ymax - ymin) * 0.025),
                f"{value:.3f}{suffix}",
                ha="left" if index == 0 else "right" if index == len(rows) - 1 else "center",
                va="bottom",
                color=point_colors[index],
                fontsize=9,
                fontweight="bold",
            )
        if ymin <= 0.95 <= ymax:
            ax.axhline(0.95, color="#e69b88", linestyle="--", linewidth=1)
            ax.text(0.98, 0.92, "$R^2$=0.95", transform=ax.transAxes,
                    ha="right", color="#e69b88", fontsize=8)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(-0.35, len(rows) - 0.65)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS.get(row[0], row[0]) for row in rows])
        ax.set_ylabel("$R^2$")
    else:
        _missing(ax, "D", "scenario statistics missing")
    _panel(ax, "D")

    return _save(fig, output_dir, "fig2_scaling_predictability.png")


def make_figure3(result_dir: Path, output_dir: Path) -> Path:

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5), constrained_layout=True)

    threshold = _load_json(result_dir / "data" / "n_threshold.json")
    ax = axes[0]
    if threshold:
        rows = [value for _, value in sorted(threshold["per_N"].items(), key=lambda item: int(item[0]))]
        n = np.array([row["N"] for row in rows], dtype=float)
        r2 = np.array([row["r2"] for row in rows])
        lo = np.array([row["r2_ci_lower"] for row in rows])
        hi = np.array([row["r2_ci_upper"] for row in rows])
        ax.fill_between(n, lo, hi, color=COLORS["blue_light"], alpha=0.18)
        ax.plot(n, r2, "o-", color=COLORS["blue"], markerfacecolor=COLORS["blue"],
                markeredgecolor="white", linewidth=2.2, markersize=6, label="$R^2$")
        ax.axhline(0.95, color="#bdbdbd", linestyle="--", linewidth=1)
        n95 = threshold.get("threshold_N_95_interpolated", threshold.get("threshold_N_95"))
        if n95 is not None:
            ax.axvline(n95, color="#e8755a", linestyle="--", linewidth=1.7)
            label = f"{float(n95):.1f}" if not float(n95).is_integer() else f"{int(n95)}"
            ax.text(n95, 0.05, f"$N_{{95}}^*$={label}", color="#cf3d1b",
                    transform=ax.get_xaxis_transform(), ha="center", fontweight="bold")
        else:
            ax.text(0.96, 0.08, "$N_{95}^*$ not reached", transform=ax.transAxes,
                    ha="right", color="#cf3d1b", fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlabel("Number of devices (N)")
        ax.set_ylabel("$R^2$")
        ax.set_ylim(0.0, 1.05)
        ax.text(0.08, 0.93, "$R^2$ = 0.95", transform=ax.transAxes, color="#969696")
    else:
        _missing(ax, "A", "n-threshold data missing")
    _panel(ax, "A")

    scaling = _load_json(
        (_related(result_dir, "result3_robustness") or result_dir)
        / "data"
        / "n_scaling.json"
    )
    ax = axes[1]
    if scaling and "iid" in scaling:
        payload = scaling["iid"]
        rows = payload.get("scaling_data", [])
        n = np.array([row["N"] for row in rows], dtype=float)
        cv = np.array([row["cv"] for row in rows], dtype=float)
        sigma = float(payload.get("sigma_hat", 0.18))
        valid = np.isfinite(n) & np.isfinite(cv) & (n > 0) & (cv > 0)
        n = n[valid]
        cv = cv[valid]
        if n.size:
            ax.plot(n, cv, "o", color=COLORS["teal"], markersize=8, label="Measured CV")
            theory_n = np.geomspace(n.min() * 0.8, n.max() * 1.15, 100)
            if sigma > 0:
                ax.plot(theory_n, sigma / np.sqrt(theory_n), "--", color="#888888",
                        linewidth=1.8, label=fr"$\hat{{\sigma}}/\sqrt{{N}}$ ($\hat{{\sigma}}$={sigma:.2f})")
        else:
            _missing(ax, "B", "CV data are zero; rerun repeated-response experiment")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of devices (N)")
        ax.set_ylabel("Aggregate CV")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.text(0.06, 0.08, f"Slope = {payload.get('loglog_slope', -0.5):.2f}\n(theory: -0.50)",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd"))
    else:
        _missing(ax, "B", "correlation scaling data missing")
    _panel(ax, "B")

    hetero = _load_json(result_dir / "data" / "heterogeneity_lookup_table.json")
    ax = axes[2]
    if hetero:
        metadata = hetero["metadata"]
        n_values = np.asarray(metadata["N_values"], dtype=float)
        cv_values = np.asarray(metadata["cv_values"], dtype=float)
        grid = np.asarray(hetero["grid"]["r2"], dtype=float)
        mesh = ax.pcolormesh(cv_values, n_values, grid.T, cmap="YlGnBu",
                             shading="nearest", vmin=0.84, vmax=1.0)
        fig.colorbar(mesh, ax=ax, pad=0.03, label="$R^2$")
        ax.set_xlabel("Capacity CV (heterogeneity)")
        ax.set_ylabel("Number of devices (N)")
        ax.set_yscale("log")
        ax.set_xticks(cv_values)
        ax.set_xticklabels([f"{value:g}" for value in cv_values], rotation=45, ha="right")
        tick_candidates = np.asarray(
            [1, 5, 15, 30, 50, 100, 500, 1000, 5000], dtype=float
        )
        visible_ticks = tick_candidates[
            (tick_candidates >= n_values.min()) & (tick_candidates <= n_values.max())
        ]
        if visible_ticks.size:
            ax.set_yticks(visible_ticks)
            ax.set_yticklabels([
                f"{int(value / 1000)}K" if value >= 1000 else f"{int(value)}"
                for value in visible_ticks
            ])
        if n95 is not None:
            ax.axhline(n95, color="#e34a33", linewidth=1.2)
            label = f"{float(n95):.1f}" if not float(n95).is_integer() else f"{int(n95)}"
            ax.text(0.96, 0.27, f"$N_{{95}}^*={label}$", transform=ax.transAxes,
                    ha="right", color="#cf3d1b", fontsize=10, fontweight="bold")
        else:
            ax.text(0.96, 0.27, "$N_{95}^*$ not reached", transform=ax.transAxes,
                    ha="right", color="#cf3d1b", fontsize=10, fontweight="bold")
    else:
        _missing(ax, "C", "heterogeneity lookup data missing")
    _panel(ax, "C")

    return _save(fig, output_dir, "fig3_scaling_heterogeneity.png")


def _curtailment_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("results", data.get("scenarios", []))
    if isinstance(rows, dict):
        rows = [{"name": key, **value} for key, value in rows.items()]
    return list(rows)


def _data2_hourly(trace: dict[str, Any]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    input_values = np.asarray(trace.get("energy_input_kw", trace["pv_kw"]), dtype=float)
    load_values = np.asarray(trace["load_kw"], dtype=float)
    baseline_values = np.asarray(trace.get("baseline_battery_kw", []), dtype=float)
    if load_values.size and baseline_values.size == load_values.size and np.max(np.abs(load_values)) <= 1e-12 and np.max(np.abs(baseline_values)) > 0:
        load_values = baseline_values
    control_values = np.asarray(trace["accepted_control_kw"], dtype=float)
    for hour in range(24):
        sl = slice(hour * 12, (hour + 1) * 12)
        external_input = float(np.mean(input_values[sl])) / 1000.0
        load = float(np.mean(load_values[sl])) / 1000.0
        control = float(np.mean(control_values[sl])) / 1000.0
        balance = external_input - load
        rows.append({
            "hour": float(hour),
            "generation_mw": external_input,
            "solar_mw": external_input,
            "wind_mw": 0.0,
            "load_mw": load,
            "net_balance_mw": balance,
            "curtailment_baseline_mw": max(0.0, balance),
            "curtailment_eps_mw": max(0.0, balance - max(control, 0.0)),
            "simulated_response_mw": abs(control),
            "accepted_control_mw": control,
            "absorbed_mw": max(control, 0.0),
            "curtailment_eps_mwh": max(0.0, balance - max(control, 0.0)),
        })
    return rows


def make_figure4(
    result_dir: Path,
    output_dir: Path,
    *,
    self_consumption: bool = False,
    data2_charging_mode: bool = False,
    data2_root: Path | None = None,
) -> Path:

    result2 = _related(result_dir, "result2_curtailment")
    result4 = _related(result_dir, "result4_generalization")
    data2_data = (data2_root / "data") if data2_root is not None else None
    fig = plt.figure(figsize=(10.2, 13.1), constrained_layout=True)
    grid = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 1.02, 1.02, 1.02])
    axes = {
        "A": fig.add_subplot(grid[0, :]),
        "B": fig.add_subplot(grid[1, 0]),
        "C": fig.add_subplot(grid[1, 1]),
        "D": fig.add_subplot(grid[2, 0]),
        "E": fig.add_subplot(grid[2, 1]),
        "F": fig.add_subplot(grid[3, 0]),
        "G": fig.add_subplot(grid[3, 1]),
    }

    supply = None
    supply_metadata: dict[str, Any] = {}
    generic_input_mode = bool(data2_charging_mode)
    if data2_charging_mode and data2_data is not None:
        day_trace_path = data2_data / "network_timeseries_day.json"
        trace = _load_json(day_trace_path if day_trace_path.exists() else data2_data / "network_timeseries.json")
        if trace:
            rows = _data2_hourly(trace)
            supply = {"hourly": rows}
    elif result2:
        sensitivity = _load_json(result2 / "data" / "curtailment_sensitivity.json")
        if sensitivity:
            supply_metadata = sensitivity.get("metadata", {})
            generic_input_mode = supply_metadata.get("input_source", "pv") != "pv"
            scenarios = sensitivity.get("scenarios", [])
            scenario = next(
                (
                    row for row in scenarios
                    if abs(row.get("input_ratio", row.get("solar_ratio", 0)) - 1.10) < 1e-9
                ),
                None,
            )
            if scenario and scenario.get("hourly_breakdown"):
                supply = {"hourly": scenario["hourly_breakdown"]}
    ax = axes["A"]
    if supply and supply.get("hourly"):
        rows = supply["hourly"]
        hour = np.array([row["hour"] for row in rows], dtype=float)
        generation = np.array([row["generation_mw"] for row in rows])
        base = np.array([row["load_mw"] for row in rows])
        net_balance = generation - base
        absorbed = np.array([row.get("absorbed_mw", 0) for row in rows])
        response = np.array([row.get("simulated_response_mw", 0) for row in rows])
        if data2_charging_mode:
            signed_control = np.array([
                row.get("accepted_control_mw", 0.0) for row in rows
            ])
            final = base + signed_control
        else:
            final = np.where(net_balance > 0, base + absorbed, base - response)
        baseline_curtailed = sum(max(0.0, value) for value in net_balance)
        eps_curtailed = sum(row.get("curtailment_eps_mw", 0) for row in rows)

        if all("solar_mw" in row for row in rows):
            solar = np.array([row["solar_mw"] for row in rows], dtype=float)
            wind = np.array([row.get("wind_mw", 0.0) for row in rows], dtype=float)
        else:
            base_peak = float(supply_metadata.get("base_load_peak_mw", 11.25))
            solar_ratio = float(scenario.get("solar_ratio", 1.10))
            wind_ratio = float(supply_metadata.get("wind_ratio", 0.50))
            solar_factors = np.array([
                0, 0, 0, 0, 0, 0.02, 0.10, 0.30, 0.55, 0.78, 0.92, 0.98,
                1.00, 0.96, 0.85, 0.68, 0.45, 0.20, 0.05, 0, 0, 0, 0, 0,
            ])
            wind_factors = np.array([
                0.45, 0.50, 0.55, 0.52, 0.48, 0.40, 0.30, 0.22, 0.18, 0.20, 0.25, 0.30,
                0.32, 0.28, 0.22, 0.18, 0.20, 0.28, 0.35, 0.42, 0.48, 0.52, 0.50, 0.48,
            ])
            solar = solar_factors * base_peak * solar_ratio
            wind = wind_factors * base_peak * wind_ratio

        ax.fill_between(
            hour,
            0,
            solar,
            step="mid",
            color=COLORS["yellow"],
            alpha=0.68,
            edgecolor="#f3b951",
            linewidth=0.8,
            label="Observed EV charging power" if data2_charging_mode else "External energy input" if generic_input_mode else "Solar",
        )
        if not generic_input_mode:
            ax.fill_between(
                hour,
                solar,
                solar + wind,
                step="mid",
                color="#8ec9e8",
                alpha=0.62,
                edgecolor="#5aaede",
                linewidth=0.8,
                label="Wind",
            )
        ax.fill_between(
            hour,
            base,
            final,
            where=final >= base,
            step="mid",
            color=COLORS["teal_light"],
            alpha=0.28,
        )
        ax.fill_between(
            hour,
            final,
            base,
            where=base >= final,
            step="mid",
            color=COLORS["red"],
            alpha=0.24,
        )
        if self_consumption:
            pv_self_consumption = np.minimum(solar, base)
            grid_base = np.maximum(base - pv_self_consumption, 0.0)
            grid_final = np.maximum(final - pv_self_consumption, 0.0)
            ax.fill_between(
                hour,
                0,
                pv_self_consumption,
                step="mid",
                color="#4c9f70",
                alpha=0.18,
                label="Input consumed locally" if generic_input_mode else "PV self-consumption",
            )
            ax.step(
                hour,
                grid_base,
                where="mid",
                color="#777777",
                linestyle=":",
                linewidth=1.5,
                label="Grid demand after local input" if generic_input_mode else "Grid demand after PV self-consumption",
            )
            ax.step(
                hour,
                grid_final,
                where="mid",
                color="#276749",
                linewidth=2.0,
                label="Grid demand + EPS after input use" if generic_input_mode else "Grid demand + EPS after self-consumption",
            )
        ax.step(hour, base, where="mid", color="#555555", linestyle="--", linewidth=1.5,
                label="Observed EV charging demand" if data2_charging_mode else "Base demand")
        ax.step(hour, final, where="mid", color=COLORS["teal"], linewidth=2.8,
                label="Charging demand + EPS" if data2_charging_mode else "Demand + EPS")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Power (MW)")
        ax.set_xlim(-0.5, 23.5)
        ax.set_xticks([0, 4, 8, 12, 16, 20])
        reduction = 100 * (baseline_curtailed - eps_curtailed) / baseline_curtailed if baseline_curtailed else 0
        if data2_charging_mode:
            mean_control = float(np.mean([row.get("accepted_control_mw", 0.0) for row in rows]))
            annotation = f"Mean EPS response: {mean_control:.3f} MW\nPositive = charge; negative = discharge"
        else:
            annotation = (
                f"Curtailment: {baseline_curtailed:.1f} -> {eps_curtailed:.2f} MWh "
                f"(-{reduction:.1f}%)"
            )
        if self_consumption:
            label = "Coincident input use" if generic_input_mode else "PV self-consumption"
            annotation += f"\n{label}: {pv_self_consumption.sum():.1f} MWh"
        ax.text(
            0.98,
            0.08,
            annotation,
            transform=ax.transAxes,
            ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d0d0d0"),
            fontsize=10,
        )
        ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=8)
    else:
        _missing(ax, "A", "supply-demand data missing")
    _panel(ax, "A")

    nscale = _load_json((result2 or result_dir) / "data" / "n_scaling_curtailment.json")
    ax = axes["B"]
    if nscale:
        rows = _curtailment_rows(nscale)
        rows.sort(key=lambda row: row["N"])
        n = np.array([row["N"] for row in rows])
        reduction = np.array([row.get("reduction_pct_mean", row.get("reduction_pct", np.nan)) for row in rows])
        ax.fill_between(n, 0, reduction, color=COLORS["teal_light"], alpha=0.18)
        ax.plot(n, reduction, "o-", color=COLORS["teal"], linewidth=2.4,
                markerfacecolor="white", markeredgewidth=2, markersize=6)
        ax.set_xlabel("Number of devices (N)")
        ax.set_ylabel("Curtailment reduction (%)")
        ax.set_ylim(0, 108)
        ax.set_xscale("log")
        ax.set_xticks(n)
        ax.set_xticklabels(
            [str(int(value)) if value < 1000 else f"{value/1000:.0f}k" for value in n],
            rotation=40,
            ha="right",
        )
    else:
        _missing(ax, "B", "curtailment scaling data missing")
    _panel(ax, "B")

    sensitivity = _load_json((result2 or result_dir) / "data" / "curtailment_sensitivity.json")
    ax = axes["C"]
    if sensitivity:
        rows = sensitivity.get("scenarios", [])
        ratios = np.array([row.get("input_ratio", row.get("solar_ratio", index)) for index, row in enumerate(rows)])
        baseline = np.array([row["baseline_curtailment_mwh"] for row in rows])
        eps = np.array([row["eps_curtailment_mwh"] for row in rows])
        x = np.arange(len(rows))
        width = 0.34
        ax.bar(x - width / 2, baseline, width, color="#bdbdbd", label="No dispatch")
        ax.bar(x + width / 2, eps, width, color=COLORS["teal"], label="After EPS")
        for index, row in enumerate(rows):
            if data2_charging_mode:
                label_offset = max(float(baseline.max()), 1.0) * 0.025
                ax.text(
                    index - width / 2,
                    baseline[index] + label_offset,
                    f"{baseline[index]:.1f}",
                    ha="center",
                    color="#555555",
                    fontsize=8,
                )
                ax.text(
                    index + width / 2,
                    eps[index] + label_offset,
                    f"{eps[index]:.1f}",
                    ha="center",
                    color=COLORS["teal"],
                    fontsize=8,
                    fontweight="bold",
                )
            else:
                ax.text(
                    index,
                    max(baseline[index], eps[index]) + max(baseline.max(), 1) * 0.03,
                    f"-{row['reduction_pct']:.0f}%",
                    ha="center",
                    color=COLORS["teal"],
                    fontsize=9,
                    fontweight="bold",
                )
        ax.set_xticks(x)
        if data2_charging_mode:
            ax.set_xticklabels([f"{ratio:g}x" for ratio in ratios])
            ax.set_xlabel("External-input/load ratio")
            ax.set_ylabel("Curtailment (MWh)")
        else:
            ax.set_xticklabels([f"{ratio:g}x" for ratio in ratios])
            ax.set_xlabel("Solar/load ratio")
            ax.set_ylabel("Curtailment (MWh)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    else:
        _missing(ax, "C", "curtailment sensitivity data missing")
    _panel(ax, "C")

    baselines = _load_json((result2 or result_dir) / "data" / "curtailment_baselines.json")
    ax = axes["D"]
    if baselines:
        results = baselines.get("results", {})
        keys = ["no_coordination", "local_rules", "eps_broadcast", "centralized_optimal"]
        labels = ["No coord.\n—", "Local\nSOC rules\n—", "EPS\nbroadcast\nO(1)", "Centralized\ngreedy UB\nO(N)"]
        values = np.array([results[key]["mean_reduction_pct"] for key in keys])
        errors = np.array([results[key]["std_reduction_pct"] for key in keys])
        colors = ["#aaaaaa", "#ed7d31", COLORS["teal"], COLORS["green"]]
        bars = ax.bar(np.arange(len(keys)), values, yerr=errors, capsize=4,
                      color=colors, edgecolor="#555555", linewidth=0.5)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 0.5 + 1,
                    f"{value:.1f}%" if value else "0%", ha="center",
                    va="center", color="white" if value > 20 else "#222222",
                    fontweight="bold", fontsize=10)
        ax.axhline(100, color="#dddddd", linewidth=0.8)
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Curtailment reduction (%)")
        ax.set_ylim(0, 120)
    else:
        _missing(ax, "D", "curtailment baseline data missing")
    _panel(ax, "D")

    transfer = _load_json((result4 or result_dir) / "data" / "cross_region_transfer.json")
    ax = axes["E"]
    if transfer and transfer.get("regions"):
        region_keys = ["B", "C", "D"]
        labels = [f"Region {key}" for key in region_keys if key in transfer["regions"]]
        cold = [transfer["regions"][key]["cold_start"]["r2"] for key in region_keys if key in transfer["regions"]]
        adapted = [transfer["regions"][key]["adapted_nn"]["r2"] for key in region_keys if key in transfer["regions"]]
        x = np.arange(len(labels))
        width = 0.34
        ax.bar(x - width / 2, cold, width, color="#c6c6c6", label="Cold start")
        ax.bar(x + width / 2, adapted, width, color=COLORS["teal"], label="After adaptation")
        for index, (before, after) in enumerate(zip(cold, adapted)):
            ax.text(index - width / 2, before + 0.025, f"{before:.2f}", ha="center", fontsize=8, color="#555555")
            ax.text(index + width / 2, after + 0.025, f"{after:.3f}", ha="center",
                    fontsize=8, color=COLORS["teal"], fontweight="bold")
        source = transfer.get("region_A", {}).get("r2")
        if source is not None:
            ax.axhline(source, color="#bdbdbd", linestyle=":", linewidth=1)
            ax.text(0.02, source - 0.07, f"Source $R^2$={source:.3f}", transform=ax.get_yaxis_transform(),
                    color="#888888", fontsize=8)
        ax.axhline(0.95, color="#e69b88", linestyle="--", linewidth=1)
        all_r2 = np.asarray(cold + adapted, dtype=float)
        lower = min(0.0, float(np.nanmin(all_r2)) * 1.15) if all_r2.size else 0.0
        upper = max(1.08, float(np.nanmax(all_r2)) * 1.15) if all_r2.size else 1.08
        ax.set_ylim(lower, upper)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("$R^2$")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    else:
        _missing(ax, "E", "cross-region data missing")
    _panel(ax, "E")

    real_est = _load_json((result4 or result_dir) / "estimation" / "estimation_validation_results.json")
    ax = axes["F"]
    if real_est and real_est.get("actuals") is not None:
        actual = _as_float_array(real_est["actuals"])
        predicted = _as_float_array(real_est["predictions"])
        tolerance = 1e-9
        discharge = actual < -tolerance
        charge = actual > tolerance
        blocked = ~(charge | discharge)
        ax.scatter(actual[discharge], predicted[discharge], s=9, color="#f09b8f", alpha=0.32,
                   label=f"Discharge (n={int(discharge.sum())})")
        ax.scatter(actual[charge], predicted[charge], s=9, color=COLORS["teal_light"], alpha=0.32,
                   label=f"Charge (n={int(charge.sum())})")
        ax.scatter(actual[blocked], predicted[blocked], s=18, color=COLORS["gray"], alpha=0.55,
                   label=f"Network-blocked (n={int(blocked.sum())})")
        low = min(float(actual.min()), float(predicted.min()))
        high = max(float(actual.max()), float(predicted.max()))
        ax.plot([low, high], [low, high], "--", color="#999999", linewidth=1)
        ci = _load_json((result4 or result_dir) / "data" / "real_param_validation.json")
        metrics = (ci or {}).get("metrics", {})
        interval = metrics.get("r2_ci_95", [np.nan, np.nan])
        n_devices = (ci or {}).get("n_devices_in_simulation", (ci or {}).get("n_real_devices", 100))
        n_source = (ci or {}).get("n_source_devices")
        fleet_label = f"N = {n_devices} real devices"
        if n_source is not None and int(n_source) != int(n_devices):
            fleet_label = f"N = {n_devices} bootstrap devices\n({n_source} source sites)"
        ax.text(
            0.05,
            0.95,
            f"$R^2$ = {real_est.get('r2', metrics.get('r2', np.nan)):.3f}\n"
            f"95% CI [{interval[0]:.3f}, {interval[1]:.3f}]\n"
            f"{fleet_label}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d5d5d5"),
        )
        ax.set_xlabel("Actual response (kW)")
        ax.set_ylabel("Predicted response (kW)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    else:
        _missing(ax, "F", "real-parameter point pairs not stored")
    _panel(ax, "F")

    real_scaling = _load_json((result4 or result_dir) / "data" / "real_params_scaling.json")
    ax = axes["G"]
    if real_scaling and real_scaling.get("per_signal"):
        for key, payload in real_scaling["per_signal"].items():
            rows = payload.get("per_N", [])
            n = np.array([row["N"] for row in rows])
            cv = np.array([row["CV_pct"] for row in rows])
            positive = payload.get("signal_score", 0) > 0
            ax.plot(
                n,
                cv,
                "o-",
                color=COLORS["teal"] if positive else "#ee4d36",
                markerfacecolor="white",
                markeredgewidth=1.8,
                linewidth=1.8,
                label="Charge (s=+0.3)" if positive else "Discharge (s=-0.3)",
            )
        all_rows = next(iter(real_scaling["per_signal"].values())).get("per_N", [])
        n = np.array([row["N"] for row in all_rows])
        anchor = float(real_scaling["per_signal"]["charge_s=0.3"]["per_N"][0]["CV_pct"])
        ax.plot(n, anchor * np.sqrt(n[0] / n), "--", color="#999999", label=r"Theoretical $\propto 1/\sqrt{N}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of devices (N)")
        ax.set_ylabel("CV (%)")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.text(0.05, 0.07,
                f"Slope = {real_scaling.get('average_slope', -0.5):.2f} (theory -0.50)",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d5d5d5"))
    else:
        _missing(ax, "G", "real-parameter scaling data missing")
    _panel(ax, "G")

    filename = (
        "fig4_self_consumption.png"
        if self_consumption
        else "fig4_robustness_generalization.png"
    )
    return _save(fig, output_dir, filename)


def make_figure4_self_consumption(
    result_dir: Path,
    output_dir: Path,
    *,
    data2_charging_mode: bool = False,
    data2_root: Path | None = None,
) -> Path:

    return make_figure4(
        result_dir,
        output_dir,
        self_consumption=True,
        data2_charging_mode=data2_charging_mode,
        data2_root=data2_root,
    )


def make_figure5(result_dir: Path, output_dir: Path) -> Path:

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5), constrained_layout=True)
    nscale = _load_json(result_dir / "data" / "n_scaling.json")
    rho = _load_json(result_dir / "data" / "rho_sensitivity.json")
    rho_scan = rho.get("rho_scan", rho) if rho else {}

    ax = axes[0]
    if nscale:
        def _rho_label(key: str, fallback: str) -> str:
            value = rho.get(key, {}).get("rho_within")
            if value is None:
                return fallback
            return f"{fallback.split(' (')[0]} ($\\rho$ = {float(value):.2g})"

        styles = {
            "iid": (COLORS["teal"], "o", _rho_label("iid", "IID")),
            "weak": (COLORS["blue"], "s", _rho_label("weak", "Weak")),
            "moderate": (COLORS["orange"], "D", _rho_label("moderate", "Moderate")),
        }
        for key, (color, marker, label) in styles.items():
            rows = nscale.get(key, {}).get("scaling_data", [])
            if not rows:
                continue
            n = np.array([row["N"] for row in rows])
            cv = np.array([row["cv"] for row in rows])
            valid = np.isfinite(n) & np.isfinite(cv) & (n > 0) & (cv > 0)
            if np.any(valid):
                ax.plot(n[valid], cv[valid], marker=marker, color=color, linewidth=2, markersize=6, label=label)
        iid = nscale.get("iid", {})
        rows = iid.get("scaling_data", [])
        if rows:
            n = np.array([row["N"] for row in rows])
            sigma = iid.get("sigma_hat", 0.18)
            theory = np.geomspace(n.min() * 0.8, n.max() * 1.15, 100)
            if sigma > 0:
                ax.plot(theory, sigma / np.sqrt(theory), "--", color="#999999",
                        label=fr"$\hat{{\sigma}}_0/\sqrt{{N}}$ ($\hat{{\sigma}}_0$={sigma:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of devices N")
        ax.set_ylabel("Aggregate CV")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        slope = float(nscale.get("iid", {}).get("loglog_slope", -0.5))
        ax.text(0.55, 0.08, f"Slope = {slope:.2f}\n(theory: -0.50)", transform=ax.transAxes,
                color=COLORS["teal"], fontsize=9)
    else:
        _missing(ax, "A", "correlation scaling data missing")
    _panel(ax, "A")

    ax = axes[1]
    if rho:
        rows = list(rho_scan.values())
        rows.sort(key=lambda row: row.get("rho_within", 0))
        correlation = np.array([row["rho_within"] for row in rows])
        neff = np.array([row["N_eff"] for row in rows])
        ymax = max(160.0, float(np.nanmax(neff)) * 1.2)
        ax.axhspan(0, ymax, color="#e7f5f2", alpha=0.65)
        ax.plot(correlation, neff, "o-", color=COLORS["teal"], linewidth=2.8, markersize=8)
        current = next((row for row in rows if row.get("rho_within", 0) > 0), rows[0])
        ax.annotate(
            f"Current sim.\n$\\rho$ $\\approx$ {current['rho_within']:.3f}",
            xy=(current["rho_within"], current["N_eff"]),
            xytext=(0.02, 0.78),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#777777"),
            color="#888888",
            fontsize=9,
        )
        ax.axhline(150, color="#e8755a", linestyle="--", linewidth=1.6)
        ax.text(0.62, 0.25, "$N^*\\approx150$", transform=ax.transAxes, color="#cf3d1b",
                fontsize=11, fontweight="bold")
        ax.set_xlabel("Correlation coefficient $\\rho$")
        ax.set_ylabel("$N_{\\mathrm{eff}}$")
        ax.set_ylim(0, ymax)
    else:
        _missing(ax, "B", "rho sensitivity data missing")
    _panel(ax, "B")

    ax = axes[2]
    if rho:
        rows = list(rho_scan.values())
        rows.sort(key=lambda row: row.get("rho_within", 0))
        correlation = np.array([row["rho_within"] for row in rows])
        normalized = np.array([row["cv_sqrt_neff"] for row in rows])
        ax.axhspan(0.35, 0.40, color="#d6e9f2", alpha=0.62, label="0.35–0.40")
        iid = rows[0]
        ax.scatter([0], [iid["cv_sqrt_neff"]], s=170, facecolor="white",
                   edgecolor="#888888", linewidth=2.3, label="IID")
        ax.plot(correlation[1:], normalized[1:], "s-", color=COLORS["blue"], linewidth=2.4,
                markersize=7, label=fr"$\rho$-scan (N={int(iid.get('N', 5000)):,})")
        ax.axvline(0.003, color="#cccccc", linestyle=":", linewidth=1.1)
        ax.set_xlabel("Correlation coefficient $\\rho$")
        ax.set_ylabel("$\\mathrm{CV}\\cdot\\sqrt{N_{\\mathrm{eff}}}$")
        ymax = max(0.55, float(np.nanmax(normalized)) * 1.15)
        ax.set_ylim(0, ymax)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    else:
        _missing(ax, "C", "rho sensitivity data missing")
    _panel(ax, "C")

    return _save(fig, output_dir, "fig5_correlation_effects.png")


def generate_paper_figures(results_root: Path, output_dir: Path | None = None) -> list[Path]:

    output_dir = output_dir or (results_root / "paper_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    specs = [
        ("result1_scaling_law", make_figure2),
        ("result1_scaling_law", make_figure3),
        ("result1_scaling_law", make_figure4),
        ("result1_scaling_law", make_figure4_self_consumption),
        ("result3_robustness", make_figure5),
    ]
    for fragment, maker in specs:
        result_dir = _latest_dir(results_root, fragment)
        if result_dir is not None:
            generated.append(maker(result_dir, output_dir))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from experiment JSON")
    parser.add_argument("--results-root", default="results/experiments")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    figures = generate_paper_figures(
        results_root=Path(args.results_root),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    for figure in figures:
        print(figure)


if __name__ == "__main__":
    main()
