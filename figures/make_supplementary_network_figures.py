from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figures.network_style import configure, C1, C2, C3, C4, C5, C6, INK, RULE, GRID, MIX_FILL, MIX_INK, FS_TITLE, FS_LABEL, FS_TICK, LW_AXIS, LW_OTHER
from figures.network_style import DATASET_COLOURS, DATASET_LABELS

DEFAULT_STYLE = {key: value for key, value in plt.rcParams.items() if key != "backend"}
configure()
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "Droid Sans Fallback", "SimSun", "DejaVu Sans"]
NETWORK_STYLE = {key: value for key, value in plt.rcParams.items() if key != "backend"}
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures/out"
RESULTS = ROOT / "results"

REF_CMAP = LinearSegmentedColormap.from_list("appendix_reference", ["#E5633E", "#E49A61", "#E8ECF0", "#9CC6CE", "#5B83AD", "#365A7C"])

APPENDIX_ALGORITHM_ORDER = (
    "no_coordination",
    "local_rules",
    "mpc_optimal",
    "mean_field_control",
    "virtual_battery",
    "packetized_energy_management",
    "transactive_control",
    "eps_ieee69_fused",
    "centralized_optimal",
)

APPENDIX_ALGORITHM_LABELS = {
    "no_coordination": "No coordination",
    "local_rules": "Local SOC rules",
    "mpc_optimal": "MPC",
    "mean_field_control": "Mean-field control",
    "virtual_battery": "Virtual battery",
    "packetized_energy_management": "Packetized EM",
    "transactive_control": "Transactive control",
    "eps_ieee69_fused": "EPS IEEE-123 direct",
    "centralized_optimal": "Centralized greedy UB",
}

APPENDIX_ALGORITHM_COLORS = {
    "no_coordination": "#9C9C9C",
    "local_rules": "#E5633E",
    "mpc_optimal": "#7057FF",
    "mean_field_control": "#4E79A7",
    "virtual_battery": "#59A14F",
    "packetized_energy_management": "#AF7AA1",
    "transactive_control": "#D55E00",
    "eps_ieee69_fused": "#365A7C",
    "centralized_optimal": "#3A9D66",
}

SCENARIO_LABELS = {
    "S0_uniform": "S0\nUniform",
    "S1_feeder_50": "S1\n50% distal feeder",
    "S2_node_80": "S2\n80% distal node",
    "S3_feeder_50_derated": "S3\n50% feeder, 70% capacity",
}


ALGORITHM_LABELS = {
    "no_coordination": "No coordination", "local_rules": "Local SOC rules", "mpc_optimal": "MPC",
    "mean_field_control": "Mean-field control", "virtual_battery": "Virtual battery",
    "packetized_energy_management": "PEM", "transactive_control": "Transactive control",
    "eps_broadcast": "EPS", "eps_e20_pooled": "EPS pooled", "eps_e21_mixed": "EPS mixed",
    "eps_ieee69_fused": "EPS", "centralized_optimal": "Centralized greedy UB",
}

ALGORITHM_COLOURS = {
    "no_coordination": "#AAB4BE", "local_rules": C6, "mpc_optimal": C4,
    "mean_field_control": C5, "virtual_battery": "#77B7D3",
    "packetized_energy_management": C3, "transactive_control": C2,
    "eps_e20_pooled": C1, "eps_e21_mixed": C4, "eps_ieee69_fused": C1,
    "centralized_optimal": "#555B60",
}

DATASET_COLOURS = dict(DATASET_COLOURS)

SUMMARY = []

COMPONENT_ALGORITHM_ORDER = (
    "no_coordination",
    "local_rules",
    "mpc_optimal",
    "mean_field_control",
    "virtual_battery",
    "packetized_energy_management",
    "transactive_control",
    "eps_e20_pooled",
    "eps_e21_mixed",
    "centralized_optimal",
)

COMPONENT_STRESS_MODES = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")

COMPONENT_PANELS = (
    ("maximum_branch_loading_mean", "Executed maximum branch loading", 0.0, 1.5, "{:.2f}"),
    ("minimum_voltage_pu_mean", "Executed minimum voltage (p.u.)", 0.94, 1.0, "{:.3f}"),
    ("maximum_transformer_loading_mean", "Executed maximum transformer loading", 0.0, 1.0, "{:.2f}"),
    ("safety_layer_gain", "Safety-layer gain in violation-free steps (pp)", 0.0, 100.0, "{:.1f}"),
)



def _configure_english_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save_generated(figure: plt.Figure, target: Path) -> None:
    figure.tight_layout()
    figure.savefig(target, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _bootstrap_interval(values: np.ndarray, label: str) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(sum(label.encode("utf-8")))
    samples = rng.integers(0, len(values), size=(2000, len(values)))
    means = values[samples].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _algorithm_tick_labels() -> list[str]:
    return [APPENDIX_ALGORITHM_LABELS[algorithm] for algorithm in APPENDIX_ALGORITHM_ORDER]


def _plot_physical_risk(target: Path) -> None:
    data = pd.read_csv(RESULTS / "network_stress_boundary/data/by_seed.csv")
    axes_info = (
        ("request_intensity", "Request intensity q", "q"),
        ("line_derating", "Line derating", "Fractional derating"),
        ("spatial_concentration", "Spatial concentration", "Distal-node share"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharey=True)
    for axis, (axis_name, title, xlabel) in zip(axes, axes_info):
        subset = data[data["axis"] == axis_name]
        for algorithm in APPENDIX_ALGORITHM_ORDER:
            rows = subset[subset["algorithm"] == algorithm]
            if rows.empty:
                continue
            grouped = rows.groupby("pressure_value", sort=True)["requested_added_violation_steps"].mean()
            values = 100.0 * grouped.to_numpy(dtype=float) / 288.0
            axis.plot(grouped.index, values, marker="o", markersize=3, linewidth=1.25,
                      color=APPENDIX_ALGORITHM_COLORS[algorithm], label=APPENDIX_ALGORITHM_LABELS[algorithm])
        axis.axhline(5.0, color="#D1495B", linestyle="--", linewidth=1.0, label="5% reference")
        axis.set_title(title, fontsize=11)
        axis.set_xlabel(xlabel)
        axis.grid(axis="y", color="#E8ECF0", linewidth=0.7)
    axes[0].set_ylabel("Requested added-violation risk (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        fontsize=7,
        frameon=False,
    )
    figure.subplots_adjust(bottom=0.24, wspace=0.08)
    figure.savefig(target, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _plot_ieee123_safety(target: Path) -> None:
    data = pd.read_csv(RESULTS / "ieee123_safety_audit/data/summary.csv")
    scenarios = list(SCENARIO_LABELS)
    figure, axes = plt.subplots(1, len(scenarios), figsize=(18, 5.6), sharey=True)
    for axis, scenario in zip(axes, scenarios):
        selected = data[data["scenario"] == scenario].set_index("algorithm")
        risk = [float(selected.loc[algorithm, "requested_added_violation_mean_pct"]) for algorithm in APPENDIX_ALGORITHM_ORDER]
        acceptance = [np.nan if str(selected.loc[algorithm, "network_acceptance_mean_pct"]) == "NA"
                      else float(selected.loc[algorithm, "network_acceptance_mean_pct"])
                      for algorithm in APPENDIX_ALGORITHM_ORDER]
        x = np.arange(len(APPENDIX_ALGORITHM_ORDER))
        axis.bar(x - 0.19, risk, 0.38, color="#D1495B", label="Requested risk")
        axis.bar(x + 0.19, acceptance, 0.38, color="#4E79A7", label="Network acceptance")
        axis.axhline(5.0, color="#D1495B", linestyle="--", linewidth=1.2, label="5% reference")
        axis.set_title(SCENARIO_LABELS[scenario].replace("\n", ": "), fontsize=12, fontweight="bold")
        axis.set_xticks(x, _algorithm_tick_labels(), rotation=55, ha="right", fontsize=8, fontweight="bold")
        axis.set_ylim(0, 110)
        axis.grid(axis="y", color="#E8ECF0", linewidth=0.7)
    axes[0].set_ylabel("Share of requested control (%)", fontsize=11, fontweight="bold")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(target, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _rerender_mixed_gamma(target: Path) -> None:
    coupled = pd.read_csv(RESULTS / "effective_scale_mixtures/gamma_mixed_boundary/data/mixed_gamma_points.csv")
    all_arms = pd.read_csv(RESULTS / "effective_scale_mixtures/gamma_mixed_boundary/data/mixed_gamma_all_arms.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    colours = {s: REF_CMAP(i / max(1, coupled.scenario.nunique() - 1)) for i, s in enumerate(sorted(coupled.scenario.dropna().unique()))}
    for ax, x in zip(axes, ["N", "Gamma"]):
        for scenario in sorted(coupled.scenario.dropna().unique()):
            for arm, ls in [("data_coupled", "-"), ("decoupled", "--")]:
                d = all_arms[(all_arms.scenario == scenario) & (all_arms.arm == arm)].sort_values(x)
                if len(d): ax.plot(d[x], d.R2, marker="o", ms=2.5, lw=1.0, ls=ls, color=colours[scenario], alpha=.8)
        ax.axhline(.95, color="#4D4D4D", ls=(0, (1, 1.5)), lw=.7); ax.set_xlabel("Physical N" if x == "N" else "Gamma"); ax.set_ylim(.8, 1.01); ax.grid(axis="y", color="#E8ECF0", lw=.4)
    axes[0].set_ylabel("Independent test $R^2$"); fig.tight_layout(); fig.savefig(target, dpi=220, facecolor="white"); plt.close(fig)


def _rerender_figure(source: Path, target: Path) -> None:
    image = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / 220.0, height / 220.0), dpi=220, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.imshow(image, interpolation="nearest", aspect="auto")
    ax.set_axis_off()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=220, facecolor="white", edgecolor="white", pad_inches=0)
    plt.close(fig)


def read(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path, low_memory=False)


def save_boxplot(fig: plt.Figure, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def record(name: str, source: str, frame: pd.DataFrame, group: str | None = None, value: str | None = None) -> None:
    item = {"figure": name, "source": source, "rows": int(len(frame))}
    if group and group in frame:
        item["groups"] = int(frame[group].nunique(dropna=True))
    if value and value in frame:
        x = pd.to_numeric(frame[value], errors="coerce").dropna()
        item.update({"finite_values": int(x.size), "median": round(float(x.median()), 6) if len(x) else None,
                     "mean": round(float(x.mean()), 6) if len(x) else None,
                     "min": round(float(x.min()), 6) if len(x) else None,
                     "max": round(float(x.max()), 6) if len(x) else None})
    SUMMARY.append(item)


def style_ax(ax, grid=True):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(LW_AXIS); ax.spines["bottom"].set_linewidth(LW_AXIS)
    ax.tick_params(width=LW_AXIS, colors=INK)
    if grid: ax.grid(axis="y", color=GRID, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def raincloud(ax, values, xpos, colour, seed=0, width=0.72, point_colour=None, max_points=900):
    values = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna(), dtype=float)
    if not len(values): return
    vp = ax.violinplot(values, positions=[xpos], widths=width, showmeans=False, showmedians=False, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(colour); body.set_edgecolor(colour); body.set_alpha(0.48); body.set_linewidth(0.8)
        path = body.get_paths()[0]; verts = path.vertices
        verts[verts[:, 0] > xpos, 0] = xpos
    rng = np.random.default_rng(seed)
    draw = values if len(values) <= max_points else values[rng.choice(len(values), max_points, replace=False)]
    jitter = rng.uniform(0.04, width * 0.48, len(draw))
    ax.scatter(xpos + jitter, draw, s=8.0, c=point_colour or colour, alpha=0.46, edgecolors="white", linewidths=0.25, zorder=5)
    ax.boxplot([values], positions=[xpos - width * 0.08], widths=width * 0.24, patch_artist=True,
               showfliers=False, boxprops={"facecolor":"white", "edgecolor":INK, "linewidth":0.8},
               whiskerprops={"color":INK, "linewidth":0.8}, capprops={"color":INK, "linewidth":0.8},
               medianprops={"color":INK, "linewidth":2.0}, zorder=6)


def distribution_axis(ax, groups, frame, group_col, value_col, colours=None, seed_offset=0, point_col=None):
    positions = np.arange(1, len(groups) + 1, dtype=float)
    for i, group in enumerate(groups):
        vals = frame.loc[frame[group_col] == group, value_col]
        colour = (colours or {}).get(group, C2)
        raincloud(ax, vals, positions[i], colour, seed=seed_offset + i, point_colour=point_col)
    ax.set_xticks(positions); ax.set_xticklabels(groups, rotation=38, ha="right")
    style_ax(ax)
    return positions


def transfer_r2_by_scale():
    folder = OUTPUT; order=["in_domain","zero_shot","target_calibrated","pooled_ridge"]; labels={"in_domain":"In-domain","zero_shot":"Zero-shot","target_calibrated":"Calibrated","pooled_ridge":"Pooled ridge"}
    c=read("results/transfer/across_datasets/across_datasets_summary.csv"); fig,axes=plt.subplots(1,4,figsize=(15,4.2),sharey=True)
    for ax,m in zip(axes,order):
        s=c[c.method==m].copy(); ns=sorted(s.N.unique()); distribution_axis(ax,ns,s,"N","R2",{x:C3 for x in ns}); ax.axhline(.95,color=RULE,ls=(0,(1,1.5)),lw=.8); ax.set_title(labels[m],fontsize=FS_TITLE); ax.tick_params(axis="x",labelrotation=60,labelsize=FS_TICK)
    axes[0].set_ylabel("Independent test $R^2$"); fig.supxlabel("Device count N"); record("r2_vs_N_boxplot.png","results/transfer/across_datasets/across_datasets_summary.csv",c,"N","R2"); save_boxplot(fig,folder,"FigS26_transfer_r2_by_scale")


def pairwise_r2_by_scale():
    folder=OUTPUT; d=read("results/effective_scale_mixtures/pairwise_r2/data/pairwise_r2_points.csv"); ns=sorted(d.N.unique()); fig,ax=plt.subplots(figsize=(9.5,5.2)); distribution_axis(ax,ns,d,"N","R2",{x:C2 for x in ns}); ax.axhline(.95,color=RULE,ls=(0,(1,1.5)),lw=.8); ax.set_xlabel("Device count N"); ax.set_ylabel("Independent test $R^2$"); record("pairwise_r2_vs_N_boxplot.png","results/effective_scale_mixtures/pairwise_r2/data/pairwise_r2_points.csv",d,"N","R2"); save_boxplot(fig,folder,"FigS31_pairwise_r2_by_scale")


def r2_by_gamma():
    folder=OUTPUT; r=read("results/effective_scale_mixtures/gamma_universal_boundary/data/gamma_points.csv"); m=read("results/effective_scale_mixtures/gamma_mixed_boundary/data/mixed_gamma_points.csv"); r["source"]="Original datasets"; m["source"]="Mixed scenarios"; all_d=pd.concat([r[["Gamma","R2","source","dataset"]],m[["Gamma","R2","source"]]],ignore_index=True); edges=np.quantile(np.log10(all_d.Gamma.clip(lower=1e-6)),np.linspace(0,1,9)); edges=np.unique(edges); all_d["bin"]=pd.cut(np.log10(all_d.Gamma.clip(lower=1e-6)),bins=edges,include_lowest=True); cats=list(all_d.bin.cat.categories); fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),sharey=True)
    for ax,src,col in zip(axes,["Original datasets","Mixed scenarios"],[C3,C2]):
        s=all_d[all_d.source==src]; distribution_axis(ax,cats,s,"bin","R2",{x:col for x in cats}); ax.set_xticklabels([f"{10**x.left:.2g}\n–\n{10**x.right:.2g}" for x in cats],fontsize=FS_TICK); ax.set_title(src,fontsize=FS_TITLE)
    axes[0].set_ylabel("Response tracking $R^2$"); fig.supxlabel("Gamma bins (equal log10 spacing)"); record("gamma_r2_boxplot.png","results/effective_scale_mixtures/gamma_universal_boundary/data/gamma_points.csv + gamma_mixed_boundary/data/mixed_gamma_points.csv",all_d,"bin","R2"); save_boxplot(fig,folder,"FigS32_r2_by_gamma")


def ieee69_effect_and_fidelity():
    folder=OUTPUT; d=read("results/ieee69_network_implementation/data/by_seed.csv"); d=d[d.topology=="ieee69"].copy(); d["group"]=d.algorithm.map(ALGORITHM_LABELS); alg=[a for a in ["no_coordination","local_rules","mpc_optimal","mean_field_control","virtual_battery","packetized_energy_management","transactive_control","eps_e20_pooled","eps_e21_mixed","centralized_optimal"] if a in set(d.algorithm)]; labels=[ALGORITHM_LABELS[a] for a in alg]; fig,axes=plt.subplots(1,2,figsize=(15,5.2)); cols={ALGORITHM_LABELS[a]:ALGORITHM_COLOURS.get(a,C2) for a in alg}
    for ax,v,y,t in zip(axes,["mean_reduction_pct","response_r2"],["Curtailment reduction (%)","Response tracking $R^2$"],["IEEE-69 algorithm effect","IEEE-69 response tracking"]): distribution_axis(ax,labels,d,"group",v,cols); ax.set_ylabel(y); ax.set_title(t,fontsize=FS_TITLE)
    record("ieee69_algorithm_boxplots.png","results/ieee69_network_implementation/data/by_seed.csv",d,"group","response_r2"); save_boxplot(fig,folder,"FigS35_ieee69_effect_and_fidelity")


def stress_axis_distribution(axis_name, name):
    folder=OUTPUT; d=read("results/network_stress_boundary/data/by_seed.csv"); d["group"]=d.algorithm.map(ALGORITHM_LABELS); alg=[a for a in ["local_rules","mpc_optimal","mean_field_control","virtual_battery","packetized_energy_management","transactive_control","eps_ieee69_fused","centralized_optimal"] if a in set(d.algorithm)]
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharey=True); sub=d[d.axis==axis_name]
    for ax,a in zip(axes.ravel(),alg):
        s=sub[sub.algorithm==a]; p=sorted(s.pressure_value.unique()); distribution_axis(ax,p,s,"pressure_value","mean_reduction_pct",{x:ALGORITHM_COLOURS.get(a,C2) for x in p}); ax.set_title(ALGORITHM_LABELS[a],fontsize=FS_TITLE); ax.tick_params(axis="x",labelrotation=60)
    for ax in axes.ravel()[len(alg):]: ax.axis("off")
    axes[0,0].set_ylabel("Curtailment reduction (%)"); axes[1,0].set_ylabel("Curtailment reduction (%)"); fig.supxlabel("Pressure parameter"); record(f"{name}.png","results/network_stress_boundary/data/by_seed.csv",sub,"pressure_value","mean_reduction_pct"); save_boxplot(fig,folder,name)


def _maximum_tie_mask(values: np.ndarray, formatter: str) -> np.ndarray:
    displayed = np.asarray(
        [[formatter.format(value) for value in row] for row in values],
        dtype=object,
    )
    mask = np.zeros(values.shape, dtype=bool)
    for column_index in range(values.shape[1]):
        maximum = np.nanmax(values[:, column_index])
        maximum_display = formatter.format(maximum)
        ties = displayed[:, column_index] == maximum_display
        if int(np.count_nonzero(ties)) > 1:
            mask[:, column_index] = ties
    return mask


def _matrix(data: pd.DataFrame, metric: str) -> tuple[np.ndarray, list[str]]:
    if metric == "safety_layer_gain":
        data = data.assign(
            safety_layer_gain=(
                data["executed_violation_free_pct_mean"]
                - data["requested_violation_free_pct_mean"]
            )
        )
    labels = (
        data[["algorithm", "algorithm_label"]]
        .drop_duplicates("algorithm")
        .set_index("algorithm")
        .reindex(COMPONENT_ALGORITHM_ORDER)["algorithm_label"]
        .tolist()
    )
    values = (
        data.pivot_table(index="algorithm", columns="stress_mode", values=metric, aggfunc="mean")
        .reindex(index=COMPONENT_ALGORITHM_ORDER, columns=COMPONENT_STRESS_MODES)
        .to_numpy(dtype=float)
    )
    if np.isnan(values).any():
        raise ValueError(f"Missing values in {metric} heatmap matrix")
    return values, labels


def component_audit_heatmap(output: Path) -> None:
    data = pd.read_csv(RESULTS / "ieee69_network_implementation/data/summary.csv")
    data = data[data["topology"].eq("ieee69")].copy()
    if data["dataset"].nunique() != 17 or data["seed_count"].nunique() != 1 or int(data["seed_count"].iloc[0]) != 30:
        raise ValueError("Expected IEEE-69 summaries for 17 datasets and 30 paired seeds")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 2, figsize=(16.0, 9.2), sharex=True, sharey=True)
    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    for panel_index, (metric, title, vmin, vmax, formatter) in enumerate(COMPONENT_PANELS):
        axis = axes.flat[panel_index]
        values, labels = _matrix(data, metric)
        maximum_ties = _maximum_tie_mask(values, formatter)
        image = axis.imshow(values, cmap=REF_CMAP, aspect="auto", vmin=vmin, vmax=vmax)
        axis.set_title(f"{panel_labels[panel_index]} {title}", fontsize=13, fontweight="bold", loc="left", pad=8)
        axis.set_xticks(np.arange(len(COMPONENT_STRESS_MODES)), COMPONENT_STRESS_MODES, fontsize=10, fontweight="bold")
        if panel_index % 2 == 0:
            axis.set_yticks(np.arange(len(labels)), labels, fontsize=9.5, fontweight="bold")
        else:
            axis.tick_params(labelleft=False)
        axis.tick_params(length=3, width=0.7)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                rgba = REF_CMAP((values[row_index, column_index] - vmin) / (vmax - vmin))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                axis.text(
                    column_index,
                    row_index,
                    formatter.format(values[row_index, column_index])
                    + ("†" if maximum_ties[row_index, column_index] else ""),
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    fontweight="bold",
                    color="white" if luminance < 0.53 else "black",
                )
        colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
        colorbar.ax.tick_params(labelsize=8)
    figure.supxlabel("Network and spatial stress mode", fontsize=12, fontweight="bold", y=0.035)
    figure.supylabel("Algorithm", fontsize=12, fontweight="bold", x=0.025)
    figure.text(
        0.5,
        0.012,
        "† Repeated displayed values at the current stress-mode maximum; "
        "the shared value is caused by that mode's maximum.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#404040",
    )
    figure.subplots_adjust(left=0.18, right=0.965, bottom=0.15, top=0.90, wspace=0.34, hspace=0.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=600, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def _pdf_from_png(png: Path) -> None:
    Image.open(png).convert("RGB").save(png.with_suffix(".pdf"), "PDF", resolution=600.0)


def _from_runner(source: Path, name: str) -> Path:
    target = OUTPUT / f"{name}.png"
    with plt.rc_context(NETWORK_STYLE):
        _rerender_figure(source, target)
    _pdf_from_png(target)
    return target


def _english(renderer, name: str) -> Path:
    target = OUTPUT / f"{name}.png"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(DEFAULT_STYLE):
        _configure_english_style()
        renderer(target)
    _pdf_from_png(target)
    return target


def _boxplot(maker, name: str, *args) -> Path:
    with plt.rc_context(NETWORK_STYLE):
        maker(*args)
    target = OUTPUT / f"{name}.png"
    _pdf_from_png(target)
    return target


def figS24() -> Path:
    return _from_runner(RESULTS / "transfer/figures/transfer_heatmap.png", "FigS24_transfer_heatmap")


def figS25() -> Path:
    return _from_runner(RESULTS / "transfer/rotations/figures/multisplit_transfer_heatmap.png", "FigS25_transfer_alternative_splits")


def figS26() -> Path:
    return _boxplot(transfer_r2_by_scale, "FigS26_transfer_r2_by_scale")


def figS27() -> Path:
    return _from_runner(RESULTS / "effective_scale_mixtures/gamma_universal_boundary/figures/gamma_scatter.png", "FigS27_effective_scale_published")


def figS28() -> Path:
    return _from_runner(RESULTS / "effective_scale_mixtures/gamma_mixed_boundary/figures/mixed_composition_heatmap.png", "FigS28_mixed_composition")


def figS29() -> Path:
    target = OUTPUT / "FigS29_mixed_scaling.png"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(DEFAULT_STYLE):
        _rerender_mixed_gamma(target)
    _pdf_from_png(target)
    return target


def figS30() -> Path:
    return _from_runner(RESULTS / "effective_scale_mixtures/pairwise_r2/figures/pairwise_r2_n95_overall.png", "FigS30_pairwise_thresholds")


def figS31() -> Path:
    return _boxplot(pairwise_r2_by_scale, "FigS31_pairwise_r2_by_scale")


def figS32() -> Path:
    return _boxplot(r2_by_gamma, "FigS32_r2_by_gamma")


def figS33() -> Path:
    return _from_runner(RESULTS / "ieee69_network_implementation/figures/ieee69_branch_time_heatmap.png", "FigS33_ieee69_constraint_activation")


def figS34() -> Path:
    return _from_runner(RESULTS / "ieee69_network_implementation/trained_eps_ieee69_direct/figures/trained_vs_transfer_dataset_heatmap_ieee69.png", "FigS34_trained_vs_transferred")


def figS35() -> Path:
    return _boxplot(ieee69_effect_and_fidelity, "FigS35_ieee69_effect_and_fidelity")


def figS36() -> Path:
    return _english(_plot_physical_risk, "FigS36_physical_violation")


def figS37() -> Path:
    return _boxplot(stress_axis_distribution, "FigS37_line_derating", "line_derating", "FigS37_line_derating")


def figS38() -> Path:
    return _boxplot(stress_axis_distribution, "FigS38_request_intensity", "request_intensity", "FigS38_request_intensity")


def figS39() -> Path:
    return _boxplot(stress_axis_distribution, "FigS39_spatial_concentration", "spatial_concentration", "FigS39_spatial_concentration")


def figS40() -> Path:
    return _english(_plot_ieee123_safety, "FigS40_ieee123_safety_audit")


def figS41() -> Path:
    target = OUTPUT / "FigS41_ieee69_component_audit.png"
    with plt.rc_context(DEFAULT_STYLE):
        component_audit_heatmap(target)
    _pdf_from_png(target)
    return target


FIGURES = (figS24, figS25, figS26, figS27, figS28, figS29, figS30, figS31, figS32, figS33, figS34, figS35, figS36, figS37, figS38, figS39, figS40, figS41)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for make in FIGURES:
        print(f"generated {make()}")


if __name__ == "__main__":
    main()
