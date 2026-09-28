#!/usr/bin/env python3
"""绘制子图03和04的目标网络直接训练补充版本。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import ncstyle as style
import nckeys as keys


OUTPUT = ROOT / "outputs/figs/direct_train"
SUPPLEMENT = ROOT / "results/E22/subpanel_direct_supplement/data"
IEEE123 = ROOT / "results/E24/trained_eps_ieee123_direct/data/trained_eps_by_seed.csv"
IEEE69 = ROOT / "results/E22/trained_eps_ieee69_direct/data/trained_eps_by_seed.csv"
EPS = "eps_ieee69_fused"

style.configure()


def _save(figure: plt.Figure, filename: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    options = {"facecolor": "white", "bbox_inches": "tight", "pad_inches": 0.035}
    figure.savefig(OUTPUT / f"{filename}.png", dpi=600, **options)
    figure.savefig(OUTPUT / f"{filename}.pdf", **options)
    plt.close(figure)


def _ieee33_original() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pattern = "results/*_ieee33_real_load/coverage_fix/network_constrained_new/data/curtailment_baselines_final_fixed5000_ieee33_data_driven.json"
    import json

    for path in sorted(ROOT.glob(pattern)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = path.parent.parent.parent.parent.name.replace("_ieee33_real_load", "")
        for row in payload.get("results", {}).get("eps_broadcast", {}).get("seed_results", []):
            rows.append({"dataset": dataset, "topology": "ieee33", "algorithm": EPS, **row})
    return pd.DataFrame(rows)


def _original(topology: str) -> pd.DataFrame:
    if topology == "ieee33":
        return _ieee33_original()
    source = IEEE69 if topology == "ieee69" else IEEE123
    data = pd.read_csv(source, low_memory=False)
    return data[(data["algorithm"] == EPS) & (data["topology"] == topology)].copy()


def _spatial_retention(data: pd.DataFrame, topology: str) -> dict[str, float]:
    selected = data[data["stress_mode"].isin(["M4", "M5", "M6"])].copy()
    grouped = selected.groupby(["dataset", "stress_mode"], observed=False)["mean_reduction_pct"].mean().unstack()
    if not {"M4", "M5", "M6"}.issubset(grouped.columns):
        raise RuntimeError(f"{topology}缺少M4-M6直接训练结果")
    valid = grouped["M4"] > 1e-12
    return {
        "M5": float((grouped.loc[valid, "M5"] / grouped.loc[valid, "M4"]).mean() * 100.0),
        "M6": float((grouped.loc[valid, "M6"] / grouped.loc[valid, "M4"]).mean() * 100.0),
    }


def make_03() -> None:
    topologies = ("ieee69", "ieee123")
    values = np.asarray([
        [_spatial_retention(_original(topology), topology)[mode] for topology in topologies for mode in ("M5", "M6")]
    ])
    figure, axis = plt.subplots(figsize=(12.8, 2.9))
    image = axis.imshow(values, cmap=style.SEQ, vmin=0.0, vmax=100.0, aspect="auto")
    axis.set_xticks(
        np.arange(4),
        ["IEEE-69\nM5", "IEEE-69\nM6", "IEEE-123\nM5", "IEEE-123\nM6"],
    )
    axis.set_yticks([0], ["EPS direct"])
    axis.tick_params(axis="both", labelsize=style.FS_SUBPANEL_TICK)
    for column, value in enumerate(values[0]):
        axis.text(
            column,
            0,
            f"{value:.1f}%",
            ha="center",
            va="center",
            fontsize=style.FS_SUBPANEL_CELL,
            color="white" if value > 58 else "#111111",
        )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    ticks = np.linspace(0, 100, 6)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{tick:.0f}%" for tick in ticks])
    colorbar.ax.tick_params(labelsize=style.FS_SUBPANEL_TICK)
    figure.tight_layout()
    source_rows = []
    for topology, pair in zip(topologies, values.reshape(2, 2)):
        for mode, value in zip(("M5", "M6"), pair):
            source_rows.append({"topology": topology, "stress_mode": mode, "algorithm": EPS, "retention_pct": value})
    source_dir = OUTPUT / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(source_dir / "03_direct_eps_spatial_retention_ieee69_ieee123.csv", index=False)
    _save(figure, "03_direct_eps_spatial_retention_ieee69_ieee123")


def _half_violin(axis: plt.Axes, values: np.ndarray, x: float, color: str, side: str) -> None:
    violin = axis.violinplot(values, positions=[x], vert=False, widths=0.72, showextrema=False)
    for body in violin["bodies"]:
        vertices = body.get_paths()[0].vertices
        if side == "upper":
            vertices[:, 1] = np.maximum(vertices[:, 1], x)
        else:
            vertices[:, 1] = np.minimum(vertices[:, 1], x)
        body.set_facecolor(color)
        body.set_edgecolor("#202020")
        body.set_linewidth(0.8)
        body.set_alpha(0.72)


def make_04() -> None:
    pairwise_path = SUPPLEMENT / "pairwise_eps_target_direct_by_seed.csv"
    if not pairwise_path.is_file():
        raise FileNotFoundError(f"补充实验尚未完成：{pairwise_path}")
    pairwise_data = pd.read_csv(pairwise_path, low_memory=False)
    figure, axis = plt.subplots(figsize=(12.8, 5.5))
    source_rows = []
    for row_index, topology in enumerate(("ieee69", "ieee123")):
        original = _original(topology)
        original = original[original["stress_mode"] == "M0"].copy()
        original_values = original["mean_reduction_pct"].dropna().to_numpy(dtype=float)
        mixed = pairwise_data[pairwise_data["topology"] == topology].copy()
        mixed_values = mixed["mean_reduction_pct"].dropna().to_numpy(dtype=float)
        if not len(original_values) or not len(mixed_values):
            raise RuntimeError(f"{topology}原始或两两组合直接训练结果为空")
        _half_violin(axis, original_values, float(row_index), "#3B82B8", "upper")
        _half_violin(axis, mixed_values, float(row_index), style.MIX_FILL, "lower")
        original_points = original.groupby("dataset", observed=False)["mean_reduction_pct"].mean()
        mixed_points = mixed.groupby("pair_id", observed=False)["mean_reduction_pct"].mean()
        original_colours = [keys.DATASET_COLOURS.get(str(name), "#3B82B8") for name in original_points.index]
        rng = np.random.default_rng(20260904 + row_index)
        axis.scatter(
            original_points.to_numpy(),
            row_index + 0.15 + rng.uniform(0.0, 0.16, len(original_points)),
            c=original_colours,
            s=22,
            edgecolor="white",
            linewidth=0.4,
            zorder=4,
        )
        axis.scatter(
            mixed_points.to_numpy(),
            row_index - 0.15 - rng.uniform(0.0, 0.16, len(mixed_points)),
            c=style.MIX_INK,
            s=14,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.25,
            zorder=4,
        )
        source_rows.extend(
            {"topology": topology, "source": "original", "item": name, "mean_reduction_pct": value}
            for name, value in original_points.items()
        )
        source_rows.extend(
            {"topology": topology, "source": "pairwise_mixed", "item": name, "mean_reduction_pct": value}
            for name, value in mixed_points.items()
        )
    axis.set_yticks([0, 1], ["IEEE-69", "IEEE-123"])
    axis.set_ylim(-0.65, 1.65)
    axis.set_xlim(0, 105)
    axis.tick_params(axis="both", labelsize=style.FS_DISTRIBUTION_TICK)
    axis.grid(axis="x", color=style.GRID, linewidth=0.6, alpha=0.8)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    axis.legend(
        handles=[
            plt.Line2D([], [], marker="o", linestyle="none", color="#3B82B8", label="Original datasets"),
            plt.Line2D([], [], marker="o", linestyle="none", color=style.MIX_INK, label="Pairwise mixed"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        fontsize=style.FS_DISTRIBUTION_LEGEND,
    )
    source_dir = OUTPUT / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(source_dir / "04_direct_eps_original_pairwise_ieee69_ieee123.csv", index=False)
    figure.tight_layout()
    _save(figure, "04_direct_eps_original_pairwise_ieee69_ieee123")


def main() -> None:
    make_03()
    make_04()
    (OUTPUT / "README_03_04_SUPPLEMENT.md").write_text(
        """# 子图03和04直接训练补充

- `03_direct_eps_spatial_retention_ieee69_ieee123`：EPS在IEEE-69和IEEE-123目标网络直接训练后，M5和M6相对于M4的弃电降低率保持比例。IEEE-123列使用其目标域直接训练模型。
- `04_direct_eps_original_pairwise_ieee69_ieee123`：IEEE-69和IEEE-123中，原始数据集与105个两两组合各自目标网络直接训练后的EPS效果。上半钢琴和彩色点为原始数据集，下半钢琴和灰点为两两组合。

原始子图目录`outputs/figs/subpanel/`不被覆盖。
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
