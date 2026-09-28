"""Build English IEEE-123 fixed-desired replay figures for the appendix."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/E22/standardized_constraint_replay_original123/data/transformer_panel_replay_summary.csv"
OUTPUT = ROOT / "outputs/figs/e22_ieee123_appendix"


def _load() -> pd.DataFrame:
    frame = pd.read_csv(SUMMARY)
    frame = frame[(frame["topology"] == "ieee123") & frame["fixed_desired"].astype(bool)].copy()
    if frame.empty:
        raise ValueError("No fixed-desired IEEE-123 replay rows found")
    return frame


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(figure: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(OUTPUT / f"{name}.png", dpi=260, facecolor="white", bbox_inches="tight")
    figure.savefig(OUTPUT / f"{name}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(figure)


def build() -> None:
    _style()
    data = _load()
    grouped = data.groupby("constraint_value", as_index=False).mean(numeric_only=True).sort_values("constraint_value", ascending=False)
    x = grouped["constraint_value"].to_numpy(float)

    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    axis.plot(x, grouped["network_acceptance_pct"], marker="o", linewidth=2.1, color="#E5633E")
    axis.set_xlabel("Transformer loading limit")
    axis.set_ylabel("Network acceptance (%)")
    axis.set_title("IEEE-123 fixed-desired replay: network acceptance")
    axis.set_xticks(x, [f"{100 * value:.0f}%" for value in x])
    axis.set_ylim(0, 105)
    axis.grid(axis="y", color="#E8ECF0", linewidth=0.7)
    _save(figure, "ieee123_transformer_acceptance")

    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    axis.plot(x, grouped["network_admitted_power_mwh"], marker="o", linewidth=2.1, color="#365A7C")
    axis.set_xlabel("Transformer loading limit")
    axis.set_ylabel("Network-admitted request (MWh)")
    axis.set_title("IEEE-123 fixed-desired replay: admitted energy")
    axis.set_xticks(x, [f"{100 * value:.0f}%" for value in x])
    axis.invert_xaxis()
    axis.grid(axis="y", color="#E8ECF0", linewidth=0.7)
    _save(figure, "ieee123_transformer_admitted_energy")

    pivot = data.pivot(index="dataset_label", columns="constraint_value", values="network_acceptance_pct")
    pivot = pivot.sort_values(1.0, ascending=False)
    retention = 100.0 * pivot[0.5] / pivot[1.0]
    figure, axis = plt.subplots(figsize=(8.0, 5.7))
    image = axis.imshow(retention.to_numpy()[:, None], cmap="RdYlBu", vmin=70, vmax=100, aspect="auto")
    axis.set_xticks([0], ["50% / 100% acceptance"])
    axis.set_yticks(np.arange(len(retention)), retention.index)
    for row, value in enumerate(retention):
        axis.text(0, row, f"{value:.1f}%", ha="center", va="center", fontsize=8)
    axis.set_title("IEEE-123 fixed-desired retention at a 50% limit")
    axis.set_xlabel("Acceptance retained relative to the 100% limit")
    axis.set_ylabel("Dataset")
    figure.colorbar(image, ax=axis, pad=0.03, label="Retention (%)")
    _save(figure, "ieee123_transformer_dataset_retention")


if __name__ == "__main__":
    build()
