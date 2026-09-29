from __future__ import annotations

import math
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


FS_PANEL = 10.0
FS_TITLE = 9.0
FS_LABEL = 8.5
FS_TICK = 8.0
FS_LEGEND = 8.0
FS_ANNOT = 7.5
FS_ANNOT_HI = 8.0
FS_SUBPANEL_TICK = 20.0
FS_SUBPANEL_LEGEND = 16.0
FS_SUBPANEL_ANNOT = 15.0
FS_SUBPANEL_CELL = 16.0
FS_AUDIT_TICK = 20.0
FS_AUDIT_CELL = 16.0
FS_DISTRIBUTION_TICK = 20.0
FS_DISTRIBUTION_LEGEND = 16.0
FS_LINE_TICK = 20.0
FS_LINE_LEGEND = 12.0
FS_SUBPANEL_CAPTION = 13.0

LW_AXIS = 0.75
LW_MAIN = 1.4
LW_OTHER = 1.1
LW_TRACE = 0.65
LW_SUMM = 1.9

MS = 4.5
MS_BIG = 5.0
DASH_SUMM = (0, (4.2, 1.8))

INK = "#1A1A1A"
RULE = "#4D4D4D"
FAINT = "#B9C2CB"
BAND = "#C7D4E1"
GRID = "#E8ECF0"
MIX_LINE = "#C3C7CB"
MIX_FILL = "#E4E6E8"
MIX_INK = "#50565C"

C1 = "#365A7C"
C2 = "#5B83AD"
C3 = "#55A6B5"
C4 = "#81729B"
C5 = "#E49A61"
C6 = "#E5633E"

SEQ = LinearSegmentedColormap.from_list(
    "nc_seq",
    [C6, C5, "#EFE3D5", "#9CC6CE", C2, C1],
)

MM = 1.0 / 25.4
WIDTH_2COL = 183.0


def configure() -> None:
    font_dir = Path(__file__).resolve().parent / "fonts"
    for font_path in font_dir.glob("*.ttf"):
        fm.fontManager.addfont(str(font_path))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic",
        "mathtext.bf": "Liberation Sans:bold",
        "mathtext.cal": "Liberation Sans:italic",
        "mathtext.sf": "Liberation Sans",
        "mathtext.tt": "DejaVu Sans Mono",
        "mathtext.default": "it",
        "axes.unicode_minus": False,
        "axes.linewidth": LW_AXIS,
        "xtick.major.width": LW_AXIS,
        "ytick.major.width": LW_AXIS,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 120,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })


def minus(value: object) -> str:
    return str(value).replace("-", "−")


def shade(hex_colour: str, dl: float = 0.0, dh: float = 0.0, ds: float = 0.0) -> str:
    import colorsys

    value = hex_colour.lstrip("#")
    rgb = tuple(int(value[index:index + 2], 16) / 255.0 for index in (0, 2, 4))
    hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
    hue = (hue + dh) % 1.0
    lightness = min(0.88, max(0.12, lightness + dl))
    saturation = min(1.0, max(0.0, saturation + ds))
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return "#%02X%02X%02X" % (round(red * 255), round(green * 255), round(blue * 255))


def ax_mm(fig: plt.Figure, x: float, y: float, width: float, height: float, canvas_height: float):
    return fig.add_axes([
        x / WIDTH_2COL,
        1.0 - (y + height) / canvas_height,
        width / WIDTH_2COL,
        height / canvas_height,
    ])


def tidy(axis: plt.Axes, grid: str | None = None) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(axis=grid, color=GRID, linewidth=0.4, zorder=0)
    axis.set_axisbelow(True)


def title(axis: plt.Axes, text: str, pad: float = 3.0) -> None:
    axis.set_title(text, fontsize=FS_TITLE, pad=pad, color=INK)


def annot(axis: plt.Axes, x: float, y: float, text_value: str, **kwargs) -> None:
    options = {"fontsize": FS_ANNOT, "color": INK, "ha": "left", "va": "center"}
    options.update(kwargs)
    axis.text(x, y, text_value, **options)


def panel(axis: plt.Axes, label: str, dx: float = -0.08, dy: float = 1.04) -> None:
    axis.text(dx, dy, label, transform=axis.transAxes, fontsize=FS_PANEL, fontweight="bold", color=INK, va="bottom")


def _density(values: np.ndarray, grid_n: int = 256, bw_floor: float = 0.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.ptp(values) <= 1e-12:
        center = float(values[0]) if len(values) else 0.0
        span = max(abs(center) * 0.015, 0.5)
        return np.linspace(center - span, center + span, grid_n), np.ones(grid_n)
    lo, hi = float(values.min()), float(values.max())
    spread = max(np.ptp(values) * 0.08, 1e-6)
    x = np.linspace(lo - spread, hi + spread, grid_n)
    bandwidth = max(float(np.std(values, ddof=1) * len(values) ** (-1.0 / 5.0)), bw_floor, 1e-6)
    distances = (x[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * distances ** 2).sum(axis=1) / (len(values) * bandwidth * math.sqrt(2.0 * math.pi))
    return x, density / max(float(density.max()), 1e-12)


def cloud(
    axis: plt.Axes,
    values,
    y0: float,
    colour: str,
    fill: str | None = None,
    height: float = 0.30,
    box_h: float = 0.16,
    points: bool = True,
    point_values=None,
    pt_colours=None,
    pt_alpha: float = 0.85,
    ms: float = 2.6,
    seed: int = 0,
    bw_floor: float = 0.0,
    zorder: int = 6,
):
    source = np.asarray(values, dtype=float)
    source = source[np.isfinite(source)]
    if len(source) == 0:
        return None
    x, density = _density(source, bw_floor=bw_floor)
    axis.fill_between(x, y0, y0 + height * density, color=fill or shade(colour, dl=0.28, ds=-0.18), alpha=0.72, linewidth=0, zorder=zorder - 2)
    axis.plot(x, y0 + height * density, color=colour, linewidth=0.9, zorder=zorder - 1)
    box = axis.boxplot(
        [source],
        positions=[y0 + box_h * 0.28],
        vert=False,
        widths=box_h,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": INK, "linewidth": 0.8},
        whiskerprops={"color": INK, "linewidth": 0.8},
        capprops={"color": INK, "linewidth": 0.8},
        medianprops={"color": INK, "linewidth": 2.0},
        zorder=zorder + 1,
    )
    if points:
        point_array = source if point_values is None else np.asarray(point_values, dtype=float)
        point_array = point_array[np.isfinite(point_array)]
        rng = np.random.default_rng(seed)
        jitter = rng.uniform(-0.16, 0.03, len(point_array))
        if pt_colours is None:
            point_colours = [colour] * len(point_array)
        else:
            point_colours = list(pt_colours)[:len(point_array)]
        axis.scatter(point_array, y0 + jitter, s=ms ** 2, c=point_colours, alpha=pt_alpha, edgecolors="white", linewidths=0.35, zorder=zorder + 2)
    return {
        "n": int(len(source)),
        "q25": float(np.quantile(source, 0.25)),
        "median": float(np.median(source)),
        "q75": float(np.quantile(source, 0.75)),
    }


DATASET_LABELS = {
    "bdg1_building_data_genome": "BDG1",
    "bdg2_building_data_genome": "BDG2",
    "complete_energy_community": "CEC",
    "danish_smart_heat_meters": "Danish",
    "european_lv_rural_2731": "EU-Rural",
    "european_lv_urban_35297": "EU-35k",
    "goiener_smart_meters": "GoiEner",
    "heapo_heat_pumps": "HEAPO",
    "low_carbon_london": "LCL",
    "norway_ami_energy_distribution": "Norway",
    "camsl_japan_smart_meters": "CAMSL",
    "european_lv_urban_8087": "EU-8k",
    "irish_domestic_smart_meters": "Irish",
    "opsd_household_data": "OPSD",
    "smart_grid_smart_city": "SGSC",
    "nextgen_device_days": "NextGen",
    "data2_charging_sessions": "data2",
}

DATASET_COLOURS = {
    "european_lv_urban_35297": "#1F3F51",
    "european_lv_urban_8087": "#315271",
    "european_lv_rural_2731": "#3A619B",
    "goiener_smart_meters": "#328481",
    "low_carbon_london": "#389294",
    "smart_grid_smart_city": "#45969E",
    "norway_ami_energy_distribution": "#4C9FAE",
    "irish_domestic_smart_meters": "#5AA3B7",
    "camsl_japan_smart_meters": "#64A9C5",
    "opsd_household_data": "#70AAC9",
    "danish_smart_heat_meters": "#CC681B",
    "heapo_heat_pumps": "#E89351",
    "bdg1_building_data_genome": "#574D7B",
    "bdg2_building_data_genome": "#786893",
    "complete_energy_community": "#987FAF",
    "nextgen_device_days": "#A36A8B",
    "data2_charging_sessions": "#6689A8",
}
