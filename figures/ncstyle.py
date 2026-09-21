from __future__ import annotations

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in glob.glob(os.path.join(HERE, "fonts", "*.ttf")):
    fm.fontManager.addfont(_p)

C1 = "#365A7C"
C2 = "#5B83AD"
C3 = "#55A6B5"
C4 = "#81729B"
C5 = "#E49A61"
C6 = "#E5633E"
PALETTE = [C1, C2, C3, C4, C5, C6]

INK = "#1A1A1A"
RULE = "#4D4D4D"
FAINT = "#B9C2CB"
BAND = "#C7D4E1"

SEQ = LinearSegmentedColormap.from_list(
    "nc_seq", ["#E5633E", "#E49A61", "#EFE3D5", "#9CC6CE", "#5B83AD", "#365A7C"])

FS_PANEL = 10.0
FS_TITLE = 9.0
FS_LABEL = 8.5
FS_TICK = 8.0
FS_LEGEND = 8.0
FS_ANNOT = 7.5
FS_ANNOT_HI = 8.0

LW_AXIS = 0.75
LW_MAIN = 1.4
LW_OTHER = 1.1
MS = 4.5
MS_BIG = 5.0

MM = 1.0 / 25.4
WIDTH_2COL = 183 * MM


def apply() -> None:
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.cal": "Arial:italic",
        "mathtext.sf": "Arial",
        "mathtext.tt": "DejaVu Sans Mono",
        "mathtext.default": "it",
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.linewidth": LW_AXIS,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlepad": 3.0,
        "axes.labelpad": 2.0,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": "#E8ECF0",
        "grid.linewidth": 0.4,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": LW_AXIS,
        "ytick.major.width": LW_AXIS,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.minor.size": 1.3,
        "ytick.minor.size": 1.3,
        "xtick.major.pad": 1.8,
        "ytick.major.pad": 1.8,
        "lines.linewidth": LW_OTHER,
        "lines.markersize": MS,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.28,
        "legend.borderpad": 0.2,
        "legend.columnspacing": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 120,
    })


def panel(ax, letter, dx=-0.085, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=FS_PANEL,
            fontweight="bold", va="baseline", ha="left", color=INK)
    return ax


def title(ax, text, pad=3.0):
    ax.set_title(text, fontsize=FS_TITLE, pad=pad, color=INK)


def tidy(ax, grid="none", top=False, right=False):
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    for s in ax.spines.values():
        s.set_linewidth(LW_AXIS)
        s.set_color(INK)
    if grid == "y":
        ax.grid(True, axis="y", color="#E8ECF0", linewidth=0.4, zorder=0)
    elif grid == "x":
        ax.grid(True, axis="x", color="#E8ECF0", linewidth=0.4, zorder=0)
    elif grid == "both":
        ax.grid(True, color="#E8ECF0", linewidth=0.4, zorder=0)
    ax.tick_params(width=LW_AXIS, length=2.4, labelsize=FS_TICK)
    return ax


def plain_log(ax, which="x"):
    from matplotlib.ticker import FuncFormatter, NullFormatter

    def fmt(v, _pos):
        if v <= 0:
            return ""
        return f"{v:g}"

    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_formatter(FuncFormatter(fmt))
    axis.set_minor_formatter(NullFormatter())
    return ax


def annot(ax, x, y, text, size=FS_ANNOT, **kw):
    kw.setdefault("transform", ax.transAxes)
    kw.setdefault("color", INK)
    kw.setdefault("va", "top")
    ax.text(x, y, text, fontsize=size, **kw)


def legend_bg(leg, alpha=0.88):
    leg.set_frame_on(True)
    fr = leg.get_frame()
    fr.set_facecolor("white")
    fr.set_edgecolor("none")
    fr.set_alpha(alpha)
    leg.set_zorder(20)
    return leg


def minus(text):
    return str(text).replace("-", "\u2212")


PUB_LINE = "#94BCDF"
PUB_FILL = "#D9E7F4"
PUB_INK = "#1F4E79"

MIX_LINE = "#C3C7CB"
MIX_FILL = "#E4E6E8"
MIX_INK = "#50565C"

LW_TRACE = 0.65
LW_SUMM = 1.9
DASH_SUMM = (0, (4.2, 1.8))


def shade(hex_colour, dl=0.0, dh=0.0, ds=0.0):
    import colorsys

    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb((h + dh) % 1.0,
                                  min(0.88, max(0.12, l + dl)),
                                  min(1.0, max(0.0, s + ds)))
    return "#%02X%02X%02X" % (round(r * 255), round(g * 255), round(b * 255))


def raincloud(ax, values, y0, colour, height=1.7, log=False, box_h=0.44,
              pt_gap=0.16, pt_h=0.95, ms=1.5, seed=0, fill=None, alpha=0.55,
              lw=0.6, box_lw=0.6, med_lw=1.0, min_n=4, pad_bw=3.0, zorder=6):
    import numpy as np
    from matplotlib.patches import Rectangle

    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return None
    t = np.log10(v) if log else v
    n = len(t)
    if n < min_n:
        rng = np.random.default_rng(seed)
        yy = y0 - box_h / 2 - pt_gap - rng.uniform(0.0, pt_h, n)
        ax.plot(v, yy, "o", ms=ms, mfc=colour, mec="none", alpha=alpha, ls="none",
                zorder=zorder)
        return {"n": int(n), "median": float(np.median(v)), "q1": float(np.median(v)),
                "q3": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    iqr = q3 - q1
    sd = t.std(ddof=1) if n > 1 else 0.0
    a = min(sd, iqr / 1.349) if iqr > 0 else sd
    bw = 0.9 * a * n ** (-0.2) if a > 0 else 1e-6
    span = max(t.max() - t.min(), 1e-9)
    bw = max(bw, span / 60.0)

    grid = np.linspace(t.min() - pad_bw * bw, t.max() + pad_bw * bw, 512)
    dens = np.exp(-0.5 * ((grid[:, None] - t[None, :]) / bw) ** 2).sum(1)
    dens = dens / dens.max()
    x = 10 ** grid if log else grid
    ax.fill_between(x, y0, y0 + height * dens, facecolor=fill or colour,
                    alpha=0.55 if fill is None else 1.0, lw=0, zorder=zorder)
    ax.plot(x, y0 + height * dens, color=colour, lw=lw, zorder=zorder + 1)

    conv = (lambda u: 10 ** u) if log else (lambda u: u)
    lo = t[t >= q1 - 1.5 * iqr].min()
    hi = t[t <= q3 + 1.5 * iqr].max()
    ax.plot([conv(lo), conv(hi)], [y0, y0], color=INK, lw=box_lw, zorder=zorder + 2)
    ax.add_patch(Rectangle((conv(q1), y0 - box_h / 2), conv(q3) - conv(q1), box_h,
                           facecolor="white", edgecolor=INK, lw=box_lw,
                           zorder=zorder + 3))
    ax.plot([conv(med), conv(med)], [y0 - box_h / 2, y0 + box_h / 2], color=INK,
            lw=med_lw, zorder=zorder + 4, solid_capstyle="butt")

    rng = np.random.default_rng(seed)
    yy = y0 - box_h / 2 - pt_gap - rng.uniform(0.0, pt_h, n)
    ax.plot(v, yy, "o", ms=ms, mfc=colour, mec="none", alpha=alpha, ls="none",
            zorder=zorder)
    return {"n": int(n), "median": float(conv(med)), "q1": float(conv(q1)),
            "q3": float(conv(q3)), "min": float(v.min()), "max": float(v.max())}


def cloud(ax, values, y0, colour, fill=None, height=0.62, log=False,
          box_h=0.185, pt_gap=0.055, pt_h=0.40, ms=2.6, seed=0,
          pt_colours=None, pt_alpha=0.85, dens_alpha=0.55, dens_lw=0.9,
          box_lw=0.8, med_lw=2.0, box_ink=None, pad_bw=3.0, zorder=6,
          points=True, grid_n=512, bw_floor=0.0, density=True, marker="o"):
    import numpy as np
    from matplotlib.patches import Rectangle

    v = np.asarray(list(values), dtype=float)
    keep = np.isfinite(v)
    v = v[keep]
    if pt_colours is not None:
        pt_colours = [c for c, k in zip(pt_colours, keep) if k]
    if len(v) == 0:
        return None
    ink = box_ink or INK
    t = np.log10(v) if log else v
    n = len(t)
    conv = (lambda u: 10 ** u) if log else (lambda u: u)
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    iqr = q3 - q1

    if n >= 4 and density:
        sd = t.std(ddof=1)
        a = min(sd, iqr / 1.349) if iqr > 0 else sd
        bw = 0.9 * a * n ** (-0.2) if a > 0 else 1e-6
        bw = max(bw, max(t.max() - t.min(), 1e-9) / 60.0, bw_floor)
        grid = np.linspace(t.min() - pad_bw * bw, t.max() + pad_bw * bw, grid_n)
        dens = np.exp(-0.5 * ((grid[:, None] - t[None, :]) / bw) ** 2).sum(1)
        dens /= dens.max()
        x = conv(grid)
        ax.fill_between(x, y0, y0 + height * dens, facecolor=fill or colour,
                        alpha=dens_alpha if fill is None else 1.0, lw=0,
                        zorder=zorder)
        ax.plot(x, y0 + height * dens, color=colour, lw=dens_lw,
                zorder=zorder + 1, solid_joinstyle="round")

    lo = t[t >= q1 - 1.5 * iqr].min()
    hi = t[t <= q3 + 1.5 * iqr].max()
    ax.plot([conv(lo), conv(hi)], [y0, y0], color=ink, lw=box_lw,
            zorder=zorder + 2, solid_capstyle="butt")
    for w in (lo, hi):
        ax.plot([conv(w), conv(w)], [y0 - box_h * 0.32, y0 + box_h * 0.32],
                color=ink, lw=box_lw, zorder=zorder + 2)
    ax.add_patch(Rectangle((conv(q1), y0 - box_h / 2), conv(q3) - conv(q1),
                           box_h, facecolor="white", edgecolor=ink, lw=box_lw,
                           zorder=zorder + 3))
    ax.plot([conv(med), conv(med)], [y0 - box_h / 2, y0 + box_h / 2], color=ink,
            lw=med_lw, zorder=zorder + 4, solid_capstyle="butt")

    if points:
        rng = np.random.default_rng(seed)
        yy = y0 - box_h / 2 - pt_gap - rng.uniform(0.0, pt_h, n)
        if pt_colours is None:
            ax.plot(v, yy, marker, ms=ms, mfc=colour, mec="none", alpha=pt_alpha,
                    ls="none", zorder=zorder + 1)
        else:
            ax.scatter(v, yy, s=ms ** 2, c=list(pt_colours), marker=marker,
                       linewidths=0, alpha=pt_alpha, zorder=zorder + 1)
    return {"n": int(n), "median": float(conv(med)), "q1": float(conv(q1)),
            "q3": float(conv(q3)), "min": float(v.min()), "max": float(v.max()),
            "whisker_lo": float(conv(lo)), "whisker_hi": float(conv(hi))}


def logistic_fit(x, p, log=True):
    import numpy as np
    from scipy.optimize import curve_fit

    x = np.asarray(x, float)
    p = np.asarray(p, float)
    m = np.isfinite(x) & np.isfinite(p) & (x > 0 if log else True)
    u, p = (np.log10(x[m]) if log else x[m]), p[m]

    def f(u, u0, k):
        return 1.0 / (1.0 + np.exp(-np.clip(k * (u - u0), -60, 60)))

    k0 = 6.0 / max(u.max() - u.min(), 1e-6)
    try:
        (u0, k), _ = curve_fit(f, u, p, p0=[np.median(u), k0], maxfev=20000)
    except Exception:
        u0, k = float(np.median(u)), k0
    return float(u0), float(k), (lambda xs: f(np.log10(xs) if log else xs, u0, k))


def vbox(ax, values, x0, colour, width=0.20, fill="white", box_lw=0.8, med_lw=2.0,
         cap=0.55, points=True, ms=1.9, jitter=0.075, pt_dx=0.0, pt_alpha=0.65,
         seed=0, zorder=6, log=False):
    import numpy as np
    from matplotlib.patches import Rectangle

    v = np.asarray(list(values), dtype=float)
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
    if len(v) == 0:
        return None
    t = np.log10(v) if log else v
    conv = (lambda u: 10 ** u) if log else (lambda u: u)
    q1, med, q3 = np.percentile(t, [25, 50, 75])
    iqr = q3 - q1
    lo = t[t >= q1 - 1.5 * iqr].min()
    hi = t[t <= q3 + 1.5 * iqr].max()

    ax.plot([x0, x0], [conv(lo), conv(hi)], color=colour, lw=box_lw,
            zorder=zorder + 1, solid_capstyle="butt")
    for w in (lo, hi):
        ax.plot([x0 - width * cap / 2, x0 + width * cap / 2], [conv(w), conv(w)],
                color=colour, lw=box_lw, zorder=zorder + 1)
    ax.add_patch(Rectangle((x0 - width / 2, conv(q1)), width, conv(q3) - conv(q1),
                           facecolor=fill, edgecolor=colour, lw=box_lw,
                           zorder=zorder + 2))
    ax.plot([x0 - width / 2, x0 + width / 2], [conv(med), conv(med)], color=colour,
            lw=med_lw, zorder=zorder + 3, solid_capstyle="butt")

    if points:
        rng = np.random.default_rng(seed)
        xx = x0 + pt_dx + rng.uniform(-jitter, jitter, len(v))
        ax.plot(xx, v, "o", ms=ms, mfc=colour, mec="none", alpha=pt_alpha, ls="none",
                zorder=zorder)
    return {"n": int(len(v)), "median": float(conv(med)), "q1": float(conv(q1)),
            "q3": float(conv(q3)), "min": float(v.min()), "max": float(v.max()),
            "whisker_lo": float(conv(lo)), "whisker_hi": float(conv(hi))}
