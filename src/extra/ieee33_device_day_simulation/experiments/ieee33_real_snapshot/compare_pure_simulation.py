from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[5]
PURE_FIGURES = ROOT / "results" / "experiments" / "Figs"

FIGURE_FILES = {
    "fig2": "fig2_scaling_predictability.png",
    "fig3": "fig3_scaling_heterogeneity.png",
    "fig4": "fig4_robustness_generalization.png",
    "fig4_self_consumption": "fig4_self_consumption.png",
    "fig5": "fig5_correlation_effects.png",
}

PANEL_BOXES = {
    "fig2": {
        "A": (0.00, 0.00, 0.50, 0.50),
        "B": (0.50, 0.00, 1.00, 0.50),
        "C": (0.00, 0.50, 0.50, 1.00),
        "D": (0.50, 0.50, 1.00, 1.00),
    },
    "fig3": {
        "A": (0.00, 0.00, 1 / 3, 1.00),
        "B": (1 / 3, 0.00, 2 / 3, 1.00),
        "C": (2 / 3, 0.00, 1.00, 1.00),
    },
    "fig4": {
        "A": (0.00, 0.00, 1.00, 0.245),
        "B": (0.00, 0.235, 0.50, 0.480),
        "C": (0.50, 0.235, 1.00, 0.480),
        "D": (0.00, 0.485, 0.50, 0.745),
        "E": (0.50, 0.485, 1.00, 0.745),
        "F": (0.00, 0.750, 0.50, 1.00),
        "G": (0.50, 0.750, 1.00, 1.00),
    },
    "fig4_self_consumption": {
        "A": (0.00, 0.00, 1.00, 0.245),
        "B": (0.00, 0.235, 0.50, 0.480),
        "C": (0.50, 0.235, 1.00, 0.480),
        "D": (0.00, 0.485, 0.50, 0.745),
        "E": (0.50, 0.485, 1.00, 0.745),
        "F": (0.00, 0.750, 0.50, 1.00),
        "G": (0.50, 0.750, 1.00, 1.00),
    },
    "fig5": {
        "A": (0.00, 0.00, 1 / 3, 1.00),
        "B": (1 / 3, 0.00, 2 / 3, 1.00),
        "C": (2 / 3, 0.00, 1.00, 1.00),
    },
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _crop(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = box
    return image.crop(
        (
            round(width * left),
            round(height * top),
            round(width * right),
            round(height * bottom),
        )
    )


def _comparison(
    pure: Image.Image,
    snapshot: Image.Image,
    *,
    figure_label: str,
    panel: str,
    mode: str,
) -> Image.Image:
    pure_width, pure_height = pure.size
    snapshot = snapshot.resize((pure_width, pure_height), Image.Resampling.LANCZOS)
    gap = 18
    header = 96
    canvas = Image.new("RGB", (pure_width * 2 + gap, pure_height + header), "white")
    canvas.paste(pure.convert("RGB"), (0, header))
    canvas.paste(snapshot.convert("RGB"), (pure_width + gap, header))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(24)
    source_font = _font(22)
    draw.text((16, 12), f"{figure_label}{panel} | pure simulation vs real snapshot ({mode})",
              fill="#222222", font=title_font)
    draw.text((pure_width // 2, header - 20), "Pure simulation", anchor="mm",
              fill="#333333", font=source_font)
    draw.text((pure_width + gap + pure_width // 2, header - 20), "Real snapshot",
              anchor="mm", fill="#333333", font=source_font)
    return canvas


def _full_comparison(pure: Image.Image, snapshot: Image.Image, *, label: str, mode: str) -> Image.Image:
    width, height = pure.size
    snapshot = snapshot.resize((width, height), Image.Resampling.LANCZOS)
    gap = 18
    header = 96
    canvas = Image.new("RGB", (width * 2 + gap, height + header), "white")
    canvas.paste(pure.convert("RGB"), (0, header))
    canvas.paste(snapshot.convert("RGB"), (width + gap, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 12), f"{label} | pure simulation vs real snapshot ({mode})",
              fill="#222222", font=_font(24))
    draw.text((width // 2, header - 20), "Pure simulation", anchor="mm",
              fill="#333333", font=_font(22))
    draw.text((width + gap + width // 2, header - 20), "Real snapshot",
              anchor="mm", fill="#333333", font=_font(22))
    return canvas


def generate_comparisons(snapshot_figures: Path, output_dir: Path, mode: str) -> list[Path]:

    panels_dir = output_dir / "panels"
    figures_dir = output_dir / "figures"
    panels_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for figure_key, filename in FIGURE_FILES.items():
        pure_path = PURE_FIGURES / filename
        snapshot_path = snapshot_figures / filename
        if not pure_path.exists() or not snapshot_path.exists():
            raise FileNotFoundError(f"Missing comparison input: {pure_path} or {snapshot_path}")
        pure_image = Image.open(pure_path).convert("RGB")
        snapshot_image = Image.open(snapshot_path).convert("RGB")

        full_path = figures_dir / f"{figure_key}_pure_vs_snapshot.png"
        _full_comparison(pure_image, snapshot_image, label=figure_key, mode=mode).save(full_path, dpi=(180, 180))
        generated.append(full_path)

        for panel, box in PANEL_BOXES[figure_key].items():
            comparison = _comparison(
                _crop(pure_image, box),
                _crop(snapshot_image, box),
                figure_label=figure_key,
                panel=panel,
                mode=mode,
            )
            panel_path = panels_dir / f"{figure_key}_{panel}_pure_vs_snapshot.png"
            comparison.save(panel_path, dpi=(220, 220))
            generated.append(panel_path)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Create panel-level comparisons between pure simulation and real snapshots.")
    parser.add_argument(
        "--snapshot-root",
        default="results/ieee33_real_snapshot_simulation",
        help="Root containing weak_correlation and network_stress results",
    )
    parser.add_argument(
        "--mode",
        choices=("weak_correlation", "network_stress", "all"),
        default="all",
    )
    args = parser.parse_args()
    root = Path(args.snapshot_root)
    modes = ("weak_correlation", "network_stress") if args.mode == "all" else (args.mode,)
    for mode in modes:
        result_dir = root / mode
        generated = generate_comparisons(
            result_dir / "Figs",
            result_dir / "comparison_pure_simulation",
            mode,
        )
        print(f"{mode}: generated {len(generated)} comparison images")


if __name__ == "__main__":
    main()
