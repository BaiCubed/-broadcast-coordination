from __future__ import annotations

from pathlib import Path

from src.extra.paper_figures import make_figure4, make_figure4_self_consumption


def rewrite_data2_figure4(result_root: Path) -> list[Path]:
    legacy_root = result_root / "legacy_compat" / "result1_scaling_law_3000dev"
    output_dir = result_root / "Figs"
    return [
        make_figure4(
            legacy_root,
            output_dir,
            data2_charging_mode=True,
            data2_root=result_root,
        ),
        make_figure4_self_consumption(
            legacy_root,
            output_dir,
            data2_charging_mode=True,
            data2_root=result_root,
        ),
    ]
