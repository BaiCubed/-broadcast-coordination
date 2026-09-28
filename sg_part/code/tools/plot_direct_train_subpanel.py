#!/usr/bin/env python3
"""用直接训练 EPS 数据重绘 E22 子图到 direct_train。"""

from pathlib import Path

from tools import generate_e22_subpanel_figures as figures


ROOT = Path(__file__).resolve().parents[1]
figures.OUT = ROOT / "outputs/figs/direct_train"
figures.DIRECT_MODE = True
figures.DIRECT_PAIRWISE = ROOT / "results/E21/pairwise_curtailment/direct_training/data/pairwise_eps_direct_by_seed.csv"


def main() -> None:
    figures.OUT.mkdir(parents=True, exist_ok=True)
    figures.make_acceptance()
    figures.make_constraint_audit()
    figures.make_effect_safety()
    figures.make_spatial_retention()
    figures.make_topology_boxpoint()
    figures.make_dataset_retention()
    figures.make_stress_lines()
    figures.make_physical_boundary()
    figures.write_readme()
    print(f"已生成直接训练子图：{figures.OUT}")


if __name__ == "__main__":
    main()
