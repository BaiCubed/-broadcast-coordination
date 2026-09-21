from __future__ import annotations

import argparse
from pathlib import Path

from ..run_experiment import run_default
from ...figures.plot_figures import plot_all


def main() -> None:
    parser = argparse.ArgumentParser(description="IEEE 33 real-snapshot experiment")
    parser.add_argument("--config", default=None)
    parser.add_argument("--mode", choices=["weak_correlation", "network_stress"], default="network_stress")
    parser.add_argument("--results-root", default=None)
    args = parser.parse_args()
    default_root = Path("results/ieee33_real_snapshot_simulation") / args.mode
    result_root = run_default(
        args.config,
        mode=args.mode,
        results_root=Path(args.results_root) if args.results_root else default_root,
        snapshot_protocol=True,
    )
    for figure in plot_all(result_root):
        print(figure)


if __name__ == "__main__":
    main()
